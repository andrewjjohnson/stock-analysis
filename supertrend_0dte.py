"""The Kalman SuperTrend strategy (kalman_supertrend.py) traded the way its Reddit author describes: same-day (0DTE)
at-the-money SPY options. Per contract at traded option prices plus slippage, no commissions or fees.

  uv run --env-file .env python supertrend_0dte.py --out output/supertrend_0dte

Fixed before any result was seen:
- Trades: kalman_supertrend.py's SPY trades, flat by the close (0DTE SPY options stop trading at 16:00), on every
  session the option data plan covers (a rolling two years) to 2026-09-30. The share trade's own entry and exits
  (stop, target, breakeven, wave, time, opposite flag, regular close) decide when the option is bought and sold.
- Contract: a call for a BUY, a put for a SELL, expiring that day, at the $1 strike nearest the share fill.
- Prices from the option's one-minute trade bars (the plan has no quotes). Entry: the open of the option's bar in
  the share fill minute. Exit: a stop, target or breakeven happens inside a minute, so the close of the option's
  bar in that minute; an exit decided at a bar's close (opposite flag, wave, time) fills at the next minute's
  open, so that bar's open; the regular close, the last bar's close before 16:00. A missing bar: the next one
  within 2 minutes (its open), else the last one in the 5 minutes before (its close); otherwise the trade is
  skipped and counted.
- Slippage $0.01, $0.02 (primary) and $0.05 a share per fill. P&L per contract = 100 x (exit - entry - 2 x slippage).
- Test: mean P&L per contract > 0 at $0.02 a fill (one-sided, standard errors clustered by session). The share P&L of
  the same trades is shown beside it.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alert_spreads as asp  # noqa: E402
import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import kalman_supertrend as ks  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKER = "SPY"
OPTIONS_FROM = pd.Timestamp("2024-10-01")      # the plan's rolling two years start about here
SLIPS = (0.01, 0.02, 0.05)                      # $ a share per fill
PRIMARY_SLIP = 0.02
WINDOW = pd.Timedelta(minutes=2)                # look ahead for a missing bar
STALE = pd.Timedelta(minutes=5)                 # look back for one
INSIDE = ("stop", "target", "breakeven")        # exits that happen inside a minute
MINUTE = pd.Timedelta(minutes=1)


def contract(day, d, fill):
    """Massive ticker of the 0DTE at-the-money option for a share trade: a call for longs, a put for shorts, at the
    $1 strike nearest the fill."""
    return asp.option_ticker(day, "C" if d > 0 else "P", math.floor(fill + 0.5), TICKER)


def price_at(bars, at, field):
    """The option's `field` ("open" or "close") in the bar starting at `at`; if there is none, the open of the next
    bar within WINDOW, else the close of the last bar in the STALE minutes before `at`; else None."""
    if bars is None or bars.empty:
        return None
    t = _ns(bars["ts"])
    a = pd.Timestamp(at).value
    k = int(np.searchsorted(t, a))
    if k < len(t) and t[k] == a:
        return float(bars[field].iloc[k])
    if k < len(t) and t[k] <= a + WINDOW.value:
        return float(bars["open"].iloc[k])
    return float(bars["close"].iloc[k - 1]) if k > 0 and t[k - 1] >= a - STALE.value else None


def option_trades(trades, load):
    """Entry and exit option prices (before slippage) for each share trade, with a status."""
    rows = []
    for r in trades.itertuples():
        tk = contract(r.day, r.dir, r.fill)
        try:
            bars = load(tk, f"{r.day:%Y-%m-%d}")
        except asp.NotInPlan:
            rows.append((tk, np.nan, np.nan, "outside the data plan"))
            continue
        entry = price_at(bars, r.entry_time, "open")
        field = "close" if r.reason in INSIDE or r.reason == "regular close" else "open"
        exit_ = price_at(bars, r.exit_time, field)
        if entry is None or exit_ is None:
            rows.append((tk, np.nan, np.nan, "no option trades"))
            continue
        rows.append((tk, entry, exit_, "ok"))
    o = pd.DataFrame(rows, columns=["contract", "opt_entry", "opt_exit", "opt_status"], index=trades.index)
    return trades.join(o)


def prefetch(trades, load, workers=8):
    """Downloads every contract-day the trades need, several at a time (each is cached on its first load)."""
    reqs = sorted({(contract(r.day, r.dir, r.fill), f"{r.day:%Y-%m-%d}") for r in trades.itertuples()})

    def one(req):
        try:
            load(*req)
        except asp.NotInPlan:
            pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, reqs))
    return len(reqs)


def pnl(o, slip):
    return 100 * (o["opt_exit"] - o["opt_entry"] - 2 * slip)


def stats(o, slip, years):
    """Per contract: count, win rate, mean with a session-clustered 95% interval and one-sided p, average win and
    loss, premium, % of premium, dollars a year (one contract a trade), worst trade and worst run."""
    x = pnl(o, slip).to_numpy(float)
    n = len(x)
    out = {"slip": slip, "trades": n}
    if n < 2:
        return out
    m, se = sm.cluster_mean(x, o["day"].to_numpy())
    z = m / se if se else np.nan
    prem = 100 * (o["opt_entry"].to_numpy(float) + slip)
    cum = np.cumsum(x[np.argsort(pd.DatetimeIndex(o["entry_time"]).asi8)])
    return {**out, "win": (x > 0).mean() * 100, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "avg_win": x[x > 0].mean() if (x > 0).any() else np.nan,
            "avg_loss": x[x <= 0].mean() if (x <= 0).any() else np.nan, "premium": prem.mean(),
            "pct_premium": (x / prem).mean() * 100, "per_year": x.sum() / years, "worst": x.min(),
            "drawdown": float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max())}


def run_study(minutes, sessions, load, *, timings=None):
    """Share trades (flat by the close) from OPTIONS_FROM on, priced as 0DTE options. No file I/O beyond the
    loader's cache."""
    timings = {} if timings is None else timings
    with timed(timings, "share trades"):
        bars, m = ks.five_minute_bars(minutes, sessions)
        bars = ks.indicators(bars)
        trades = ks.strategy(bars, m, flat_by_close=True)
        trades = trades[(trades["day"] >= OPTIONS_FROM) & (trades["reason"] != "end of data")].reset_index(drop=True)
    with timed(timings, "option data"):
        if hasattr(load, "state"):
            prefetch(trades, load)
    with timed(timings, "option prices"):
        o = option_trades(trades, load)
    ok = o[o["opt_status"] == "ok"]
    years = (ok["day"].max() - ok["day"].min()).days / 365.25 if len(ok) else np.nan
    st = pd.DataFrame([stats(ok, s, years) for s in SLIPS])
    share = ks.stats(ok, years) if len(ok) > 1 else {}
    by_dir = pd.DataFrame([{"side": "long (calls)" if d > 0 else "short (puts)", **stats(g, PRIMARY_SLIP, years),
                            "share_bps": g["bps"].mean()} for d, g in ok.groupby("dir")])
    hour = pd.DatetimeIndex(ok["entry_time"]).tz_convert(NY).hour
    by_hour = pd.DataFrame([{"hour": f"{h}:00", **stats(g, PRIMARY_SLIP, years), "share_bps": g["bps"].mean()}
                            for h, g in ok.groupby(hour)])
    return {"trades": o, "stats": st, "share": share, "by_dir": by_dir, "by_hour": by_hour, "years": years,
            "status": o["opt_status"].value_counts().to_dict(),
            "first": ok["day"].min() if len(ok) else None, "last": ok["day"].max() if len(ok) else None}


# ---------------------------------------------------------------- report

def money(v):
    return "n/a" if v is None or pd.isna(v) else f"{'-' if v < 0 else '+'}${abs(v):,.0f}"


def render_report(res):
    st, share = res["stats"], res["share"]
    lines = []
    w = lines.append
    w("# The Kalman SuperTrend strategy as 0DTE at-the-money SPY options\n")
    w(f"{res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}. Each of the strategy's SPY trades (flat by the close) is "
      "taken as a same-day option at the $1 strike nearest the share fill: a call for a BUY, a put for a SELL, bought "
      "and sold when the share trade enters and exits. Prices are one-minute option trades (no quotes on the data "
      "plan) plus slippage on each fill; per contract (100 shares), no commissions.\n")
    p = st[st["slip"] == PRIMARY_SLIP].iloc[0]
    w(f"**Bottom line.** {int(p['trades']):,} trades, {p['win']:.0f}% winners, {money(p['mean'])} per contract a "
      f"trade at ${PRIMARY_SLIP:.2f} a fill ({money(p['lo'])} to {money(p['hi'])}), {p['pct_premium']:+.1f}% of the "
      f"premium; one contract a trade: {money(p['per_year'])} a year. The same trades in shares: "
      f"{share.get('mean', np.nan):+.1f} bps each.\n")
    rows = [[f"${r.slip:.2f}", f"{int(r.trades):,}", f"{r.win:.0f}%", f"{money(r.avg_win)} / {money(r.avg_loss)}",
             f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})", f"{r.pct_premium:+.1f}%", money(r.premium),
             money(r.per_year), f"{money(r.worst)} / {money(-r.drawdown)}"] for r in st.itertuples()]
    w(alerts.md_table(["Slippage per fill", "Trades", "Winners", "Avg win / loss", "Per contract (95%)",
                       "% of premium", "Avg premium", "A year (1 contract a trade)", "Worst trade / worst run"],
                      rows) + "\n")
    w(f"## By side and by hour of entry (${PRIMARY_SLIP:.2f} a fill)\n")
    rows = [[r.side, f"{int(r.trades):,}", f"{r.win:.0f}%", f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})",
             f"{r.pct_premium:+.1f}%", f"{r.share_bps:+.1f}"] for r in res["by_dir"].itertuples()]
    rows += [[f"entered {r.hour}", f"{int(r.trades):,}", f"{r.win:.0f}%",
              f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})", f"{r.pct_premium:+.1f}%", f"{r.share_bps:+.1f}"]
             for r in res["by_hour"].itertuples()]
    w(alerts.md_table(["Group", "Trades", "Winners", "Per contract (95%)", "% of premium", "Shares, bps"], rows) + "\n")
    w("## Notes\n")
    w("- Status of the trades: " + ", ".join(f"{k} {v:,}" for k, v in res["status"].items()) + ".")
    w("- Option prices are trades, which print at the bid or the ask; the slippage settings stand in for the spread "
      "(SPY 0DTE at-the-money options usually quote a cent or two wide).")
    w("- Stops and targets come from the share trade (as the strategy labels them), not from the option's price.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    o = res["trades"]
    o = o[o["opt_status"] == "ok"].sort_values("entry_time")
    fig = plt.figure(figsize=(9, 4.4), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.1, 0.18, 0.85, 0.62])
    for slip, color in zip(SLIPS, (SERIES[0], INK_2, SERIES[1])):
        ax.plot(o["day"], pnl(o, slip).cumsum(), color=color, lw=1.2, label=f"${slip:.2f} a fill")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative $, one contract a trade", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(0.02, 0.97, "Kalman SuperTrend as 0DTE at-the-money SPY options", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Every strategy trade as a same-day call (BUY) or put (SELL), entered and exited with the "
             "share signal; traded option prices plus slippage.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="The Kalman SuperTrend strategy as 0DTE at-the-money SPY options.")
    p.add_argument("--out", default="output/supertrend_0dte")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        minutes, _ = gr.load_minutes(TICKER, sessions, args.cache_dir, args.refresh)
    load = asp.option_loader(args.cache_dir, args.refresh)
    res = run_study(minutes, sessions, load, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    plot(res, out / "options.png")
    (out / "report.md").write_text(render_report(res))
    print(f"option downloads: {load.state['fetched']}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
