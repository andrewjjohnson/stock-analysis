"""Intraday momentum: does the morning's move predict the last half hour? A signal study on minute bars, no P&L.

  uv run --env-file .env python intraday_momentum.py --out output/intraday_momentum                # design period
  uv run --env-file .env python intraday_momentum.py --final-test --out output/intraday_momentum   # adds the holdout

Published for SPY (Gao, Han, Li and Zhou, "Market intraday momentum", 2018): the return from the previous close
to 10:00 predicts the return over the last half hour, and the 15:00-15:30 return adds to it.

Fixed before any result was seen:
- Tickers: SPY (primary), QQQ, IWM.
- Signals, known at 15:30 ET (30 minutes before the close on early-close days too): the sign of the move from
  the previous close to 10:00 (primary; on an ex-dividend morning the payout is taken out of the previous
  close); the sign of the move over the half hour before (15:00-15:30); and both agreeing (no trade otherwise).
- Outcome: the move from 15:30 to the close, in basis points, in the signal's direction (long after an up
  morning, short after a down one).
- Volatile mornings: |move to 10:00| in the ticker's top third of the design period, against the rest.
- Design: sessions to 2024-12-31. Primary result: SPY, morning signal, average outcome > 0 (one observation a
  day). The holdout (2025-01-01 on) is read only with --final-test.
- An illustrative 1 bp round-trip cost is shown next to the averages.

Prices are Massive split-adjusted minute bars: the price at a time is the close of the minute bar ending then
(the close is the last regular minute, near but not exactly the closing auction). A day with a missing needed
minute is left out.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import outcomes  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
NS = outcomes.NS_PER_MINUTE
TICKERS = ("SPY", "QQQ", "IWM")
HOLDOUT_START = pd.Timestamp("2025-01-01")
COST_BPS = 1.0
SIGNALS = {"Morning (previous close to 10:00)": "morning", "Half hour before (15:00-15:30)": "late",
           "Both agree": "both"}
PRIMARY = ("SPY", "Morning (previous close to 10:00)")


def price_at(rth, times):
    """Close of the minute bar that ends at each time (it starts one minute earlier); NaN if that bar is missing."""
    t = outcomes._ns(rth["ts"])
    c = rth["close"].to_numpy(float)
    start = outcomes._ns(times) - NS
    k = np.searchsorted(t, start)
    ok = (k < len(t)) & (t[np.minimum(k, len(t) - 1)] == start)
    out = np.full(len(start), np.nan)
    out[ok] = c[k[ok]]
    return out


def day_table(rth, sessions, dividends):
    """One row per session: the morning move (previous close, net of a dividend paid that morning, to 10:00), the
    15:00-15:30 move (half hours before the close) and the last half hour's move, in bps."""
    d = pd.DataFrame(index=sessions.index)
    d["p_10"] = price_at(rth, sessions["open"] + pd.Timedelta(minutes=30))
    d["p_m60"] = price_at(rth, sessions["close"] - pd.Timedelta(minutes=60))
    d["p_m30"] = price_at(rth, sessions["close"] - pd.Timedelta(minutes=30))
    d["p_close"] = price_at(rth, sessions["close"])
    paid = dividends.groupby("ex_date")["cash_amount"].sum() if len(dividends) else pd.Series(dtype=float)
    prev = d["p_close"].shift(1) - paid.reindex(d.index).fillna(0.0).to_numpy()
    d["r_morning"] = (d["p_10"] / prev - 1) * 1e4
    d["r_late"] = (d["p_m30"] / d["p_m60"] - 1) * 1e4
    d["r_last"] = (d["p_close"] / d["p_m30"] - 1) * 1e4
    return d


def signal_outcomes(d, kind):
    """Signed last-half-hour move for one signal: +1 x move after an up signal, -1 x after a down one; NaN on days
    without a trade (a zero move, disagreement for "both", or missing data)."""
    s1, s2 = np.sign(d["r_morning"]), np.sign(d["r_late"])
    s = s1 if kind == "morning" else s2 if kind == "late" else s1.where(s1 == s2, 0)
    return (s * d["r_last"]).where((s != 0) & d["r_last"].notna())


def stats(x):
    """Days traded, average (bps) with a t-test, hit rate, and the total."""
    x = x.dropna()
    n = len(x)
    out = {"days": n}
    if n < 3:
        return out
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t = m / se if se else np.nan
    lo, hi = alerts.wilson(int((x > 0).sum()), n)
    out.update(avg=m, se=se, lo=m - 1.96 * se, hi=m + 1.96 * se, t=t, p=2 * norm.sf(abs(t)), p_up=norm.sf(t),
               hit=(x > 0).mean() * 100, hit_lo=lo, hit_hi=hi, total=x.sum(), net=m - COST_BPS)
    return out


def slope(d):
    """OLS slope of the last half hour's move on the morning move, with its t-statistic."""
    v = d[["r_morning", "r_last"]].dropna()
    if len(v) < 10:
        return np.nan, np.nan
    x, y = v["r_morning"].to_numpy(), v["r_last"].to_numpy()
    x0 = x - x.mean()
    b = (x0 * (y - y.mean())).sum() / (x0**2).sum()
    resid = y - y.mean() - b * x0
    se = np.sqrt((resid**2).sum() / (len(v) - 2) / (x0**2).sum())
    return b, b / se


def run_study(tables, *, final=False, timings=None):
    """tables: {ticker: day_table}. No file I/O."""
    timings = {} if timings is None else timings
    rows, outcomes_by = [], {}
    with timed(timings, "statistics"):
        for tk, d in tables.items():
            d = d[d.index > d.index[0]]  # the first session has no previous close
            design = d.index < HOLDOUT_START
            cut = d.loc[design, "r_morning"].abs().quantile(2 / 3)
            volatile = d["r_morning"].abs() >= cut
            for name, kind in SIGNALS.items():
                x = signal_outcomes(d, kind)
                outcomes_by[tk, name] = x
                for period, keep in (("design", design), ("holdout", ~design)):
                    if period == "holdout" and not final:
                        continue
                    for part, mask in (("all days", np.ones(len(d), bool)), ("volatile mornings", volatile),
                                       ("calmer mornings", ~volatile)):
                        b, bt = slope(d[keep & mask]) if kind == "morning" else (np.nan, np.nan)
                        rows.append({"ticker": tk, "signal": name, "period": period, "part": part,
                                     "volatile_cut": cut, "slope": b, "slope_t": bt,
                                     **stats(x[keep & mask])})
        table = pd.DataFrame(rows)
    return {"stats": table, "outcomes": outcomes_by, "tables": tables, "final": final}


# ---------------------------------------------------------------- report

def bps(v, sign=True, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def pick(s, ticker, signal, period="design", part="all days"):
    m = s[(s["ticker"] == ticker) & (s["signal"] == signal) & (s["period"] == period) & (s["part"] == part)]
    return m.iloc[0] if len(m) else None


def render_report(res):
    s = res["stats"]
    lines = []
    w = lines.append
    w("# Intraday momentum: does the morning predict the last half hour?\n")
    w("At 15:30 ET, go long if the signal was up and short if it was down, and measure the move to the close in basis "
      "points (1 bp = 0.01%) in that direction. Signals: the move from the previous close to 10:00 (*morning*, the "
      "published one), the move from 15:00 to 15:30 (*half hour before*), or both agreeing. A signal study: no fills "
      f"or spreads beyond the illustrative {COST_BPS:g} bp round trip shown as *after cost*. Design: sessions to "
      "2024-12-31" + ("; holdout 2025-01-02 on." if res["final"] else " (the 2025+ holdout is not read).") + "\n")
    r = pick(s, *PRIMARY)
    if r is not None:
        w(f"**Primary (SPY, morning signal, design period):** {int(r['days'])} days, average {bps(r['avg'])} bps "
          f"({bps(r['lo'])} to {bps(r['hi'])}), t = {r['t']:.2f}, right direction {r['hit']:.0f}% of days; "
          f"{bps(r['net'])} bps after the cost.")
        if res["final"]:
            h = pick(s, *PRIMARY, period="holdout")
            if h is not None:
                w(f" **Holdout:** {int(h['days'])} days, {bps(h['avg'])} bps ({bps(h['lo'])} to {bps(h['hi'])}), "
                  f"t = {h['t']:.2f}, one-sided p = {h['p_up']:.3f}.")
        w("\n")
    for period in (("design", "holdout") if res["final"] else ("design",)):
        w(f"## {'Design period' if period == 'design' else 'Holdout'}\n")
        rows = []
        for tk in res["tables"]:
            for name in SIGNALS:
                for part in ("all days", "volatile mornings", "calmer mornings"):
                    x = pick(s, tk, name, period, part)
                    if x is None or not x.get("days"):
                        continue
                    rows.append([tk, name, part, f"{int(x['days'])}", f"{bps(x['avg'])} ({bps(x['lo'])} to {bps(x['hi'])})",
                                 f"{x['t']:.2f}", f"{x['hit']:.0f}%", bps(x["net"]),
                                 "" if pd.isna(x["slope"]) else f"{x['slope']:+.3f} (t {x['slope_t']:.1f})"])
        w(alerts.md_table(["Ticker", "Signal", "Days", "Traded", "Avg bps (95%)", "t", "Right direction",
                           "After cost", "Slope on the morning move"], [[r[0], r[1], r[2], r[3], *r[4:]] for r in rows])
          + "\n")
    w("## SPY by year (morning signal)\n")
    x = res["outcomes"][PRIMARY]
    rows = []
    for year, g in x.dropna().groupby(x.dropna().index.year):
        if not res["final"] and year >= HOLDOUT_START.year:
            continue
        st = stats(g)
        rows.append([str(year), f"{st['days']}", bps(st.get("avg")), f"{st.get('hit', np.nan):.0f}%",
                     bps(st.get("total"), digits=0)])
    w(alerts.md_table(["Year", "Days", "Avg bps", "Right direction", "Sum of bps"], rows) + "\n")
    w("*Volatile mornings*: the morning move's size in the ticker's top third of the design period. *Slope*: how "
      "many bps the last half hour moved per bp of morning move (positive = momentum).\n")
    return "\n".join(lines) + "\n"


def plot(res, path):
    fig = plt.figure(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.1, 0.2, 0.86, 0.6])
    colors = {"SPY": SERIES[0], "QQQ": SERIES[1], "IWM": INK_2}
    for tk in res["tables"]:
        x = res["outcomes"][tk, PRIMARY[1]].dropna()
        if not res["final"]:
            x = x[x.index < HOLDOUT_START]
        ax.plot(x.index, x.cumsum(), color=colors.get(tk, MUTED), lw=1.6, label=f"{tk} ({len(x)} days)")
    if res["final"]:
        ax.axvline(HOLDOUT_START, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.axhline(0, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Cumulative bps (signal study)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "Morning direction, traded from 15:30 to the close", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Long after an up morning (previous close to 10:00), short after a down one; before costs."
             + (" Dashed: start of the holdout." if res["final"] else ""), color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Intraday momentum: the morning move vs the last half hour.")
    p.add_argument("--out", default="output/intraday_momentum")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read the 2025+ holdout (run once)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    with timed(timings, "data"):
        tables = {}
        for tk in TICKERS:
            minutes, _ = gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)
            rth, _ = features.regular_session_minutes(minutes, sessions)
            tables[tk] = day_table(rth, sessions, mr.load_dividends(tk, first, last, args.cache_dir, args.refresh))
    res = run_study(tables, final=args.final_test, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(out / "stats.csv", index=False)
    pd.concat({tk: d for tk, d in tables.items()}, names=["ticker", "session"]).to_csv(out / "days.csv")
    plot(res, out / "momentum.png")
    (out / "report.md").write_text(render_report(res))
    r = pick(res["stats"], *PRIMARY)
    print(f"SPY morning signal, design: {int(r['days'])} days, {r['avg']:+.2f} bps, t = {r['t']:.2f}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
