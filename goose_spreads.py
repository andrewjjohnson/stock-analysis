"""Green Goose as $1 credit spreads: at 15:50 sell a $1 vertical in Green Goose's direction (bullish: a put spread,
bearish: a call spread) instead of buying an at-the-money option. SPY, the option data plan's two years.

  uv run --env-file .env python goose_spreads.py --out output/goose_spreads

Fixed before any result was seen:
- Direction: Green Goose version 1 at 15:50 (green_goose.daily_signals, ADX/DMI 5).
- Spread: $1 wide, expiring the next session. Placement (alert_spreads.spread_legs): offset 0 = the user's (bullish:
  sell the put at the strike just above the 15:50 price, buy the one $1 below; bearish: sell the call just below, buy
  the one $1 above); offsets -1 and -2 move both legs $1 and $2 further out of the money (for reading).
- Prices: the opens of the first minute within 5 minutes of the time in which both legs traded
  (alert_spreads.spread_path); failing that, each leg's first open in those 5 minutes.
- Entry: 15:50; the credit must be between 0 and $1.
- Exits: buy back at 9:35 the next morning; the user's 0DTE rule (take profit at 80% of the credit on any minute close
  after the entry, else buy back at 15:30); or hold to expiry, worth the strikes' distance from SPY's closing price,
  capped at $0-$1, with no closing fills. A trade counts only when its entry, 9:35 and 15:30 prices exist, so every
  exit uses the same trades.
- Costs: slippage $0.02 a share per leg per fill (primary; $0, $0.01 and $0.03 shown); no commission.
- Baselines on the same sessions: sell a put spread every night, a call spread every night, and the opposite of
  Green Goose.
- Tests (offset 0, $0.02): Green Goose's mean P&L per spread > 0 for each exit, one-sided, BH q < 0.10 across the
  three.
"""

import argparse
import math
import warnings
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
import green_goose as gg  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
NS_MIN = asp.NS_MIN
WIDTH, SHARES = asp.WIDTH, asp.SHARES
OFFSETS = (0, -1, -2)
PRIMARY_OFFSET = 0
SLIPS = (0.0, 0.01, 0.02, 0.03)
PRIMARY_SLIP = 0.02
WINDOW = 5                       # minutes to find a price after each time
MORNING = 5                      # the morning exit: minutes after the open (9:35)
TAKE_PROFIT = 80                 # % of the credit, the user's 0DTE rule
LATE_EXIT = "15:30"
EXITS = ("9:35", "80% take profit, else 15:30", "hold to expiry")
RULES = ("Green Goose", "against Green Goose", "always put spreads", "always call spreads")
FDR = 0.10
BOOT = 5000


# ---------------------------------------------------------------- one spread

def value_at(short, long, at_ns, window=WINDOW):
    """(spread value, minute, how): short minus long at the opens of the first minute from at_ns, within `window`
    minutes, in which both legs traded; failing that, each leg's first open in the window. NaN if a leg didn't trade."""
    both, v_open, _ = asp.spread_path(short, long, at_ns, at_ns + window * NS_MIN)
    if len(both):
        return float(v_open[0]), int(both[0]), "both legs"
    a, b = gg.price_from(short, at_ns, look=window - 1), gg.price_from(long, at_ns, look=window - 1)
    return (a - b, at_ns, "legs apart") if np.isfinite(a) and np.isfinite(b) else (np.nan, at_ns, "no price")


def settle(right, short_k, close):
    """A $1 credit spread's value at expiry from the underlying's closing price."""
    return float(np.clip(short_k - close if right == "P" else close - short_k, 0.0, WIDTH))


def spread_trade(day, nxt, side, spot, sessions, load, close_next, offset=0):
    """One $1 spread sold at 15:50 on `day`, expiring `nxt` (the next session): the credit, the 9:35 value, the minute
    closes to watch for a take profit (after the entry through 15:30 the next day), the 15:30 value and the value at
    expiry. Prices before slippage."""
    right, ks, kl = asp.spread_legs(side, spot, offset)
    tks = asp.option_ticker(nxt, right, ks), asp.option_ticker(nxt, right, kl)
    row = {"session": day, "side": side, "offset": offset, "right": right, "short_strike": ks, "long_strike": kl,
           "spot": spot}
    try:
        s1, l1 = (load(tk, f"{day:%Y-%m-%d}") for tk in tks)
        s2, l2 = (load(tk, f"{nxt:%Y-%m-%d}") for tk in tks)
    except asp.NotInPlan:
        return {**row, "status": "outside the data plan"}
    c1 = sessions.loc[day, "close"].value
    credit, entry_ns, entry_how = value_at(s1, l1, c1 - gg.DECISION.value)
    if not 0 < credit < WIDTH:
        return {**row, "status": "no entry price" if np.isnan(credit) else "credit outside $0-$1", "credit": credit}
    o2, c2 = sessions.loc[nxt, "open"].value, sessions.loc[nxt, "close"].value
    late = min(pd.Timestamp(f"{nxt:%Y-%m-%d} {LATE_EXIT}", tz=NY).value, c2 - asp.EXIT_BEFORE_CLOSE_MIN * NS_MIN)
    v935, _, how935 = value_at(s2, l2, o2 + MORNING * NS_MIN)
    v_late, _, how_late = value_at(s2, l2, late)
    watch = np.r_[asp.spread_path(s1, l1, entry_ns + NS_MIN, c1)[2], asp.spread_path(s2, l2, o2, late)[2]]
    row.update(credit=credit, entry_how=entry_how, v935=v935, how935=how935, v_late=v_late, how_late=how_late,
               watch=watch, close_next=close_next,
               expiry_value=settle(right, ks, close_next) if np.isfinite(close_next) else np.nan)
    ok = np.isfinite(v935) and np.isfinite(v_late) and np.isfinite(row["expiry_value"])
    return {**row, "status": "ok" if ok else "no exit price"}


def trade_pnl(t, exit_name, slip):
    """P&L in $ per spread for one exit at `slip` a share per leg per fill."""
    if exit_name == "hold to expiry":
        return (t["credit"] - 2 * slip - t["expiry_value"]) * SHARES
    if exit_name == "9:35":
        trade, tp = {"credit_traded": t["credit"], "watch": np.array([]), "exit_value": t["v935"]}, None
    else:
        trade, tp = {"credit_traded": t["credit"], "watch": t["watch"], "exit_value": t["v_late"]}, TAKE_PROFIT
    return asp.simulate(trade, tp, None, slip, 0.0)[0]


# ---------------------------------------------------------------- statistics

def spread_stats(pnl, risk, years):
    """pnl, risk: $ per spread (risk = the most a spread could lose)."""
    x = np.asarray(pnl, float)
    n = len(x)
    out = {"trades": n}
    if n < 10:
        return out
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    win, loss = x[x > 0], x[x <= 0]
    cum = np.cumsum(x)
    return {**out, "win": len(win) / n * 100, "avg_win": win.mean() if len(win) else np.nan,
            "avg_loss": loss.mean() if len(loss) else np.nan, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(m / se) if se else np.nan, "on_risk": m / np.mean(risk) * 100, "risk": np.mean(risk),
            "per_year": x.sum() / years, "worst": x.min(),
            "drawdown": float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()),
            "profit_factor": win.sum() / -loss.sum() if loss.sum() < 0 else np.nan}


def choose(trades, direction, rule):
    """The ok trades a rule takes: Green Goose's side (+1 put spread, -1 call spread), its opposite, or always one."""
    ok = trades[trades["status"] == "ok"]
    d = direction.reindex(ok["session"]).fillna(0).to_numpy()
    want = {"Green Goose": d, "against Green Goose": -d, "always put spreads": np.ones(len(ok)),
            "always call spreads": -np.ones(len(ok))}[rule]
    return ok[(want != 0) & (np.sign(want) == ok["side"].to_numpy())]


def diff_interval(a, b, reps=BOOT, seed=0):
    """95% interval of mean(a) - mean(b), resampling sessions (a, b: P&L indexed by session)."""
    days = a.index.union(b.index)
    A, B = a.reindex(days).to_numpy(float), b.reindex(days).to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(days), (reps, len(days)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a resample can miss every trade of one rule
        d = np.nanmean(A[idx], axis=1) - np.nanmean(B[idx], axis=1)
    return np.nanpercentile(d, [2.5, 97.5])


# ---------------------------------------------------------------- study

def run_study(minutes, dividends, sessions, load, *, timings=None):
    """SPY minute bars and dividends; load = the option loader. No file I/O beyond the loader's cache."""
    timings = {} if timings is None else timings
    with timed(timings, "signals"):
        raw, _ = mr.daily_table(minutes, sessions)
        adj, _ = mr.adjust_dividends(raw, dividends)
        direction = gg.daily_signals(adj, gg.VERSIONS["version 1"])["direction"]
    spot, close = raw["snap"], raw["close"]
    idx = sessions.index
    days = [(d, idx[i + 1]) for i, d in enumerate(idx[:-1]) if d >= gg.OPTIONS_FROM and pd.notna(spot.get(d))]
    plan = [(d, n, side, off) for d, n in days for side in (1, -1) for off in OFFSETS]
    if hasattr(load, "state"):
        with timed(timings, "option downloads"):
            reqs = set()
            for d, n, side, off in plan:
                right, ks, kl = asp.spread_legs(side, spot[d], off)
                for k in (ks, kl):
                    reqs |= {(asp.option_ticker(n, right, k), f"{d:%Y-%m-%d}"),
                             (asp.option_ticker(n, right, k), f"{n:%Y-%m-%d}")}
            gg._fetch(reqs, load)
    rows = []
    with timed(timings, "spreads"):
        for d, n in days:
            memo = {}

            def cached(tk, day):
                if (tk, day) not in memo:
                    memo[(tk, day)] = load(tk, day)
                return memo[(tk, day)]

            for side in (1, -1):
                for off in OFFSETS:
                    rows.append(spread_trade(d, n, side, float(spot[d]), sessions, cached, float(close.get(n, np.nan)),
                                             off))
    trades = pd.DataFrame(rows)
    ok = trades[trades["status"] == "ok"]
    years = (ok["session"].max() - ok["session"].min()).days / 365.25 if len(ok) else np.nan
    stats, pnls = [], {}
    for off in OFFSETS:
        t_off = trades[trades["offset"] == off]
        for rule in RULES:
            picked = choose(t_off, direction, rule)
            for exit_name in EXITS:
                for slip in SLIPS:
                    pnl = pd.Series([trade_pnl(t, exit_name, slip) for t in picked.to_dict("records")],
                                    index=picked["session"].to_numpy(), dtype=float)
                    risk = (WIDTH - picked["credit"].to_numpy(float) + 2 * slip) * SHARES
                    pnls[(off, rule, exit_name, slip)] = pnl
                    stats.append({"offset": off, "rule": rule, "exit": exit_name, "slip": slip,
                                  "credit": picked["credit"].mean() * SHARES if len(picked) else np.nan,
                                  **spread_stats(pnl, risk, years)})
    stats = pd.DataFrame(stats)
    stats["minus_puts_lo"], stats["minus_puts_hi"] = np.nan, np.nan
    for exit_name in EXITS:
        key = (PRIMARY_OFFSET, "Green Goose", exit_name, PRIMARY_SLIP)
        base = (PRIMARY_OFFSET, "always put spreads", exit_name, PRIMARY_SLIP)
        lo, hi = diff_interval(pnls[key], pnls[base])
        sel = ((stats["offset"] == PRIMARY_OFFSET) & (stats["rule"] == "Green Goose") & (stats["exit"] == exit_name)
               & (stats["slip"] == PRIMARY_SLIP))
        stats.loc[sel, ["minus_puts_lo", "minus_puts_hi"]] = lo, hi
    tests = []
    for exit_name in EXITS:
        r = stats[(stats["offset"] == PRIMARY_OFFSET) & (stats["rule"] == "Green Goose") & (stats["exit"] == exit_name)
                  & (stats["slip"] == PRIMARY_SLIP)].iloc[0]
        tests.append({"test": f"Green Goose $1 spreads, {exit_name}: mean per spread", "value": r["mean"],
                      "p": r["p_up"]})
    tests = pd.DataFrame(tests)
    tests["q"] = mr.bh_qvalues(tests["p"].fillna(1).to_numpy())
    entered = trades[trades["status"].isin(["ok", "no exit price"]) & (trades["offset"] == PRIMARY_OFFSET)]
    all_expiry = {}
    for rule in RULES:
        p = choose(entered.assign(status="ok"), direction, rule)
        p = p[np.isfinite(p["expiry_value"])]
        all_expiry[rule] = spread_stats([trade_pnl(t, "hold to expiry", PRIMARY_SLIP) for t in p.to_dict("records")],
                                        (WIDTH - p["credit"].to_numpy(float) + 2 * PRIMARY_SLIP) * SHARES, years)
    return {"trades": trades, "stats": stats, "pnls": pnls, "tests": tests, "years": years,
            "all_expiry": pd.DataFrame(all_expiry).T.reset_index(names="rule")}


# ---------------------------------------------------------------- report

def money(v):
    return gg.money(v)


def placement_text(off):
    return {0: "your placement (short leg at or just in the money)", -1: "$1 further out of the money",
            -2: "$2 further out of the money"}[off]


def render_report(res):
    s, trades = res["stats"], res["trades"]
    lines = []
    w = lines.append
    w("# Green Goose as $1 credit spreads\n")
    w("At 15:50 sell a $1 vertical in Green Goose's direction (version 1): a put spread when it says calls, a call "
      "spread when it says puts, expiring the next session. SPY, 2024-10 to 2026-09, per spread (100 shares), traded "
      f"prices, ${PRIMARY_SLIP:.2f} a share per leg per fill unless noted, no commission.\n")
    w("**Tests (your placement, one-sided).**\n")
    rows = [[r.test, money(r.value), f"{r.p:.3f}", f"{r.q:.3f}"] for r in res["tests"].itertuples()]
    w(alerts.md_table(["Test", "Mean per spread", "p", "q"], rows) + "\n")
    for off in OFFSETS:
        w(f"## {placement_text(off).capitalize()}\n")
        rows = []
        for r in s[(s["offset"] == off) & (s["slip"] == PRIMARY_SLIP)].itertuples():
            if pd.isna(getattr(r, "mean", np.nan)):
                continue
            vs = (f"{money(r.minus_puts_lo)} to {money(r.minus_puts_hi)}" if pd.notna(r.minus_puts_lo) else "")
            rows.append([r.exit, r.rule, f"{int(r.trades):,}", money(r.credit), f"{r.win:.0f}%",
                         f"{money(r.avg_win)} / {money(r.avg_loss)}", f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})",
                         f"{r.on_risk:+.1f}%", money(r.per_year), f"{money(r.worst)} / {money(-r.drawdown)}",
                         "n/a" if pd.isna(r.profit_factor) else f"{r.profit_factor:.2f}", vs])
        w(alerts.md_table(["Exit", "Rule", "Trades", "Avg credit", "Wins", "Avg win / loss", "Per spread (95%)",
                           "Of max risk", "A year (1 spread)", "Worst / worst run", "Profit factor",
                           "Goose minus always puts (95%)"], rows) + "\n")
    w("## Slippage (Green Goose)\n")
    rows = []
    for r in s[s["rule"] == "Green Goose"].itertuples():
        if pd.isna(getattr(r, "mean", np.nan)):
            continue
        rows.append([placement_text(r.offset), r.exit, f"${r.slip:.2f}", f"{r.win:.0f}%",
                     f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})", money(r.per_year)])
    w(alerts.md_table(["Placement", "Exit", "Slippage per leg per fill", "Wins", "Per spread (95%)",
                       "A year (1 spread)"], rows) + "\n")
    a = res["all_expiry"]
    w("## Hold to expiry on every entered spread (your placement)\n")
    w("Including spreads dropped above for a missing 9:35 or 15:30 price, to check the dropping doesn't flatter the "
      "results.\n")
    rows = [[r.rule, f"{int(r.trades):,}", f"{r.win:.0f}%", f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})"]
            for r in a.itertuples() if pd.notna(getattr(r, "mean", np.nan))]
    w(alerts.md_table(["Rule", "Trades", "Wins", "Per spread (95%)"], rows) + "\n")
    w("## Notes\n")
    for off in OFFSETS:
        t = trades[trades["offset"] == off]
        w(f"- {placement_text(off).capitalize()}: " + ", ".join(f"{k} {v:,}" for k, v in t["status"].value_counts()
                                                                   .items()) + " (spreads, both sides).")
    ok = trades[(trades["status"] == "ok") & (trades["offset"] == PRIMARY_OFFSET)]
    w(f"- Prices at your placement: entry from one minute with both legs {(ok['entry_how'] == 'both legs').mean():.0%}, "
      f"9:35 {(ok['how935'] == 'both legs').mean():.0%}, 15:30 {(ok['how_late'] == 'both legs').mean():.0%}; the rest "
      "from each leg's own first trade in the 5 minutes.")
    w("- At your placement the put spread and the call spread use the same two strikes, so selling one is close to the "
      "mirror image of selling the other (before costs): a $1 credit spread is the same position as a $1 debit spread "
      "on the same strikes.")
    w("- Hold to expiry settles at SPY's closing price with no closing fills. In practice a spread finishing between "
      "its strikes leaves an assignment to deal with.")
    w("- Option prices are trades (no quotes on the data plan); slippage stands in for the bid-ask spread.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    fig = plt.figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.2, 0.86, 0.6])
    styles = [("Green Goose", "9:35", SERIES[0], "-", 1.3), ("Green Goose", EXITS[1], SERIES[1], "-", 1.1),
              ("Green Goose", "hold to expiry", INK_2, "-", 1.0), ("always put spreads", "9:35", MUTED, (0, (3, 2)), 1.0)]
    for rule, exit_name, color, ls, lw in styles:
        pnl = res["pnls"][(PRIMARY_OFFSET, rule, exit_name, PRIMARY_SLIP)].sort_index()
        if len(pnl):
            ax.plot(pnl.index, pnl.cumsum(), color=color, ls=ls, lw=lw, label=f"{rule}, {exit_name}")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative $ per spread", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(0.02, 0.97, "Green Goose as $1 credit spreads on SPY (your strike placement)", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"One spread a trade, ${PRIMARY_SLIP:.2f} a share per leg per fill, 2024-10 to 2026-09.",
             color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Green Goose's direction as $1 SPY credit spreads sold at 15:50.")
    p.add_argument("--out", default="output/goose_spreads")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    with timed(timings, "data"):
        minutes = gr.load_minutes("SPY", sessions, args.cache_dir, args.refresh)[0]
        dividends = mr.load_dividends("SPY", first, last, args.cache_dir, args.refresh)
    load = asp.option_loader(args.cache_dir, args.refresh)
    res = run_study(minutes, dividends, sessions, load, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    res["trades"].drop(columns="watch").to_parquet(out / "trades.parquet", index=False)
    plot(res, out / "spreads.png")
    (out / "report.md").write_text(render_report(res))
    print(f"option downloads: {load.state['fetched']:,}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
