"""Breakout trades at widely watched levels: buy when price breaks above yesterday's or last week's high, or trade
through yesterday's close, with a stop, a target and costs. A trade simulation on Massive minute bars: gross of
commissions, per share of the underlying, not options P&L.

  uv run --env-file .env python breakouts.py --out output/breakouts                  # design period
  uv run --env-file .env python breakouts.py --final-test --out output/breakouts     # adds the holdout, once

A follow-up to levels.py's measurements of price running through levels (last week's high, yesterday's close,
yesterday's high): this asks whether trading through them pays once fills and costs are counted.

Fixed before any result was seen:
- Tickers TSLA, SPY, QQQ, IWM. Levels and their side come from levels.py (`level_table`, `real_levels`): a level
  above the session's open is resistance, below it support; levels within 0.05 ATR of the open are skipped; ATR =
  daily ATR(14) through yesterday.
- Rules, trading in the direction through the level: yesterday's high and last week's high when the session opens
  below them (long); yesterday's close from either side (long when the open is below it, short when above). The
  mirror breakdowns (short below yesterday's low and last week's low when the session opens above them) are
  reported as a check.
- Entry: a stop order at the level at its first touch, from 9:30, starting at least 30 minutes before the close.
  It fills at the level, or at the touching minute's open if that is already through it.
- Exits, measured from the level: (primary) a stop 0.1 ATR back through the level and a target 0.1 ATR beyond it;
  (b) both 0.25 ATR; (c) the 0.1 ATR stop and no target. Otherwise out at the session's last close. A minute that
  reaches both the stop and the target counts as stopped, and so does the entry minute when it reaches the stop
  (the order inside a minute is unknown); a later minute that opens beyond the stop fills there.
- Costs: 0, 1 (primary) and 2 bps per side. P&L is in bps of the fill price: 1 bp = $1 per $10,000 traded.
- Baseline: the same trades at levels.py's random fake levels (10 per real level, same session and side).
- Test: mean net P&L per trade at 1 bp per side and the primary exit, for each of the three rules x four tickers,
  one-sided (positive), clustered by session; qualifies with BH q < 0.10 across those 12. Design to 2024-12-31;
  the holdout (2025-01-01 on) is read only with --final-test, for the qualifiers (one-sided, BH across them).

Added after the first results: the worst case inside the fill minute turned out to be heavily biased. Trades at
random fake levels lost 3-4 bps (ETFs) to 12-16 bps (TSLA) under it, where random levels should be near zero.
Every result is therefore also shown for the best case (the fill minute's low or high came before the fill, so
the stop counts only from the next minute); the truth lies between the two. Under the best case the fakes come out
near zero. The test above stays on the worst case, as fixed.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import levels as lv  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("TSLA", "SPY", "QQQ", "IWM")
# rule -> (levels.py level, sides traded: -1 = resistance above the open, traded long; +1 = support, short)
RULES = {
    "yesterday's high": ("prev_high", (-1,)),
    "last week's high": ("week_high", (-1,)),
    "yesterday's close": ("prev_close", (-1, 1)),
    "yesterday's low (check)": ("prev_low", (1,)),
    "last week's low (check)": ("week_low", (1,)),
}
TESTED = ("yesterday's high", "last week's high", "yesterday's close")
# exit -> (stop, target) in ATR from the level; target None = hold to the close unless stopped
EXITS = {"0.1 ATR stop and target": (0.1, 0.1), "0.25 ATR stop and target": (0.25, 0.25),
         "0.1 ATR stop, hold to the close": (0.1, None)}
PRIMARY_EXIT = "0.1 ATR stop and target"
COSTS = (0, 1, 2)                           # bps per side
FILLS = ("worst", "best")                   # inside the fill minute: its stop touch came after / before the fill
PRIMARY_COST = 1
HOLDOUT_START = lv.HOLDOUT_START
FDR = 0.10
COLORS = {"yesterday's high": SERIES[0], "last week's high": SERIES[1], "yesterday's close": INK_2}


# ---------------------------------------------------------------- trades

def simulate(open_, high, low, close, n_touch, price, side, stop_w, target_w, best=False):
    """Breakout trades on one session's minute bars (time order) for levels `price` with `side` (+1 support below
    the open, -1 resistance above). Each trade goes through the level (direction -side) at its first touch within
    the first n_touch bars: a stop order filled at the level, or at the touching minute's open if that is already
    through it. Exits, from the level: the stop `stop_w` back on the other side, the target `target_w` beyond it
    (None: no target), else the last close. A minute reaching both counts as stopped, and so does the entry minute
    reaching the stop. With `best`, the entry minute is read in the trade's favour instead: a target reached in it
    comes first, and a stop touch counts only if the minute closed beyond the stop (otherwise the dip came before
    the fill). A later minute opening beyond the stop fills at its open. Returns a dict of arrays: entry (bar index,
    -1 if untouched), fill, exit, exit_bar and reason (1 target, -1 stop, 0 close)."""
    n, k = len(high), len(price)
    d = -np.asarray(side, float)
    up = d > 0
    reach = np.where(up[:, None], high[None, :] >= price[:, None], low[None, :] <= price[:, None])
    reach[:, n_touch:] = False
    touched = reach.any(axis=1)
    entry = np.where(touched, reach.argmax(axis=1), -1)
    e = np.maximum(entry, 0)
    fill = np.where(up, np.maximum(price, open_[e]), np.minimum(price, open_[e]))
    j = np.arange(n)[None, :]
    after = j >= entry[:, None]
    stop = price - d * stop_w
    when = after
    if best:  # the entry minute's stop touch counts only when the minute closed beyond the stop
        definite = np.where(up, close[e] <= stop, close[e] >= stop)
        when = (j > entry[:, None]) | ((j == entry[:, None]) & definite[:, None])
    hit = when & np.where(up[:, None], low[None, :] <= stop[:, None], high[None, :] >= stop[:, None])
    fs = np.where(hit.any(axis=1), hit.argmax(axis=1), n)
    if target_w is None:
        target, ft = np.full(k, np.nan), np.full(k, n)
    else:
        target = price + d * target_w
        hit = after & np.where(up[:, None], high[None, :] >= target[:, None], low[None, :] <= target[:, None])
        ft = np.where(hit.any(axis=1), hit.argmax(axis=1), n)
    stopped = (fs < n) & ((fs < ft) | ((fs == ft) & ~(best & (ft == entry))))
    on_target = ~stopped & (ft < n)
    gap = open_[np.minimum(fs, n - 1)]
    stop_fill = np.where(fs > entry, np.where(up, np.minimum(stop, gap), np.maximum(stop, gap)), stop)
    target_fill = np.where(up, np.maximum(target, np.where(ft == entry, fill, target)),
                           np.minimum(target, np.where(ft == entry, fill, target)))
    exit_ = np.where(stopped, stop_fill, np.where(on_target, target_fill, close[n - 1]))
    out = {"entry": entry, "fill": np.where(touched, fill, np.nan), "exit": np.where(touched, exit_, np.nan),
           "exit_bar": np.where(touched, np.where(stopped, fs, np.where(on_target, ft, n - 1)), -1),
           "reason": np.where(touched, np.where(stopped, -1, np.where(on_target, 1, 0)), 0)}
    return out


def run_trades(rth, levels, table, exits=EXITS):
    """One row per touched level and exit rule: entry and exit times (minutes after the open), fill, exit price,
    reason and gross P&L in bps (through the level's direction)."""
    o, h, l, c = (rth[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t, op, cl = (_ns(rth[k]) for k in ("ts", "session_open", "session_close"))
    bars_by_day = rth.groupby("session").indices
    price, side = levels["price"].to_numpy(float), levels["side"].to_numpy()
    frames = []
    for day, rows in levels.groupby("session").indices.items():
        bars = bars_by_day.get(day)
        a = table["atr"].get(day, np.nan)
        if bars is None or not np.isfinite(a):
            continue
        n_touch = int((t[bars] < cl[bars[0]] - lv.TOUCH_END.value).sum())
        minutes = (t[bars] - op[bars[0]]) / 60e9
        for (name, (stop_w, target_w)), fills in ((e, f) for e in exits.items() for f in FILLS):
            s = simulate(o[bars], h[bars], l[bars], c[bars], n_touch, price[rows], side[rows], stop_w * a,
                         None if target_w is None else target_w * a, best=fills == "best")
            ok = s["entry"] >= 0
            if not ok.any():
                continue
            d = -side[rows][ok]
            frames.append(pd.DataFrame({
                "row": rows[ok], "exit_rule": name, "fills": fills, "entry_minute": minutes[s["entry"][ok]],
                "exit_minute": minutes[s["exit_bar"][ok]], "fill": s["fill"][ok], "exit": s["exit"][ok],
                "reason": s["reason"][ok], "gross": d * (s["exit"][ok] / s["fill"][ok] - 1) * 1e4}))
    if not frames:
        return pd.DataFrame(columns=["row", "exit_rule", "fills", "entry_minute", "exit_minute", "fill", "exit",
                                     "reason", "gross"])
    trades = pd.concat(frames, ignore_index=True)
    keep = ["session", "level", "price", "dist", "side", "period", "fake", "control", "rule"]
    return trades.join(levels[keep].reset_index(drop=True), on="row").drop(columns="row")


def rule_levels(real):
    """Real levels that one of RULES trades, with the rule's name."""
    parts = [real[(real["level"] == level) & real["side"].isin(sides)].assign(rule=name)
             for name, (level, sides) in RULES.items()]
    return pd.concat(parts, ignore_index=True)


def ticker_trades(minutes, sessions, final):
    """Trades at the real levels and at their random fakes for one ticker. Without `final`, holdout sessions are
    not traded at all."""
    rth, _ = features.regular_session_minutes(minutes, sessions)
    table = lv.level_table(rth, minutes, sessions)
    real = rule_levels(lv.real_levels(table))
    if not final:
        real = real[real["period"] == "design"].reset_index(drop=True)
    fakes = lv.fake_levels(real.drop(columns="rule"), table)
    rule_of = {(level, s): name for name, (level, sides) in RULES.items() for s in sides}
    fakes["rule"] = [rule_of[(a, b)] for a, b in zip(fakes["level"], fakes["side"])]
    levels = pd.concat([real, fakes], ignore_index=True)
    return run_trades(rth, levels, table), table


# ---------------------------------------------------------------- statistics

def trade_stats(t, cost, years):
    """Net P&L per trade (bps) at `cost` bps per side: count, per year, win rate, mean with a session-clustered
    95% interval and one-sided p (mean > 0), and the yearly sum (the % a year if every trade used the whole
    account)."""
    net = t["gross"].to_numpy(float) - 2 * cost
    n = len(net)
    out = {"trades": n, "per_year": n / years if years else np.nan}
    if n < 2:
        return out
    m, se = sm.cluster_mean(net, t["session"].to_numpy())
    z = m / se if se else np.nan
    return {**out, "win": (net > 0).mean() * 100, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "pct_year": net.sum() / 100 / years,
            "targets": (t["reason"] == 1).mean() * 100, "stops": (t["reason"] == -1).mean() * 100}


def statistics(trades, years_by_period):
    rows = []
    group = ["ticker", "period", "rule", "exit_rule", "fills", "control"]
    for (tk, period, rule, exit_rule, fills, control), g in trades.groupby(group, sort=False):
        for cost in COSTS:
            rows.append({"ticker": tk, "period": period, "rule": rule, "exit_rule": exit_rule, "fills": fills,
                         "control": control, "cost": cost, **trade_stats(g, cost, years_by_period[period])})
    return pd.DataFrame(rows)


def run_study(minutes_by_ticker, sessions, *, final=False, timings=None):
    """minutes_by_ticker: {ticker: minute bars}. No file I/O."""
    timings = {} if timings is None else timings
    frames, atr = [], {}
    with timed(timings, "levels and trades"):
        for tk, minutes in minutes_by_ticker.items():
            t, table = ticker_trades(minutes, sessions, final)
            frames.append(t.assign(ticker=tk))
            atr[tk] = float(table.loc[table.index < HOLDOUT_START, "atr"].median())
        trades = pd.concat(frames, ignore_index=True)
    with timed(timings, "statistics"):
        first = trades.loc[trades["period"] == "design", "session"].min()
        years = {"design": (HOLDOUT_START - first).days / 365.25,
                 "holdout": (sessions.index[-1] - HOLDOUT_START).days / 365.25}
        stats = statistics(trades, years)
        key = ((stats["exit_rule"] == PRIMARY_EXIT) & (stats["cost"] == PRIMARY_COST) & (stats["control"] == "real")
               & (stats["fills"] == "worst"))
        tests = stats[key & (stats["period"] == "design") & stats["rule"].isin(TESTED)].copy()
        tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
        tests["qualifies"] = (tests["q"] < FDR) & (tests["mean"] > 0)
        held = None
        if final:
            q = tests.loc[tests["qualifies"], ["ticker", "rule", "mean"]].rename(columns={"mean": "design_mean"})
            held = stats[key & (stats["period"] == "holdout")].merge(q, on=["ticker", "rule"])
            if len(held):
                held["holdout_q"] = mr.bh_qvalues(held["p_up"].fillna(1).to_numpy())
                held["holds_up"] = (held["holdout_q"] < FDR) & (held["mean"] > 0)
    return {"trades": trades, "stats": stats, "tests": tests, "holdout": held, "final": final,
            "tickers": list(minutes_by_ticker), "median_atr": atr, "years": years}


# ---------------------------------------------------------------- report

def bps(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def find(stats, **match):
    return lv.find(stats, **match)


def val(row, key):
    return None if row is None else row.get(key)


def cell(stats, **match):
    r = find(stats, **match)
    return "n/a" if r is None or pd.isna(r.get("mean")) else (
        f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})")


def render_report(res):
    s, tests = res["stats"], res["tests"]
    lines = []
    w = lines.append
    w("# Breakout trades at key levels\n")
    w("Each trade is a stop order at a level known before the open, filled at its first touch (at the level, or at "
      "the touching minute's open if that is already through it), in the direction through the level. It exits at "
      "a stop back through the level, a target beyond it, or the session's last close. P&L is per trade in basis "
      "points of the fill price after costs: 1 bp = $1 per $10,000 traded. *% a year* adds up the trades as if each "
      "used the whole account (intraday, so no compounding). *Fake levels* are levels.py's random fakes, traded the "
      "same way. Design period to 2024-12-31" + (", holdout 2025-01-01 on." if res["final"] else
                                                   " (holdout not read).") + "\n")
    w("Minute bars can't show what happened inside the minute the stop order filled. *Worst*: a dip to the stop in "
      "that minute came after the fill (the rule fixed in advance). *Best*: it came before, so the stop counts only "
      "from the next minute. The truth lies between. Random fake levels should earn about nothing before costs: "
      "under the worst case they lose several bps a trade, under the best case they come out near zero, so the "
      "best case is the closer of the two.\n")
    q = tests[tests["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(tests)} tests qualified (worst case, mean net P&L > 0 at {PRIMARY_COST} bp "
      f"per side with the {PRIMARY_EXIT} exit, BH q < {FDR:g})" + (": " + "; ".join(
          f"{r.ticker} {r.rule} ({bps(r.mean)} bps a trade, {r.per_year:.0f} trades a year)" for r in q.itertuples())
          if len(q) else ".") + "\n")
    w(f"## Primary exit ({PRIMARY_EXIT}), design period, {PRIMARY_COST} bp per side\n")
    base = s[(s["period"] == "design") & (s["exit_rule"] == PRIMARY_EXIT)]
    rows = []
    for tk in res["tickers"]:
        for rule in RULES:
            rw, rb = (find(base, ticker=tk, rule=rule, control="real", cost=PRIMARY_COST, fills=f) for f in FILLS)
            if rw is None:
                continue
            fw, fb = (find(base, ticker=tk, rule=rule, control="random", cost=PRIMARY_COST, fills=f) for f in FILLS)
            b0, b2 = (find(base, ticker=tk, rule=rule, control="real", cost=c, fills="best") for c in (0, 2))
            t = find(tests, ticker=tk, rule=rule)
            rows.append([tk, rule, f"{int(rw['trades']):,} ({rw['per_year']:.0f})",
                         cell(base, ticker=tk, rule=rule, control="real", cost=PRIMARY_COST, fills="worst"),
                         cell(base, ticker=tk, rule=rule, control="real", cost=PRIMARY_COST, fills="best"),
                         f"{bps(val(b0, 'mean'))} / {bps(val(b2, 'mean'))}",
                         f"{lv.pct(val(rw, 'win'))} / {lv.pct(val(rb, 'win'))}",
                         f"{bps(val(fw, 'mean'))} / {bps(val(fb, 'mean'))}", "" if t is None else f"{t['q']:.2f}",
                         f"{bps(val(rw, 'pct_year'))}% / {bps(val(rb, 'pct_year'))}%"])
    w(alerts.md_table(["Ticker", "Rule", "Trades (a year)", "Net bps a trade, worst (95%)", "Best (95%)",
                       "Best at 0 / 2 bp", "Wins, worst / best", "Fake levels, worst / best", "q (worst)",
                       "% a year, worst / best"], rows) + "\n")
    w(f"## Other exits, design period (net bps a trade at {PRIMARY_COST} bp per side, worst / best)\n")
    rows = []
    design = s[(s["period"] == "design") & (s["cost"] == PRIMARY_COST)]
    for tk in res["tickers"]:
        for rule in TESTED:
            cells = []
            for control in ("real", "random"):
                for e in EXITS:
                    m = [bps(val(find(design, ticker=tk, rule=rule, control=control, exit_rule=e, fills=f), "mean"))
                         for f in FILLS]
                    cells.append(" / ".join(m))
            rows.append([tk, rule, *cells])
    w(alerts.md_table(["Ticker", "Rule", *EXITS, *[f"Fakes: {e}" for e in EXITS]], rows) + "\n")
    w(f"## Yesterday's close by side, design period (primary exit, {PRIMARY_COST} bp, worst / best)\n")
    rows = []
    tr = res["trades"]
    for tk in res["tickers"]:
        cells = []
        for side in (-1, 1):
            parts = []
            for f in FILLS:
                g = tr[(tr["ticker"] == tk) & (tr["rule"] == "yesterday's close") & (tr["side"] == side)
                       & (tr["exit_rule"] == PRIMARY_EXIT) & (tr["control"] == "real") & (tr["period"] == "design")
                       & (tr["fills"] == f)]
                st = trade_stats(g, PRIMARY_COST, res["years"]["design"])
                parts.append(bps(st.get("mean")))
            cells.append(f"{' / '.join(parts)}, n={st['trades']}")
        rows.append([tk, *cells])
    w(alerts.md_table(["Ticker", "Long: price rose to it", "Short: price fell to it"], rows) + "\n")
    if res["final"]:
        w("## Holdout: did the qualifiers hold up?\n")
        h = res["holdout"]
        if h is None or h.empty:
            w("No qualifiers to test.\n")
        else:
            rows = [[r.ticker, r.rule, bps(r.design_mean), f"{bps(r.mean)} ({bps(r.lo)} to {bps(r.hi)})",
                     f"{int(r.trades):,}", f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"]
                    for r in h.itertuples()]
            w(alerts.md_table(["Ticker", "Rule", "Design bps", "Holdout bps (95%)", "Trades", "Holdout q",
                               "Holds up"], rows) + "\n")
    w("## Notes\n")
    w("- Median ATR(14) in the design period: " + ", ".join(
        f"{tk} ${a:.2f} (0.1 ATR = ${0.1 * a:.2f})" for tk, a in res["median_atr"].items()) + ".")
    w("- Fills are modelled from minute bars: a stop order at the level, no queue or spread beyond the cost "
      "setting. Real stop orders in fast moves can fill worse. One-second bars (on the data plan) can settle "
      "the order inside the fill minute.")
    w("- Results are per share of the stock or ETF, gross of commissions; not options P&L.")
    w("- Rules can overlap on the same day (yesterday's and last week's high can be close together).")
    return "\n".join(lines) + "\n"


def plot(res, path):
    """Cumulative net % (each trade using the whole account, primary exit, 1 bp per side) for each tested rule:
    solid = best case inside the fill minute, dotted = worst case."""
    tr = res["trades"]
    tr = tr[(tr["exit_rule"] == PRIMARY_EXIT) & (tr["control"] == "real") & (tr["period"] == "design")]
    n = len(res["tickers"])
    fig = plt.figure(figsize=(10, 4.6), dpi=150, facecolor=SURFACE)
    for k, tk in enumerate(res["tickers"]):
        ax = sds._axes(fig, [0.07 + k * (0.9 / n), 0.2, 0.9 / n - 0.05, 0.6])
        for rule in TESTED:
            for f, style in (("best", "-"), ("worst", (0, (1, 1.5)))):
                g = tr[(tr["ticker"] == tk) & (tr["rule"] == rule) & (tr["fills"] == f)].sort_values("session")
                if g.empty:
                    continue
                cum = ((g["gross"] - 2 * PRIMARY_COST) / 100).cumsum()
                ax.plot(g["session"], cum, color=COLORS[rule], lw=1.2 if f == "best" else 1.0, ls=style,
                        label=rule if k == 0 and f == "best" else None)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(tk, color=INK, fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=6.5, rotation=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if k == 0:
            ax.set_ylabel("Cumulative net %, whole account per trade", color=INK_2, fontsize=7)
    fig.text(0.02, 0.97, "Breakout trades at key levels", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{PRIMARY_EXIT}, {PRIMARY_COST} bp per side, design period 2021-2024. Solid: best case "
             "inside the fill minute; dotted: worst case. Per share of the underlying.", color=INK_2, fontsize=7.5,
             va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Breakout trades at key levels (trade simulation).")
    p.add_argument("--out", default="output/breakouts")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
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
    with timed(timings, "data"):
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    res = run_study(data, sessions, final=args.final_test, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    if res["final"] and res["holdout"] is not None:
        res["holdout"].to_csv(out / "holdout.csv", index=False)
    plot(res, out / "breakouts.png")
    (out / "report.md").write_text(render_report(res))
    print(f"tests: {len(res['tests'])}, qualified: {int(res['tests']['qualifies'].sum())}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
