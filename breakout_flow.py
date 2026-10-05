"""TSLA breakouts with a buying-pressure filter: take the long through yesterday's high only when buying volume led in
the minutes before the touch, and the short through yesterday's low only when selling led. Order flow is approximated
from Massive one-second bars (the plan has no quotes). A trade simulation, per share of TSLA, gross of commissions.

  uv run --env-file .env python breakout_flow.py --out output/breakout_flow                  # 2021-24
  uv run --env-file .env python breakout_flow.py --final-test --out output/breakout_flow     # 2025-26, once

breakout_check.py's one-time check already used 2025-26 for the two rules, so this filter is tested on
2021-24, and 2025-26 is read only with --final-test.

Fixed on 2026-10-03, before any result was seen:
- Trades: breakout_check.py's long through yesterday's high and short through yesterday's low, with its entries and
  one-second replay, net of 1 bp per side (0 and 2 shown).
- Pressure: over the 5 minutes before the fill second, or from 9:30:01 when the fill comes sooner (the 9:30:00
  second, which carries the opening auction, only sets the starting price). Each second's volume counts as buying
  when its close is above the previous second's close, as selling when below, and as the last change when unchanged;
  volume before the first change isn't counted. Pressure = (buying - selling) / (buying + selling), with the sign
  flipped for the short, so positive always means pressure in the trade's direction. It needs at least 10 seconds
  with trades; otherwise there is no measure and the filter skips the trade.
- Filter: take a trade only when its pressure is above 0.
- Test: mean net P&L of trades with pressure above 0 minus trades with pressure at or below 0, the two rules pooled,
  one-sided (positive), standard errors clustered by session.
- For reading: each rule alone, 1- and 15-minute windows, pressure in thirds, and pressure's effect after allowing
  for the price move over the same 5 minutes.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import breakout_check as bc  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import levels as lv  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKER = bc.TICKER
RULES = {"long through yesterday's high": "yesterday's high",
         "short through yesterday's low": "yesterday's low (check)"}   # -> breakouts.RULES
WINDOWS = (1, 5, 15)                          # minutes before the fill
PRIMARY_WINDOW = 5
MIN_SECONDS = 10
COSTS, PRIMARY_COST = bc.COSTS, bc.PRIMARY_COST
MINUTE_MS, SECOND_MS = 60_000, 1_000


# ---------------------------------------------------------------- pressure

def pressure(sec, start_ms, end_ms, d, min_seconds=MIN_SECONDS):
    """(pressure in the trade's direction, seconds counted, the price move over the window in bps, signed the same
    way) from one-second bars `sec` (ms, open, high, low, close, volume; time order) starting in [start_ms, end_ms).
    Bars before start_ms only set the starting price and the last change. d = +1 long, -1 short."""
    if len(sec) == 0:
        return np.nan, 0, np.nan
    ms, close, vol = sec[:, 0], sec[:, 4], sec[:, 5]
    step = np.r_[0.0, np.sign(np.diff(close))]
    last = pd.Series(np.where(step != 0, step, np.nan)).ffill().fillna(0.0).to_numpy()
    inside = (ms >= start_ms) & (ms < end_ms)
    n = int(inside.sum())
    if n < min_seconds:
        return np.nan, n, np.nan
    buy, sell = vol[inside & (last > 0)].sum(), vol[inside & (last < 0)].sum()
    p = (buy - sell) / (buy + sell) if buy + sell > 0 else np.nan
    before = np.flatnonzero(ms < start_ms)
    ref = close[before[-1]] if before.size else sec[np.argmax(inside), 1]
    return d * p, n, d * (close[inside][-1] / ref - 1) * 1e4


def flow_columns(trades, sessions, flow_of):
    """Adds pressure_W, seconds_W and move_W for each window W (minutes) to replayed trades (status ok)."""
    open_ms = dict(zip(sessions.index, _ns(sessions["open"]) // 1_000_000))
    reqs = [(TICKER, max(int(r.fill_ms) - max(WINDOWS) * MINUTE_MS - SECOND_MS, int(open_ms[r.session])),
             int(r.fill_ms) - 1) for r in trades.itertuples()]
    if hasattr(flow_of, "prefetch"):
        flow_of.prefetch(reqs)
    rows = []
    for r, req in zip(trades.itertuples(), reqs):
        sec, d, row = flow_of(*req), -r.side, {}
        for w in WINDOWS:
            start = max(int(r.fill_ms) - w * MINUTE_MS, int(open_ms[r.session]) + SECOND_MS)
            p, n, move = pressure(sec, start, int(r.fill_ms), d)
            row.update({f"pressure_{w}": p, f"seconds_{w}": n, f"move_{w}": move})
        rows.append(row)
    return trades.join(pd.DataFrame(rows, index=trades.index))


# ---------------------------------------------------------------- statistics

def cluster_ols(y, X, groups):
    """OLS of y on a constant and the columns of X: (coefficients, standard errors clustered by group, small-sample
    corrected)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    u = y - X @ beta
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices.values()
    meat = sum(np.outer(X[i].T @ u[i], X[i].T @ u[i]) for i in idx)
    g, n, k = len(idx), len(y), X.shape[1]
    v = inv @ meat @ inv * g / (g - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(v))


def split_test(t, w, cost):
    """Mean net P&L with pressure above 0 minus at or below 0 (trades with a measure): (difference, se, p one-sided)."""
    m = t[t[f"pressure_{w}"].notna()]
    if m[f"pressure_{w}"].gt(0).nunique() < 2:
        return np.nan, np.nan, np.nan
    beta, se = cluster_ols(m["gross"] - 2 * cost, m[f"pressure_{w}"].gt(0).astype(float), m["session"])
    return beta[1], se[1], norm.sf(beta[1] / se[1]) if se[1] > 0 else np.nan


def groups_of(t, w):
    p = t[f"pressure_{w}"]
    return {"all trades": t, "pressure in the trade's direction (taken)": t[p > 0],
            "pressure against it (skipped)": t[p <= 0], "no measure (skipped)": t[p.isna()]}


# ---------------------------------------------------------------- study

def run_study(minutes, sessions, seconds_of, flow_of, *, final=False, timings=None):
    """No file I/O apart from the injected one-second loaders' caches."""
    timings = {} if timings is None else timings
    period = "holdout" if final else "design"
    with timed(timings, "levels and entries"):
        cand, rth = bc.candidates(minutes, sessions, period)
        real = cand[~cand["fake"] & cand["rule"].isin(RULES.values())]
    with timed(timings, "share replay"):
        shares = bc.replay_all(real, rth, seconds_of)
    ok = shares[shares["status"] == "ok"].copy()
    ok["rule_label"] = ok["rule"].map({v: k for k, v in RULES.items()})
    with timed(timings, "order flow"):
        t = flow_columns(ok, sessions, flow_of)
    first = sessions.index[sessions.index >= (lv.HOLDOUT_START if final else t["session"].min())][0]
    last = sessions.index[-1] if final else lv.HOLDOUT_START - pd.Timedelta(days=1)
    years = (last - first).days / 365.25
    rows = []
    for scope, part in [("both rules", t), *[(label, t[t["rule_label"] == label]) for label in RULES]]:
        for w in WINDOWS:
            for name, g in groups_of(part, w).items():
                for cost in COSTS:
                    rows.append({"scope": scope, "window": w, "group": name, "cost": cost,
                                 **bc.share_stats(g, cost, years)})
    stats = pd.DataFrame(rows)
    for col in ("mean", "lo", "hi", "win", "avg_win", "avg_loss", "pct_year"):
        if col not in stats:
            stats[col] = np.nan
    splits = []
    for scope, part in [("both rules", t), *[(label, t[t["rule_label"] == label]) for label in RULES]]:
        for w in WINDOWS:
            diff, se, p = split_test(part, w, PRIMARY_COST)
            splits.append({"scope": scope, "window": w, "diff": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
                           "p_up": p, "share_taken": part[f"pressure_{w}"].gt(0).mean() * 100})
    splits = pd.DataFrame(splits)
    m = t[t[f"pressure_{PRIMARY_WINDOW}"].notna()].copy()
    m["third"] = pd.qcut(m[f"pressure_{PRIMARY_WINDOW}"], 3, labels=["lowest third", "middle third", "highest third"])
    thirds = pd.DataFrame([{"third": str(k), "pressure_lo": g[f"pressure_{PRIMARY_WINDOW}"].min(),
                            "pressure_hi": g[f"pressure_{PRIMARY_WINDOW}"].max(), **bc.share_stats(g, PRIMARY_COST, years)}
                           for k, g in m.groupby("third", observed=True)])
    net = m["gross"] - 2 * PRIMARY_COST
    beta, se = cluster_ols(net, m[[f"pressure_{PRIMARY_WINDOW}", f"move_{PRIMARY_WINDOW}"]], m["session"])
    adjusted = {"per_tenth": beta[1] / 10, "per_tenth_se": se[1] / 10, "p": 2 * norm.sf(abs(beta[1] / se[1])),
                "move_per_10bps": beta[2] * 10, "move_p": 2 * norm.sf(abs(beta[2] / se[2])),
                "corr": float(np.corrcoef(m[f"pressure_{PRIMARY_WINDOW}"], m[f"move_{PRIMARY_WINDOW}"])[0, 1])}
    primary = splits[(splits["scope"] == "both rules") & (splits["window"] == PRIMARY_WINDOW)].iloc[0]
    return {"trades": t, "stats": stats, "splits": splits, "thirds": thirds, "adjusted": adjusted, "primary": primary,
            "final": final, "period": period, "years": years, "first": first, "last": last,
            "status": shares["status"].value_counts().to_dict()}


# ---------------------------------------------------------------- report

def fmt(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def render_report(res):
    s, sp, pr = res["stats"], res["splits"], res["primary"]
    span = f"{res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}"
    lines = []
    w = lines.append
    w(f"# TSLA breakouts with a buying-pressure filter ({'2025-26 check' if res['final'] else '2021-24'})\n")
    w(f"{span}. The long through yesterday's high and the short through yesterday's low from breakout_check.py, "
      "replayed second by second. *Pressure* = buying minus selling volume over total in the minutes before the fill, "
      "each second's volume classed by its price change from the second before, signed so that positive is the "
      "trade's direction. The filter takes a trade only when the 5-minute pressure is above 0. P&L in bps of the fill "
      "after 1 bp per side (1 bp = $1 per $10,000 traded).\n")
    verdict = "passes" if pd.notna(pr["p_up"]) and pr["p_up"] < 0.05 and pr["diff"] > 0 else "does not pass"
    w(f"**Test (fixed in advance).** Trades with pressure in their direction minus trades against it, both rules: "
      f"{fmt(pr['diff'])} bps a trade (95% {fmt(pr['lo'])} to {fmt(pr['hi'])}), one-sided p = {pr['p_up']:.3f}: "
      f"the filter {verdict} at p < 0.05. {pr['share_taken']:.0f}% of trades had pressure in their direction.\n")
    w("## The filter, 5-minute window\n")
    rows = []
    for scope in ["both rules", *RULES]:
        for r in s[(s["scope"] == scope) & (s["window"] == PRIMARY_WINDOW) & (s["cost"] == PRIMARY_COST)].itertuples():
            if pd.isna(r.mean):
                rows.append([scope, r.group, f"{int(r.trades):,}"] + ["n/a"] * 4)
                continue
            rows.append([scope, r.group, f"{int(r.trades):,} ({r.per_year:.0f})", f"{r.win:.0f}%",
                         f"{fmt(r.avg_win)} / {fmt(r.avg_loss)}", f"{fmt(r.mean)} ({fmt(r.lo)} to {fmt(r.hi)})",
                         f"{r.pct_year:+.1f}%"])
    w(alerts.md_table(["Rule", "Trades", "Count (a year)", "Wins", "Avg win / loss, bps", "Net bps a trade (95%)",
                       "% a year (whole account)"], rows) + "\n")
    w("## With minus against, by window and rule\n")
    rows = [[r.scope, f"{r.window} min", f"{r.share_taken:.0f}%", f"{fmt(r.diff)} ({fmt(r.lo)} to {fmt(r.hi)})",
             "n/a" if pd.isna(r.p_up) else f"{r.p_up:.3f}"] for r in sp.itertuples()]
    w(alerts.md_table(["Rule", "Window", "Share with pressure in the trade's direction",
                       "With minus against, bps (95%)", "p (one-sided)"], rows) + "\n")
    w("## Pressure in thirds (5 minutes, both rules)\n")
    rows = [[r.third, f"{r.pressure_lo:+.2f} to {r.pressure_hi:+.2f}", f"{int(r.trades):,}", f"{r.win:.0f}%",
             f"{fmt(r.mean)} ({fmt(r.lo)} to {fmt(r.hi)})"] for r in res["thirds"].itertuples()]
    w(alerts.md_table(["Third", "Pressure", "Trades", "Wins", "Net bps a trade (95%)"], rows) + "\n")
    a = res["adjusted"]
    w(f"**Beyond the price move.** Pressure and the price move over the same 5 minutes correlate at {a['corr']:+.2f}. "
      f"Allowing for the move, each 0.1 of pressure goes with {fmt(a['per_tenth'], 2)} bps a trade "
      f"(± {1.96 * a['per_tenth_se']:.2f}, p = {a['p']:.3f}); each 10 bps of move with {fmt(a['move_per_10bps'], 2)} "
      f"bps (p = {a['move_p']:.3f}).\n")
    w("## Notes\n")
    t = res["trades"]
    w(f"- Trades: {len(t):,} replayed ({', '.join(f'{k} {v:,}' for k, v in res['status'].items())}); without a "
      f"5-minute measure (fewer than {MIN_SECONDS} seconds with trades, i.e. a fill right at the open): "
      f"{int(t[f'pressure_{PRIMARY_WINDOW}'].isna().sum())}.")
    w("- Order flow from one-second bars is an approximation: without quotes, each second is classed by its price "
      "change, and off-exchange prints near the middle of the spread can be classed either way.")
    if not res["final"]:
        w("- 2025-26 not read (breakout_check.py's one-time check already used it for the rules themselves).")
    return "\n".join(lines) + "\n"


def plot(res, path):
    th = res["thirds"]
    fig = plt.figure(figsize=(7.5, 4.4), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.12, 0.14, 0.82, 0.66])
    x = np.arange(len(th))
    ax.bar(x, th["mean"], color=[MUTED, MUTED, SERIES[0]], width=0.6)
    ax.errorbar(x, th["mean"], yerr=[th["mean"] - th["lo"], th["hi"] - th["mean"]], fmt="none", ecolor=INK_2, lw=1,
                capsize=4)
    allr = res["stats"]
    allr = allr[(allr["scope"] == "both rules") & (allr["window"] == PRIMARY_WINDOW) & (allr["group"] == "all trades")
                & (allr["cost"] == PRIMARY_COST)]["mean"].iloc[0]
    ax.axhline(allr, color=SERIES[1], lw=1, ls=(0, (3, 2)), label=f"all trades {allr:+.1f} bps")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.third}\n{r.pressure_lo:+.2f} to {r.pressure_hi:+.2f}" for r in th.itertuples()],
                       fontsize=7, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Net bps a trade (1 bp per side)", color=INK_2, fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
    title = "2025-26" if res["final"] else "2021-24"
    fig.text(0.02, 0.97, f"TSLA breakouts by buying pressure before the touch ({title})", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Pressure over the 5 minutes before the fill, in the trade's direction; both rules.",
             color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TSLA breakouts filtered by order-flow pressure before the touch.")
    p.add_argument("--out", default="output/breakout_flow")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="read 2025-26 (once)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        minutes, _ = gr.load_minutes(TICKER, sessions, args.cache_dir, args.refresh)
    seconds_of = bc.second_loader(args.cache_dir, args.refresh)
    flow_of = bc.second_loader(args.cache_dir, args.refresh, volume=True)
    try:
        res = run_study(minutes, sessions, seconds_of, flow_of, final=args.final_test, timings=timings)
    finally:
        seconds_of.save()
        flow_of.save()
    out = Path(args.out) / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["splits"].to_csv(out / "splits.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": res["period"], "first": str(res["first"].date()),
        "last": str(res["last"].date()), "seconds_fetched": seconds_of.state["fetched"],
        "flow_seconds_fetched": flow_of.state["fetched"]}, indent=2))
    plot(res, out / "pressure.png")
    (out / "report.md").write_text(render_report(res))
    print(f"one-second requests: {seconds_of.state['fetched']} (replay), {flow_of.state['fetched']} (order flow)")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
