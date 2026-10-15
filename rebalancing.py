"""Month-end rebalancing flows: when stocks have beaten bonds during the month, balanced funds sell stocks in the
last days of the month (and buy them when bonds have won). Does trading ahead of them work on SPY, IWM and QQQ?
A signal study on 15:50 ET daily snapshots: P&L per trade in bps of the entry price, after costs.

  uv run --env-file .env python rebalancing.py --out output/rebalancing                 # design screen, 2021-24
  uv run --env-file .env python rebalancing.py --final-test --out output/rebalancing    # held-back check, once

Mechanism. Pension funds, balanced and target-date funds hold fixed stock/bond weights and many rebalance on a
calendar, mostly at month-end. A month in which stocks outrun bonds leaves them overweight stocks, so they sell
stocks and buy bonds over the last days of the month; a month in which bonds win makes them buy stocks. They trade
for their mandate, not for price, so whoever takes the other side ahead of them earns the price pressure. Harvey,
Mazzoleni and Melone ("The Unintended Consequences of Rebalancing", NBER w33554, 2025) find this in S&P 500 and
10-year Treasury futures over 1997-2023: a 60/40 portfolio's drift from target since the last month-end predicts
lower stock-minus-bond returns in the last week of the month, strongest in the last four days, and the effect
reverses on the first days of the next month. It should last as long as trillions stay in calendar-rebalanced
mandates. It is both bullish and bearish, depending on the month.

Declared on 2026-10-04, before any result was computed:
- Prices: 15:50 ET snapshots (meanrev.daily_table), dividend-adjusted, so returns are total returns. Bond leg: IEF
  (7-10 year Treasuries, the closest ETF to the paper's 10-year note).
- Signal (the paper's calendar signal): at 15:50 on session t, the stock weight of a 60/40 SPY/IEF portfolio that
  was rebalanced to 60/40 at the previous month's last 15:50 snapshot, minus 0.60. Positive = stocks overweight =
  rebalancers will sell stocks.
- Trade: at 15:50 on the 5th-last session of the month (L-4, where L is the month's last session), go short the
  stock ETF if the signal is positive and long it if negative; exit at 15:50 on L. That holds through the last four
  sessions' moves (L-3 to L). Costs 1 bp a side (shares; a short would be an inverse ETF or a put on Robinhood).
- Baselines: the same rule slid to every other 4-session window from L-19 to L+4 with the signal measured at its
  entry ("slid copies"), and always long / always short in the month-end window.
- Tests (8, one-sided, Student t on the monthly trades, which don't overlap; Benjamini-Hochberg q < 0.10 across
  all 8), design months 2021-11 to 2024-12:
    1. SPY, month-end window: mean net P&L > 0 (primary)
    2. IWM, the same
    3. QQQ, the same
    4. SPY minus IEF (the paper's stock-minus-bond trade), the same
    5. SPY, the paper's full week (L-4 to the first session of the next month, L+1)
    6. SPY, quarter-end months only (March, June, September, December)
    7. SPY, the reversal: from L to L+2, in the direction rebalancers just traded (buy what they sold)
    8. SPY, month-end window minus the average of its slid copies > 0 (the month end is special)
- The idea passes the screen if test 1 or 4 has q < 0.10 with a mean net P&L of at least twice the round-trip cost,
  IWM's test points the same way, at least 2 of 2022-2024 are positive, and test 8's difference is positive.
- Reported, not tested: each year, the signal's size (thirds), the account view (one trade a month) and the
  daily-P&L correlation with SPY dip buying and the TSLA breakouts.
- Held-back check (--final-test): SPY's 2025+ data has already been used for daily SPY ideas, so the clean check is
  IWM and QQQ, 2025-01 to 2026-09 (their daily data was used only for trend rules in trend_etfs.py). SPY 2025-26 is
  shown for context and labeled as already used. The rules for it are in FINAL below.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import portfolio_check as pc  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
EQUITIES = ("SPY", "IWM", "QQQ")
BOND = "IEF"
TICKERS = EQUITIES + (BOND,)
TARGET = 0.60
ENTRY, EXIT = -4, 0              # sessions relative to the month's last session L
FULL_WEEK_EXIT = 1
REVERSAL = (0, 2)
SLIDES = [k for k in range(-15, 5) if k != 0]   # window shifts: entry L-4+k (in the month), exit L+k
COST_BPS = 1.0                  # per side
DESIGN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
YEARS = (2022, 2023, 2024)
FDR = 0.10
TESTS = [("SPY", "month end"), ("IWM", "month end"), ("QQQ", "month end"), ("SPY-IEF", "month end"),
         ("SPY", "full week"), ("SPY", "quarter ends"), ("SPY", "reversal"), ("SPY", "vs slid copies")]
FINAL = None                    # set only if the screen passes, before --final-test


# ---------------------------------------------------------------- signal and trades

def month_last(index):
    """Position (in `index`) of each session's month's last session, and each session's offset from it."""
    month = index.to_period("M")
    pos = np.arange(len(index))
    last = pd.Series(pos, index=index).groupby(month).transform("max").to_numpy()
    return last, pos - last


def calendar_signal(stock, bond, target=TARGET):
    """Stock weight minus target of a target/(1-target) portfolio rebalanced at the previous month's last
    snapshot, at each session's snapshot. Uses only snapshots up to that session."""
    idx = stock.index
    s, b = stock.to_numpy(float), bond.to_numpy(float)
    first = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).transform("min").to_numpy()
    ref = first - 1                       # the previous month's last session
    ok = ref >= 0
    r = np.maximum(ref, 0)
    gs = np.where(ok, s / s[r], np.nan)
    gb = np.where(ok, b / b[r], np.nan)
    return pd.Series(target * gs / (target * gs + (1 - target) * gb) - target, index=idx)


def window_trades(snap, signal, entry=ENTRY, exit_=EXIT, direction=-1, cost_bps=COST_BPS):
    """One trade per month: enter at the snapshot `entry` sessions from the month's last session L, exit `exit_`
    sessions from L (can be in the next month). Position = direction x sign(signal at entry): -1 trades ahead of
    the rebalancers. Returns rows of entry/exit dates, signal, position, gross and net bps."""
    idx = snap.index
    last, off = month_last(idx)
    rows = []
    for L in np.unique(last):
        i, j = L + entry, L + exit_
        if i < 0 or j >= len(idx) or i >= j or idx[i].to_period("M") != idx[L].to_period("M"):
            continue
        sig = signal.iloc[i]
        a, b = snap.iloc[i], snap.iloc[j]
        if not (np.isfinite(sig) and np.isfinite(a) and np.isfinite(b)) or sig == 0:
            continue
        pos = direction * np.sign(sig)
        gross = pos * (b / a - 1) * 1e4
        rows.append({"month": idx[L].to_period("M"), "entry": idx[i], "exit": idx[j], "signal": sig,
                     "position": pos, "gross": gross, "net": gross - 2 * cost_bps})
    return pd.DataFrame(rows, columns=["month", "entry", "exit", "signal", "position", "gross", "net"])


def reversal_trades(snap, month_end, start=REVERSAL[0], end=REVERSAL[1], cost_bps=COST_BPS):
    """After each month-end trade, the opposite position (the rebalancers' own direction, expecting their price
    pressure to fade) from `start` to `end` sessions after the month's last session."""
    idx = snap.index
    rows = []
    for t in month_end.itertuples():
        L = int(idx.get_indexer([t.exit])[0]) - EXIT
        i, j = L + start, L + end
        if j >= len(idx) or not (np.isfinite(snap.iloc[i]) and np.isfinite(snap.iloc[j])):
            continue
        pos = -t.position
        gross = pos * (snap.iloc[j] / snap.iloc[i] - 1) * 1e4
        rows.append({"month": t.month, "entry": idx[i], "exit": idx[j], "signal": t.signal, "position": pos,
                     "gross": gross, "net": gross - 2 * cost_bps})
    return pd.DataFrame(rows, columns=["month", "entry", "exit", "signal", "position", "gross", "net"])


def spread_snap(stock, bond):
    """A daily-rebalanced long stock / short bond price series (its returns are stock minus bond returns)."""
    r = stock.pct_change() - bond.pct_change()
    return (1 + r.fillna(0)).cumprod() * 100


def stats(net, years=None):
    """Mean net bps a trade, 95% interval, one-sided p (Student t), win rate, count."""
    x = np.asarray(net, float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"trades": n}
    if n < 3:
        return out
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    tq = sps.t.ppf(0.975, n - 1)
    return {**out, "mean": m, "lo": m - tq * se, "hi": m + tq * se, "p_up": sps.t.sf(m / se, n - 1),
            "win": (x > 0).mean() * 100, "per_year": n / years if years else np.nan}


def run_study(snaps, *, period="design", timings=None):
    """snaps: {ticker: 15:50 dividend-adjusted snapshot series over all sessions}. No file I/O."""
    timings = {} if timings is None else timings
    idx = snaps["SPY"].index
    keep = (idx <= DESIGN_END) if period == "design" else (idx >= HOLDOUT_START)
    sig = calendar_signal(snaps["SPY"], snaps[BOND])
    series = {**{t: snaps[t] for t in EQUITIES}, "SPY-IEF": spread_snap(snaps["SPY"], snaps[BOND])}
    in_period = lambda tr: tr[tr["entry"].isin(idx[keep]) & tr["exit"].isin(idx[keep])]  # noqa: E731
    with timed(timings, "trades"):
        trades = {}
        for name, s in series.items():
            trades[(name, "month end")] = in_period(window_trades(s, sig))
        trades[("SPY", "full week")] = in_period(window_trades(series["SPY"], sig, exit_=FULL_WEEK_EXIT))
        q = trades[("SPY", "month end")]
        trades[("SPY", "quarter ends")] = q[q["month"].dt.month.isin((3, 6, 9, 12))]
        rev = reversal_trades(series["SPY"], window_trades(series["SPY"], sig))
        trades[("SPY", "reversal")] = in_period(rev)
        slid = {}
        for name in ("SPY", "IWM", "QQQ"):
            for k in SLIDES:
                slid[(name, k)] = in_period(window_trades(series[name], sig, entry=ENTRY + k, exit_=EXIT + k))
        for name in EQUITIES:
            base = window_trades(series[name], pd.Series(-1.0, index=idx))   # position +1: always long
            trades[(name, "always long")] = in_period(base)
    years = (idx[keep][-1] - trades[("SPY", "month end")]["entry"].min()).days / 365.25
    rows = []
    for (name, kind), t in trades.items():
        rows.append({"series": name, "window": kind, **stats(t["net"], years)})
    table = pd.DataFrame(rows)
    slid_rows = [{"series": n, "shift": k, **stats(t["net"], years)} for (n, k), t in slid.items()]
    slid_table = pd.DataFrame(slid_rows)
    # test 8: the month-end trade minus the average slid copy, months paired
    me = trades[("SPY", "month end")].set_index("month")["net"]
    sl = pd.concat([slid[("SPY", k)].set_index("month")["net"].rename(k) for k in SLIDES], axis=1).mean(axis=1)
    diff = (me - sl.reindex(me.index)).dropna()
    tests = []
    for name, kind in TESTS:
        s = stats(diff) if kind == "vs slid copies" else stats(trades[(name, kind)]["net"])
        tests.append({"series": name, "window": kind, **s})
    tests = pd.DataFrame(tests)
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["qualifies"] = (tests["q"] < FDR) & (tests["mean"] > 0)
    by_year = pd.DataFrame([{"series": n, "year": y, **stats(g["net"])}
                            for n in ("SPY", "IWM", "QQQ", "SPY-IEF")
                            for y, g in trades[(n, "month end")].groupby(trades[(n, "month end")]["exit"].dt.year)])
    t1 = trades[("SPY", "month end")]
    terc = pd.qcut(t1["signal"].abs(), 3, labels=["small", "middle", "large"]) if len(t1) >= 6 else None
    by_size = pd.DataFrame([{"size": k, **stats(g["net"])} for k, g in t1.groupby(terc, observed=True)]) \
        if terc is not None else pd.DataFrame()
    return {"trades": trades, "table": table, "slid": slid_table, "tests": tests, "by_year": by_year,
            "by_size": by_size, "signal": sig, "period": period, "years": years}


def verdict(res):
    t = res["tests"].set_index(["series", "window"])
    by_year = res["by_year"]
    yrs = by_year[(by_year["series"] == "SPY") & by_year["year"].isin(YEARS)]
    primary = [t.loc[("SPY", "month end")], t.loc[("SPY-IEF", "month end")]]
    sig = any(r["q"] < FDR and r["mean"] >= 2 * 2 * COST_BPS for r in primary)
    c = {"significant_and_2x_costs": bool(sig), "iwm_same_sign": bool(t.loc[("IWM", "month end")]["mean"] > 0),
         "years_positive": int((yrs["mean"] > 0).sum()),
         "beats_slid_copies": bool(t.loc[("SPY", "vs slid copies")]["mean"] > 0)}
    c["passes"] = c["significant_and_2x_costs"] and c["iwm_same_sign"] and c["years_positive"] >= 2 \
        and c["beats_slid_copies"]
    return c


def daily_returns(trades, snap, index):
    """Daily snapshot-to-snapshot P&L of one trade a month (whole account), costs on entry and exit days."""
    r = snap.pct_change()
    out = pd.Series(0.0, index=index)
    for t in trades.itertuples():
        days = index[(index > t.entry) & (index <= t.exit)]
        out.loc[days] += t.position * r.reindex(days).fillna(0).to_numpy()
        if t.entry in out.index:
            out.loc[t.entry] -= COST_BPS / 1e4
        if t.exit in out.index:
            out.loc[t.exit] -= COST_BPS / 1e4
    return out


# ---------------------------------------------------------------- report

def bps(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.0f}"


def ci(r):
    if r is None or pd.isna(r.get("mean")):
        return "n/a"
    return f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})"


def render_report(res, port, v):
    tests, table, slid = res["tests"], res["table"], res["slid"]
    lines = []
    w = lines.append
    title = "design screen, 2021-24" if res["period"] == "design" else "held-back check, 2025-26"
    w(f"# Month-end rebalancing flows: {title}\n")
    w("At 15:50 on the 5th-last session of each month: if a 60/40 SPY/IEF portfolio rebalanced at the last month-end "
      "is now overweight stocks, short the stock ETF (long if underweight) until 15:50 on the month's last session. "
      "P&L per trade in bps of the entry price after 1 bp a side; dividend-adjusted 15:50 prices. Rules fixed in "
      "advance (see `rebalancing.py`). A signal study, not account P&L except in the account view.\n")
    w("## Verdict\n")
    w(alerts.md_table(["SPY or SPY-IEF significant, >= 2x costs", "IWM same sign", "Years positive (of 3)",
                       "Beats slid copies", "Passes"],
                      [["yes" if v["significant_and_2x_costs"] else "no", "yes" if v["iwm_same_sign"] else "no",
                        str(v["years_positive"]), "yes" if v["beats_slid_copies"] else "no",
                        "**yes**" if v["passes"] else "no"]]) + "\n")
    w("## The declared tests\n")
    w(alerts.md_table(["#", "Series", "Window", "Trades", "Wins", "Net bps a trade (95%)", "p", "q", "Qualifies"],
                      [[str(i + 1), r["series"], r["window"], f"{int(r['trades'])}", alerts_pct(r.get("win")),
                        ci(r), f"{r['p_up']:.3f}", f"{r['q']:.3f}", "yes" if r["qualifies"] else "no"]
                       for i, r in tests.iterrows()]) + "\n")
    w("## Baselines\n")
    rows = []
    for name in EQUITIES:
        a = table[(table["series"] == name) & (table["window"] == "always long")].iloc[0]
        m = table[(table["series"] == name) & (table["window"] == "month end")].iloc[0]
        s = slid[slid["series"] == name]
        rows.append([name, ci(m), ci(a), f"{bps(s['mean'].mean())} (range {bps(s['mean'].min())} to "
                     f"{bps(s['mean'].max())})", f"{int((s['mean'] >= m['mean']).sum())} of {len(s)}"])
    w(alerts.md_table(["Series", "Rule, month end", "Always long, month end", "Rule slid to other windows",
                       "Slid windows doing as well"], rows) + "\n")
    w("## By year (month-end window)\n")
    by = res["by_year"]
    w(alerts.md_table(["Series", "Year", "Trades", "Net bps (95%)"],
                      [[r["series"], str(r["year"]), f"{int(r['trades'])}", ci(r)] for _, r in by.iterrows()]) + "\n")
    if len(res["by_size"]):
        w("## By the size of the drift (SPY)\n")
        w(alerts.md_table(["Drift from 60/40", "Trades", "Net bps (95%)"],
                          [[r["size"], f"{int(r['trades'])}", ci(r)] for _, r in res["by_size"].iterrows()]) + "\n")
    if port is not None:
        w("## Account view and correlations (SPY rule, one trade a month)\n")
        rows = []
        for _, r in port["account"].iterrows():
            yrs = " / ".join(f"{r[c] * 100:+.1f}%" for c in r.index if str(c).startswith("y") and pd.notna(r[c]))
            rows.append([r["what"], f"{r['total'] * 100:+.1f}%", f"{r['per_year'] * 100:+.1f}%",
                         f"{r['max_dd'] * 100:.1f}%", yrs])
        w(alerts.md_table(["", "Total", "A year", "Worst drop", "Each year"], rows) + "\n")
        w(alerts.md_table(["Daily P&L correlation with", "Correlation", "Sessions both had a position"],
                          [[r["with"], "n/a" if pd.isna(r["corr"]) else f"{r['corr']:+.2f}", f"{r['both_active']}"]
                           for _, r in port["corr"].iterrows()]) + "\n")
    w("## Notes\n")
    w("- Shorts would be an inverse ETF (for example SH for SPY) or a put on Robinhood; the SPY-IEF row needs both "
      "legs. Costs are 1 bp a side for shares.")
    w("- With about 12 trades a year, the design period has fewer than 40 trades, so only a large effect can pass.")
    return "\n".join(lines) + "\n"


def alerts_pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def plot(res, path):
    slid = res["slid"]
    fig = plt.figure(figsize=(9, 4), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.18, 0.86, 0.62])
    for name, color in zip(("SPY", "IWM", "QQQ"), (SERIES[0], SERIES[1], MUTED)):
        s = slid[slid["series"] == name].sort_values("shift")
        me = res["table"][(res["table"]["series"] == name) & (res["table"]["window"] == "month end")]["mean"].iloc[0]
        x = np.r_[s["shift"].to_numpy(), 0]
        y = np.r_[s["mean"].to_numpy(), me]
        o = np.argsort(x)
        ax.plot(x[o], y[o], color=color, lw=1.3, marker="o", ms=3, label=name)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.axvline(0, color=INK_2, lw=0.6, ls="--")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("Window shift in sessions (0 = the last four sessions of the month)", color=INK_2, fontsize=7.5)
    ax.set_ylabel("Net bps a trade", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, f"Rebalancing rule at each window of the month ({res['period']})", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.legend(loc="upper right", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Month-end rebalancing flows on SPY, IWM and QQQ.")
    p.add_argument("--out", default="output/rebalancing")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.final_test and FINAL is None:
        raise SystemExit("FINAL is not set: write the held-back rules down before running --final-test.")
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    with timed(timings, "data"):
        daily, notes = mr.load_daily(sessions, args.cache_dir, args.refresh, TICKERS)
        snaps = {t: daily[t]["snap"] for t in TICKERS}
    period = "holdout" if args.final_test else "design"
    res = run_study(snaps, period=period, timings=timings)
    v = verdict(res)
    idx = snaps["SPY"].index
    index = idx[(idx <= DESIGN_END) if period == "design" else (idx >= HOLDOUT_START)]
    index = index[index >= res["trades"][("SPY", "month end")]["entry"].min()]
    port = None
    with timed(timings, "account view"):
        lead = "SPY" if period == "design" else "IWM"
        daily_pnl = daily_returns(res["trades"][(lead, "month end")], snaps[lead], index)
        spy = snaps["SPY"].pct_change().reindex(index).fillna(0)
        dip = pc.spy_dip_daily(index, args.cache_dir)
        acct = pd.DataFrame([{"what": f"{lead} rule, one trade a month", **pc.account(daily_pnl)},
                             {"what": "holding SPY", **pc.account(spy)},
                             {"what": "SPY dip buying (shares)", **pc.account(dip)}])
        corr = pc.correlations(daily_pnl, {"SPY dip buying": dip, "TSLA breakouts": pc.tsla_breakout_daily(index),
                                           "SPY": spy})
        port = {"account": acct, "corr": corr}
    out = Path(args.out) / period
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([t.assign(series=n, window=k) for (n, k), t in res["trades"].items()], ignore_index=True).to_csv(
        out / "trades.csv", index=False)
    for k in ("table", "slid", "tests", "by_year", "by_size"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    (out / "settings.json").write_text(json.dumps({"run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": period,
                                                   "verdict": v, "notes": notes}, indent=2, default=str))
    plot(res, out / "windows.png")
    (out / "report.md").write_text(render_report(res, port, v))
    print(res["tests"].to_string(index=False))
    print(v)
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
