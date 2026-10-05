"""Trend following on ETFs: long in uptrends, short (or out) in downtrends. Does it make money, and does it do well
when SPY dip buying does badly? A daily study on Massive minute data (no options); returns are hypothetical.

  uv run --env-file .env python trend_etfs.py --out output/trend_etfs

Fixed before any result was seen:
- ETFs: SPY, QQQ, IWM (stocks), TLT (long Treasuries), GLD (gold), UUP (US dollar), HYG (high-yield bonds),
  dividend-adjusted. Decisions at 15:50 ET each session (meanrev's snapshot), held to the next snapshot.
- Trend rules (classic settings, not tuned): the sign of the 1-month (21-session) and 3-month (63-session)
  return, and the price above or below its 50- and 100-session average; each long/short and long/flat.
- Portfolio: each ETF sized to the same risk (10% a year from its 60-session volatility, at most 2x) with a 1/7
  sleeve each. Primary: 3-month momentum, long/short.
- Costs: 2 bps per unit of position traded, in every result (gross figures in the CSV).
- Period: decisions from 2022-03-01 (once the 100-session average exists) to the last session. Benchmarks: SPY
  bought and held, the same equal-risk sleeves bought and held, and SPY dip buying (meanrev's "3 down days; exit
  on the first up day", at most 5 sessions), alone and 50/50 with the trend portfolio.
- Only one bear market fits in five years and the 2022 trend is well known, so this checks the idea rather than
  proving it; the 6-12-month and 200-day rules need more history than the data plan's five years.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
ETFS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "HYG")
EVAL_START = pd.Timestamp("2022-03-01")
TARGET_VOL, MAX_WEIGHT, VOL_WINDOW = 0.10, 2.0, 60
COST_BPS = 2.0
RULES = {"1-month momentum": ("mom", 21), "3-month momentum": ("mom", 63),
         "50-day average": ("sma", 50), "100-day average": ("sma", 100)}
VERSIONS = ("long/short", "long/flat")
PRIMARY = ("3-month momentum", "long/short")


def trend_signal(prices, rule, n):
    """+1 / -1 at each session from prices up to and including it (NaN until the lookback is filled)."""
    if rule == "mom":
        return np.sign(prices / prices.shift(n) - 1)
    return np.sign(prices - prices.rolling(n, min_periods=n).mean())


def risk_weights(returns):
    """Position size per ETF for a 10%-a-year risk target from the last 60 daily returns, capped at 2x."""
    vol = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * math.sqrt(252)
    return (TARGET_VOL / vol).clip(upper=MAX_WEIGHT)


def sleeve_returns(positions, returns, cost_bps=COST_BPS):
    """Daily return of each sleeve: the position decided at session t earns the move to t+1, less the cost of
    changing the position at t. Missing positions count as flat."""
    pos = positions.fillna(0.0)
    traded = pos.diff().abs().fillna(pos.abs())
    return pos.shift(1) * returns - cost_bps / 1e4 * traded.shift(1)


def portfolio(positions, returns, start=EVAL_START, cost_bps=COST_BPS):
    """Equal sleeves (1/N each) of the positions; daily returns from the first decision on or after `start`."""
    r = sleeve_returns(positions, returns, cost_bps).mean(axis=1)
    first = positions.index[positions.index >= start][0]
    return r[r.index > first]


def dip_positions(f, max_hold=5):
    """SPY dip buying, one position at a time: enter at a snapshot after 3+ down days, exit at the first later
    snapshot after an up day or after max_hold sessions. 1 = holding into the next session."""
    streak, up = f["streak"].to_numpy(), (f["ret_1d"] > 0).to_numpy()
    pos, held, holding = np.zeros(len(f)), 0, False
    for i in range(len(f)):
        if holding:
            held += 1
            if up[i] or held >= max_hold:
                holding = False
        if not holding and streak[i] <= -3:
            holding, held = True, 0
        pos[i] = holding
    return pd.Series(pos, index=f.index)


def perf(r):
    """Annualized return (compounded), volatility, Sharpe (no risk-free rate), worst drawdown, share of up months
    and the return in each calendar year."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    months = r.groupby(r.index.to_period("M")).apply(lambda g: (1 + g).prod() - 1)
    out = {"days": len(r), "cagr": eq.iloc[-1] ** (1 / years) - 1, "vol": r.std() * math.sqrt(252),
           "max_dd": (eq / eq.cummax() - 1).min(), "up_months": (months > 0).mean() * 100}
    out["sharpe"] = r.mean() * 252 / out["vol"] if out["vol"] else np.nan
    for y, g in r.groupby(r.index.year):
        out[f"y{y}"] = (1 + g).prod() - 1
    return out


def run_study(prices, spy_features, *, timings=None):
    """prices: dividend-adjusted 15:50 snapshots, one column per ETF. No file I/O."""
    timings = {} if timings is None else timings
    with timed(timings, "study"):
        returns = prices.pct_change(fill_method=None)
        weights = risk_weights(returns)
        series, rows, sleeves = {}, [], {}
        for name, (rule, n) in RULES.items():
            signal = trend_signal(prices, rule, n)
            for version in VERSIONS:
                pos = (signal if version == "long/short" else signal.clip(lower=0)) * weights
                for costs in (COST_BPS, 0.0):
                    r = portfolio(pos, returns, cost_bps=costs)
                    if costs:
                        series[name, version] = r
                        if (name, version) == PRIMARY:
                            sleeves = sleeve_returns(pos, returns)
                    rows.append({"rule": name, "version": version, "costs": "after costs" if costs else "gross",
                                 **perf(r)})
        hold = portfolio(weights, returns)
        spy = returns["SPY"][returns.index > EVAL_START]
        dip = (dip_positions(spy_features).shift(1) * returns["SPY"])[returns.index > EVAL_START]
        trend = series[PRIMARY]
        combo = 0.5 * trend + 0.5 * dip.reindex(trend.index).fillna(0.0)
        bench = {"SPY bought and held": spy, "The 7 ETFs bought and held (equal risk)": hold,
                 "SPY dip buying": dip, "Trend (primary)": trend, "50/50 trend and dip buying": combo}
        for name, r in bench.items():
            rows.append({"rule": name, "version": "benchmark", "costs": "after costs", **perf(r)})
        table = pd.DataFrame(rows)
        corr = pd.DataFrame({k: v for k, v in bench.items()}).corr()
        corr_2022 = pd.DataFrame({k: v[v.index.year == 2022] for k, v in bench.items()}).corr()
        per_etf = sleeves[sleeves.index > EVAL_START].groupby(sleeves.index[sleeves.index > EVAL_START].year).apply(
            lambda g: (1 + g).prod() - 1)
    return {"table": table, "series": series, "bench": bench, "corr": corr, "corr_2022": corr_2022,
            "per_etf": per_etf, "first": EVAL_START, "last": prices.index[-1]}


# ---------------------------------------------------------------- report

def pc(v, digits=1, sign=True):
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"


def row_of(table, rule, version, costs="after costs"):
    m = table[(table["rule"] == rule) & (table["version"] == version) & (table["costs"] == costs)]
    return m.iloc[0] if len(m) else None


def render_report(res):
    t = res["table"]
    years = sorted(int(c[1:]) for c in t.columns if c.startswith("y") and c[1:].isdigit())
    lines = []
    w = lines.append
    w("# Trend following on ETFs\n")
    w(f"Daily decisions at 15:50 ET from {res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}, dividend-adjusted, after "
      f"{COST_BPS:g} bps per unit traded. Each of SPY, QQQ, IWM, TLT, GLD, UUP and HYG gets a 1/7 sleeve sized to 10% "
      "a year of risk. Hypothetical returns of a signal study: no slippage model, no borrow costs for shorts, and no "
      "cash interest.\n")
    w(bottom_line(res))
    w("## 1. Every rule, all seven ETFs together\n")
    head = ["Rule", "Version", "A year", "Volatility", "Sharpe", "Worst drawdown", *[str(y) for y in years],
            "Corr. with SPY"]
    rows = []
    for name in RULES:
        for version in VERSIONS:
            r = row_of(t, name, version)
            c = res["series"][name, version].corr(res["bench"]["SPY bought and held"])
            rows.append([name, version, pc(r["cagr"]), pc(r["vol"], sign=False), f"{r['sharpe']:.2f}", pc(r["max_dd"]),
                         *[pc(r.get(f"y{y}")) for y in years], f"{c:+.2f}"])
    for name in res["bench"]:
        if name == "Trend (primary)":
            continue
        r = row_of(t, name, "benchmark")
        c = res["bench"][name].corr(res["bench"]["SPY bought and held"])
        rows.append([f"*{name}*", "", pc(r["cagr"]), pc(r["vol"], sign=False), f"{r['sharpe']:.2f}", pc(r["max_dd"]),
                     *[pc(r.get(f"y{y}")) for y in years], f"{c:+.2f}"])
    w(alerts.md_table(head, rows))
    w(f"\n{years[-1]} is a partial year. Sharpe here is return over volatility, without a risk-free rate.\n")
    w("## 2. The primary rule by ETF\n")
    w(f"{PRIMARY[0]}, {PRIMARY[1]}: each sleeve's return by year (a sleeve is 1/7 of the portfolio, so the "
      "portfolio's year is roughly the average):\n")
    pe = res["per_etf"]
    w(alerts.md_table(["ETF", *[str(y) for y in pe.index]],
                      [[c, *[pc(pe.loc[y, c]) for y in pe.index]] for c in pe.columns]) + "\n")
    w("## 3. Next to SPY dip buying\n")
    rows = []
    for name in ("SPY dip buying", "Trend (primary)", "50/50 trend and dip buying", "SPY bought and held"):
        r = row_of(t, name, "benchmark")
        rows.append([name, pc(r["cagr"]), pc(r["vol"], sign=False), f"{r['sharpe']:.2f}", pc(r["max_dd"]),
                     f"{r['up_months']:.0f}%", *[pc(r.get(f"y{y}")) for y in years]])
    w(alerts.md_table(["", "A year", "Volatility", "Sharpe", "Worst drawdown", "Up months",
                       *[str(y) for y in years]], rows) + "\n")
    c, c22 = res["corr"], res["corr_2022"]
    w(f"Daily correlation of the trend portfolio with dip buying: {c.loc['Trend (primary)', 'SPY dip buying']:+.2f} "
      f"overall, {c22.loc['Trend (primary)', 'SPY dip buying']:+.2f} in 2022; with SPY: "
      f"{c.loc['Trend (primary)', 'SPY bought and held']:+.2f} overall, "
      f"{c22.loc['Trend (primary)', 'SPY bought and held']:+.2f} in 2022. The dip rule holds SPY only on a few days, "
      "so its own correlation with SPY is low too.\n")
    w("## 4. Notes\n")
    w("- Rules use classic settings, untuned; the 6-12-month and 200-day versions need more than five years of data.")
    w("- Only one bear market (2022) fits, and trend following's good 2022 was widely reported, so treat this as a "
      "check of the idea, not proof.")
    w("- The short side needs margin to short ETFs (Robinhood does not allow short selling); with options it would be "
      "puts or put spreads.")
    w("- `rules.csv` has every rule before and after costs; `daily.csv` the daily returns.")
    return "\n".join(lines) + "\n"


def bottom_line(res):
    t = res["table"]
    p = row_of(t, *PRIMARY)
    spy, dip, combo = (row_of(t, n, "benchmark") for n in ("SPY bought and held", "SPY dip buying",
                                                           "50/50 trend and dip buying"))
    c = res["corr"].loc["Trend (primary)", "SPY dip buying"]
    return (f"**Bottom line.** The primary rule ({PRIMARY[0]}, {PRIMARY[1]}) returned {pc(p['cagr'])} a year "
            f"(Sharpe {p['sharpe']:.2f}, worst drawdown {pc(p['max_dd'])}); in 2022 {pc(p.get('y2022'))} while SPY "
            f"returned {pc(spy.get('y2022'))} and dip buying {pc(dip.get('y2022'))}. Its daily correlation with dip "
            f"buying is {c:+.2f}. Half and half with dip buying: {pc(combo['cagr'])} a year, worst drawdown "
            f"{pc(combo['max_dd'])}, against {pc(dip['max_dd'])} for dip buying alone.\n")


def plot(res, path):
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.1, 0.2, 0.86, 0.6])
    styles = {"SPY bought and held": (MUTED, 1.4), "SPY dip buying": (SERIES[1], 1.6),
              "Trend (primary)": (SERIES[0], 2.0), "50/50 trend and dip buying": (TARGET, 2.0)}
    for name, (color, lw) in styles.items():
        r = res["bench"][name].dropna()
        ax.plot(r.index, (1 + r).cumprod(), color=color, lw=lw, label=name)
    ax.axvspan(pd.Timestamp("2022-03-01"), pd.Timestamp("2022-10-12"), color=GRID, alpha=0.6, lw=0)
    ax.axhline(1, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Growth of $1 (hypothetical)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "Trend following next to SPY dip buying", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Trend: {PRIMARY[0]}, {PRIMARY[1]}, seven ETFs at equal risk, after costs. Shaded: the 2022 "
             "decline to its October low.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Trend following on ETFs, next to SPY dip buying.")
    p.add_argument("--out", default="output/trend_etfs")
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
        daily, _ = mr.load_daily(sessions, args.cache_dir, args.refresh, ETFS)
        prices = pd.DataFrame({t: daily[t]["snap"] for t in ETFS})
        spy_features = mr.ticker_features(daily["SPY"]).reindex(sessions.index)
    res = run_study(prices, spy_features, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "rules.csv", index=False)
    pd.DataFrame(res["bench"]).to_csv(out / "daily.csv")
    plot(res, out / "trend.png")
    (out / "report.md").write_text(render_report(res))
    print(bottom_line(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
