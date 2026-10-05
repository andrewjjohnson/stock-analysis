"""A scalping strategy built on the five indicators the Reddit author lists, rebuilt and combined into one rule, on
SPY 5-minute bars. A trade simulation per share of the underlying (1-tick slippage, no commissions), not options P&L.

  uv run --env-file .env python confluence_scalper.py --out output/confluence_scalper                # 2021-24
  uv run --env-file .env python confluence_scalper.py --final-test --out output/confluence_scalper   # adds 2025-26, once

The indicators (each reimplemented here from its published logic; LuxAlgo's scripts are CC BY-NC-SA 4.0, used for
non-commercial research):
- MACD Custom (ChrisMoody, CM_MacD_Ult_MTF): EMA 12 - EMA 26, signal = SMA 9 of the MACD (the original uses an SMA).
- Squeeze Momentum (LazyBear): momentum = linear regression (20) of close minus the average of the 20-bar midrange
  and SMA 20; squeeze = Bollinger inside Keltner (the original code uses the 1.5 Keltner multiplier for the Bollinger width
  too, kept here).
- SuperTrend AI (LuxAlgo, clustering): SuperTrends for factors 1.0-5.0 (step 0.5) on ATR 10, each scored by an
  exponentially smoothed measure of how well it has called the next move (memory 10); k-means (3 clusters, started
  at the 25th/50th/75th percentiles) on those scores; the "best" cluster's average factor drives the final
  SuperTrend, whose direction is the trend.
- Pure Price Action (LuxAlgo): approximated, as its code isn't at hand, with Larry Williams swing structure:
  short-term swing highs/lows (a bar beyond both neighbours), intermediate-term ones (a short-term swing beyond the
  short-term swings either side); the structure is bullish after a close above the last intermediate swing high,
  bearish after a close below the last intermediate swing low.
- ADX Volatility Waves (BOSWaves): the approximation from kalman_supertrend.py (VWMA 50 equilibrium, outer waves at
  2 x the RMA 10 of stdev 20 x 1.5 x (1 + 0.8 ADX 14 / 100)).

Fixed before any result was seen:
- Bars and fills as kalman_supertrend.py: SPY 5-minute bars 04:00-20:00 ET, indicators over all of them, entries on
  bars starting 9:30-15:55 ET filled at the next minute's open plus a tick; flat by the close (built for 0DTE
  options). QQQ, IWM and TSLA reported for reading only.
- BUY on the bar where all of these become true together (they were not all true on the bar before): SuperTrend AI
  bullish; MACD above its signal; squeeze momentum above zero and rising; structure bullish; close below the upper
  outer wave. SELL is the mirror.
- Exits: (A, primary) kalman_supertrend.py's: stop 1.5 ATR(7), target 2 ATR, breakeven after 1 ATR, the opposite
  outer wave, 12 bars, an opposite signal, the regular close; (B) hold until SuperTrend AI turns against the trade
  (out at the next minute's open) or the regular close.
- Comparisons: plain SuperTrend AI flips (no filters) with the same exits; random entries (20 per trade, same session
  and direction, at random regular-hours bars where SuperTrend AI points that way) with the same exits.
- Tests on 2021-10 to 2024-12: mean net bps per trade > 0 and signal minus random > 0 (one-sided, clustered by
  session), for exits A and B; BH q < 0.10 across those four. 2025-26 is read only with --final-test.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import kalman_supertrend as ks  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM", "TSLA")
FACTORS = tuple(np.arange(1.0, 5.01, 0.5))
ST_ATR, PERF_MEMORY = 10, 10
DESIGN_END = pd.Timestamp("2024-12-31")
REPS = 20
SEED = 9
FDR = 0.10


# ---------------------------------------------------------------- indicators

def cm_macd(close, fast=12, slow=26, signal=9):
    """ChrisMoody's MACD: (macd, signal, histogram) with an SMA signal line."""
    macd = talib.EMA(close, fast) - talib.EMA(close, slow)
    sig = talib.SMA(macd, signal)
    return macd, sig, macd - sig


def squeeze_momentum(high, low, close, length=20, mult_kc=1.5):
    """LazyBear's Squeeze Momentum: (momentum value, squeeze on). Bollinger width uses mult_kc, as in the original code."""
    basis = talib.SMA(close, length)
    dev = mult_kc * talib.STDDEV(close, length, 1)
    rangema = talib.SMA(talib.TRANGE(high, low, close), length)
    on = (basis - dev > basis - rangema * mult_kc) & (basis + dev < basis + rangema * mult_kc)
    mid = (talib.MAX(high, length) + talib.MIN(low, length)) / 2
    val = talib.LINEARREG(close - (mid + basis) / 2, length)
    return val, on


def _percentile(sorted_vals, p):
    pos = p / 100 * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def kmeans3(values, labels, max_iter=1000):
    """LuxAlgo's 1-D k-means with 3 clusters started at the 25th/50th/75th percentiles: (value groups, label
    groups), cluster 0 lowest. Ties go to the lower cluster; an empty cluster keeps a missing centroid."""
    s = sorted(values)
    cent = [_percentile(s, 25), _percentile(s, 50), _percentile(s, 75)]
    for _ in range(max_iter + 1):
        groups, lab = ([], [], []), ([], [], [])
        for v, f in zip(values, labels):
            d = [abs(v - c) if c == c else math.inf for c in cent]
            k = d.index(min(d))
            groups[k].append(v)
            lab[k].append(f)
        new = [sum(g) / len(g) if g else math.nan for g in groups]
        if all(a == b or (a != a and b != b) for a, b in zip(new, cent)):
            break
        cent = new
    return groups, lab


def supertrend_ai(high, low, close, factors=FACTORS, atr_len=ST_ATR, memory=PERF_MEMORY, cluster=2):
    """LuxAlgo's SuperTrend AI: (trend 1/0, trailing stop, target factor, performance index)."""
    n, m = len(close), len(factors)
    f = np.asarray(factors, float)
    atr = talib.ATR(high, low, close, atr_len)
    hl2 = (high + low) / 2
    alpha = 2 / (memory + 1)
    upper, lower = np.full(m, hl2[0]), np.full(m, hl2[0])
    output, perf, trend = np.full(m, np.nan), np.zeros(m), np.zeros(m)
    den = talib.EMA(np.abs(np.r_[np.nan, np.diff(close)]), memory)
    os_, ts, tf, pidx = np.zeros(n, int), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    target, up_f, lo_f, state, prev_idx = np.nan, np.nan, np.nan, 0, np.nan
    for i in range(n):
        c, cp = close[i], close[i - 1] if i else np.nan
        up, dn = hl2[i] + atr[i] * f, hl2[i] - atr[i] * f
        trend = np.where(c > upper, 1.0, np.where(c < lower, 0.0, trend))
        upper = np.where(cp < upper, np.minimum(up, upper), up)
        lower = np.where(cp > lower, np.maximum(dn, lower), dn)
        diff = np.nan_to_num(np.sign(cp - output))
        move = 0.0 if np.isnan(cp) else c - cp
        perf = perf + alpha * (move * diff - perf)
        output = np.where(trend == 1, lower, upper)
        if np.isfinite(atr[i]):
            groups, labs = kmeans3(perf.tolist(), f.tolist())
            if labs[cluster]:
                target = sum(labs[cluster]) / len(labs[cluster])
            if np.isfinite(den[i]) and den[i] > 0:
                prev_idx = max(sum(groups[cluster]) / len(groups[cluster]) if groups[cluster] else 0.0, 0.0) / den[i]
        tf[i], pidx[i] = target, prev_idx
        if np.isnan(target):
            continue
        u, d = hl2[i] + atr[i] * target, hl2[i] - atr[i] * target
        up_f = min(u, up_f) if (i and close[i - 1] < up_f) else u   # comparisons with a missing band are False
        lo_f = max(d, lo_f) if (i and close[i - 1] > lo_f) else d
        state = 1 if c > up_f else 0 if c < lo_f else state
        os_[i], ts[i] = state, lo_f if state else up_f
    return os_, ts, tf, pidx


def swing_structure(high, low, close):
    """Larry Williams swings as a stand-in for LuxAlgo's Pure Price Action: the structure (+1 bullish, -1 bearish,
    0 not yet set) after closes beyond the last intermediate-term swing high or low. A short-term swing is known one
    bar after it prints; an intermediate one when the next short-term swing on its side is known."""
    n = len(close)
    out = np.zeros(n, int)
    st_hi, st_lo = [], []          # (bar, price) of short-term swings, known so far
    level_hi = level_lo = np.nan   # last intermediate swing high / low not yet broken
    state = 0
    for i in range(2, n):
        j = i - 1                  # bar j is a short-term swing once bar i (after it) is known
        if high[j] > high[j - 1] and high[j] > high[i]:
            st_hi.append((j, high[j]))
            if len(st_hi) >= 3 and st_hi[-2][1] > st_hi[-3][1] and st_hi[-2][1] > st_hi[-1][1]:
                level_hi = st_hi[-2][1]
        if low[j] < low[j - 1] and low[j] < low[i]:
            st_lo.append((j, low[j]))
            if len(st_lo) >= 3 and st_lo[-2][1] < st_lo[-3][1] and st_lo[-2][1] < st_lo[-1][1]:
                level_lo = st_lo[-2][1]
        if close[i] > level_hi:
            state, level_hi = 1, np.nan
        elif close[i] < level_lo:
            state, level_lo = -1, np.nan
        out[i] = state
    return out


def indicators(bars):
    """kalman_supertrend.indicators (waves, ATR 7, session flags) plus the five indicators and the signals: `sig`
    (the confluence rule), `st_sig` (plain SuperTrend AI flips), `st_dir` (+1/-1)."""
    b = ks.indicators(bars)
    h, l, c = (b[k].to_numpy(float) for k in ("high", "low", "close"))
    b["macd"], b["macd_sig"], _ = cm_macd(c)
    b["sqz_val"], b["sqz_on"] = squeeze_momentum(h, l, c)
    os_, b["st_ts"], b["st_factor"], b["st_perf"] = supertrend_ai(h, l, c)
    b["st_dir"] = np.where(np.isnan(b["st_factor"]), 0, np.where(os_ == 1, 1, -1))
    b["structure"] = swing_structure(h, l, c)
    rising = b["sqz_val"] > b["sqz_val"].shift(1)
    falling = b["sqz_val"] < b["sqz_val"].shift(1)
    long_ok = ((b["st_dir"] == 1) & (b["macd"] > b["macd_sig"]) & (b["sqz_val"] > 0) & rising
               & (b["structure"] == 1) & (b["close"] < b["upper_outer"]))
    short_ok = ((b["st_dir"] == -1) & (b["macd"] < b["macd_sig"]) & (b["sqz_val"] < 0) & falling
                & (b["structure"] == -1) & (b["close"] > b["lower_outer"]))
    start_long = long_ok & ~long_ok.shift(1, fill_value=False)
    start_short = short_ok & ~short_ok.shift(1, fill_value=False)
    b["sig"] = np.where(b["rth"], np.where(start_long, 1, np.where(start_short, -1, 0)), 0)
    flip = np.r_[0, np.diff(b["st_dir"].to_numpy())]
    b["st_sig"] = np.where(b["rth"] & (b["st_dir"].shift(1).fillna(0) != 0), np.sign(flip), 0).astype(int)
    return b


# ---------------------------------------------------------------- trades with exit B

def trend_trade_from(i, d, ctx, st_dir, tick=ks.PARAMS["tick"]):
    """Exit B: from the close of bar i, hold until SuperTrend AI points against the trade at a bar's close (out at
    the next minute's open) or the regular close."""
    t, mo, mc = ctx["t"], ctx["mo"], ctx["mc"]
    last_rth, m_hi, ends = ctx["last_rth"], ctx["m_hi"], ctx["ends"]
    k0 = int(np.searchsorted(t, ends[i]))
    if k0 >= len(t):
        return None
    fill = mo[k0] + d * tick
    j = i
    while j + 1 < ctx["n"]:
        j += 1
        if last_rth[j]:
            k = m_hi[j] - 1
            return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": mc[k] - d * tick, "exit_bar": j,
                    "reason": "regular close"}
        if st_dir[j] == -d:
            k = int(np.searchsorted(t, ends[j]))
            k = min(k, len(t) - 1)
            return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": mo[k] - d * tick, "exit_bar": j,
                    "reason": "trend turned"}
    return None


def trend_strategy(bars, m, sig):
    """Exit-B trades for signal array `sig` (one at a time; signals while a trade is open are ignored)."""
    ctx = ks.context(bars, m)
    st_dir, last_rth = bars["st_dir"].to_numpy(), ctx["last_rth"]
    rows, busy_until = [], -1
    for s in np.flatnonzero(sig != 0):
        if s < busy_until or last_rth[s]:
            continue
        tr = trend_trade_from(s, int(sig[s]), ctx, st_dir)
        if tr is None:
            continue
        rows.append({**tr, "dir": int(sig[s]), "bar": s})
        busy_until = tr["exit_bar"]
    return ks.trade_log(rows, bars, m)


def random_entries(real, bars, m, exit_b=False, reps=REPS, seed=SEED):
    """`reps` trades per real trade on its session, from random regular-hours bars where SuperTrend AI points the
    trade's way (not the last bar), with the same exits."""
    ctx = ks.context(bars, m)
    st_dir = bars["st_dir"].to_numpy()
    ok = bars["rth"].to_numpy() & ~bars["last_rth"].to_numpy()
    day = bars["day"].to_numpy()
    pools = {}
    for d in pd.DatetimeIndex(real["day"].unique()):
        today = (day == d.to_datetime64()) & ok
        pools[d] = {+1: np.flatnonzero(today & (st_dir == 1)), -1: np.flatnonzero(today & (st_dir == -1))}
    rng = np.random.default_rng(seed)
    rows = []
    for r in real.itertuples():
        pool = pools.get(pd.Timestamp(r.day), {}).get(int(r.dir))
        if pool is None or not len(pool):
            continue
        for i in rng.choice(pool, size=reps):
            tr = (trend_trade_from(int(i), int(r.dir), ctx, st_dir) if exit_b else
                  ks.trade_from(int(i), int(r.dir), ctx, flat_by_close=True))
            if tr is not None:
                rows.append({**tr, "dir": int(r.dir), "bar": int(i)})
    return ks.trade_log(rows, bars, m)


# ---------------------------------------------------------------- study

def run_study(minutes_by_ticker, sessions, *, final=False, primary="SPY", timings=None, reps=REPS):
    """minutes_by_ticker: {ticker: minute bars with extended hours}. No file I/O. Without `final`, trades after
    2024-12-31 are dropped before any statistic."""
    timings = {} if timings is None else timings
    rows, cmp, moves, trades, signals = [], [], [], {}, []
    for tk, minutes in minutes_by_ticker.items():
        with timed(timings, "bars and indicators"):
            bars, m = ks.five_minute_bars(minutes, sessions)
            bars = indicators(bars)
        keep = (lambda t: t) if final else (lambda t: t[t["day"] <= DESIGN_END])
        periods = ["2021-24", "2025-26"] if final else ["2021-24"]
        rth_days = bars.loc[bars["rth"], "day"]
        for rule, col in (("all five", "sig"), ("SuperTrend AI alone", "st_sig")):
            sig = bars[col].to_numpy()
            signals.append({"ticker": tk, "rule": rule,
                            "per_session": (keep(bars.loc[sig != 0, ["day"]])["day"].size
                                            / max(keep(pd.DataFrame({"day": rth_days.unique()})).shape[0], 1))})
            for exit_name in ("A", "B"):
                if tk != primary and (rule != "all five" or exit_name != "A"):
                    continue
                with timed(timings, "trades"):
                    b2 = bars.assign(sig=sig)
                    t = keep(ks.strategy(b2, m, flat_by_close=True) if exit_name == "A" else trend_strategy(b2, m, sig))
                trades[(tk, rule, exit_name)] = t
                for period in periods:
                    part = t[t["day"] <= DESIGN_END] if period == "2021-24" else t[t["day"] > DESIGN_END]
                    days = rth_days[rth_days <= DESIGN_END] if period == "2021-24" else rth_days[rth_days > DESIGN_END]
                    rows.append({"ticker": tk, "rule": rule, "exit": exit_name, "period": period,
                                 **ks.stats(part, ks.years_between(days))})
                if tk == primary or rule == "all five":
                    with timed(timings, "random entries"):
                        rand = keep(random_entries(t, b2, m, exit_b=exit_name == "B", reps=reps))
                    for period in periods:
                        sel = (lambda x: x[x["day"] <= DESIGN_END]) if period == "2021-24" else (
                            lambda x: x[x["day"] > DESIGN_END])
                        cmp.append({"ticker": tk, "rule": rule, "exit": exit_name, "period": period,
                                    **ks.compare(sel(t), sel(rand))})
        with timed(timings, "signal moves"):
            mv = ks.flag_moves(bars, m)
            mv = mv if final else mv[mv["day"] <= DESIGN_END]
            moves.append(ks.move_stats(mv).assign(ticker=tk))
    stats, compare = pd.DataFrame(rows), pd.DataFrame(cmp)
    tests = stats[(stats["ticker"] == primary) & (stats["rule"] == "all five") & (stats["period"] == "2021-24")].merge(
        compare[["ticker", "rule", "exit", "period", "excess", "excess_lo", "excess_hi", "excess_p_up",
                 "random_mean"]], on=["ticker", "rule", "exit", "period"], how="left")
    p = np.r_[tests["p_up"].fillna(1).to_numpy(), tests["excess_p_up"].fillna(1).to_numpy()]
    q = mr.bh_qvalues(p)
    tests["q_mean"], tests["q_excess"] = q[:len(tests)], q[len(tests):]
    return {"stats": stats, "compare": compare, "tests": tests, "moves": pd.concat(moves, ignore_index=True),
            "trades": trades, "signals": pd.DataFrame(signals), "final": final, "primary": primary,
            "tickers": list(minutes_by_ticker)}


# ---------------------------------------------------------------- report

bps, find = ks.bps, ks.find


def render_report(res):
    s, c, tests, mv, sg = res["stats"], res["compare"], res["tests"], res["moves"], res["signals"]
    pk = res["primary"]
    lines = []
    w = lines.append
    w("# A scalper built on the five indicators\n")
    w("MACD (ChrisMoody), Squeeze Momentum (LazyBear), SuperTrend AI (LuxAlgo), a Larry Williams swing-structure "
      "stand-in for Pure Price Action (LuxAlgo) and the ADX Volatility Waves approximation, combined: BUY when "
      "SuperTrend AI is bullish, MACD is above its signal, squeeze momentum is positive and rising, structure is "
      "bullish and price is below the upper wave, all together for the first time; SELL the mirror. SPY 5-minute bars, "
      "regular-hours entries a minute after the signal, 1-tick slippage, flat by the close. Exit A: stop 1.5 ATR, "
      "target 2 ATR, breakeven after 1 ATR, the opposite wave, 12 bars or an opposite signal. Exit B: hold until "
      "SuperTrend AI turns. *Random* = entries at random regular-hours bars of the same session while SuperTrend "
      "AI points the same way, same exits. P&L in basis points per trade (1 bp = $1 per $10,000). "
      + ("2021-24 and 2025-26." if res["final"] else "2021-24 only (2025-26 not read).") + "\n")
    w("**Tests (SPY, 2021-24).**\n")
    rows = [[r.exit, f"{int(r.trades):,} ({r.per_year:.0f})", f"{r.win:.0f}%",
             f"{bps(r.mean)} ({bps(r.lo)} to {bps(r.hi)})", f"{r.q_mean:.2f}", bps(r.random_mean),
             f"{bps(r.excess)} ({bps(r.excess_lo)} to {bps(r.excess_hi)})", f"{r.q_excess:.2f}"]
            for r in tests.itertuples()]
    w(alerts.md_table(["Exit", "Trades (a year)", "Wins", "Net bps a trade (95%)", "q", "Random, bps",
                       "Signal minus random (95%)", "q"], rows) + "\n")
    w(f"## {pk}: every rule, exit and period\n")
    rows = []
    for r in s[s["ticker"] == pk].itertuples():
        cr = find(c, ticker=pk, rule=r.rule, exit=r.exit, period=r.period)
        rows.append([r.rule, r.exit, r.period, f"{int(r.trades):,} ({r.per_year:.0f})", f"{r.win:.0f}%",
                     f"{bps(r.avg_win)} / {bps(r.avg_loss)}", f"{bps(r.mean)} ({bps(r.lo)} to {bps(r.hi)})",
                     f"{r.pf:.2f}", "n/a" if cr is None else bps(cr.get("random_mean")),
                     "n/a" if cr is None else f"{bps(cr.get('excess'))} ({bps(cr.get('excess_lo'))} to "
                                              f"{bps(cr.get('excess_hi'))})",
                     f"{r.script_net:+,.0f} $", f"{r.whole_pct:+.1f}%"])
    w(alerts.md_table(["Rule", "Exit", "Period", "Trades (a year)", "Wins", "Avg win / loss", "Net bps (95%)",
                       "Profit factor", "Random", "Minus random (95%)", "$3k a trade", "Whole account"], rows) + "\n")
    w("## Signals per session\n")
    rows = [[r.ticker, r.rule, f"{r.per_session:.2f}"] for r in sg.itertuples()]
    w(alerts.md_table(["Ticker", "Rule", "Signals per session"], rows) + "\n")
    w("## Other tickers (all five, exit A, reading only)\n")
    rows = []
    for tk in res["tickers"]:
        if tk == pk:
            continue
        for period in ("2021-24", "2025-26"):
            r = find(s, ticker=tk, rule="all five", exit="A", period=period)
            cr = find(c, ticker=tk, rule="all five", exit="A", period=period)
            if r is None or pd.isna(r.get("mean")):
                continue
            rows.append([tk, period, f"{int(r['trades']):,}", f"{r['win']:.0f}%",
                         f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})",
                         "n/a" if cr is None else bps(cr.get("excess"))])
    w(alerts.md_table(["Ticker", "Period", "Trades", "Wins", "Net bps (95%)", "Minus random"], rows) + "\n")
    w("## The move after each signal (all five), against any bar at the same time of day\n")
    rows = [[r.ticker, f"{r.minutes} min", f"{r.flags:,}", f"{bps(r.excess)} ({bps(r.lo)} to {bps(r.hi)})",
             f"{r.hit:.0f}%"] for r in mv.itertuples()]
    w(alerts.md_table(["Ticker", "After", "Signals", "Excess move (95%)", "Moved the signal's way"], rows) + "\n")
    w("## Notes\n")
    w("- Pure Price Action and ADX Volatility Waves are approximations (their exact code isn't available); MACD, "
      "Squeeze Momentum and SuperTrend AI follow their published logic.")
    w("- Shares, not options: per share of the ETF, gross of commissions.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    pk = res["primary"]
    fig = plt.figure(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.18, 0.86, 0.62])
    for (rule, exit_name), color, style in ((("all five", "A"), SERIES[0], "-"), (("all five", "B"), SERIES[1], "-"),
                                            (("SuperTrend AI alone", "A"), INK_2, (0, (3, 2))),
                                            (("SuperTrend AI alone", "B"), MUTED, (0, (3, 2)))):
        t = res["trades"].get((pk, rule, exit_name))
        if t is None or t.empty:
            continue
        t = t.sort_values("entry_time")
        ax.plot(t["day"], t["bps"].cumsum() / 100, color=color, lw=1.2, ls=style, label=f"{rule}, exit {exit_name}")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative net %, whole account each trade", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.02, 0.97, f"Five-indicator scalper on {pk} 5-minute bars", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.915, "MACD + Squeeze Momentum + SuperTrend AI + swing structure + volatility waves; per share, "
             "1-tick slippage, flat by the close.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="A scalper built on the five indicators the Reddit author lists.")
    p.add_argument("--out", default="output/confluence_scalper")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read 2025-26 (run once)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    res = run_study(data, sessions, final=args.final_test, primary=args.tickers[0], timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([t.assign(ticker=k[0], rule=k[1], exit=k[2]) for k, t in res["trades"].items()]).to_parquet(
        out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["compare"].to_csv(out / "random_compare.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    plot(res, out / "strategy.png")
    (out / "report.md").write_text(render_report(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
