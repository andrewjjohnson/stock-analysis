"""Bollinger Bands + RSI mean-reversion scalp with "trap" filters, tested as written on 5-minute bars of SPY, QQQ and
12 mega caps. A trade simulation on Massive minute bars (one-second bars for order flow): per share, gross of
commissions, with the costs per side stated below; no options.

  uv run --env-file .env python band_scalp.py --out output/band_scalp                  # design, 2022-24
  uv run --env-file .env python band_scalp.py --final-test --out output/band_scalp     # 2025-26, once

The spec (a user-supplied prompt): go long when a bar's low pierces the lower band with RSI(14) <= 30 and a later bar
closes back inside; the short mirrors it. Filters against "walking the bands": a higher-timeframe 200 EMA, a cap on
band expansion, cumulative volume delta, and a prior-session POC/VAH/VAL at the pierce. Stop beyond the pierce wick,
target the middle band, out after 10 bars.

Fixed before any result was seen ("choice" marks a point the spec leaves open):
- Tickers (choice): SPY and QQQ (stand-ins for ES and NQ), plus NVDA, MSFT, AAPL, GOOGL, AMZN, META, AVGO, TSLA,
  JPM, WMT, LLY and V. Regular-hours minutes. 5-minute bars from the session open (choice: 5 minutes rather than 1),
  each needing 80% of its minutes. TA-Lib indicators run continuously across sessions: BBANDS(20, 2) and RSI(14)
  on closes.
- Setup (long; the short mirrors it): a pierce bar whose low is below its lower band, with RSI(14) <= 30 at its
  close. Entry is decided at the close of the first later bar in the same session that closes above its lower
  band, within 6 bars (choice); otherwise the setup lapses. Bars that pierce again before then belong to the same
  setup. The "pierce wick" is the lowest low from the pierce bar to the entry bar.
- Filters, judged at the entry bar's close:
  - Trend anchor (choice: 1-hour bars): the close must be above (long) or below (short) the 200 EMA of hourly
    closes, using the last hourly bar completed by then.
  - Band expansion (choice: 50%): band width, (upper - lower) / middle, may not have grown more than 50% over the
    last 3 bars. It was first set at 20%. Before any outcome was seen, a count showed that the median setup's band
    width had already grown 34% (the pierce itself widens the bands), so 20% rejected three setups in four; 50%
    rejects roughly the most explosive quarter.
  - Level (choice: 0.1 daily ATR(14)): one of the previous session's POC, VAH or VAL must lie within 0.1 ATR of the
    pierce wick. The levels come from levels.py's 48-bin volume profile, but the previous session needs only 80% of
    its minutes. levels.py asks for every minute, which (found before any outcome was seen) left AVGO and LLY
    without levels on about 70% of days.
  - CVD (choices; measured for setups that pass the other three filters): from one-second bars, each second's volume
    classed by its price change as in breakout_flow.py.
    - Invalid if selling accelerates: the entry bar's delta is below 0 and below the pierce bar's.
    - Otherwise valid only on a divergence or on absorption.
    - Divergence: the pierce wick is below the lowest low of up to 12 earlier bars in the session, while cumulative
      delta at the wick bar's close is above its value at the earlier low's bar.
    - Absorption: the wick bar's delta is below 0 while the bar closes in the upper half of its range.
- Trade:
  - Market entry at the open of the next minute.
  - Stop at the pierce wick minus 0.05% (plus 0.05% for shorts).
  - Target: the middle band at the entry bar's close (choice: fixed at entry, as "fixed at the middle band" reads).
  - Out at the close 50 minutes (10 bars) after entry, or at the session's last minute if that comes first.
  - Skipped when the entry is already beyond the stop or the target.
  - Exits are checked on minute bars. A stop fills at its level, or at the open if a minute opens through it; the
    target fills at its level; a minute reaching both counts as stopped.
  - One position at a time per ticker (choice).
- Costs: 1 bp per side (primary); 0 and 2 shown. P&L is in bps of the entry price.
- Tests: the strategy as written (all four filters), mean net P&L per trade > 0 at 1 bp per side, one-sided,
  clustered by session, for the index ETFs and for the mega caps; BH q < 0.10 across the two. Design 2022-01-03 to
  2024-12-31; 2025-01-02 on only with --final-test (the same two tests, once).
- For reading: the filters added one at a time in the spec's order (core; + trend; + expansion; + level; + CVD,
  which is the strategy as written), each ticker, long and short, exit reasons, and every trade's indicator and
  filter states (trades.parquet).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402

import alerts  # noqa: E402
import breakout_check as bc  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import levels as lv  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
import volume_profile as vp  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT", "LLY", "V")
GROUPS = {"index ETFs": TICKERS[:2], "mega caps": TICKERS[2:]}
BAR_MIN, MIN_COVERAGE = 5, 0.8
BB_N, BB_K, RSI_N, RSI_LO, RSI_HI = 20, 2.0, 14, 30, 70
ARM_BARS = 6
HTF_MIN, HTF_EMA = 60, 200
EXPANSION_BARS, EXPANSION_MAX = 3, 0.50
LEVELS = ("poc", "vah", "val")
LEVEL_TOL = 0.1
CVD_LOOKBACK = 12
STOP_BUFFER = 0.0005
TIME_BARS = 10
COSTS, PRIMARY_COST = (0, 1, 2), 1
DESIGN_START, HOLDOUT_START = pd.Timestamp("2022-01-03"), pd.Timestamp("2025-01-02")
FDR = 0.10
ROWS = {  # the filters added one at a time, in the spec's order
    "core": [],
    "+ trend anchor": ["trend_ok"],
    "+ band expansion": ["trend_ok", "expansion_ok"],
    "+ level": ["trend_ok", "expansion_ok", "level_ok"],
    "+ CVD (as written)": ["trend_ok", "expansion_ok", "level_ok", "cvd_ok"],
}
PRIMARY_ROW = "+ CVD (as written)"
REASONS = {1: "target", -1: "stop", 0: "time or close"}


# ---------------------------------------------------------------- bars, setups, filters

def five_minute_bars(rth):
    """5-minute bars with the bands, RSI(14), band width and its growth over 3 bars, and the 1-hour 200 EMA known at
    each bar's close."""
    bars, _ = features.resample_bars(rth, BAR_MIN, MIN_COVERAGE)
    c = bars["close"].to_numpy(float)
    up, mid, lo = talib.BBANDS(c, BB_N, BB_K, BB_K, 0)
    bars["upper"], bars["middle"], bars["lower"], bars["rsi"] = up, mid, lo, talib.RSI(c, RSI_N)
    width = (up - lo) / mid
    bars["width"] = width
    bars["expansion"] = width / np.r_[np.full(EXPANSION_BARS, np.nan), width[:-EXPANSION_BARS]] - 1
    hourly, _ = features.resample_bars(rth, HTF_MIN, MIN_COVERAGE)
    hourly["anchor"] = talib.EMA(hourly["close"].to_numpy(float), HTF_EMA)
    known = pd.merge_asof(bars[["bar_end"]], hourly[["bar_end", "anchor"]].dropna(), on="bar_end",
                          direction="backward")
    bars["anchor"] = known["anchor"].to_numpy()
    return bars


def find_setups(bars):
    """One row per setup: side (+1 long, -1 short) and the positions in `bars` of the pierce bar (p), the entry bar
    (c) and the bar holding the pierce wick (w), with the wick price."""
    s = bars["session"].to_numpy()
    lo, hi, cl = (bars[k].to_numpy(float) for k in ("low", "high", "close"))
    lower, upper, rsi = (bars[k].to_numpy(float) for k in ("lower", "upper", "rsi"))
    n, rows = len(bars), []
    for side in (1, -1):
        pierce = (lo < lower) & (rsi <= RSI_LO) if side > 0 else (hi > upper) & (rsi >= RSI_HI)
        inside = cl > lower if side > 0 else cl < upper
        busy = -1
        for p in np.flatnonzero(pierce):
            if p <= busy:
                continue
            c, j = None, p + 1
            while j < n and j <= p + ARM_BARS and s[j] == s[p]:
                if inside[j]:
                    c = j
                    break
                j += 1
            busy = c if c is not None else j - 1
            if c is None:
                continue
            seg = slice(p, c + 1)
            w = p + int(np.argmin(lo[seg]) if side > 0 else np.argmax(hi[seg]))
            rows.append({"side": side, "p": p, "c": c, "w": w, "wick": lo[w] if side > 0 else hi[w]})
    return pd.DataFrame(rows, columns=["side", "p", "c", "w", "wick"]).sort_values("c", ignore_index=True)


def profile_levels(rth, sessions):
    """Per session: the previous session's POC, VAH and VAL (levels.py's 48-bin volume profile of its regular-hours
    minutes, from a session with at least 80% of its minutes) and the daily ATR(14) through yesterday."""
    daily = features.daily_features(rth, sessions, min_coverage=MIN_COVERAGE, atr_period=14)
    rows = rth.groupby("session").indices
    lo, hi, vol = (rth[k].to_numpy(float) for k in ("low", "high", "volume"))
    out = pd.DataFrame(np.nan, index=sessions.index, columns=list(LEVELS))
    for prev, day in zip(sessions.index[:-1], sessions.index[1:]):
        if not daily.at[prev, "usable"] or prev not in rows:
            continue
        i = rows[prev]
        prof = vp.bar_profile(lo[i], hi[i], vol[i], lv.PROFILE_BINS)
        if prof:
            val, poc, vah = vp.value_area(prof[1], prof[0])
            if val < poc < vah:
                out.loc[day, ["poc", "vah", "val"]] = poc, vah, val
    out["atr"] = daily["prev_atr_14"]
    return out


def setup_states(st, bars, table):
    """Adds times and the indicator states at the entry bar, and the trend, expansion and level filters."""
    c, p = st["c"].to_numpy(), st["p"].to_numpy()
    side = st["side"].to_numpy()
    st = st.assign(session=bars["session"].to_numpy()[c], pierce_time=bars["bar_start"].to_numpy()[p],
                   entry_bar_end=bars["bar_end"].to_numpy()[c], close=bars["close"].to_numpy(float)[c],
                   middle=bars["middle"].to_numpy(float)[c], rsi_pierce=bars["rsi"].to_numpy(float)[p],
                   width=bars["width"].to_numpy(float)[c], expansion=bars["expansion"].to_numpy(float)[c],
                   anchor=bars["anchor"].to_numpy(float)[c])
    st["anchor_dist_bps"] = (st["close"] / st["anchor"] - 1) * 1e4
    st["trend_ok"] = np.where(side > 0, st["close"] > st["anchor"], st["close"] < st["anchor"])
    st["expansion_ok"] = st["expansion"] <= EXPANSION_MAX
    lev = table.reindex(st["session"])
    dist = np.column_stack([(st["wick"].to_numpy() - lev[k].to_numpy(float)) / lev["atr"].to_numpy(float)
                            for k in LEVELS])
    near = np.nanargmin(np.where(np.isnan(dist), np.inf, np.abs(dist)), axis=1)
    st["level"] = np.array(LEVELS)[near]
    st["level_dist_atr"] = dist[np.arange(len(st)), near]
    st["level_ok"] = np.abs(st["level_dist_atr"]) <= LEVEL_TOL
    return st


def bar_deltas(sec, edges, open_ms):
    """Buying minus selling volume in each bar [edges[i], edges[i+1]) from one-second bars (ms, o, h, l, c, v): each
    second classed by its close against the previous second's (flat seconds keep the last change). The session's
    first second (the opening auction) only sets the price."""
    if len(sec) == 0:
        return np.full(len(edges) - 1, np.nan)
    ms, close, vol = sec[:, 0], sec[:, 4], sec[:, 5]
    step = np.r_[0.0, np.sign(np.diff(close))]
    last = pd.Series(np.where(step != 0, step, np.nan)).ffill().fillna(0.0).to_numpy()
    signed = np.where(ms == open_ms, 0.0, vol * last)
    cum = np.r_[0.0, np.cumsum(signed)]
    idx = np.searchsorted(ms, edges)
    return np.diff(cum[idx])


def cvd_flags(side, lo, hi, cl, deltas, p, c, w):
    """(divergence, absorption, accelerating) for one setup; positions are within the window arrays (lookback bars
    first, then the pierce bar p .. the entry bar c; w holds the wick)."""
    cvd = np.cumsum(deltas)
    up = side > 0
    divergence = False
    if p > 0:
        prior = int(np.argmin(lo[:p]) if up else np.argmax(hi[:p]))
        lower_low = lo[w] < lo[prior] if up else hi[w] > hi[prior]
        divergence = bool(lower_low and (cvd[w] > cvd[prior] if up else cvd[w] < cvd[prior]))
    mid = (hi[w] + lo[w]) / 2
    absorption = bool(deltas[w] < 0 and cl[w] >= mid) if up else bool(deltas[w] > 0 and cl[w] <= mid)
    accelerating = bool(deltas[c] < 0 and deltas[c] < deltas[p]) if up else bool(deltas[c] > 0 and deltas[c] > deltas[p])
    return divergence, absorption, accelerating


def add_cvd(st, bars, ticker, flow_of, open_ms):
    """CVD states for setups passing the trend, expansion and level filters (others get NaN and cvd_ok False)."""
    for col in ("delta_pierce", "delta_entry"):
        st[col] = np.nan
    for col in ("divergence", "absorption", "accelerating", "cvd_ok"):
        st[col] = False
    todo = st.index[st["trend_ok"] & st["expansion_ok"] & st["level_ok"]]
    starts, ends = _ns(bars["bar_start"]) // 1_000_000, _ns(bars["bar_end"]) // 1_000_000
    s = bars["session"].to_numpy()
    plan = {}
    for i in todo:
        p, c = int(st.at[i, "p"]), int(st.at[i, "c"])
        a = p
        while a > 0 and p - a < CVD_LOOKBACK and s[a - 1] == s[p]:
            a -= 1
        plan[i] = (a, p, c, (ticker, int(starts[a]) - 1_000, int(ends[c]) - 1))
    if hasattr(flow_of, "prefetch"):
        flow_of.prefetch([v[3] for v in plan.values()])
    lo, hi, cl = (bars[k].to_numpy(float) for k in ("low", "high", "close"))
    for i, (a, p, c, req) in plan.items():
        sec = flow_of(*req)
        edges = np.r_[starts[a:c + 1], ends[c]]
        d = bar_deltas(sec, edges, open_ms[st.at[i, "session"]])
        if np.isnan(d).any():
            continue
        w = int(st.at[i, "w"])
        div, absb, acc = cvd_flags(int(st.at[i, "side"]), lo[a:c + 1], hi[a:c + 1], cl[a:c + 1], d, p - a, c - a,
                                   w - a)
        st.loc[i, ["delta_pierce", "delta_entry"]] = d[p - a], d[c - a]
        st.loc[i, ["divergence", "absorption", "accelerating"]] = div, absb, acc
        st.at[i, "cvd_ok"] = (div or absb) and not acc
    return st


# ---------------------------------------------------------------- trades

def simulate(st, rth):
    """Trades for the setups in time order, one position at a time: entry at the next minute's open, stop beyond the
    wick, target the middle band, out after 10 bars or at the session's last minute. Exits on minute bars (a minute
    reaching the stop and the target counts as stopped)."""
    ts = _ns(rth["ts"])
    sess = rth["session"].to_numpy()
    op, hi, lo, cl = (rth[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rows, free = [], -1
    for r in st.itertuples():
        e = int(np.searchsorted(ts, _ns(pd.Series([r.entry_bar_end]))[0]))
        if e >= len(ts) or sess[e] != r.session or ts[e] < free:
            continue
        d, entry = r.side, op[e]
        stop = r.wick * (1 - STOP_BUFFER * d)
        target = r.middle
        if not (d * (entry - stop) > 0 and d * (target - entry) > 0):
            continue
        end = ts[e] + TIME_BARS * BAR_MIN * 60_000_000_000
        k, exit_, reason = e, np.nan, 0
        while k < len(ts) and sess[k] == r.session and ts[k] < end:
            hit_stop = lo[k] <= stop if d > 0 else hi[k] >= stop
            if hit_stop:
                gapped = k > e and (op[k] <= stop if d > 0 else op[k] >= stop)
                exit_, reason = (op[k] if gapped else stop), -1
                break
            if (hi[k] >= target) if d > 0 else (lo[k] <= target):
                exit_, reason = target, 1
                break
            k += 1
        if reason == 0:
            k -= 1
            exit_ = cl[k]
        free = ts[k] + 60_000_000_000
        rows.append({**r._asdict(), "entry_time": pd.Timestamp(ts[e], tz="UTC"), "entry": entry, "stop": stop,
                     "target": target, "exit_time": pd.Timestamp(ts[k], tz="UTC"), "exit": exit_, "reason": reason,
                     "gross": d * (exit_ / entry - 1) * 1e4, "risk_bps": d * (entry - stop) / entry * 1e4,
                     "reward_bps": d * (target - entry) / entry * 1e4})
    t = pd.DataFrame(rows)
    if len(t):
        t["r_multiple"] = t["gross"] / t["risk_bps"]
        t["exit_reason"] = t["reason"].map(REASONS)
        t = t.drop(columns=["Index"])
    return t


# ---------------------------------------------------------------- study

def run_study(data, sessions, flow_of, *, final=False, timings=None):
    """data: {ticker: minute bars}. No file I/O apart from the injected one-second loader's cache."""
    timings = {} if timings is None else timings
    first = HOLDOUT_START if final else DESIGN_START
    last = sessions.index[-1] if final else sessions.index[sessions.index < HOLDOUT_START][-1]
    open_ms = dict(zip(sessions.index, _ns(sessions["open"]) // 1_000_000))
    setups, trades = [], []
    for tk, minutes in data.items():
        with timed(timings, "bars and setups"):
            rth, _ = features.regular_session_minutes(minutes, sessions)
            bars = five_minute_bars(rth)
            st = find_setups(bars)
            table = profile_levels(rth, sessions)
            st = setup_states(st, bars, table)
            st = st[(st["session"] >= first) & (st["session"] <= last)].copy()
        with timed(timings, "order flow"):
            st = add_cvd(st, bars, tk, flow_of, open_ms)
        st.insert(0, "ticker", tk)
        setups.append(st)
        with timed(timings, "trades"):
            for row, cols in ROWS.items():
                keep = st[np.logical_and.reduce([st[k].to_numpy(bool) for k in cols])] if cols else st
                t = simulate(keep, rth)
                if len(t):
                    trades.append(t.assign(row=row))
    setups = pd.concat(setups, ignore_index=True)
    trades = pd.concat(trades, ignore_index=True)
    years = (last - first).days / 365.25
    rows = []
    scopes = [*GROUPS.items(), *[(tk, (tk,)) for tk in data]]
    for row in ROWS:
        t_row = trades[trades["row"] == row]
        for scope, members in scopes:
            for sides, label in (((1, -1), "both"), ((1,), "long"), ((-1,), "short")):
                g = t_row[t_row["ticker"].isin(members) & t_row["side"].isin(sides)]
                for cost in COSTS:
                    rows.append({"row": row, "scope": scope, "sides": label, "cost": cost,
                                 **bc.share_stats(g, cost, years)})
    stats = pd.DataFrame(rows)
    for col in ("mean", "lo", "hi", "p_up", "win", "avg_win", "avg_loss", "targets", "stops", "pct_year"):
        if col not in stats:
            stats[col] = np.nan
    tests = stats[(stats["row"] == PRIMARY_ROW) & stats["scope"].isin(GROUPS) & (stats["sides"] == "both")
                  & (stats["cost"] == PRIMARY_COST)].copy()
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["passes"] = (tests["q"] < FDR) & (tests["mean"] > 0)
    return {"setups": setups, "trades": trades, "stats": stats, "tests": tests, "final": final, "first": first,
            "last": last, "years": years}


# ---------------------------------------------------------------- report

def fmt(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def stat_cells(r):
    """Table cells from a stats row (a Series or an itertuples row: `mean` is a Series method, so use brackets)."""
    r = r._asdict() if hasattr(r, "_asdict") else r
    if pd.isna(r["mean"]):
        return [f"{int(r['trades']):,}", "n/a", "n/a", "n/a", "n/a"]
    return [f"{int(r['trades']):,} ({r['per_year']:.0f})", f"{r['win']:.0f}%",
            f"{fmt(r['avg_win'])} / {fmt(r['avg_loss'])}", f"{fmt(r['mean'])} ({fmt(r['lo'])} to {fmt(r['hi'])})",
            f"{r['targets']:.0f}% / {r['stops']:.0f}%"]


def render_report(res):
    s, t, st = res["stats"], res["tests"], res["setups"]
    pick = lambda **kw: s[np.logical_and.reduce([s[k] == v for k, v in kw.items()])]  # noqa: E731
    lines = []
    w = lines.append
    w(f"# Bollinger + RSI scalp with trap filters ({'2025-26 check' if res['final'] else '2022-24'})\n")
    w(f"{res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}, 5-minute bars, regular hours. Long when a bar pierces the "
      "lower band with RSI(14) <= 30 and a later bar closes back inside (shorts mirror it); filters: the 1-hour 200 "
      f"EMA, band width growth of at most {EXPANSION_MAX:.0%} over 3 bars, a prior-session POC/VAH/VAL within "
      f"{LEVEL_TOL} ATR of the wick, and "
      "order flow (divergence or absorption, no accelerating pressure); stop beyond the wick, target the middle band, "
      "out after 50 minutes. Net bps a trade after 1 bp per side (1 bp = $1 per $10,000 traded). Every choice the "
      "spec left open is listed in band_scalp.py.\n")
    w("**Tests (as written, one-sided, BH across the two).**\n")
    rows = [[r.scope, *stat_cells(r), "n/a" if pd.isna(r.p_up) else f"{r.p_up:.3f}", f"{r.q:.3f}",
             "passes" if r.passes else "fails"] for r in t.itertuples()]
    w(alerts.md_table(["Group", "Trades (a year)", "Wins", "Avg win / loss, bps", "Net bps a trade (95%)",
                       "Targets / stops", "p", "q", ""], rows) + "\n")
    w("## The filters added one at a time\n")
    rows = []
    for row in ROWS:
        for scope in GROUPS:
            r = pick(row=row, scope=scope, sides="both", cost=PRIMARY_COST).iloc[0]
            rows.append([row, scope, *stat_cells(r)])
    w(alerts.md_table(["Filters", "Group", "Trades (a year)", "Wins", "Avg win / loss, bps", "Net bps a trade (95%)",
                       "Targets / stops"], rows) + "\n")
    w("## Costs (as written and core)\n")
    rows = []
    for row in ("core", PRIMARY_ROW):
        for scope in GROUPS:
            cells = [row, scope]
            for cost in COSTS:
                r = pick(row=row, scope=scope, sides="both", cost=cost).iloc[0]
                cells.append(fmt(r["mean"]))
            rows.append(cells)
    w(alerts.md_table(["Filters", "Group", "0 bp", "1 bp", "2 bp per side"], rows) + "\n")
    w("## Long and short (as written)\n")
    rows = []
    for scope in GROUPS:
        for sides in ("long", "short"):
            r = pick(row=PRIMARY_ROW, scope=scope, sides=sides, cost=PRIMARY_COST).iloc[0]
            rows.append([scope, sides, *stat_cells(r)])
    w(alerts.md_table(["Group", "Side", "Trades (a year)", "Wins", "Avg win / loss, bps", "Net bps a trade (95%)",
                       "Targets / stops"], rows) + "\n")
    w("## Each ticker\n")
    rows = []
    for tk in TICKERS:
        cells = [tk]
        for row in ("core", PRIMARY_ROW):
            r = pick(row=row, scope=tk, sides="both", cost=PRIMARY_COST)
            r = r.iloc[0] if len(r) else None
            cells += (["n/a", "n/a"] if r is None or pd.isna(r["mean"]) else
                      [f"{int(r['trades']):,}, {r['win']:.0f}% wins",
                       f"{fmt(r['mean'])} ({fmt(r['lo'])} to {fmt(r['hi'])})"])
        rows.append(cells)
    w(alerts.md_table(["Ticker", "Core: trades, wins", "Core: net bps (95%)", "As written: trades, wins",
                       "As written: net bps (95%)"], rows) + "\n")
    w("## How often each filter let a setup through\n")
    rows = []
    three = st["trend_ok"] & st["expansion_ok"] & st["level_ok"]
    for name, share, base in (("trend anchor", st["trend_ok"].mean(), "all setups"),
                              ("band expansion", st["expansion_ok"].mean(), "all setups"),
                              ("level", st["level_ok"].mean(), "all setups"),
                              ("CVD", st.loc[three, "cvd_ok"].mean(), "setups passing the other three"),
                              ("all four", (three & st["cvd_ok"]).mean(), "all setups")):
        rows.append([name, f"{share * 100:.0f}%", base])
    w(alerts.md_table(["Filter", "Passed", "Of"], rows) + "\n")
    w(f"Setups found: {len(st):,} ({int((st['side'] > 0).sum()):,} long, {int((st['side'] < 0).sum()):,} short). "
      "Among setups passing the other three filters, CVD showed a divergence in "
      f"{st.loc[three, 'divergence'].mean() * 100:.0f}%, absorption in "
      f"{st.loc[three, 'absorption'].mean() * 100:.0f}% and accelerating pressure in "
      f"{st.loc[three, 'accelerating'].mean() * 100:.0f}%. (Each trade row is simulated on its own, one position "
      "at a time, so a setup another row skipped can't block it.)\n")
    w("## Notes\n")
    w("- Trade simulation on minute bars, per share, gross of commissions; entries at the next minute's open, stops "
      "and targets at their levels (a minute reaching both counts as stopped).")
    w("- Order flow is approximated from one-second trade bars (no quotes on the plan).")
    if not res["final"]:
        w("- 2025-26 not read.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    s = res["stats"]
    fig = plt.figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.08, 0.13, 0.88, 0.69])
    x = np.arange(len(ROWS))
    for k, (scope, color) in enumerate(zip(GROUPS, SERIES)):
        r = s[(s["scope"] == scope) & (s["sides"] == "both") & (s["cost"] == PRIMARY_COST)].set_index("row").loc[list(ROWS)]
        xs = x + (k - 0.5) * 0.36
        ax.bar(xs, r["mean"], width=0.34, color=color, label=scope)
        ax.errorbar(xs, r["mean"], yerr=[r["mean"] - r["lo"], r["hi"] - r["mean"]], fmt="none", ecolor=INK_2, lw=0.8,
                    capsize=2)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" + ", "\n+ ") for k in ROWS], fontsize=6.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Net bps a trade (1 bp per side)", color=INK_2, fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower left")
    title = "2025-26" if res["final"] else "2022-24"
    fig.text(0.02, 0.97, f"Bollinger + RSI scalp, filters added one at a time ({title})", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "5-minute bars; 95% intervals clustered by session.", color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bollinger + RSI mean-reversion scalp with trap filters (5-minute bars).")
    p.add_argument("--out", default="output/band_scalp")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
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
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    flow_of = bc.second_loader(args.cache_dir, args.refresh, volume=True)
    try:
        res = run_study(data, sessions, flow_of, final=args.final_test, timings=timings)
    finally:
        flow_of.save()
    out = Path(args.out) / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["setups"].to_parquet(out / "setups.parquet", index=False)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "first": str(res["first"].date()),
        "last": str(res["last"].date()), "tickers": args.tickers,
        "flow_seconds_fetched": flow_of.state["fetched"]}, indent=2))
    plot(res, out / "filters.png")
    (out / "report.md").write_text(render_report(res))
    print(f"order-flow requests: {flow_of.state['fetched']:,}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
