"""Intraday mean reversion for a scalper, on stock minute bars: a signal study, no options, no P&L.

  uv run --env-file .env python scalp_meanrev.py --out output/scalp_meanrev                # design period
  uv run --env-file .env python scalp_meanrev.py --final-test --out output/scalp_meanrev   # adds the holdout, once

Question: after a 5-minute bar that looks stretched (far from a moving average or VWAP, outside a Bollinger
band, at an RSI extreme, after a run of same-colour bars or a sharp 15-minute move), does price snap back over
the next 5-30 minutes by more than a scalper's costs? Signals only from 10:30 to 15:00 ET, an hour after the
open and an hour before the close.

Fixed before any result was seen:
- Tickers SPY, QQQ, IWM (index ETFs) and MSFT, AAPL, AMZN, META, on 5-minute bars. Indicators come from the
  bars (TA-Lib, continuous across sessions; runs of bars and 15-minute moves count within the session); the
  VWAP band uses the session's minute bars.
- Signals (`signal_masks`): 20 single-indicator thresholds and 7 combinations, each long (buy the stretch
  below) and short (fade the stretch above, the mirror). An event is the first bar of a run in which the
  condition holds, at that bar's close.
- Outcome: the move from the signal bar's close over the next 15 minutes, in basis points, in the signal's
  direction. Also 5, 10 and 30 minutes, and a symmetric bracket: did price go 1 ATR (5-minute ATR(14)) the
  signal's way before 1 ATR the other way, within 30 minutes? About 50% is a coin flip.
- Excess: the move minus the same ticker's average move from every bar in the same half hour of the day.
- Design: 2022-01-03 to 2024-12-31. A signal qualifies, for a ticker or a pooled group (index ETFs, stocks),
  with at least 300 events (800 pooled), a positive excess with a Benjamini-Hochberg q below 0.10 across every
  design test, and an average move above an illustrative round-trip cost: 1 bp for the ETFs, 2 bps for stocks.
- Holdout: 2025-01-02 on, read only with --final-test. A qualifier holds up if its holdout average still clears
  the cost and its excess is positive with a BH q below 0.10 across the qualifiers.
Uncertainty is clustered by day: events on the same day (across tickers too, when pooled) are not independent.

Round 2, fixed after round 1's design results (nothing qualified) and before any round-2 result was seen, with
its own Benjamini-Hochberg correction and the same rules (`round2_masks`, `fast_masks`): stretches on 2x normal
volume for the time of day (capitulation) or on quiet volume; stretches on volatile days (the previous daily
ATR in the ticker's top third); intraday stretches after 3+ daily down (up) days or on 0.5%+ gap days; and the
core signals on 2-minute bars.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
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
TICKERS = ("SPY", "QQQ", "IWM", "MSFT", "AAPL", "AMZN", "META")
GROUPS = {"index ETFs": ("SPY", "QQQ", "IWM"), "stocks": ("MSFT", "AAPL", "AMZN", "META")}
COST_BPS = {"SPY": 1.0, "QQQ": 1.0, "IWM": 1.0, "MSFT": 2.0, "AAPL": 2.0, "AMZN": 2.0, "META": 2.0}
GROUP_COST = {"index ETFs": 1.0, "stocks": 2.0}
STUDY_START = pd.Timestamp("2022-01-03")      # the minutes from 2021-10-04 warm the indicators up
HOLDOUT_START = pd.Timestamp("2025-01-01")
BAR_MINUTES = 5
WINDOW = (10 * 60 + 30, 15 * 60)              # signal bar ends, minutes after midnight ET
HORIZONS = (5, 10, 15, 30)                    # minutes after the signal bar's close
PRIMARY = 15
RACE_MINUTES = 30
MIN_EVENTS, MIN_POOLED, FDR = 300, 800, 0.10
SIDES = {"long": 1.0, "short": -1.0}


# ---------------------------------------------------------------- indicators

def run_lengths(move):
    """Signed length of the current run of same-direction moves (+3 = three up closes in a row; 0 resets)."""
    out, run = np.zeros(len(move)), 0.0
    for i, m in enumerate(move):
        run = 0.0 if m == 0 else m if run == 0 or np.sign(run) != m else run + m
        out[i] = run
    return out


def session_vwap(rth, bar_end):
    """Session-anchored VWAP and its volume-weighted standard deviation at each bar end, from minute bars (each
    minute's own VWAP, or its typical price when missing)."""
    m = rth.sort_values("ts")
    px = m["vwap"].astype(float).fillna((m["high"] + m["low"] + m["close"]) / 3).to_numpy(float)
    v = m["volume"].to_numpy(float)
    g = m["session"].to_numpy()
    cum = pd.DataFrame({"v": v, "pv": px * v, "p2v": px * px * v}).groupby(g).cumsum()
    vwap = (cum["pv"] / cum["v"]).to_numpy()
    sd = np.sqrt(np.maximum((cum["p2v"] / cum["v"]).to_numpy() - vwap**2, 0.0))
    k = np.searchsorted(outcomes._ns(m["ts"]), outcomes._ns(bar_end) - NS, side="right") - 1  # the bar's last minute
    return vwap[k], sd[k]


def add_indicators(bars, rth):
    """Indicator columns at each bar's close (causal: bars up to and including that one)."""
    b = bars.copy()
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    b["atr"] = talib.ATR(h, l, c, 14)
    b["rsi2"], b["rsi14"] = talib.RSI(c, 2), talib.RSI(c, 14)
    b["bb_z"] = (c - talib.SMA(c, 20)) / talib.STDDEV(c, 20, 1)
    for p in (9, 20):
        b[f"ema{p}_atr"] = (c - b[f"ema_{p}"]) / b["atr"]
    sess = b["session"].to_numpy()
    new = np.r_[True, sess[1:] != sess[:-1]]
    move = np.sign(np.r_[0.0, np.diff(c)])
    move[new] = 0
    b["streak"] = run_lengths(move)
    r = np.r_[np.nan, np.diff(np.log(c))]
    r[new] = np.nan  # no overnight moves
    vol = pd.Series(r).rolling(50, min_periods=30).std().to_numpy()
    back = pd.Series(sess).shift(3).to_numpy() == sess
    c3 = pd.Series(c).shift(3).to_numpy()
    b["move15_z"] = np.where(back, np.log(c / c3) / (vol * math.sqrt(3)), np.nan)
    b["vwap"], b["vwap_sd"] = session_vwap(rth, b["bar_end"])
    b["vwap_z"] = (c - b["vwap"]) / b["vwap_sd"].replace(0, np.nan)
    b["up_trend"] = b["ema_20"] > b["ema_50"]
    b["down_trend"] = b["ema_20"] < b["ema_50"]
    b["daily_up"] = b["prev_close"] > b["prev_ema_50"]
    b["daily_down"] = b["prev_close"] < b["prev_ema_50"]
    b["green"], b["red"] = c > o, c < o
    prev = pd.Series(b["bb_z"].to_numpy()).shift(1).where(~pd.Series(new)).to_numpy()
    b["bb_z_prev"] = prev
    return b


def signal_masks(b):
    """{name: (family, long condition, short condition)}: the pre-registered signals (short = the mirror)."""
    S = {}

    def add(name, family, long, short):
        S[name] = (family, np.asarray(long, bool), np.asarray(short, bool))

    for k in (1.0, 1.5, 2.0):
        add(f"9 EMA stretch {k:g} ATR", "EMA stretch", b["ema9_atr"] <= -k, b["ema9_atr"] >= k)
    for k in (1.5, 2.0, 3.0):
        add(f"20 EMA stretch {k:g} ATR", "EMA stretch", b["ema20_atr"] <= -k, b["ema20_atr"] >= k)
    for k in (1.5, 2.0, 2.5):
        add(f"VWAP band {k:g} sd", "VWAP band", b["vwap_z"] <= -k, b["vwap_z"] >= k)
    for k in (10, 5):
        add(f"RSI(2) {k}/{100 - k}", "RSI", b["rsi2"] <= k, b["rsi2"] >= 100 - k)
    for k in (30, 25):
        add(f"RSI(14) {k}/{100 - k}", "RSI", b["rsi14"] <= k, b["rsi14"] >= 100 - k)
    for k in (2.0, 2.5):
        add(f"Bollinger {k:g} sd", "Bollinger", b["bb_z"] <= -k, b["bb_z"] >= k)
    for k in (4, 5, 6):
        add(f"{k}+ bars in a row", "Run of bars", b["streak"] <= -k, b["streak"] >= k)
    for k in (2, 3):
        add(f"15-minute move {k} sd", "Sharp move", b["move15_z"] <= -k, b["move15_z"] >= k)
    add("RSI(2) 10/90 and VWAP band 1.5 sd", "Combination", (b["rsi2"] <= 10) & (b["vwap_z"] <= -1.5),
        (b["rsi2"] >= 90) & (b["vwap_z"] >= 1.5))
    add("Bollinger 2 sd with the 20/50 EMA trend", "Combination", (b["bb_z"] <= -2) & b["up_trend"],
        (b["bb_z"] >= 2) & b["down_trend"])
    add("Bollinger 2 sd against the 20/50 EMA trend", "Combination", (b["bb_z"] <= -2) & b["down_trend"],
        (b["bb_z"] >= 2) & b["up_trend"])
    add("RSI(2) 10/90 with the 20/50 EMA trend", "Combination", (b["rsi2"] <= 10) & b["up_trend"],
        (b["rsi2"] >= 90) & b["down_trend"])
    add("20 EMA stretch 2 ATR with the daily trend", "Combination", (b["ema20_atr"] <= -2) & b["daily_up"],
        (b["ema20_atr"] >= 2) & b["daily_down"])
    add("Bollinger hook: back inside after 2 sd", "Combination",
        (b["bb_z_prev"] <= -2) & (b["bb_z"] > -2) & b["green"], (b["bb_z_prev"] >= 2) & (b["bb_z"] < 2) & b["red"])
    add("4+ bars in a row and VWAP band 1 sd", "Combination", (b["streak"] <= -4) & (b["vwap_z"] <= -1),
        (b["streak"] >= 4) & (b["vwap_z"] >= 1))
    return S


def add_context(bars, daily, design_end=HOLDOUT_START):
    """Round-2 columns: relative volume for the bar's time of day (vs the previous 20 sessions), the daily run of
    up/down closes through the previous session, today's opening gap (%), and the previous daily ATR as a % of
    the previous close with its top-third cut from the design sessions."""
    b = bars.copy()
    slot = b["bar_start"].dt.tz_convert(NY).dt.strftime("%H:%M")
    vol = b.pivot_table(index="session", columns=slot, values="volume", aggfunc="sum")
    avg = vol.rolling(20, min_periods=10).mean().shift(1)
    b["rvol"] = b["volume"].to_numpy() / avg.stack().reindex(pd.MultiIndex.from_arrays([b["session"], slot])).to_numpy()
    close = daily["close"]
    move = np.sign(close.diff()).fillna(0).to_numpy()
    runs = pd.Series(run_lengths(move), index=daily.index).shift(1)  # through the previous session
    b["daily_streak_prev"] = b["session"].map(runs).to_numpy()
    b["gap_today"] = b["session"].map((daily["open"] / daily["prev_close"] - 1) * 100).to_numpy()
    b["prev_atr_pct"] = b["prev_atr_14"] / b["prev_close"] * 100
    design = b["session"] < design_end
    b.attrs["atr_cut"] = b.loc[design, "prev_atr_pct"].groupby(b.loc[design, "session"]).first().quantile(2 / 3)
    return b


def round2_masks(b):
    """Round 2 on 5-minute bars: volume, volatile days and daily context (long; short = the mirror)."""
    S = {}

    def add(name, family, long, short):
        S[name] = (family, np.asarray(long, bool), np.asarray(short, bool))

    rv, loud, quiet = b["rvol"], b["rvol"] >= 2, b["rvol"] < 1
    add("Bollinger 2 sd on 2x volume", "Volume", (b["bb_z"] <= -2) & loud, (b["bb_z"] >= 2) & loud)
    add("VWAP band 2 sd on 2x volume", "Volume", (b["vwap_z"] <= -2) & loud, (b["vwap_z"] >= 2) & loud)
    add("Bollinger 2 sd on quiet volume", "Volume", (b["bb_z"] <= -2) & quiet, (b["bb_z"] >= 2) & quiet)
    hot = b["prev_atr_pct"] >= b.attrs["atr_cut"]
    add("Bollinger 2 sd on volatile days", "Volatile days", (b["bb_z"] <= -2) & hot, (b["bb_z"] >= 2) & hot)
    add("VWAP band 2 sd on volatile days", "Volatile days", (b["vwap_z"] <= -2) & hot, (b["vwap_z"] >= 2) & hot)
    dd = b["daily_streak_prev"]
    add("VWAP band 1.5 sd after 3+ daily down days", "Daily context", (b["vwap_z"] <= -1.5) & (dd <= -3),
        (b["vwap_z"] >= 1.5) & (dd >= 3))
    add("RSI(2) 10/90 after 3+ daily down days", "Daily context", (b["rsi2"] <= 10) & (dd <= -3),
        (b["rsi2"] >= 90) & (dd >= 3))
    gap = b["gap_today"]
    add("VWAP band 1.5 sd on a 0.5%+ gap day", "Daily context", (b["vwap_z"] <= -1.5) & (gap <= -0.5),
        (b["vwap_z"] >= 1.5) & (gap >= 0.5))
    return S


def fast_masks(b):
    """Round 2 on 2-minute bars: the core signals, faster."""
    S = {}

    def add(name, long, short):
        S[name] = ("2-minute bars", np.asarray(long, bool), np.asarray(short, bool))

    add("2-min VWAP band 2 sd", b["vwap_z"] <= -2, b["vwap_z"] >= 2)
    add("2-min RSI(2) 10/90", b["rsi2"] <= 10, b["rsi2"] >= 90)
    add("2-min Bollinger 2 sd", b["bb_z"] <= -2, b["bb_z"] >= 2)
    add("2-min 9 EMA stretch 1.5 ATR", b["ema9_atr"] <= -1.5, b["ema9_atr"] >= 1.5)
    return S


def first_of_run(cond, session):
    """True on the first bar of each run of consecutive True bars within a session."""
    cond = np.asarray(cond, bool)
    same = np.r_[False, session[1:] == session[:-1]]
    return cond & ~(np.r_[False, cond[:-1]] & same)


# ---------------------------------------------------------------- outcomes

def bar_outcomes(rth, bars, horizons=HORIZONS, race_minutes=RACE_MINUTES):
    """Long-signed moves (bps) from each bar's close at each horizon, the largest moves up and down within
    race_minutes, and the race: +1 if price went 1 ATR up before 1 ATR down within race_minutes, -1 for the
    reverse, 0 for neither, NaN if both happened in the same minute. NaN wherever a needed minute is missing
    or the window passes the session close (the rules of outcomes.forward_outcomes)."""
    t = outcomes._ns(rth["ts"])
    c, h, l = (rth[k].to_numpy(float) for k in ("close", "high", "low"))
    T = outcomes._ns(bars["bar_end"])
    ref = bars["close"].to_numpy(float)
    limit = outcomes._ns(bars["session_close"])
    first = np.searchsorted(t, T)
    n, last_i = len(T), len(t) - 1

    def window(w):
        last = first + w - 1
        ok = (last <= last_i) & (T + w * NS <= limit)
        f, e = np.minimum(first, last_i), np.minimum(last, last_i)
        return ok & (t[f] == T) & (t[e] == T + (w - 1) * NS), e

    out = {}
    for hz in horizons:
        ok, e = window(hz)
        r = np.full(n, np.nan)
        r[ok] = (c[e[ok]] / ref[ok] - 1) * 1e4
        out[f"ret_{hz}"] = r
    ok, _ = window(race_minutes)
    idx = np.flatnonzero(ok)
    hw = sliding_window_view(h, race_minutes)[first[idx]]
    lw = sliding_window_view(l, race_minutes)[first[idx]]
    d = bars["atr"].to_numpy(float)[idx]
    up, dn = hw >= (ref[idx] + d)[:, None], lw <= (ref[idx] - d)[:, None]
    up_i = np.where(up.any(1), up.argmax(1), race_minutes)
    dn_i = np.where(dn.any(1), dn.argmax(1), race_minutes)
    race = np.where(up_i < dn_i, 1.0, np.where(dn_i < up_i, -1.0, np.where(up_i == race_minutes, 0.0, np.nan)))
    race[np.isnan(d)] = np.nan
    for name, vals in (("race", race), ("max_up", (hw.max(1) / ref[idx] - 1) * 1e4),
                       ("max_down", (lw.min(1) / ref[idx] - 1) * 1e4)):
        col = np.full(n, np.nan)
        col[idx] = vals
        out[name] = col
    return pd.DataFrame(out, index=bars.index)


def window_bars(bars):
    end = bars["bar_end"].dt.tz_convert(NY)
    mins = end.dt.hour * 60 + end.dt.minute
    return ((mins >= WINDOW[0]) & (mins <= WINDOW[1])).to_numpy()


def half_hour(bars):
    end = bars["bar_end"].dt.tz_convert(NY)
    return ((end.dt.hour * 60 + end.dt.minute - WINDOW[0]) // 30).to_numpy()


def ticker_events(ticker, rth, bars, masks=None, round_label="1"):
    """(events, baseline): one row per signal event (all signals, both sides, signed outcomes), and the
    every-bar averages by period and half hour used for the excess. masks: signal_masks(bars) by default."""
    use = window_bars(bars) & (bars["session"] >= STUDY_START).to_numpy()
    out = bar_outcomes(rth, bars)
    period = np.where(bars["session"] >= HOLDOUT_START, "holdout", "design")
    slot = half_hour(bars)
    base = (pd.DataFrame({"period": period, "slot": slot, **{f"ret_{hz}": out[f"ret_{hz}"] for hz in HORIZONS}})[use]
            .groupby(["period", "slot"]).mean())
    sess = bars["session"].to_numpy()
    rows = []
    for name, (family, long, short) in (signal_masks(bars) if masks is None else masks).items():
        for side, cond in (("long", long), ("short", short)):
            sel = np.flatnonzero(first_of_run(cond, sess) & use)
            if not len(sel):
                continue
            s = SIDES[side]
            ev = pd.DataFrame({"round": round_label, "ticker": ticker, "signal": name, "family": family, "side": side,
                               "day": sess[sel], "time": bars["bar_end"].to_numpy()[sel], "period": period[sel],
                               "slot": slot[sel]})
            for hz in HORIZONS:
                ev[f"ret_{hz}"] = s * out[f"ret_{hz}"].to_numpy()[sel]
            b = base.reindex(pd.MultiIndex.from_arrays([ev["period"], ev["slot"]]))[f"ret_{PRIMARY}"].to_numpy()
            ev["excess"] = ev[f"ret_{PRIMARY}"] - s * b
            ev["race"] = s * out["race"].to_numpy()[sel]
            up, down = out["max_up"].to_numpy()[sel], out["max_down"].to_numpy()[sel]
            ev["mfe"], ev["mae"] = (up, down) if s > 0 else (-down, -up)
            rows.append(ev)
    return pd.concat(rows, ignore_index=True), base


# ---------------------------------------------------------------- statistics

def cluster_mean(x, groups):
    """(mean, standard error clustered by group): residuals summed within each group, small-sample corrected."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (x.mean() if n else np.nan), np.nan
    m = x.mean()
    sums = pd.Series(x - m).groupby(np.asarray(groups)).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return m, np.nan
    return m, math.sqrt((sums**2).sum() * g / (g - 1)) / n


def event_stats(ev, months):
    """Counts, average moves by horizon, the clustered excess test, hit rate, the race and excursions."""
    v = ev.dropna(subset=[f"ret_{PRIMARY}"])
    n = len(v)
    out = {"events": n, "per_month": n / months if months else np.nan}
    if n < 2:
        return out
    for hz in HORIZONS:
        out[f"avg_{hz}"] = v[f"ret_{hz}"].mean()
    out["avg"], out["avg_se"] = cluster_mean(v[f"ret_{PRIMARY}"], v["day"])
    out["excess"], out["excess_se"] = cluster_mean(v["excess"], v["day"])
    z = out["excess"] / out["excess_se"] if out["excess_se"] else np.nan
    out.update(excess_lo=out["excess"] - 1.96 * out["excess_se"], excess_hi=out["excess"] + 1.96 * out["excess_se"],
               z=z, p=2 * norm.sf(abs(z)) if not np.isnan(z) else np.nan, p_up=norm.sf(z) if not np.isnan(z) else np.nan,
               hit=(v[f"ret_{PRIMARY}"] > 0).mean() * 100, mfe=v["mfe"].median(), mae=v["mae"].median())
    wins, losses = int((v["race"] == 1).sum()), int((v["race"] == -1).sum())
    lo, hi = alerts.wilson(wins, wins + losses)
    out.update(race_n=wins + losses, race=wins / (wins + losses) * 100 if wins + losses else np.nan,
               race_lo=lo, race_hi=hi)
    return out


def months_in(period, first, last):
    lo, hi = (first, HOLDOUT_START - pd.Timedelta(days=1)) if period == "design" else (HOLDOUT_START, last)
    return (hi - lo).days / 30.44


def stats_table(events, first, last):
    """One row per who (each ticker and each pooled group), signal, side and period."""
    rows = []
    whos = [(tk, events["ticker"] == tk, COST_BPS[tk], False) for tk in TICKERS if tk in set(events["ticker"])]
    whos += [(g, events["ticker"].isin(tks), GROUP_COST[g], True) for g, tks in GROUPS.items()]
    keys = ["round", "signal", "family", "side", "period"]
    for who, sel, cost, pooled in whos:
        for (rnd, sig, fam, side, period), ev in events[sel].groupby(keys, sort=False):
            rows.append({"round": rnd, "who": who, "pooled": pooled, "signal": sig, "family": fam, "side": side,
                         "period": period, "cost": cost, **event_stats(ev, months_in(period, first, last))})
    return pd.DataFrame(rows)


def select(stats):
    """Design-period qualifiers: enough events, positive excess with BH q < FDR across every design test, and an
    average move above the cost."""
    d = stats[stats["period"] == "design"].copy()
    d["q"] = np.nan
    for rnd, g in d.groupby("round"):  # each round is its own family of tests
        d.loc[g.index, "q"] = mr.bh_qvalues(g["p"].fillna(1).to_numpy())
    need = np.where(d["pooled"], MIN_POOLED, MIN_EVENTS)
    d["qualifies"] = (d["events"] >= need) & (d["q"] < FDR) & (d["excess"] > 0) & (d["avg"] > d["cost"])
    return d


def final_test(stats, design):
    """Holdout results for the design qualifiers, with BH q-values (one-sided, excess > 0) among them."""
    keys = ["round", "who", "signal", "side"]
    q = design[design["qualifies"]][keys + ["avg", "excess", "q"]].rename(
        columns={"avg": "design_avg", "excess": "design_excess", "q": "design_q"})
    h = stats[stats["period"] == "holdout"].merge(q, on=keys)
    if len(h):
        h["holdout_q"] = mr.bh_qvalues(h["p_up"].fillna(1).to_numpy())
        h["holds_up"] = (h["holdout_q"] < FDR) & (h["avg"] > h["cost"])
    return h


# ---------------------------------------------------------------- the study

def run_study(data, *, final=False, first=None, last=None, timings=None):
    """data: {ticker: (regular-hours minutes, 5-minute bars with indicators and context, 2-minute bars or None)}.
    No file I/O."""
    timings = {} if timings is None else timings
    with timed(timings, "events"):
        frames, bases = [], {}
        for tk, (rth, bars, fast) in data.items():
            ev, bases[tk] = ticker_events(tk, rth, bars)
            frames.append(ev)
            if "rvol" in bars:
                frames.append(ticker_events(tk, rth, bars, round2_masks(bars), "2")[0])
            if fast is not None:
                frames.append(ticker_events(tk, rth, fast, fast_masks(fast), "2")[0])
        events = pd.concat(frames, ignore_index=True)
        if not final:
            events = events[events["period"] == "design"]
    with timed(timings, "statistics"):
        stats = stats_table(events, first, last)
        design = select(stats)
        held = final_test(stats, design) if final else None
    return {"events": events, "stats": stats, "design": design, "holdout": held, "final": final,
            "tickers": list(data), "first": first, "last": last, "baseline": bases}


# ---------------------------------------------------------------- report

def bps(v, digits=1, sign=True):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def find(table, who, signal, side, period="design"):
    m = table[(table["who"] == who) & (table["signal"] == signal) & (table["side"] == side)
              & (table["period"] == period)]
    return m.iloc[0] if len(m) else None


def render_report(res):
    d = res["design"]
    lines = []
    w = lines.append
    w("# Intraday mean reversion for a scalper\n")
    w(f"5-minute bars, signals from 10:30 to 15:00 ET; design period {STUDY_START:%Y-%m-%d} to 2024-12-31"
      + (f", holdout 2025-01-02 to {res['last']:%Y-%m-%d}" if res["final"] else " (holdout not read)") + ". "
      "*Avg* is the move over the next 15 minutes from the signal bar's close, in basis points (1 bp = 0.01%), in "
      "the signal's direction: positive means price snapped back. *Excess* subtracts the ticker's average move from "
      "any bar in the same half hour. *Bracket* is the share of events that went 1 ATR the signal's way before 1 ATR "
      "against it within 30 minutes (50% is a coin flip). Ranges are 95% intervals clustered by day. Costs are "
      "illustrative round trips: 1 bp for SPY, QQQ and IWM, 2 bps for the stocks.\n")
    w(bottom_line(res))
    w("## 1. Round 1: every signal, design period, pooled\n")
    w(overview_section(res, "1"))
    w("## 2. Round 2: volume, volatile days, daily context and 2-minute bars\n")
    w("Fixed after round 1 came back empty, before any round-2 result was seen; corrected for its own tests.\n")
    w(overview_section(res, "2"))
    w("## 3. Signals that qualified (design period)\n")
    w(qualifier_section(res))
    n = 4
    if res["final"]:
        w("## 4. Holdout: did they hold up?\n")
        w(holdout_section(res))
        n = 5
    w(f"## {n}. Best signals per ticker (design period)\n")
    w(ticker_section(res))
    w(f"## {n + 1}. Data and method\n")
    w(data_section(res))
    return "\n".join(lines) + "\n"


def bottom_line(res):
    d = res["design"]
    q = d[d["qualifies"]]
    per = d.groupby("round").size()
    out = [f"**Bottom line.** {len(d)} design tests: round 1 {per.get('1', 0)} (27 signals x 2 sides x 7 tickers and "
           f"2 pooled groups), round 2 {per.get('2', 0)}. "]
    if q.empty:
        out.append("None qualified: no signal beat the time-of-day average clearly and also cleared the cost. ")
    else:
        out.append(f"{len(q)} qualified: " + "; ".join(
            f"{r.who} {r.side} {r.signal} ({bps(r.avg)} bps, n={int(r.events)})" for r in
            q.sort_values("avg", ascending=False).head(6).itertuples()) + ("; ..." if len(q) > 6 else "") + ". ")
    pos = d[(d["q"] < FDR) & (d["excess"] > 0)]
    out.append(f"{len(pos)} had a clear positive excess before the cost check, ")
    out.append(f"and {len(d[(d['q'] < FDR) & (d['excess'] < 0)])} a clear negative one (price kept going). ")
    if res["final"] and res["holdout"] is not None and len(res["holdout"]):
        h = res["holdout"]
        out.append(f"In the holdout, {int(h['holds_up'].sum())} of {len(h)} qualifiers held up.")
    return "".join(out) + "\n"


def overview_section(res, rnd):
    d = res["design"]
    out = ["Average 15-minute move in bps (bracket share in brackets); **bold** = qualified, *q* marks a clear "
           "excess (q < 0.10) in either direction:\n"]
    rows = []
    for (sig, fam), _ in d[d["round"] == rnd].groupby(["signal", "family"], sort=False):
        row = [sig]
        for who in GROUPS:
            for side in SIDES:
                r = find(d, who, sig, side)
                if r is None or not r["events"]:
                    row.append("")
                    continue
                mark = "**" if r["qualifies"] else ""
                flag = " *q*" if r["q"] < FDR else ""
                row.append(f"{mark}{bps(r['avg'])}{mark} ({pct(r['race'])}){flag}, n={int(r['events'])}")
        rows.append(row)
    out.append(alerts.md_table(["Signal", "ETFs, long", "ETFs, short", "Stocks, long", "Stocks, short"], rows))
    return "\n".join(out) + "\n"


def qualifier_section(res):
    d = res["design"]
    q = d[d["qualifies"]].sort_values(["pooled", "avg"], ascending=[False, False])
    if q.empty:
        return "None.\n"
    rows = [[r.who, r.side, r.signal + (" (round 2)" if r.round == "2" else ""), f"{int(r.events)}", f"{r.per_month:.0f}",
             " / ".join(bps(getattr(r, f"avg_{hz}")) for hz in HORIZONS),
             f"{bps(r.excess)} ({bps(r.excess_lo)} to {bps(r.excess_hi)})", f"{r.q:.3f}", pct(r.hit),
             f"{pct(r.race)} ({pct(r.race_lo)}-{pct(r.race_hi)})", f"{bps(r.mfe)} / {bps(r.mae)}"]
            for r in q.itertuples()]
    return alerts.md_table(["Who", "Side", "Signal", "Events", "Per month", "Avg 5/10/15/30 min (bps)",
                            "Excess, 15 min (95%)", "q", "Up after 15 min", "Bracket (95%)",
                            "Median best / worst within 30 min"], rows) + "\n"


def holdout_section(res):
    h = res["holdout"]
    if h is None or h.empty:
        return "No qualifiers to test.\n"
    rows = [[r.who, r.side, r.signal, f"{int(r.events)}", f"{bps(r.design_avg)} -> {bps(r.avg)}",
             f"{bps(r.excess)} ({bps(r.excess_lo)} to {bps(r.excess_hi)})", f"{r.holdout_q:.3f}",
             f"{pct(r.race)} ({pct(r.race_lo)}-{pct(r.race_hi)})", "yes" if r.holds_up else "no"]
            for r in h.sort_values("avg", ascending=False).itertuples()]
    return alerts.md_table(["Who", "Side", "Signal", "Events", "Avg 15 min, design -> holdout (bps)",
                            "Holdout excess (95%)", "Holdout q", "Bracket (95%)", "Holds up"], rows) + "\n"


def ticker_section(res):
    d = res["design"]
    rows = []
    for tk in res["tickers"]:
        t = d[(d["who"] == tk) & (d["events"] >= MIN_EVENTS)].sort_values("excess", ascending=False)
        for r in t.head(3).itertuples():
            rows.append([tk, r.side, r.signal, f"{int(r.events)}", bps(r.avg), f"{bps(r.excess)}", f"{r.q:.2f}",
                         pct(r.race), "yes" if r.qualifies else ""])
    return ("The three largest excess moves per ticker among signals with at least 300 events:\n\n"
            + alerts.md_table(["Ticker", "Side", "Signal", "Events", "Avg 15 min", "Excess", "q", "Bracket",
                               "Qualified"], rows) + "\n")


def data_section(res):
    ev = res["events"]
    out = [f"- Events (design{' and holdout' if res['final'] else ''}): {len(ev):,}; per ticker: "
           + ", ".join(f"{tk} {n:,}" for tk, n in ev["ticker"].value_counts().reindex(res['tickers']).items()) + "."]
    out.append("- A signal fires on the first bar of each run in which its condition holds; later bars of the same run "
               "are not new events. Outcomes need every minute of their window and end before the session close.")
    out.append("- Indicators: EMA 9/20/50 and ATR(14), RSI(2)/RSI(14), Bollinger(20, 2) on 5-minute closes; VWAP and "
               "its standard deviation from the session's minute bars; runs and the 15-minute move (over its 50-bar "
               "volatility) within the session; the daily trend is the previous close against the daily 50 EMA.")
    out.append("- Not modeled: fills, spreads beyond the illustrative cost, borrow for shorts, and overlap between "
               "signals; this is a signal study, not P&L.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- charts

def plot_design(res, path):
    d = res["design"]
    sigs = list(dict.fromkeys(d["signal"]))
    fig = plt.figure(figsize=(8, 0.9 + 0.24 * len(sigs) + 1.2), dpi=150, facecolor=SURFACE)
    height = fig.get_figheight()
    for k, who in enumerate(GROUPS):
        ax = sds._axes(fig, [0.36 + k * 0.33, 0.9 / height, 0.29, 1 - 2.1 / height])
        y = np.arange(len(sigs))[::-1]
        for side, color, off in (("long", SERIES[0], 0.17), ("short", SERIES[1], -0.17)):
            for yy, sig in zip(y, sigs):
                r = find(d, who, sig, side)
                if r is None or not r["events"] or pd.isna(r["avg_se"]):
                    continue
                ax.errorbar(r["avg"], yy + off, xerr=1.96 * r["avg_se"], fmt="o", ms=3, color=color, ecolor=color,
                            elinewidth=1, capsize=0, label=side if yy == y[0] else None)
        ax.axvline(0, color=BASELINE, lw=1)
        ax.axvline(GROUP_COST[who], color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(y, sigs if k == 0 else [""] * len(sigs), fontsize=6.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(f"{who[0].upper() + who[1:]}", color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("Avg 15-minute move, bps (95%)", color=INK_2, fontsize=7)
    fig.text(0.02, 1 - 0.25 / height, "Does price snap back 15 minutes after a stretched 5-minute bar?", color=INK,
             fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / height, "Design period 2022-2024, signals 10:30-15:00 ET. Dashed line: the illustrative "
             "round-trip cost.\nBlue: long (buy the stretch below). Orange: short (fade the stretch above). Bars: "
             "95% intervals.", color=INK_2, fontsize=7, va="top", linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_holdout(res, path):
    h = res["holdout"]
    if h is None or h.empty:
        return
    h = h.sort_values("design_avg")
    fig = plt.figure(figsize=(8, 1.6 + 0.3 * len(h)), dpi=150, facecolor=SURFACE)
    height = fig.get_figheight()
    ax = sds._axes(fig, [0.45, 0.5 / height, 0.5, 1 - 1.3 / height])
    y = np.arange(len(h))
    ax.scatter(h["design_avg"], y, color=MUTED, s=18, label="Design 2022-2024", zorder=3)
    ax.errorbar(h["avg"], y, xerr=1.96 * h["avg_se"], fmt="o", ms=4, color=SERIES[0], ecolor=SERIES[0], elinewidth=1,
                label="Holdout 2025-2026", zorder=4)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y, [f"{r.who} {r.side}: {r.signal}" for r in h.itertuples()], fontsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Avg 15-minute move, bps", color=INK_2, fontsize=7.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK_2)
    fig.text(0.02, 1 - 0.25 / height, "Qualifiers: design vs holdout", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def load_bars(ticker, sessions, cache_dir, refresh):
    """(regular-hours minutes, 5-minute bars with indicators and round-2 context, 2-minute bars, note)."""
    minutes, note = gr.load_minutes(ticker, sessions, cache_dir, refresh)
    rth, bars, daily, _ = features.build_features(minutes, sessions, bar_minutes=BAR_MINUTES, ema_periods=(9, 20, 50))
    bars = add_context(add_indicators(bars, rth), daily)
    _, fast, _, _ = features.build_features(minutes, sessions, bar_minutes=2, ema_periods=(9, 20, 50))
    return rth, bars, add_indicators(fast, rth), note


def write_outputs(out_dir, res):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["design"].to_csv(out / "design.csv", index=False)
    if res["final"] and res["holdout"] is not None:
        res["holdout"].to_csv(out / "holdout.csv", index=False)
        plot_holdout(res, out / "holdout.png")
    res["events"].to_parquet(out / "events.parquet", index=False)
    plot_design(res, out / "design.png")
    (out / "report.md").write_text(render_report(res))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Intraday mean-reversion signals for a scalper (signal study).")
    p.add_argument("--out", default="output/scalp_meanrev")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read the 2025+ holdout (run once)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today].copy()
    sessions["in_study"] = sessions.index >= STUDY_START
    with timed(timings, "data"):
        data, notes = {}, []
        for tk in TICKERS:
            rth, bars, fast, note = load_bars(tk, sessions, args.cache_dir, args.refresh)
            data[tk] = (rth, bars, fast)
            notes += [note] if note else []
    res = run_study(data, final=args.final_test, first=STUDY_START, last=sessions.index[-1], timings=timings)
    res["notes"] = notes
    with timed(timings, "outputs"):
        out = write_outputs(args.out, res)
    q = res["design"][res["design"]["qualifies"]]
    print(f"design tests: {len(res['design'])}, qualified: {len(q)}")
    if args.final_test and res["holdout"] is not None:
        print(f"holdout: {int(res['holdout']['holds_up'].sum()) if len(res['holdout']) else 0} of "
              f"{len(res['holdout'])} held up")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
