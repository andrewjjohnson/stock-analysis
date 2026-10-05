"""The user's Pine Script strategy "Kalman SuperTrend + ADX Volatility Waves" (v3.7), rebuilt from its source and
tested on Massive minute bars. A trade simulation: per share of the underlying, after a 1-tick slippage, no
commissions; not options P&L.

  uv run --env-file .env python kalman_supertrend.py --out output/kalman_supertrend

What the script does (5-minute SPY chart with extended hours):
- Signal: a scalar Kalman filter over the close (Q 0.01, R 0.2; with these constants it settles into an exponential
  average with weight 0.2, about a 9-bar EMA) is the centre of a SuperTrend with ATR(7) x 2.0. A close beyond the
  ratcheting band flips the trend: BUY on an up flip, SELL on a down flip, entries only on bars starting 9:30-15:55
  ET. An opposite flag reverses the position; a flag in the same direction is ignored.
- Exits: stop 1.5 ATR and target 2.0 ATR from the signal bar's close; the stop moves to that close once a bar's high
  (low for shorts) reaches 1 ATR in favour; close when a bar closes beyond the opposite outer "volatility wave"
  (VWMA(50) +/- 2 x the RMA(10) of stdev(20) x 1.5 x (1 + 0.8 ADX(14)/100)); close 12 bars after the signal bar.
- Everything else in the script (gradient fills, compression highlight, TP markers, labels, dashboard) is display.

Fixed before any result was seen:
- SPY (the instrument the script was fitted on); QQQ, IWM and TSLA reported for reading only. Bars: 5 minutes,
  clock-aligned, 04:00-20:00 ET, indicators computed continuously over all of them (TA-Lib ATR, ADX and STDDEV;
  the Kalman filter, SuperTrend recurrence, VWMA and RMA as written in the script).
- Fills: the script fills at the signal bar's close (process_orders_on_close); a live order can only fill after the
  bar closes, so entries and market exits fill at the next minute's open. Stops and targets are checked minute by
  minute (stop first if both fall in one minute); a stop gapped through (even at the fill) fills at that minute's
  open. The script's 1-tick ($0.01) slippage applies to market and stop fills, not targets. As written, exits can
  happen in after-hours bars.
- Variants: "flat by the close" (options-realistic: no entry on the last regular bar, any open trade closed at the
  regular close) and "regular-hours chart" (indicators over 9:30-16:00 bars only).
- Baselines: 20 random entries per real trade, same session, random regular-hours bar, same direction and the same
  exits (including the real opposite flags); and the move after each flag (15/30/60 minutes, same session) against
  every regular-hours bar at the same time of day.
- The parameters were fitted to match another indicator's flags, not to returns, so the whole 2021-10 to 2026-09
  span is out of sample for returns; 2021-24 and 2025-26 are shown separately. Test: SPY as written, mean net P&L
  per trade > 0 and real minus random > 0 (one-sided, standard errors clustered by session).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM", "TSLA")
PARAMS = {"kf_q": 0.01, "kf_r": 0.2, "atr_period": 7, "factor": 2.0, "vwma_len": 50, "bb_len": 20, "bb_mult": 1.5,
          "adx_len": 14, "adx_gain": 0.8, "smooth_len": 10, "offset": 1.0, "expansion": 1.0, "stop_atr": 1.5,
          "target_atr": 2.0, "breakeven_atr": 1.0, "max_bars": 12, "tick": 0.01, "zone_exit": True}
VARIANTS = {"as written": {"flat_by_close": False, "rth_chart": False},
            "flat by the close": {"flat_by_close": True, "rth_chart": False},
            "regular-hours chart": {"flat_by_close": False, "rth_chart": True}}
PRIMARY = "as written"
BAR = pd.Timedelta(minutes=5)
ETH_START, ETH_END = 4 * 60, 20 * 60       # minutes after midnight ET
HORIZONS = (15, 30, 60)                     # minutes, for the move after each flag
RANDOM_REPS = 20
POSITION_USD, CAPITAL = 3000.0, 25000.0     # the script's sizing
DESIGN_END = pd.Timestamp("2024-12-31")
SEED = 3


# ---------------------------------------------------------------- bars and indicators

def five_minute_bars(minutes, sessions, rth_chart=False):
    """Clock-aligned 5-minute bars 04:00-20:00 ET (or regular hours only), with each bar's session, whether it is
    a regular-hours bar, whether it is the session's last regular bar, and its minutes' index range in the cleaned
    minute frame (returned too)."""
    m = minutes.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    ts = m["ts"].dt.tz_convert("UTC").dt.as_unit("ns")
    local = ts.dt.tz_convert(NY)
    clock = local.dt.hour * 60 + local.dt.minute
    day = local.dt.tz_localize(None).dt.normalize()
    m = m.assign(ts=ts, day=day)[(clock >= ETH_START) & (clock < ETH_END) & day.isin(sessions.index)]
    m = m.join(sessions[["open", "close"]].rename(columns={"open": "s_open", "close": "s_close"}), on="day")
    m["rth"] = (m["ts"] >= m["s_open"]) & (m["ts"] < m["s_close"])
    if rth_chart:
        m = m[m["rth"]]
    m = m.reset_index(drop=True)
    start = m["ts"].dt.floor("5min")
    g = m.assign(bar=start).groupby("bar", sort=True)
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum(), "day": g["day"].first(),
                         "s_close": g["s_close"].first(), "rth": g["rth"].first()}).reset_index()
    t = _ns(m["ts"])
    b = _ns(bars["bar"])
    bars["m_lo"] = np.searchsorted(t, b)
    bars["m_hi"] = np.searchsorted(t, b + BAR.value)
    end = bars["bar"] + BAR
    bars["last_rth"] = bars["rth"] & (end >= bars["s_close"])
    return bars, m


def kalman(close, q, r):
    """The script's scalar Kalman filter: starts at the first close with error 1."""
    out = np.empty(len(close))
    x, p = close[0], 1.0
    for i, c in enumerate(close):
        pp = p + q
        k = pp / (pp + r)
        x = x + k * (c - x)
        p = (1 - k) * pp
        out[i] = x
    return out


def supertrend(close, basis, atr, factor):
    """The script's SuperTrend recurrence on a custom centre: (up band, down band, trend +1/-1)."""
    n = len(close)
    up, dn, trend = np.full(n, np.nan), np.full(n, np.nan), np.ones(n, int)
    for i in range(n):
        ub, db = basis[i] - factor * atr[i], basis[i] + factor * atr[i]
        if i == 0 or np.isnan(up[i - 1]) or np.isnan(ub):
            up[i] = ub
        else:
            up[i] = max(ub, up[i - 1]) if close[i - 1] > up[i - 1] else ub
        if i == 0 or np.isnan(dn[i - 1]) or np.isnan(db):
            dn[i] = db
        else:
            dn[i] = min(db, dn[i - 1]) if close[i - 1] < dn[i - 1] else db
        if i:
            t = trend[i - 1]
            if t == -1 and close[i] > dn[i - 1]:
                t = 1
            elif t == 1 and close[i] < up[i - 1]:
                t = -1
            trend[i] = t
    return up, dn, trend


def rma(x, n):
    """Pine's ta.rma: seeded with the mean of the first n consecutive values, then (prev * (n - 1) + x) / n."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    run = 0
    for i in range(len(x)):
        if not np.isnan(out[i - 1]) if i else False:
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n if ok[i] else out[i - 1]
            continue
        run = run + 1 if ok[i] else 0
        if run == n:
            out[i] = x[i - n + 1:i + 1].mean()
    return out


def indicators(bars, p=PARAMS):
    """Adds the script's indicators and in-session flags: kalman, atr, st_up/st_dn/trend, sig (+1 BUY, -1 SELL on
    regular-hours flip bars), vwma, width, upper/lower outer waves, ob/os (close beyond them)."""
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    b = bars.copy()
    b["kalman"] = kalman(c, p["kf_q"], p["kf_r"])
    b["atr"] = talib.ATR(h, l, c, p["atr_period"])
    b["st_up"], b["st_dn"], b["trend"] = supertrend(c, b["kalman"].to_numpy(), b["atr"].to_numpy(), p["factor"])
    flip = np.r_[0, np.diff(b["trend"].to_numpy())]
    b["sig"] = np.where(b["rth"], np.sign(flip), 0).astype(int)
    pv = pd.Series(c * v).rolling(p["vwma_len"]).sum() / pd.Series(v).rolling(p["vwma_len"]).sum()
    b["vwma"] = pv.to_numpy()
    raw = talib.STDDEV(c, p["bb_len"], 1) * p["bb_mult"] * (1 + p["adx_gain"] * talib.ADX(h, l, c, p["adx_len"]) / 100)
    b["width"] = rma(raw, p["smooth_len"])
    reach = p["offset"] + p["expansion"]
    b["upper_outer"], b["lower_outer"] = b["vwma"] + reach * b["width"], b["vwma"] - reach * b["width"]
    b["ob"], b["os"] = c >= b["upper_outer"].to_numpy(), c <= b["lower_outer"].to_numpy()
    return b


# ---------------------------------------------------------------- trades

def context(bars, m):
    """The arrays trade_from needs, built once."""
    ctx = {k: m[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    ctx = {"t": _ns(m["ts"]), "mo": ctx["open"], "mh": ctx["high"], "ml": ctx["low"], "mc": ctx["close"]}
    for k in ("high", "low", "close", "atr"):
        ctx["b" + k] = bars[k].to_numpy(float)
    for k in ("sig", "ob", "os", "last_rth", "m_lo", "m_hi"):
        ctx[k] = bars[k].to_numpy()
    ctx["ends"] = _ns(bars["bar"]) + BAR.value
    ctx["n"] = len(bars)
    return ctx


def trade_from(i, d, ctx, p=PARAMS, flat_by_close=False):
    """One trade in direction d from the close of bar i, with the script's exits (ctx from context()). Returns a
    dict: fill and exit (minute index and price), exit reason and bar, or None if no minute follows the bar."""
    t, mo, mh, ml, mc = ctx["t"], ctx["mo"], ctx["mh"], ctx["ml"], ctx["mc"]
    bh, bl, bc, atr = ctx["bhigh"], ctx["blow"], ctx["bclose"], ctx["batr"]
    sig, ob, os_, last_rth = ctx["sig"], ctx["ob"], ctx["os"], ctx["last_rth"]
    m_lo, m_hi, ends = ctx["m_lo"], ctx["m_hi"], ctx["ends"]
    tick = p["tick"]
    k0 = int(np.searchsorted(t, ends[i]))
    if k0 >= len(t):
        return None
    fill = mo[k0] + d * tick
    ref, a = bc[i], atr[i]
    stop, target = ref - d * p["stop_atr"] * a, ref + d * p["target_atr"] * a
    be = False

    def market(j):  # a market exit decided at bar j's close: the next minute's open, less a tick
        k = int(np.searchsorted(t, ends[j]))
        return (k, mo[k] - d * tick) if k < len(t) else (len(t) - 1, mc[-1] - d * tick)

    j = i
    while j + 1 < ctx["n"]:
        j += 1
        for k in range(max(m_lo[j], k0), m_hi[j]):
            if (ml[k] <= stop) if d > 0 else (mh[k] >= stop):
                px = min(stop, mo[k]) if d > 0 else max(stop, mo[k])  # an open beyond the stop fills there
                return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": px - d * tick, "exit_bar": j,
                        "reason": "breakeven" if be else "stop"}
            if (mh[k] >= target) if d > 0 else (ml[k] <= target):
                px = max(target, mo[k]) if d > 0 else min(target, mo[k])
                return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": px, "exit_bar": j, "reason": "target"}
        if not be and ((bh[j] >= ref + p["breakeven_atr"] * a) if d > 0 else (bl[j] <= ref - p["breakeven_atr"] * a)):
            be, stop = True, ref
        reason = ("regular close" if flat_by_close and last_rth[j] else
                  "opposite flag" if sig[j] == -d else
                  "wave" if p["zone_exit"] and ((ob[j] and d > 0) or (os_[j] and d < 0)) else
                  "time" if j - i >= p["max_bars"] else None)
        if reason:
            if reason == "regular close":
                k = m_hi[j] - 1
                return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": mc[k] - d * tick, "exit_bar": j,
                        "reason": reason}
            k, px = market(j)
            return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": px, "exit_bar": j, "reason": reason}
    k, px = len(t) - 1, mc[-1] - d * tick
    return {"fill_k": k0, "fill": fill, "exit_k": k, "exit": px, "exit_bar": ctx["n"] - 1, "reason": "end of data"}


def strategy(bars, m, p=PARAMS, flat_by_close=False):
    """The script's trades: each regular-hours flag opens a trade unless one in the same direction is still open;
    an opposite flag closes the open trade (inside trade_from) and opens the new one."""
    ctx = context(bars, m)
    sig, last_rth = ctx["sig"], ctx["last_rth"]
    rows, cur = [], None
    for s in np.flatnonzero(sig != 0):
        if flat_by_close and last_rth[s]:
            continue
        d = int(sig[s])
        if cur is not None:
            open_at_close = s < cur["exit_bar"] or (s == cur["exit_bar"] and cur["reason"] in
                                                    ("wave", "time", "regular close"))
            if open_at_close and d == cur["dir"]:
                continue  # same direction while the trade is open (pyramiding off)
        tr = trade_from(s, d, ctx, p, flat_by_close)
        if tr is None:
            continue
        cur = {**tr, "dir": d, "bar": s}
        rows.append(cur)
    return trade_log(rows, bars, m)


def trade_log(rows, bars, m):
    if not rows:
        return pd.DataFrame(columns=["day", "dir", "entry_time", "fill", "exit_time", "exit", "reason", "bars",
                                     "pnl", "bps"])
    r = pd.DataFrame(rows)
    ts, bar = pd.DatetimeIndex(m["ts"]), pd.DatetimeIndex(bars["bar"])
    out = pd.DataFrame({"day": bars["day"].to_numpy()[r["bar"]], "dir": r["dir"].to_numpy(),
                        "signal_time": bar[r["bar"]] + BAR, "entry_time": ts[r["fill_k"]],
                        "fill": r["fill"].to_numpy(), "exit_time": ts[r["exit_k"]], "exit": r["exit"].to_numpy(),
                        "reason": r["reason"].to_numpy(), "bars": (r["exit_bar"] - r["bar"]).to_numpy()})
    out["pnl"] = out["dir"] * (out["exit"] - out["fill"])
    out["bps"] = out["pnl"] / out["fill"] * 1e4
    return out


def random_trades(real, bars, m, reps=RANDOM_REPS, seed=SEED, p=PARAMS, flat_by_close=False):
    """For each real trade, `reps` trades on the same session from random regular-hours bars (not the last one),
    same direction, same exits."""
    ctx = context(bars, m)
    rng = np.random.default_rng(seed)
    pool = {d: np.flatnonzero((bars["day"] == d).to_numpy() & bars["rth"].to_numpy() & ~bars["last_rth"].to_numpy())
            for d in real["day"].unique()}
    rows = []
    for r in real.itertuples():
        bars_today = pool.get(r.day)
        if bars_today is None or not len(bars_today):
            continue
        for i in rng.choice(bars_today, size=reps):
            tr = trade_from(int(i), int(r.dir), ctx, p, flat_by_close)
            if tr is not None:
                rows.append({**tr, "dir": int(r.dir), "bar": int(i)})
    return trade_log(rows, bars, m)


def flag_moves(bars, m, horizons=HORIZONS):
    """Move (bps, in the flag's direction) from the next minute's open after each regular-hours bar to `h` minutes
    later in the same session, for flags and for every bar; NaN past the regular close."""
    t = _ns(m["ts"])
    mo, mc = m["open"].to_numpy(float), m["close"].to_numpy(float)
    ends = _ns(bars["bar"]) + BAR.value
    s_close = _ns(bars["s_close"])
    rth = bars["rth"].to_numpy() & ~bars["last_rth"].to_numpy()
    idx = np.flatnonzero(rth)
    k0 = np.searchsorted(t, ends[idx])
    ok = k0 < len(t)
    local = pd.DatetimeIndex(bars["bar"]).tz_convert(NY)
    slot = (local.hour * 60 + local.minute).to_numpy()
    out = pd.DataFrame({"bar": idx[ok], "day": bars["day"].to_numpy()[idx[ok]], "slot": slot[idx[ok]],
                        "sig": bars["sig"].to_numpy()[idx[ok]]})
    entry = mo[k0[ok]]
    for h in horizons:
        target = ends[idx[ok]] + h * 60 * 10**9
        kh = np.searchsorted(t, target) - 1  # the last minute that starts before entry + h
        valid = (kh >= k0[ok]) & (target <= s_close[idx[ok]])
        out[f"ret_{h}"] = np.where(valid, (mc[np.clip(kh, 0, len(t) - 1)] / entry - 1) * 1e4, np.nan)
    return out


# ---------------------------------------------------------------- statistics

def stats(t, years):
    """Per trade: count, a year, win rate, mean bps with a session-clustered interval and one-sided p (> 0), profit
    factor, and the account results of the script's sizing ($3,000 a trade on $25,000) and of the whole account."""
    x = t["bps"].to_numpy(float)
    n = len(x)
    out = {"trades": n, "per_year": n / years if years else np.nan}
    if n < 2:
        return out
    m, se = sm.cluster_mean(x, t["day"].to_numpy())
    z = m / se if se else np.nan
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    dollars = POSITION_USD * x / 1e4
    eq = np.cumprod(1 + x / 1e4)
    script_eq = CAPITAL + np.cumsum(dollars)
    return {**out, "win": (x > 0).mean() * 100, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "pf": gains / losses if losses else np.nan,
            "avg_win": x[x > 0].mean() if (x > 0).any() else np.nan,
            "avg_loss": x[x < 0].mean() if (x < 0).any() else np.nan,
            "script_net": dollars.sum(), "script_pct": dollars.sum() / CAPITAL * 100,
            "script_dd": float((np.maximum.accumulate(np.r_[CAPITAL, script_eq])[1:] - script_eq).max()),
            "whole_pct": (eq[-1] - 1) * 100, "whole_dd": float((1 - eq / np.maximum.accumulate(eq)).max() * 100)}


def compare(real, rand):
    """Real mean bps minus the random trades' mean, with an interval clustered by session (each session's real trades
    against its random ones)."""
    if real.empty or rand.empty:
        return {}
    r = rand.groupby("day")["bps"].mean()
    diff = real["bps"].to_numpy(float) - r.reindex(real["day"]).to_numpy()
    keep = np.isfinite(diff)
    m, se = sm.cluster_mean(diff[keep], real["day"].to_numpy()[keep])
    z = m / se if se else np.nan
    return {"random_mean": rand["bps"].mean(), "excess": m, "excess_lo": m - 1.96 * se, "excess_hi": m + 1.96 * se,
            "excess_p_up": norm.sf(z) if np.isfinite(z) else np.nan, "random_win": (rand["bps"] > 0).mean() * 100}


def move_stats(moves, horizons=HORIZONS):
    """Flag moves in the flag's direction against every bar at the same time of day (direction-adjusted)."""
    rows = []
    flags = moves[moves["sig"] != 0]
    for h in horizons:
        base = moves.groupby("slot")[f"ret_{h}"].mean()
        x = flags["sig"] * (flags[f"ret_{h}"] - base.reindex(flags["slot"]).to_numpy())
        keep = x.notna().to_numpy()
        m, se = sm.cluster_mean(x[keep].to_numpy(), flags["day"].to_numpy()[keep])
        rows.append({"minutes": h, "flags": int(keep.sum()), "raw": (flags["sig"] * flags[f"ret_{h}"]).mean(),
                     "excess": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
                     "hit": ((flags["sig"] * flags[f"ret_{h}"])[keep] > 0).mean() * 100})
    return pd.DataFrame(rows)


def years_between(days):
    return max((days.max() - days.min()).days, 1) / 365.25


def run_study(minutes_by_ticker, sessions, *, primary="SPY", timings=None, reps=RANDOM_REPS):
    """minutes_by_ticker: {ticker: minute bars with extended hours}. No file I/O."""
    timings = {} if timings is None else timings
    out = {"trades": {}, "stats": [], "compare": [], "moves": [], "years": [], "random": {}}
    for tk, minutes in minutes_by_ticker.items():
        for variant, cfg in VARIANTS.items():
            if tk != primary and variant != PRIMARY:
                continue
            with timed(timings, "bars and indicators"):
                bars, m = five_minute_bars(minutes, sessions, cfg["rth_chart"])
                bars = indicators(bars)
            with timed(timings, "trades"):
                t = strategy(bars, m, flat_by_close=cfg["flat_by_close"])
            out["trades"][(tk, variant)] = t
            span = years_between(bars.loc[bars["rth"], "day"])
            for period, part, yrs in (("2021-24", t[t["day"] <= DESIGN_END], None),
                                      ("2025-26", t[t["day"] > DESIGN_END], None), ("all", t, None)):
                days = bars.loc[bars["rth"], "day"]
                days = days[days <= DESIGN_END] if period == "2021-24" else days[days > DESIGN_END] if period == "2025-26" else days
                out["stats"].append({"ticker": tk, "variant": variant, "period": period,
                                     **stats(part, years_between(days))})
            if variant == PRIMARY or (tk == primary and variant == "flat by the close"):
                with timed(timings, "random entries"):
                    rand = random_trades(t, bars, m, reps=reps, flat_by_close=cfg["flat_by_close"])
                out["random"][(tk, variant)] = rand
                for period in ("2021-24", "2025-26", "all"):
                    sel = (lambda x: x[x["day"] <= DESIGN_END]) if period == "2021-24" else (
                        lambda x: x[x["day"] > DESIGN_END]) if period == "2025-26" else (lambda x: x)
                    out["compare"].append({"ticker": tk, "variant": variant, "period": period,
                                           **compare(sel(t), sel(rand))})
            if variant == PRIMARY:
                with timed(timings, "flag moves"):
                    mv = flag_moves(bars, m)
                out["moves"].append(move_stats(mv).assign(ticker=tk))
                for y, g in t.groupby(t["day"].dt.year):
                    out["years"].append({"ticker": tk, "year": y, **stats(g, 1.0)})
    out["stats"] = pd.DataFrame(out["stats"])
    out["compare"] = pd.DataFrame(out["compare"])
    out["moves"] = pd.concat(out["moves"], ignore_index=True)
    out["years"] = pd.DataFrame(out["years"])
    out["tickers"] = list(minutes_by_ticker)
    out["primary"] = primary
    return out


# ---------------------------------------------------------------- report

def bps(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def find(table, **match):
    m = table
    for k, v in match.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


def render_report(res):
    s, c, mv, yr = res["stats"], res["compare"], res["moves"], res["years"]
    pk = res["primary"]
    lines = []
    w = lines.append
    w("# Kalman SuperTrend + ADX Volatility Waves (the user's Pine Script, v3.7)\n")
    w("Rebuilt from the script and run on Massive minute bars: 5-minute bars 04:00-20:00 ET, BUY/SELL on SuperTrend "
      "flips during regular hours, the script's stop (1.5 ATR), target (2 ATR), breakeven (after 1 ATR), wave exit "
      "and 12-bar time stop. Fills at the next minute's open after a signal (the script assumes the bar's close), "
      "stops and targets checked minute by minute, 1-tick slippage. P&L per trade in basis points of the fill "
      "(1 bp = $1 per $10,000 traded); *script sizing* = $3,000 a trade on $25,000, as in the script's settings. "
      "*Random entries* = 20 trades per real trade on the same session at random regular-hours bars, same direction "
      "and exits.\n")
    r = find(s, ticker=pk, variant=PRIMARY, period="all")
    cr = find(c, ticker=pk, variant=PRIMARY, period="all")
    if r is not None and pd.notna(r.get("mean")):
        w(f"**Bottom line ({pk}, as written, 2021-10 to 2026-09).** {int(r['trades']):,} trades ({r['per_year']:.0f} a "
          f"year), {r['win']:.0f}% winners, {bps(r['mean'])} bps a trade ({bps(r['lo'])} to {bps(r['hi'])}); random "
          f"entries with the same exits {bps(cr.get('random_mean'))} bps, so the flags add {bps(cr.get('excess'))} "
          f"({bps(cr.get('excess_lo'))} to {bps(cr.get('excess_hi'))}). With the script's sizing: "
          f"{r['script_net']:+,.0f} $ ({r['script_pct']:+.1f}% of $25,000).\n")
    w(f"## {pk}: every variant and period\n")
    rows = []
    for variant in VARIANTS:
        for period in ("2021-24", "2025-26", "all"):
            r, cr = find(s, ticker=pk, variant=variant, period=period), find(c, ticker=pk, variant=variant,
                                                                             period=period)
            if r is None or pd.isna(r.get("mean")):
                continue
            rows.append([variant, period, f"{int(r['trades']):,} ({r['per_year']:.0f})", f"{r['win']:.0f}%",
                         f"{bps(r['avg_win'])} / {bps(r['avg_loss'])}",
                         f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})", f"{r['pf']:.2f}",
                         "n/a" if cr is None else bps(cr.get("random_mean")),
                         "n/a" if cr is None else f"{bps(cr.get('excess'))} ({bps(cr.get('excess_lo'))} to "
                                                  f"{bps(cr.get('excess_hi'))})",
                         f"{r['script_net']:+,.0f} $ ({r['script_pct']:+.1f}%)", f"{r['whole_pct']:+.1f}%"])
    w(alerts.md_table(["Variant", "Period", "Trades (a year)", "Wins", "Avg win / loss, bps", "Net bps a trade (95%)",
                       "Profit factor", "Random entries, bps", "Flags minus random (95%)",
                       "Script sizing ($3k on $25k)", "Whole account each trade"], rows) + "\n")
    w(f"## {pk} as written, by year\n")
    rows = [[int(r.year), f"{int(r.trades):,}", f"{r.win:.0f}%", bps(r.mean), f"{r.pf:.2f}",
             f"{r.script_net:+,.0f} $", f"{r.whole_pct:+.1f}%"] for r in yr[yr["ticker"] == pk].itertuples()]
    w(alerts.md_table(["Year", "Trades", "Wins", "Net bps a trade", "Profit factor", "Script sizing",
                       "Whole account"], rows) + "\n")
    w("## How trades end (as written)\n")
    t = res["trades"][(pk, PRIMARY)]
    g = t.groupby("reason")["bps"].agg(["size", "mean"])
    rows = [[k, f"{int(v['size']):,} ({v['size'] / len(t) * 100:.0f}%)", bps(v["mean"])] for k, v in g.iterrows()]
    w(alerts.md_table(["Exit", "Trades", "Average bps"], rows) + "\n")
    w("## The move after each flag, against any bar at the same time of day\n")
    rows = []
    for r in mv.itertuples():
        rows.append([r.ticker, f"{r.minutes} min", f"{r.flags:,}", bps(r.raw), f"{bps(r.excess)} ({bps(r.lo)} to "
                     f"{bps(r.hi)})", f"{r.hit:.0f}%"])
    w(alerts.md_table(["Ticker", "After", "Flags", "Move in the flag's direction, bps", "Excess (95%)",
                       "Moved the flag's way"], rows) + "\n")
    w("## Other tickers (as written, reading only)\n")
    rows = []
    for tk in res["tickers"]:
        if tk == pk:
            continue
        r, cr = find(s, ticker=tk, variant=PRIMARY, period="all"), find(c, ticker=tk, variant=PRIMARY, period="all")
        if r is None or pd.isna(r.get("mean")):
            continue
        rows.append([tk, f"{int(r['trades']):,}", f"{r['win']:.0f}%", f"{bps(r['mean'])} ({bps(r['lo'])} to "
                     f"{bps(r['hi'])})", "n/a" if cr is None else bps(cr.get("random_mean")),
                     "n/a" if cr is None else bps(cr.get("excess")), f"{r['script_pct']:+.1f}%"])
    w(alerts.md_table(["Ticker", "Trades", "Wins", "Net bps a trade (95%)", "Random entries", "Flags minus random",
                       "Script sizing, % of $25k"], rows) + "\n")
    w("## Notes\n")
    w("- The Kalman filter (Q 0.01, R 0.2) settles within a few bars to a fixed gain of 0.2: an exponential average "
      "of the close, about a 9-bar EMA. So the engine is a SuperTrend around a 9-EMA.")
    w("- The script's own backtest fills at the signal bar's close and resolves stops and targets with TradingView's "
      "within-bar assumptions; here fills come a minute later and stops/targets use minute bars, so results differ.")
    w("- As written, trades can end in after-hours bars (12-bar time stop); 'flat by the close' is what an options "
      "trader could actually do.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    """Cumulative bps (whole account each trade) of the primary ticker's variants against its random entries."""
    pk = res["primary"]
    fig = plt.figure(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.18, 0.86, 0.62])
    for variant, color in zip(VARIANTS, (SERIES[0], SERIES[1], INK_2)):
        t = res["trades"].get((pk, variant))
        if t is None or t.empty:
            continue
        t = t.sort_values("entry_time")
        ax.plot(t["day"], t["bps"].cumsum() / 100, color=color, lw=1.2, label=variant)
    rand = res["random"].get((pk, PRIMARY))
    if rand is not None and len(rand):
        daily = rand.groupby("day")["bps"].mean() * res["trades"][(pk, PRIMARY)].groupby("day").size()
        ax.plot(daily.index, daily.fillna(0).cumsum() / 100, color=MUTED, lw=1.0, ls=(0, (3, 2)),
                label="random entries, same exits")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative net %, whole account each trade", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.02, 0.97, f"Kalman SuperTrend + volatility waves on {pk} 5-minute bars", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "The user's Pine Script rebuilt; fills a minute after each signal, 1-tick slippage, per "
             "share of the ETF.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="The user's Kalman SuperTrend + ADX Volatility Waves Pine strategy, rebuilt.")
    p.add_argument("--out", default="output/kalman_supertrend")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
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
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    res = run_study(data, sessions, primary=args.tickers[0], timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([t.assign(ticker=k[0], variant=k[1]) for k, t in res["trades"].items()]).to_parquet(
        out / "trades.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["compare"].to_csv(out / "random_compare.csv", index=False)
    res["moves"].to_csv(out / "flag_moves.csv", index=False)
    res["years"].to_csv(out / "years.csv", index=False)
    plot(res, out / "strategy.png")
    (out / "report.md").write_text(render_report(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
