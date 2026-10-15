"""TG Capital's "Trident" (Chart Fanatics playbook): a fair value gap, a doji wicking into its midpoint, and a
confirmation candle, on 30-minute bars with stacked EMAs. The playbook trades forex and gold in the London kill
zone; this tests it on SPY, QQQ, IWM and 12 mega caps at the New York open. A signal study in R (the entry-to-stop
distance): how often and how far price moves the trade's way before the stop, against random entries with the
same filters and stop size. No P&L, no position limits; every pattern is measured on its own.

  uv run --env-file .env python trident.py --out output/trident                  # design, 2021-11 to 2024-12
  uv run --env-file .env python trident.py --final-test --out output/trident     # 2025-26, once, only if design passes

The playbook (longs; shorts mirror): EMAs 5/9/13/21 stacked and price above the 200 EMA; the daily chart aligned;
a 3-candle FVG forms between 2:30 and 4:00 ET; the next candle is a small-bodied doji whose wick reaches the FVG's
50% level; the candle after that closes below the doji's high; enter at that close (or a limit at the 50% level),
stop below the FVG candle's low, hold while the daily EMAs stay stacked. It claims about 90% wins.

Fixed before any result was seen, and confirmed by the user ("choice" marks a point the playbook leaves open):
- Tickers (choice): SPY, QQQ, IWM, NVDA, MSFT, AAPL, GOOGL, AMZN, META, AVGO, TSLA, JPM, WMT, LLY, V. Regular-hours
  minutes, 30-minute bars from the open, each needing 80% of its minutes. TA-Lib EMAs run continuously across
  sessions.
- Kill zone (choice): London's 2:30-4:00 FVG window and 3:00-6:30 entry window become the New York open. With
  regular-hours bars the earliest FVG completes at 11:00, so candle 3 must close at 11:00, 11:30 or 12:00 ET and
  entries come by 13:00 (3.5 hours after the open, like 3:00-6:30).
- FVG (long): candle 3's low above candle 1's high, all three candles consecutive in one session. The 50% level is
  the gap's midpoint. No direction rule for the middle candle (choice).
- Doji: candle 4, immediately after candle 3. Body at most 25% of its range (choice), low at or below the 50%
  level, body entirely above it (a wick, not a body, reaches the level).
- Confirmation: candle 5 closes below the doji's high (as written).
- Entry at candle 5's close. Stop at candle 1's low (choice: the usual ICT reading of "the candle that forms the
  FVG"). R = entry - stop; entries already past the stop are skipped.
- 30-minute trend, at candle 5's close: EMA 5 > 9 > 13 > 21 and the close above the 200 EMA.
- Daily bias (choice): daily EMA 5 > 9 > 13 > 21 at the previous session's close. The "Candle Strength Indicator"
  is proprietary and left out.
- Outcomes, from the first minute after the entry, scanning minute bars up to 20 sessions after the entry session:
  - Did price reach +1R, +2R, +3R and +5R before the stop? A minute reaching both counts as the stop.
  - The furthest move before the stop, in R, within the session, 5 sessions and 20 sessions.
  - A +2R / -1R bracket in R: +2 at the target, the stop (or a worse open) at the stop, else marked at the close
    20 sessions later. This is the primary statistic.
  - Returns with no stop at the session's close and 5 and 20 sessions later.
  - The playbook's exit: hold until a daily close where the daily EMAs are no longer stacked (the stop still
    applies), capped at 120 sessions.
- Baseline (matched random): for each trade, 20 entries on random design sessions of the same ticker, side and
  time of day where the same trend filters held at the last 30-minute bar completed by then. Entry at the previous
  minute's close, stop the same number of 30-minute ATR(14)s away.
- Test: the pattern as written, mean bracket R minus the mean of its own random matches, paired per trade,
  clustered by session, one-sided, for the index ETFs and the mega caps; BH q < 0.10 across the two and an excess
  above 0 to pass. Design 2021-11-01 to 2024-12-31 (data after 2024-12-31 is not loaded into any calculation);
  2025-01-02 on only with --final-test.
- Pre-declared variants (reported with their own random matches, BH across the 10, for reading): doji body <= 10%
  and <= 40%, stop at candle 2's low, a gap of at least 0.1 x the 30-minute ATR(14) at candle 3, and a limit at
  the 50% level (filters at candle 3's close, live until 13:00; a fill minute that also reaches the stop counts
  as stopped, and targets count from the next minute).
- Diagnostics: no daily bias, no trend filters at all, and every FVG with the filters but no trident (entry at
  candle 3's close).
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
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM", "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT",
           "LLY", "V")
GROUPS = {"index ETFs": TICKERS[:3], "mega caps": TICKERS[3:]}
BAR_MIN, MIN_COVERAGE = 30, 0.8
STACK, TREND_EMA, ATR_N = (5, 9, 13, 21), 200, 14
FVG_ENDS = ("11:00", "11:30", "12:00")         # candle 3's close, ET
LAST_ENTRY = "13:00"
DOJI_BODY = 0.25
TARGETS, BRACKET = (1, 2, 3, 5), 2
HORIZON, MFE_SESSIONS, RETURN_SESSIONS = 20, (0, 5, 20), (0, 5, 20)
SWING_CAP = 120
RANDOM_DRAWS, SEED = 20, 7
DESIGN_START, DESIGN_END = pd.Timestamp("2021-11-01"), pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-02")
FDR = 0.10
PRIMARY = "as written"
ROWS = {
    "as written": {},
    "doji body <= 10%": {"doji": 0.10},
    "doji body <= 40%": {"doji": 0.40},
    "stop at candle 2": {"stop": 2},
    "gap >= 0.1 ATR": {"min_gap": 0.1},
    "limit at the 50% level": {"entry": "limit"},
    "no daily bias": {"bias": False},
    "no trend filters": {"trend": False, "bias": False},
    "every FVG, no trident": {"entry": "fvg"},
}
VARIANTS = list(ROWS)[1:6]


# ---------------------------------------------------------------- bars and filters

def daily_states(rth, sessions):
    """Per session: whether the daily EMAs 5/9/13/21 are stacked up (long_now) or down (short_now) at its close (NaN
    for sessions under 80% coverage), the bias for trading in it from the previous session's close, and whether the
    previous session's candle closed above (green_long) or below (green_short) its open."""
    d = features.daily_features(rth, sessions, MIN_COVERAGE)
    usable = d["usable"].to_numpy()
    close = d["close"].to_numpy(float)
    emas = np.full((len(d), len(STACK)), np.nan)
    for k, p in enumerate(STACK):
        emas[usable, k] = talib.EMA(close[usable], timeperiod=p)
    ok = usable & np.isfinite(emas).all(axis=1)
    out = pd.DataFrame(index=d.index)
    out["long_now"] = np.where(ok, np.all(emas[:, :-1] > emas[:, 1:], axis=1), np.nan)
    out["short_now"] = np.where(ok, np.all(emas[:, :-1] < emas[:, 1:], axis=1), np.nan)
    out["bias_long"] = out["long_now"].shift(1).eq(1.0)
    out["bias_short"] = out["short_now"].shift(1).eq(1.0)
    up, down = usable & (close > d["open"].to_numpy(float)), usable & (close < d["open"].to_numpy(float))
    out["green_long"] = pd.Series(up, index=d.index).shift(1, fill_value=False)
    out["green_short"] = pd.Series(down, index=d.index).shift(1, fill_value=False)
    return out


def thirty_minute_bars(rth, daily):
    """30-minute bars with the EMA stack, the 200 EMA, ATR(14), the ET time of each bar's close and the daily bias."""
    bars, _ = features.resample_bars(rth, BAR_MIN, MIN_COVERAGE)
    h, lo, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    emas = np.column_stack([talib.EMA(c, timeperiod=p) for p in STACK])
    trend = talib.EMA(c, timeperiod=TREND_EMA)
    bars["trend_long"] = np.all(emas[:, :-1] > emas[:, 1:], axis=1) & (c > trend)
    bars["trend_short"] = np.all(emas[:, :-1] < emas[:, 1:], axis=1) & (c < trend)
    bars["atr"] = talib.ATR(h, lo, c, timeperiod=ATR_N)
    bars["tod"] = bars["bar_end"].dt.tz_convert(NY).dt.strftime("%H:%M")
    cols = ["bias_long", "bias_short", "green_long", "green_short"]
    bars = bars.join(daily[cols], on="session")
    bars[cols] = bars[cols].fillna(False).astype(bool)
    return bars


def fvg_table(bars, ends=FVG_ENDS, doji_window=1):
    """Every 3-candle FVG whose third candle closes at one of `ends` (ET), both directions, with what every row of
    ROWS needs: the doji and confirmation candles (when both exist in the session), both stops and the filters at
    candles 3 and 5. With doji_window = 1 the doji is candle 4 and its measures are kept for filtering later (the
    variants change its body limit). With doji_window > 1 it is the first of the next `doji_window` candles that is a
    doji (body <= DOJI_BODY) wicking into the 50% level with its body beyond it, and the setup is void if a candle
    closes beyond the gap's far side first. The confirmation candle follows the doji (`doji_k` candles after
    candle 3)."""
    n = len(bars)
    s = bars["session"].to_numpy()
    start, end = _ns(bars["bar_start"]), _ns(bars["bar_end"])
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = bars["atr"].to_numpy(float)
    linked = np.r_[(s[1:] == s[:-1]) & (start[1:] == end[:-1]), False]  # bar k+1 follows bar k in the session
    i = np.flatnonzero(bars["tod"].isin(ends).to_numpy())
    i = i[i >= 2]
    i = i[linked[i - 2] & linked[i - 1]]
    frames = []
    for side in (1, -1):
        up = side > 0
        gap = (lo[i] - h[i - 2]) if up else (lo[i - 2] - h[i])
        ce = (h[i - 2] + lo[i]) / 2 if up else (lo[i - 2] + h[i]) / 2
        far = h[i - 2] if up else lo[i - 2]

        def measures(j):
            body = np.abs(c[j] - o[j]) / np.where(h[j] > lo[j], h[j] - lo[j], np.nan)
            wick = (lo[j] <= ce) & (np.minimum(o[j], c[j]) > ce) if up else \
                (h[j] >= ce) & (np.maximum(o[j], c[j]) < ce)
            return body, wick
        if doji_window == 1:
            ok4, i4 = (i + 1 < n) & linked[i], np.minimum(i + 1, n - 1)
        else:
            i4, chain, void = np.full(len(i), -1), np.ones(len(i), bool), np.zeros(len(i), bool)
            for k in range(1, doji_window + 1):
                j = np.minimum(i + k, n - 1)
                chain &= (i + k < n) & linked[np.minimum(i + k - 1, n - 1)]
                live = chain & ~void & (i4 < 0)
                body, wick = measures(j)
                hit = live & (body <= DOJI_BODY) & wick
                i4 = np.where(hit, j, i4)
                void |= live & ~hit & ((c[j] < far) if up else (c[j] > far))
            ok4 = i4 >= 0
            i4 = np.where(ok4, i4, i)
        five = ok4 & (i4 + 1 < n) & linked[i4]
        i5 = np.minimum(i4 + 1, n - 1)
        body, wick = measures(i4)
        confirm = c[i5] < h[i4] if up else c[i5] > lo[i4]
        trend = bars["trend_long" if up else "trend_short"].to_numpy()
        f = pd.DataFrame({
            "side": side, "i3": i, "session": s[i], "ce": ce, "gap": gap, "gap_atr": gap / atr[i],
            "stop1": lo[i - 2] if up else h[i - 2], "stop2": lo[i - 1] if up else h[i - 1],
            "c3": c[i], "end3": end[i], "atr3": atr[i], "trend3": trend[i],
            "bias": bars["bias_long" if up else "bias_short"].to_numpy()[i],
            "green": bars["green_long" if up else "green_short"].to_numpy()[i],
            "five_ok": five, "doji_k": np.where(five, i4 - i, 0), "body4": np.where(five, body, np.nan),
            "wick4": five & wick, "confirm": five & confirm,
            "c5": np.where(five, c[i5], np.nan), "end5": np.where(five, end[i5], 0), "atr5": atr[i5],
            "trend5": five & trend[i5]})
        frames.append(f[gap > 0])
    return pd.concat(frames, ignore_index=True).sort_values(["i3", "side"], ignore_index=True)


# ---------------------------------------------------------------- minutes, entries and outcomes

def minute_arrays(rth, sessions, daily, bars=None):
    """Plain arrays for the outcome scans: prices, each minute's session ordinal, `ends[o]` = one past session o's
    last minute, and the daily EMA states per ordinal. With `bars`, stops trigger only on a 30-minute close beyond
    them (the playbook's rule for gold): `stop_lo` / `stop_hi` hold each bar's close at its last minute and never
    trigger elsewhere."""
    ordinal = pd.Series(np.arange(len(sessions)), index=sessions.index)
    ordv = ordinal.reindex(rth["session"]).to_numpy()
    m = {"ts": _ns(rth["ts"]), "op": rth["open"].to_numpy(float), "hi": rth["high"].to_numpy(float),
         "lo": rth["low"].to_numpy(float), "cl": rth["close"].to_numpy(float), "ordv": ordv,
         "ends": np.searchsorted(ordv, np.arange(len(sessions)), side="right"), "ordinal": ordinal,
         "long_now": daily["long_now"].reindex(sessions.index).to_numpy(float),
         "short_now": daily["short_now"].reindex(sessions.index).to_numpy(float), "close_stop": bars is not None}
    if bars is not None:
        last = np.searchsorted(m["ts"], _ns(bars["bar_end"])) - 1
        m["stop_lo"], m["stop_hi"] = np.full(len(m["ts"]), np.inf), np.full(len(m["ts"]), -np.inf)
        m["stop_lo"][last] = m["stop_hi"][last] = m["cl"][last]
    return m


def ny_time_ns(sessions, hhmm):
    """UTC nanoseconds of `hhmm` ET on each session date."""
    t = pd.DatetimeIndex(sessions) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    return _ns(pd.Series(t.tz_localize(NY).tz_convert("UTC")))


def entries(fvg, rule, m, last_entry=LAST_ENTRY):
    """Entries for one row of ROWS: side, session, price, stop, the 30-minute ATR known at the entry, the minute
    targets start counting from (`first`), the minute stops start counting from (`stop_first`) and the ET time."""
    kind = rule.get("entry", "pattern")
    f = fvg
    if rule.get("trend", True):
        f = f[f["trend5"] if kind == "pattern" else f["trend3"]]
    if rule.get("bias", True):
        f = f[f["bias"]]
    if rule.get("green"):
        f = f[f["green"]]
    if kind == "pattern":
        f = f[f["five_ok"] & (f["body4"] <= rule.get("doji", DOJI_BODY)) & f["wick4"] & f["confirm"]]
        if rule.get("min_gap", 0) > 0:
            f = f[f["gap_atr"] >= rule["min_gap"]]
        price, t, atr = (np.array(f[k], dtype=dt) for k, dt in (("c5", float), ("end5", np.int64), ("atr5", float)))
    else:
        price, t, atr = (np.array(f[k], dtype=dt) for k, dt in (("c3", float), ("end3", np.int64), ("atr3", float)))
    stop = f["stop2" if rule.get("stop") == 2 else "stop1"].to_numpy(float)
    side = f["side"].to_numpy()
    ts, ordv = m["ts"], m["ordv"]
    o = m["ordinal"].reindex(f["session"]).to_numpy()
    first = np.searchsorted(ts, t)
    ok = (first < len(ts)) & (ordv[np.minimum(first, len(ts) - 1)] == o)
    ok &= t <= ny_time_ns(f["session"], last_entry)  # a late doji can push the entry past the window
    stop_first = first.copy()
    if kind == "limit":
        last = np.searchsorted(ts, ny_time_ns(f["session"], last_entry))
        ce = f["ce"].to_numpy(float)
        for k in np.flatnonzero(ok):
            seg = m["lo"][first[k]:last[k]] if side[k] > 0 else m["hi"][first[k]:last[k]]
            hit = np.flatnonzero(seg <= ce[k] if side[k] > 0 else seg >= ce[k])
            if not len(hit):
                ok[k] = False
                continue
            fm = first[k] + hit[0]
            op = m["op"][fm]
            price[k] = min(ce[k], op) if side[k] > 0 else max(ce[k], op)
            stop_first[k], first[k], t[k] = fm, fm + 1, ts[fm]
    ok &= side * (price - stop) > 0
    out = pd.DataFrame({"side": side, "session": f["session"].to_numpy(), "ordinal": o, "price": price, "stop": stop,
                        "atr": atr, "first": first, "stop_first": stop_first, "entry_ns": t})[ok]
    out["tod"] = pd.to_datetime(out["entry_ns"]).dt.tz_localize("UTC").dt.tz_convert(NY).dt.strftime("%H:%M")
    return out.reset_index(drop=True)


def first_hit(arr, a, b, level, below, chunk=4000):
    """First index in [a, b) where arr <= level (below) or >= level, else b."""
    k = a
    while k < b:
        e = min(b, k + chunk)
        seg = arr[k:e]
        hit = seg <= level if below else seg >= level
        if hit.any():
            return k + int(hit.argmax())
        k = e
    return b


def outcomes(e, m):
    """Per entry, in R: the +kR-before-stop races, the furthest move before the stop, the +2R/-1R bracket, returns
    with no stop and the hold-while-the-daily-EMAs-stack exit. Minutes after the data's last session are never
    read; scans that run into it are flagged `truncated`. With `m["close_stop"]` the stop is a 30-minute close
    beyond it, exited at that close."""
    op, hi, lo, cl, ends, ordv = m["op"], m["hi"], m["lo"], m["cl"], m["ends"], m["ordv"]
    stop_lo, stop_hi = m.get("stop_lo", lo), m.get("stop_hi", hi)
    last_o = len(ends) - 1
    rows = []
    for r in e.itertuples(index=False):
        d, p, s, f, sf, o = r.side, r.price, r.stop, int(r.first), int(r.stop_first), int(r.ordinal)
        risk = d * (p - s)
        adv, fav = (stop_lo, hi) if d > 0 else (stop_hi, lo)
        b20 = ends[min(o + HORIZON, last_o)]
        ks = first_hit(adv, sf, b20, s, d > 0)
        row = {"r_bps": risk / p * 1e4, "truncated": o + HORIZON > last_o, "stopped": ks < b20}
        for k in TARGETS:
            row[f"hit{k}"] = first_hit(fav, f, ks, p + d * k * risk, d < 0) < ks
        for h in MFE_SESSIONS:
            seg = fav[f:min(ks, ends[min(o + h, last_o)])]
            row[f"mfe{h}"] = max(0.0, d * ((seg.max() if d > 0 else seg.min()) - p) / risk) if len(seg) else 0.0

        def stop_exit(k):
            if m.get("close_stop"):
                return cl[k]
            return op[k] if d * (op[k] - s) < 0 else s
        if row[f"hit{BRACKET}"]:
            row["bracket"] = float(BRACKET)
        elif row["stopped"]:
            row["bracket"] = d * (stop_exit(ks) - p) / risk
        else:
            row["bracket"] = d * (cl[b20 - 1] - p) / risk
        for h in RETURN_SESSIONS:
            idx = ends[o + h] - 1 if o + h <= last_o else -1
            row[f"ret{h}"] = d * (cl[idx] - p) / risk if idx >= f and ordv[idx] == o + h else np.nan
        state = m["long_now"] if d > 0 else m["short_now"]
        cap = min(o + SWING_CAP, last_o)
        flips = np.flatnonzero(state[o:cap + 1] == 0)
        o2 = o + int(flips[0]) if len(flips) else cap
        x = ends[o2] - 1
        kx = first_hit(adv, sf, x + 1, s, d > 0)
        row["swing"] = d * (stop_exit(kx) - p) / risk if kx <= x else d * (cl[x] - p) / risk
        row["swing_sessions"] = o2 - o
        row["swing_open"] = not len(flips) and kx > x
        rows.append(row)
    return pd.concat([e, pd.DataFrame(rows, index=e.index)], axis=1)


def random_matches(e, bars, m, rule, rng, draws, first, last):
    """For each entry, `draws` entries on random sessions in [first, last] of the same ticker, side and ET time,
    where the row's trend filters held at the last 30-minute bar completed by then: entry at the previous minute's
    close, stop the same number of 30-minute ATRs away. `pid` links each draw to its entry."""
    days = m["ordinal"].index[(m["ordinal"].index >= first) & (m["ordinal"].index <= last)]
    ts, ordv, cl = m["ts"], m["ordv"], m["cl"]
    out, pools = [], {}
    keep = bars[["session", "bar_end", "trend_long", "trend_short", "bias_long", "bias_short", "green_long",
                 "green_short", "atr"]].copy()
    keep["bar_ns"] = _ns(keep["bar_end"])
    for tod in e["tod"].unique():
        grid = pd.DataFrame({"session": days, "t": ny_time_ns(days, tod)}).sort_values("t")
        g = pd.merge_asof(grid, keep.drop(columns="bar_end").sort_values("bar_ns"), left_on="t", right_on="bar_ns",
                          by="session", direction="backward")
        idx = np.searchsorted(ts, g["t"].to_numpy())
        o = m["ordinal"].reindex(g["session"]).to_numpy()
        safe = np.clip(idx, 1, len(ts) - 1)
        valid = (idx > 0) & (idx < len(ts)) & (ordv[safe] == o) & (ordv[safe - 1] == o) & np.isfinite(g["atr"])
        for side, name in ((1, "long"), (-1, "short")):
            ok = valid.copy()
            if rule.get("trend", True):
                ok &= g[f"trend_{name}"].fillna(False).to_numpy(bool)
            if rule.get("bias", True):
                ok &= g[f"bias_{name}"].fillna(False).to_numpy(bool)
            if rule.get("green"):
                ok &= g[f"green_{name}"].fillna(False).to_numpy(bool)
            pools[tod, side] = (g["session"].to_numpy()[ok], o[ok], idx[ok], g["atr"].to_numpy(float)[ok])
    for r in e.itertuples():
        sess, o, idx, atr = pools[r.tod, r.side]
        r_atr = r.side * (r.price - r.stop) / r.atr
        if not len(idx) or not np.isfinite(r_atr):
            continue
        pick = rng.integers(0, len(idx), draws)
        price = cl[idx[pick] - 1]
        out.append(pd.DataFrame({"pid": r.pid, "side": r.side, "session": sess[pick], "ordinal": o[pick],
                                 "price": price, "stop": price - r.side * r_atr * atr[pick], "atr": atr[pick],
                                 "first": idx[pick], "stop_first": idx[pick], "entry_ns": ts[idx[pick]],
                                 "tod": r.tod}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- study

def study_minutes(minutes, sessions, final):
    """(regular-hours minutes, the sessions they span, first and last trade session). The design run drops every
    session after DESIGN_END before anything is computed."""
    if final:
        first, used = HOLDOUT_START, sessions
    else:
        first, used = DESIGN_START, sessions[sessions.index <= DESIGN_END]
    rth, _ = features.regular_session_minutes(minutes, used)
    return rth, used, first, used.index[-1]


def run_study(data, sessions, *, final=False, draws=RANDOM_DRAWS, seed=SEED, timings=None, prepare=study_minutes,
              ends=FVG_ENDS, last_entry=LAST_ENTRY, groups=GROUPS, close_stop=(), spread_of=None, doji_window=1,
              rows=None, exclude=()):
    """data: {ticker: minute bars, or a function returning them, so tickers load one at a time}. `prepare` turns
    minutes into (regular minutes with session columns, sessions used, first and last trade session); `ends` and
    `last_entry` set the window; tickers in `close_stop` stop on 30-minute closes; `spread_of(ticker, entry_ns)`
    gives the quoted spread at each as-written entry; `doji_window` lets the doji come up to that many candles after
    the FVG; `rows` replaces ROWS; FVGs on the (ticker, session date) pairs in `exclude` are dropped. No file I/O
    apart from the injected loaders'."""
    rows = ROWS if rows is None else rows
    primary = next(iter(rows))
    timings = {} if timings is None else timings
    rng = np.random.default_rng(seed)
    trades, randoms, funnel, spans, pid = [], [], [], [], 0
    for tk, minutes in data.items():
        with timed(timings, "data"):
            minutes = minutes() if callable(minutes) else minutes
        with timed(timings, "bars and patterns"):
            rth, used, first, last = prepare(minutes, sessions, final)
            spans.append((first, last))
            daily = daily_states(rth, used)
            bars = thirty_minute_bars(rth, daily)
            fvg = fvg_table(bars, ends, doji_window)
            fvg = fvg[(fvg["session"] >= first) & (fvg["session"] <= last)]
            skip = [pd.Timestamp(d) for t_, d in exclude if t_ == tk]
            fvg = fvg[~fvg["session"].isin(skip)]
            m = minute_arrays(rth, used, daily, bars if tk in close_stop else None)
            funnel.append(funnel_counts(fvg).assign(ticker=tk))
        for row, rule in rows.items():
            with timed(timings, "entries and outcomes"):
                e = entries(fvg, rule, m, last_entry)
                if e.empty:
                    continue
                e["pid"] = np.arange(pid, pid + len(e))
                pid += len(e)
                t = outcomes(e, m).assign(ticker=tk, row=row)
            if spread_of is not None and row == primary:
                with timed(timings, "spreads"):
                    t["spread"] = spread_of(tk, t["entry_ns"].to_numpy())
                    t["spread_r"] = t["spread"] / (t["side"] * (t["price"] - t["stop"]))
            trades.append(t)
            with timed(timings, "random matches"):
                rnd = random_matches(e, bars, m, rule, rng, draws, first, last)
                if len(rnd):
                    randoms.append(outcomes(rnd, m).assign(ticker=tk, row=row))
    trades = pd.concat(trades, ignore_index=True)
    randoms = pd.concat(randoms, ignore_index=True)
    first, last = min(a for a, _ in spans), max(b for _, b in spans)  # tickers can start later (data gaps)
    years = (last - first).days / 365.25
    stats, tests = statistics(trades, randoms, years, groups, primary)
    return {"primary": primary, "trades": trades, "random": randoms, "stats": stats, "tests": tests,
            "funnel": pd.concat(funnel, ignore_index=True), "final": final, "first": first, "last": last,
            "years": years, "groups": groups, "tickers": list(data)}


def funnel_counts(fvg):
    """How many FVGs survive each step of the as-written pattern, per side."""
    steps = {"FVG in the window": pd.Series(True, index=fvg.index),
             "doji and confirmation candles exist": fvg["five_ok"]}
    steps["doji (body <= 25%)"] = steps["doji and confirmation candles exist"] & (fvg["body4"] <= DOJI_BODY)
    steps["wick into the 50% level"] = steps["doji (body <= 25%)"] & fvg["wick4"]
    steps["confirmation"] = steps["wick into the 50% level"] & fvg["confirm"]
    steps["30-minute trend"] = steps["confirmation"] & fvg["trend5"]
    steps["daily bias"] = steps["30-minute trend"] & fvg["bias"]
    return pd.DataFrame([{"step": k, "side": side, "count": int((v & (fvg["side"] == side)).sum())}
                         for k, v in steps.items() for side in (1, -1)])


def row_stats(t, years):
    out = {"trades": len(t), "per_year": len(t) / years if years else np.nan}
    if len(t) < 2:
        return out
    m, se = sm.cluster_mean(t["bracket"].to_numpy(float), t["session"].to_numpy())
    out.update({f"hit{k}": t[f"hit{k}"].mean() * 100 for k in TARGETS})
    out.update({f"mfe{h}": t[f"mfe{h}"].median() for h in MFE_SESSIONS})
    out.update({f"ret{h}": t[f"ret{h}"].mean() for h in RETURN_SESSIONS})
    out.update({"bracket": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "stopped": t["stopped"].mean() * 100,
                "swing": t["swing"].mean(), "swing_sessions": t["swing_sessions"].median(),
                "r_bps": t["r_bps"].median(), "cost_r": (2 / t["r_bps"]).median()})
    if "spread_r" in t and t["spread_r"].notna().any():
        out.update({"spread_r": t["spread_r"].median(), "net": (t["bracket"] - t["spread_r"]).mean()})
    return out


def excess(t, rnd):
    """Bracket R minus the mean of each trade's own random matches: mean, 95% interval and one-sided p, clustered
    by the trade's session."""
    base = rnd.groupby("pid")["bracket"].mean()
    t = t[t["pid"].isin(base.index)]
    diff = t["bracket"].to_numpy(float) - base.reindex(t["pid"]).to_numpy()
    if len(diff) < 2:
        return {"paired": len(diff)}
    m, se = sm.cluster_mean(diff, t["session"].to_numpy())
    z = m / se if se else np.nan
    return {"paired": len(diff), "excess": m, "ex_lo": m - 1.96 * se, "ex_hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan}


def statistics(trades, randoms, years, groups=GROUPS, primary=PRIMARY):
    rows = []
    names = list(trades["ticker"].unique())
    scopes = [("all", names), *groups.items(), *[(tk, (tk,)) for tk in names]]
    for row in [r for r in ROWS if r in set(trades["row"])] + [r for r in trades["row"].unique() if r not in ROWS]:
        t_row, r_row = trades[trades["row"] == row], randoms[randoms["row"] == row]
        for scope, members in scopes:
            for sides, label in (((1, -1), "both"), ((1,), "long"), ((-1,), "short")):
                t = t_row[t_row["ticker"].isin(members) & t_row["side"].isin(sides)]
                r = r_row[r_row["pid"].isin(t["pid"])]
                rows.append({"row": row, "scope": scope, "sides": label, "who": "pattern", **row_stats(t, years),
                             **excess(t, r)})
                rows.append({"row": row, "scope": scope, "sides": label, "who": "random", **row_stats(r, years)})
    stats = pd.DataFrame(rows)
    for col in ("bracket", "excess", "ex_lo", "ex_hi", "p_up"):
        if col not in stats:
            stats[col] = np.nan
    pick = (stats["who"] == "pattern") & stats["scope"].isin(groups) & (stats["sides"] == "both")
    tests = stats[pick & (stats["row"] == primary)].copy()
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["passes"] = (tests["q"] < FDR) & (tests["excess"] > 0)
    variants = stats[pick & stats["row"].isin(VARIANTS)].copy()
    variants["q"] = mr.bh_qvalues(variants["p_up"].fillna(1).to_numpy())
    tests = pd.concat([tests.assign(kind="primary"), variants.assign(kind="variant")], ignore_index=True)
    return stats, tests


# ---------------------------------------------------------------- report

def fmt(v, digits=2):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def get(stats, **kw):
    r = stats[np.logical_and.reduce([stats[k] == v for k, v in kw.items()])]
    return r.iloc[0] if len(r) else None


def race_cells(r):
    if r is None or r["trades"] < 2:
        return [str(0 if r is None else int(r["trades"]))] + ["n/a"] * 6
    return [f"{int(r['trades']):,} ({r['per_year']:.0f})", *[pct(r[f'hit{k}']) for k in TARGETS],
            f"{r['mfe0']:.2f} / {r['mfe5']:.2f} / {r['mfe20']:.2f}",
            f"{fmt(r['bracket'])} ({fmt(r['lo'])} to {fmt(r['hi'])})"]


RACE_HEAD = ["Trades (a year)", "+1R first", "+2R first", "+3R first", "+5R first",
             "Median furthest move before the stop, R (session / 5 / 20 sessions)", "+2R/-1R bracket, mean R (95%)"]


STOCK_INTRO = ("30-minute regular-hours bars. Long: a 3-candle FVG completing at 11:00-12:00 ET, a doji (body <= 25% "
               "of its range) wicking into the gap's 50% level with its body above it, then a candle closing below the "
               "doji's high; EMA 5 > 9 > 13 > 21, close above the 200 EMA, daily EMAs stacked the same way at "
               "yesterday's close. Entry at the confirmation close, stop at candle 1's low; shorts mirror it. "
               "Everything is in R (entry to stop). A signal study: no costs, no position limits, overlapping trades "
               "all counted. Every open choice is listed in trident.py.")
STOCK_NOTES = ("Minute bars only: a minute reaching the stop and a target counts as the stop; a stop gapped through "
               "exits at that minute's open.",)


def render_report(res, title="TG Capital Trident on stocks", intro=STOCK_INTRO, notes=STOCK_NOTES, held="2025-26"):
    s, t = res["stats"], res["tests"]
    groups, tickers, primary = res["groups"], res["tickers"], res.get("primary", PRIMARY)
    lines = []
    w = lines.append
    span = f"{held} check" if res["final"] else "design"
    w(f"# {title} ({span}: {res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d})\n")
    w(intro + "\n")
    w(f"**Test: the pattern {primary} vs its matched random entries (same ticker, side, time of day and filters, "
      "stop the same number of ATRs away), +2R/-1R bracket, paired, one-sided, BH across the two groups.**\n")
    rows = []
    for r in t[t["kind"] == "primary"].itertuples():
        base = get(s, row=primary, scope=r.scope, sides="both", who="random")
        rows.append([r.scope, f"{int(r.trades):,}", fmt(r.bracket), fmt(base["bracket"]) if base is not None else "n/a",
                     f"{fmt(r.excess)} ({fmt(r.ex_lo)} to {fmt(r.ex_hi)})",
                     "n/a" if pd.isna(r.p_up) else f"{r.p_up:.3f}", f"{r.q:.3f}", "passes" if r.passes else "fails"])
    w(alerts.md_table(["Group", "Trades", "Pattern, mean R", "Random, mean R", "Excess R (95%)", "p", "q", ""], rows)
      + "\n")
    w("## How far and how often (all tickers, both sides)\n")
    rows = []
    for row in s["row"].unique():
        for who in ("pattern", "random"):
            r = get(s, row=row, scope="all", sides="both", who=who)
            rows.append([row if who == "pattern" else "", who, *race_cells(r)])
    w(alerts.md_table(["Rule", "", *RACE_HEAD], rows) + "\n")
    w("With no edge, +kR before a -1R stop happens about 1/(k+1) of the time (50%, 33%, 25%, 17%), less any drift. "
      "Random rows are the pattern's own matches.\n")
    w(f"## Variants against their own random matches (BH across the {int((t['kind'] == 'variant').sum())}, for reading)\n")
    rows = [[r.row, r.scope, f"{int(r.trades):,}", f"{fmt(r.excess)} ({fmt(r.ex_lo)} to {fmt(r.ex_hi)})",
             f"{r.q:.3f}"] for r in t[t["kind"] == "variant"].itertuples()]
    w(alerts.md_table(["Variant", "Group", "Trades", "Excess R (95%)", "q"], rows) + "\n")
    w(f"## No stop: where price is later, in R ({primary})\n")
    rows = []
    for scope in ("all", *groups):
        for who in ("pattern", "random"):
            r = get(s, row=primary, scope=scope, sides="both", who=who)
            if r is None or r["trades"] < 2:
                continue
            rows.append([scope, who, fmt(r["ret0"]), fmt(r["ret5"]), fmt(r["ret20"]), fmt(r["swing"]),
                         f"{r['swing_sessions']:.0f}"])
    w(alerts.md_table(["Group", "", "Session close", "5 sessions", "20 sessions",
                       "Playbook exit (daily EMAs unstack, stop kept), mean R", "Median hold, sessions"], rows) + "\n")
    w(f"## Long and short ({primary})\n")
    rows = []
    for scope in groups:
        for sides in ("long", "short"):
            r = get(s, row=primary, scope=scope, sides=sides, who="pattern")
            b = get(s, row=primary, scope=scope, sides=sides, who="random")
            rows.append([scope, sides, *race_cells(r), fmt(b["bracket"]) if b is not None and b["trades"] > 1
                         else "n/a", f"{fmt(r['excess'])} ({fmt(r['ex_lo'])} to {fmt(r['ex_hi'])})"
                         if r is not None and "excess" in r and pd.notna(r["excess"]) else "n/a"])
    w(alerts.md_table(["Group", "Side", *RACE_HEAD, "Random, mean R", "Excess R (95%)"], rows) + "\n")
    w(f"## Each ticker ({primary})\n")
    rows = []
    for tk in tickers:
        r = get(s, row=primary, scope=tk, sides="both", who="pattern")
        rows.append([tk, *race_cells(r), fmt(r["excess"]) if r is not None and "excess" in r else "n/a"])
    w(alerts.md_table(["Ticker", *RACE_HEAD, "Excess R"], rows) + "\n")
    spread = s[(s["row"] == primary) & (s["who"] == "pattern") & (s["sides"] == "both")]
    if "spread_r" in spread and spread["spread_r"].notna().any():
        w(f"## After the quoted spread ({primary})\n")
        rows = [[r.scope, f"{int(r.trades):,}", f"{r.spread_r:.3f}", fmt(r.bracket), fmt(r.net)]
                for r in spread.itertuples() if pd.notna(r.spread_r) and r.scope in ("all", *groups, *tickers)]
        w(alerts.md_table(["Scope", "Trades", "Median spread, R", "Bracket, mean R", "Bracket less one spread, "
                           "mean R"], rows) + "\n")
        w("One full spread a round trip (half at the entry, half at the exit), from the median quoted spread in the "
          "minute before the entry.\n")
    w(f"## By year ({primary}, all tickers)\n")
    tr = res["trades"][res["trades"]["row"] == primary]
    rnd = res["random"][res["random"]["row"] == primary]
    rows = []
    for y, g in tr.groupby(tr["session"].dt.year):
        r = {**row_stats(g, 1.0), **excess(g, rnd[rnd["pid"].isin(g["pid"])])}
        rows.append([y, len(g), pct(r.get("hit1")), pct(r.get("hit2")), fmt(r.get("bracket")), fmt(r.get("excess"))])
    w(alerts.md_table(["Year", "Trades", "+1R first", "+2R first", "Bracket, mean R", "Excess R"], rows) + "\n")
    w(f"## How many FVGs survive each step ({primary})\n")
    f = res["funnel"].groupby(["step", "side"], sort=False)["count"].sum().unstack()
    rows = [[k, f"{int(v.loc[1]):,}", f"{int(v.loc[-1]):,}"] for k, v in f.iterrows()]
    w(alerts.md_table(["Step", "Long", "Short"], rows) + "\n")
    a = get(s, row=primary, scope="all", sides="both", who="pattern")
    w("## Notes\n")
    if a is not None and a["trades"] > 1 and pd.isna(a.get("spread_r", np.nan)):
        w(f"- The median stop is {a['r_bps']:.0f} bps from the entry, so 1 bp a side costs about "
          f"{a['cost_r']:.2f} R a trade.")
    for note in notes:
        w(f"- {note}")
    w("- Unresolved brackets are marked at the close 20 sessions after entry; scans stop at the data's last "
      "session (flagged `truncated` in trades.parquet).")
    w("- The playbook claims about 90% wins; the closest measure here is +1R before the stop.")
    if not res["final"]:
        w(f"- {held} not read.")
    return "\n".join(lines) + "\n"


def plot(res, path, title="TG Capital Trident on 15 stocks and ETFs",
         subtitle="30-minute bars, entries by 13:00 ET; targets scanned up to 20 sessions."):
    s = res["stats"]
    fig = plt.figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.08, 0.13, 0.88, 0.69])
    x = np.arange(len(TARGETS))
    primary = res.get("primary", PRIMARY)
    series = [(primary, "pattern", SERIES[0]), (primary, "random", INK_2),
              ("every FVG, no trident", "pattern", SERIES[1])]
    for k, (row, who, color) in enumerate(series):
        r = get(s, row=row, scope="all", sides="both", who=who)
        if r is None or r["trades"] < 2:
            continue
        label = f"{row}, {who} ({int(r['trades']):,})" if who == "pattern" else f"{primary}, random matches"
        ax.bar(x + (k - 1) * 0.27, [r[f"hit{t}"] for t in TARGETS], width=0.25, color=color, label=label)
    ax.plot(x, [100 / (t + 1) for t in TARGETS], "_", color=BASELINE, markersize=30, mew=1.5, label="no edge, 1/(k+1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{t}R before the stop" for t in TARGETS], fontsize=7.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("% of trades", color=INK_2, fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="upper right")
    span = f"{res['first']:%Y-%m} to {res['last']:%Y-%m}"
    fig.text(0.02, 0.97, f"{title} ({span})", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, subtitle, color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TG Capital Trident pattern (FVG + doji + confirmation) signal study.")
    p.add_argument("--out", default="output/trident")
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
    res = run_study(data, sessions, final=args.final_test, timings=timings)
    out = Path(args.out) / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["random"].to_parquet(out / "random.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "first": str(res["first"].date()),
        "last": str(res["last"].date()), "tickers": args.tickers, "random_draws": RANDOM_DRAWS, "seed": SEED},
        indent=2))
    plot(res, out / "races.png")
    (out / "report.md").write_text(render_report(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
