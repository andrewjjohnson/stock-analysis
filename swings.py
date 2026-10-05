"""Swings on a 5-minute chart: do the session's own swing highs and lows act as support and resistance, and is the
time between peaks and valleys more regular than chance? A signal study on Massive minute bars (no options, no P&L).

  uv run --env-file .env python swings.py --out output/swings                      # design period
  uv run --env-file .env python swings.py --final-test --out output/swings         # adds the holdout, once
  uv run --env-file .env python swings.py --chart 2024-03-05 --out output/swings   # charts only

Swings (`zigzag`): regular-hours 5-minute bars, restarting each session. A peak is confirmed when a later bar's low
is R below the highest high since the last valley, and a valley the other way round; R = 0.15 daily ATR(14)
through yesterday (about $0.90 on SPY). A bar that sets a new extreme cannot also confirm the turn, because the
order of its high and low is unknown; a swing is known at the close of its confirming bar.

Fixed before any result was seen:
- Part 1, swing levels. Each confirmed peak (swing high) and valley (swing low) becomes a level at the end of its
  confirming bar or at 11:30 ET, whichever is later: nothing is drawn before 11:30. Its side (support below the
  price, resistance above) and distance come from the last 1-minute close before that. From then on it is measured
  like levels.py: the first touch on 1-minute bars, starting at least 30 minutes before the close, then the race of
  +/-0.1 ATR (and 0.25 ATR); levels within 0.05 ATR of that close are skipped. Controls: 10 random fakes per level
  (its session, start and side, at distances drawn from real levels of the same type, origin (drawn at 11:30 or
  later) and side on other sessions) and near fakes 0.2/0.3/0.4 ATR either side of it. Test: swing high and swing
  low x SPY, QQQ, IWM against the random fakes at 0.1 ATR, BH q < 0.10 across those 6 and the same sign at 0.25
  ATR; the near fakes are reported beside them.
- Part 2, swing timing. A leg is the time from one swing's extreme to the next one's (valley to peak = up-leg);
  complete legs only. Null: 100 random-direction copies of each session, in which every 5-minute bar keeps its
  open, high, low and close relative to the previous close but is mirrored at random (`flipped`): the same
  volatility pattern (busy open, quiet lunch), no directional memory. Tests per ticker at R = 0.15: (a) spread,
  the standard deviation of log leg durations (lower = more regular); (b) lag-1, the Spearman correlation between
  the durations of consecutive legs (does one leg's length predict the next?). Monte Carlo p-values against the
  null, BH q < 0.10 across the 6. R = 0.10 and 0.25 are reported for reading only.
- Design to 2024-12-31; the holdout (2025-01-01 on) is read only with --final-test, for the qualifiers.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import levels as lv  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = lv.TICKERS
BAR = pd.Timedelta(minutes=5)
SLOTS = 78                                  # 5-minute bars in a full session
R = 0.15                                    # zigzag reversal, daily ATR
R_CHECKS = (0.10, 0.25)
DRAW_FROM = pd.Timedelta(hours=2)           # lines start at 11:30 ET
TYPES = ("swing high", "swing low")
ORIGINS = ("drawn at 11:30", "drawn later")
NULL_REPS = 100
CHUNK = 25                                  # null copies built at a time
MAX_AGE = 24                                # bars (2 hours), for the turn hazard and duration shares
TESTS = ("spread", "lag1")
SEED = 11
STAT_LABELS = {"legs_per_session": "Legs per session", "mean": "Average leg, minutes",
               "median": "Median leg, minutes",
               "q1": "Quarter of legs shorter than, minutes", "q3": "Quarter of legs longer than, minutes",
               "spread": "Spread (SD of log duration)", "up_median": "Median up-leg, minutes",
               "down_median": "Median down-leg, minutes", "lag1": "Consecutive-leg correlation"}


# ---------------------------------------------------------------- swings

def grid(rth, sessions, days):
    """5-minute bars of `days` as (len(days), SLOTS) arrays of open/high/low/close: NaN where a bar is missing or
    after an early close."""
    bars, _ = features.resample_bars(rth[rth["session"].isin(days)], 5, min_coverage=0.8)
    pos = days.get_indexer(bars["session"])
    slot = (_ns(bars["bar_start"]) - _ns(sessions["open"].reindex(days))[pos]) // BAR.value
    out = {k: np.full((len(days), SLOTS), np.nan) for k in ("open", "high", "low", "close")}
    for k, a in out.items():
        a[pos, slot] = bars[k].to_numpy(float)
    return out


def zigzag(high, low, r):
    """Swing points of each row of `high`/`low` (5-minute bars in time order) with reversal size `r` (one per row).
    Returns a DataFrame sorted by row and bar: row, kind (+1 peak, -1 valley), price, bar (where the extreme was)
    and confirmed (the bar whose close confirmed it)."""
    n, m = high.shape
    r = np.broadcast_to(np.asarray(r, float), (n,))
    state = np.zeros(n, np.int8)    # 0 undecided, +1 rising (tracking a peak), -1 falling (tracking a valley)
    hi, hi_at = np.full(n, -np.inf), np.zeros(n, np.int64)
    lo, lo_at = np.full(n, np.inf), np.zeros(n, np.int64)
    found = []
    for j in range(m):
        h, l = high[:, j], low[:, j]
        ok = np.isfinite(h) & np.isfinite(l)
        rising, falling = ok & (state >= 0), ok & (state <= 0)  # undecided rows track both
        new_hi, new_lo = rising & (h > hi), falling & (l < lo)
        hi, hi_at = np.where(new_hi, h, hi), np.where(new_hi, j, hi_at)
        lo, lo_at = np.where(new_lo, l, lo), np.where(new_lo, j, lo_at)
        peak = rising & ~new_hi & (l <= hi - r)
        valley = falling & ~new_lo & (h >= lo + r)
        both = peak & valley  # undecided rows only: the earlier extreme came first (a tie decides nothing yet)
        peak &= ~both | (hi_at < lo_at)
        valley &= ~both | (lo_at < hi_at)
        for mask, kind, price, at in ((peak, 1, hi, hi_at), (valley, -1, lo, lo_at)):
            idx = np.flatnonzero(mask)
            if idx.size:
                found.append(pd.DataFrame({"row": idx, "kind": kind, "price": price[idx], "bar": at[idx],
                                           "confirmed": j}))
        state = np.where(peak, -1, np.where(valley, 1, state)).astype(np.int8)
        lo, lo_at = np.where(peak, l, lo), np.where(peak, j, lo_at)  # the new leg starts from this bar
        hi, hi_at = np.where(valley, h, hi), np.where(valley, j, hi_at)
    if not found:
        return pd.DataFrame({"row": [], "kind": [], "price": [], "bar": [], "confirmed": []}, dtype=float)
    return pd.concat(found, ignore_index=True).sort_values(["row", "bar"], kind="stable").reset_index(drop=True)


def last_bars(g):
    """Index of each row's last bar."""
    return SLOTS - 1 - np.argmax(np.isfinite(g["high"])[:, ::-1], axis=1)


def legs(sw, last):
    """Complete legs between consecutive swings of each row (kind +1 = up-leg, valley to peak; length in bars),
    and the unfinished leg each row ends with (age = bars from its last swing to the row's last bar `last[row]`)."""
    row, bar, kind = (sw[c].to_numpy().astype(np.int64) for c in ("row", "bar", "kind"))
    same = row[1:] == row[:-1]
    done = pd.DataFrame({"row": row[1:][same], "kind": kind[1:][same], "start": bar[:-1][same],
                         "bars": (bar[1:] - bar[:-1])[same]})
    end = np.r_[~same, True] if len(row) else np.zeros(0, bool)
    unfinished = pd.DataFrame({"row": row[end], "age": np.asarray(last)[row[end]] - bar[end]})
    return done, unfinished


def flipped(g, reps, rng):
    """`reps` random-direction copies of the sessions in grid `g`: every 5-minute bar keeps its open, high, low and
    close relative to the previous bar's close (the first bar's: its own open) but is mirrored at random. Returns
    (high, low) with copy k in rows k * sessions to (k + 1) * sessions - 1."""
    o, h, l, c = g["open"], g["high"], g["low"], g["close"]
    pc = np.concatenate([o[:, :1], c[:, :-1]], axis=1)
    s = rng.choice(np.array([-1.0, 1.0]), size=(reps,) + o.shape)
    close = o[:, :1] + np.nancumsum(np.where(s > 0, c - pc, pc - c), axis=2)
    prev = np.concatenate([np.broadcast_to(o[:, :1], (reps,) + o[:, :1].shape), close[:, :, :-1]], axis=2)
    high = prev + np.where(s > 0, h - pc, pc - l)
    low = prev + np.where(s > 0, l - pc, pc - h)
    return high.reshape(-1, o.shape[1]), low.reshape(-1, o.shape[1])


# ---------------------------------------------------------------- part 1: swing levels

def swing_levels(sw, days, sessions, rth, atr):
    """Real levels from confirmed swings (`sw` rows index `days`). Each starts at the end of its confirming bar or
    at 11:30 ET, whichever is later, with its side and distance (ATR) relative to the last 1-minute close before
    that. Levels starting within 30 minutes of the close or within MIN_DIST ATR of that close are dropped."""
    day = days[sw["row"].to_numpy().astype(np.int64)]
    open_ns, close_ns = _ns(sessions["open"].reindex(day)), _ns(sessions["close"].reindex(day))
    drawn = open_ns + (sw["confirmed"].to_numpy().astype(np.int64) + 1) * BAR.value
    start = np.maximum(drawn, open_ns + DRAW_FROM.value)
    t = _ns(rth["ts"])
    i = np.searchsorted(t, start, side="left") - 1  # the last minute that starts before `start`
    ok = (i >= 0) & (start < close_ns - lv.TOUCH_END.value)
    i = np.where(ok, i, 0)
    ok &= _ns(rth["session"])[i] == _ns(day)
    ref = np.where(ok, rth["close"].to_numpy(float)[i], np.nan)
    a = atr.reindex(day).to_numpy(float)
    dist = (sw["price"].to_numpy(float) - ref) / a
    out = pd.DataFrame({"session": day, "level": np.where(sw["kind"].to_numpy() > 0, TYPES[0], TYPES[1]),
                        "price": sw["price"].to_numpy(float), "dist": dist, "ref": ref, "atr": a,
                        "start": pd.to_datetime(start, utc=True),
                        "origin": np.where(drawn <= open_ns + DRAW_FROM.value, ORIGINS[0], ORIGINS[1]),
                        "swing_bar": sw["bar"].to_numpy().astype(int),
                        "confirmed": sw["confirmed"].to_numpy().astype(int)})
    out = out[ok & np.isfinite(dist) & (np.abs(dist) >= lv.MIN_DIST)].reset_index(drop=True)
    out["side"] = np.where(out["dist"] < 0, 1, -1)
    out["period"] = np.where(out["session"] >= lv.HOLDOUT_START, "holdout", "design")
    out["fake"], out["control"] = False, "real"
    return out


def random_fakes(real, k=lv.FAKES, seed=SEED):
    """k fakes per real level: its session, start, reference price and side, at a distance (ATR) drawn from real
    levels of the same type, origin, side and period on other sessions."""
    rng = np.random.default_rng(seed)
    frames = []
    for _, g in real.groupby(["level", "origin", "side", "period"], sort=True):
        sess = g["session"].to_numpy()
        if len(np.unique(sess)) < 2:
            continue
        idx = rng.integers(0, len(g), size=(len(g), k))
        bad = sess[idx] == sess[:, None]
        while bad.any():
            idx[bad] = rng.integers(0, len(g), size=int(bad.sum()))
            bad = sess[idx] == sess[:, None]
        f = g.loc[g.index.repeat(k)].assign(dist=g["dist"].to_numpy()[idx].ravel(), fake=True, control="random")
        frames.append(f.assign(price=f["ref"] + f["dist"] * f["atr"]))
    return pd.concat(frames, ignore_index=True) if frames else real.iloc[:0]


def near_fakes(real, offsets=lv.NEAR):
    """Fakes `offsets` ATR above and below each real level, same session, start and side of the reference price."""
    f = pd.concat([real.assign(dist=real["dist"] + s * o) for o in offsets for s in (1, -1)], ignore_index=True)
    f = f[((f["dist"] < 0) == (f["side"] > 0)) & (f["dist"].abs() >= lv.MIN_DIST)]
    return f.assign(price=f["ref"] + f["dist"] * f["atr"], fake=True, control="near").reset_index(drop=True)


# ---------------------------------------------------------------- part 2: swing timing

def timing_stats(done, sessions):
    """Leg durations (minutes) of one set of sessions: legs per session, quartiles, spread (SD of log duration),
    up- and down-leg medians, and the Spearman correlation between consecutive legs of a session."""
    m = done["bars"].to_numpy(float) * 5
    out = {"legs_per_session": len(m) / sessions}
    if len(m) < 20:
        return out
    q1, med, q3 = np.percentile(m, [25, 50, 75])
    row = done["row"].to_numpy()
    same = row[1:] == row[:-1]
    up = done["kind"].to_numpy() > 0
    out.update(mean=m.mean(), median=med, q1=q1, q3=q3, spread=np.log(m).std(), up_median=np.median(m[up]),
               down_median=np.median(m[~up]),
               lag1=spearmanr(m[:-1][same], m[1:][same]).statistic if same.sum() > 10 else np.nan)
    return out


def hazard(done, unfinished, max_age=MAX_AGE):
    """Share of legs that end at each age 1..max_age (bars since the last swing) among those still running then."""
    a = np.arange(1, max_age + 1)
    d, c = done["bars"].to_numpy(), unfinished["age"].to_numpy()
    at_risk = (d[:, None] >= a).sum(axis=0) + (c[:, None] >= a).sum(axis=0)
    return (d[:, None] == a).sum(axis=0) / np.maximum(at_risk, 1)


def duration_shares(done, max_age=MAX_AGE):
    """Share of legs lasting 1, 2, ..., max_age bars and longer."""
    d = np.minimum(done["bars"].to_numpy().astype(np.int64), max_age + 1)
    return np.bincount(d, minlength=max_age + 2)[1:] / max(len(d), 1)


def mc_p(real, null, side=0):
    """Monte Carlo p-value of `real` against null draws: two-sided (distance from the null mean) or one-sided
    (side +1: real above the null, -1: below)."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(real) or not len(null):
        return np.nan
    if side == 0:
        hits = (np.abs(null - null.mean()) >= abs(real - null.mean())).sum()
    else:
        hits = (side * null >= side * real).sum()
    return (1 + hits) / (1 + len(null))


def timing(g, atr, reps=NULL_REPS, seed=SEED):
    """Leg timing of the sessions in grid `g` for each reversal size, real against `reps` random-direction copies:
    one row per (r, statistic). At R, also the turn hazard by leg age and the shares of legs by duration."""
    n, last = len(atr), last_bars(g)
    rng = np.random.default_rng(seed)
    rows, extra = [], {}
    for r in (R, *R_CHECKS):
        thr = r * atr
        done, unfinished = legs(zigzag(g["high"], g["low"], thr), last)
        nd, nu = [], []
        for k in range(0, reps, CHUNK):
            c = min(CHUNK, reps - k)
            h, l = flipped(g, c, rng)
            d, u = legs(zigzag(h, l, np.tile(thr, c)), np.tile(last, c))
            nd.append(d.assign(rep=k + d["row"] // n))
            nu.append(u.assign(rep=k + u["row"] // n))
        nd, nu = pd.concat(nd, ignore_index=True), pd.concat(nu, ignore_index=True)
        null = pd.DataFrame([timing_stats(x, n) for _, x in nd.groupby("rep")])
        for stat, v in timing_stats(done, n).items():
            col = null[stat].to_numpy(float) if stat in null else np.full(1, np.nan)
            rows.append({"r": r, "stat": stat, "real": v, "null_mean": np.nanmean(col),
                         "null_lo": np.nanpercentile(col, 2.5), "null_hi": np.nanpercentile(col, 97.5),
                         "p": mc_p(v, col), "p_above": mc_p(v, col, 1), "p_below": mc_p(v, col, -1)})
        if r == R:
            by_rep = dict(tuple(nu.groupby("rep")))
            extra = {"hazard": hazard(done, unfinished), "shares": duration_shares(done),
                     "null_hazard": np.array([hazard(x, by_rep.get(rep, nu.iloc[:0])) for rep, x in nd.groupby("rep")]),
                     "null_shares": np.array([duration_shares(x) for _, x in nd.groupby("rep")])}
    return pd.DataFrame(rows), extra


# ---------------------------------------------------------------- study

def ticker_data(minutes, sessions, periods):
    """(regular-hours minutes, ATR by session (NaN where today's session is under 80% covered), study days, grid)."""
    rth, _ = features.regular_session_minutes(minutes, sessions)
    daily = features.daily_features(rth, sessions, atr_period=lv.ATR)
    atr = daily[f"prev_atr_{lv.ATR}"].where(daily["usable"])
    period = np.where(sessions.index >= lv.HOLDOUT_START, "holdout", "design")
    days = sessions.index[np.isin(period, periods) & np.isfinite(atr.to_numpy(float))]
    return rth, atr, days, grid(rth, sessions, days)


def run_study(minutes_by_ticker, sessions, *, final=False, timings=None, reps=lv.REPS, null_reps=NULL_REPS):
    """minutes_by_ticker: {ticker: minute bars}. No file I/O."""
    timings = {} if timings is None else timings
    periods = ["design", "holdout"] if final else ["design"]
    events, timing_rows, extras = [], [], {}
    for tk, minutes in minutes_by_ticker.items():
        with timed(timings, "bars and swings"):
            rth, atr, days, g = ticker_data(minutes, sessions, periods)
            sw = zigzag(g["high"], g["low"], R * atr.reindex(days).to_numpy(float))
        with timed(timings, "swing levels"):
            real = swing_levels(sw, days, sessions, rth, atr)
            ev = pd.concat([real, random_fakes(real), near_fakes(real)], ignore_index=True)
            events.append(lv.measure(rth, ev, pd.DataFrame({"atr": atr})).assign(ticker=tk))
        with timed(timings, "timing and null"):
            for period in periods:
                rows = np.flatnonzero((days >= lv.HOLDOUT_START) == (period == "holdout"))
                t, extra = timing({k: v[rows] for k, v in g.items()}, atr.reindex(days[rows]).to_numpy(float),
                                  null_reps)
                timing_rows.append(t.assign(ticker=tk, period=period))
                if period == "design":
                    extras[tk] = extra
    events = pd.concat(events, ignore_index=True)
    k1, k2 = lv.KEYS
    with timed(timings, "statistics"):
        stats, pooled, near = lv.statistics(events, sessions, periods, reps)
        design = stats[stats["period"] == "design"].copy()
        design["q"] = mr.bh_qvalues(design[f"p_{k1}"].fillna(1).to_numpy())
        design["qualifies"] = (design["q"] < lv.FDR) & (np.sign(design[f"diff_{k1}"]) == np.sign(design[f"diff_{k2}"]))
        origin = events.assign(level=events["level"] + ", " + events["origin"])
        by_origin = lv.statistics(origin, sessions, ["design"], reps)
        tm = pd.concat(timing_rows, ignore_index=True)
        tests = tm[(tm["period"] == "design") & (tm["r"] == R) & tm["stat"].isin(TESTS)].copy()
        tests["q"] = mr.bh_qvalues(tests["p"].fillna(1).to_numpy())
        tests["qualifies"] = tests["q"] < lv.FDR
        held = held_t = None
        if final:
            keys = ["ticker", "level"]
            q = design.loc[design["qualifies"], keys + [f"diff_{k1}"]].rename(columns={f"diff_{k1}": "design_diff"})
            held = stats[stats["period"] == "holdout"].merge(q, on=keys)
            if len(held):
                up = held[f"p_up_{k1}"]
                held["holdout_q"] = mr.bh_qvalues(np.where(held["design_diff"] > 0, up, 1 - up))
                held["holds_up"] = held["holdout_q"] < lv.FDR
            q = tests.loc[tests["qualifies"], ["ticker", "stat", "real", "null_mean"]]
            held_t = tm[(tm["period"] == "holdout") & (tm["r"] == R)].merge(
                q.rename(columns={"real": "design_real", "null_mean": "design_null"}), on=["ticker", "stat"])
            if len(held_t):
                above = held_t["design_real"] > held_t["design_null"]
                held_t["holdout_q"] = mr.bh_qvalues(np.where(above, held_t["p_above"], held_t["p_below"]))
                held_t["holds_up"] = held_t["holdout_q"] < lv.FDR
    return {"events": events, "stats": stats, "design": design, "pooled": pooled, "near": near,
            "by_origin": by_origin, "timing": tm, "tests": tests, "extras": extras, "holdout": held,
            "holdout_timing": held_t, "final": final, "tickers": list(minutes_by_ticker), "null_reps": null_reps,
            "median_atr": {tk: float(e.loc[~e["fake"] & (e["period"] == "design"), "atr"].median())
                           for tk, e in events.groupby("ticker")}}


# ---------------------------------------------------------------- report

pct, pts, find = lv.pct, lv.pts, lv.find


def num(v, digits=2):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def render_report(res):
    d, nr = res["design"], res["near"]
    k1, k2 = lv.KEYS
    x1, x2 = lv.SIZES
    lines = []
    w = lines.append
    w("# Swings on a 5-minute chart\n")
    w(f"Swings come from a zigzag on 5-minute bars that restarts each session: a peak is confirmed once price falls "
      f"{R:g} of the daily ATR from the highest high since the last valley, and a valley the other way round. "
      + "; ".join(f"{tk} {R:g} ATR was about ${R * a:.2f}" for tk, a in res["median_atr"].items())
      + ". Design period to 2024-12-31" + (", holdout 2025-01-01 on." if res["final"] else " (holdout not read).")
      + "\n")
    w("## Part 1: do swing highs and lows act as support and resistance?\n")
    w("Each confirmed swing high and low becomes a line when it is confirmed, but not before 11:30 ET; its side "
      "(support or resistance) comes from the price at that moment. Then, as in `levels.py`: the first touch, and "
      f"a race from the line: it **held** if price moved {x1:g} daily ATR back away before going as far through it. "
      f"*Random fakes*: {lv.FAKES} per line on the same session, start and side, at distances taken from other "
      "sessions' lines. *Near fakes*: lines 0.2-0.4 ATR either side of the real one. Differences are held % at real "
      "lines minus at fakes, in points, with 95% intervals from resampling sessions; negative = the line broke more "
      "often than a fake.\n")
    q = d[d["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(d)} design tests qualified (q < {lv.FDR:g} against random fakes at "
      f"+/-{x1:g} ATR, same sign at +/-{x2:g})" + (": " + "; ".join(
          f"{r.ticker} {r.level} ({pts(r[f'diff_{k1}'])} pts)" for _, r in q.iterrows()) if len(q) else ".") + "\n")
    rows = []
    for tk in res["tickers"]:
        for name in TYPES:
            r, n = find(d, ticker=tk, level=name), find(nr, ticker=tk, level=name)
            if r is None:
                continue
            rows.append([tk, name, f"{int(r['level_days']):,}", f"{pct(r['touched'])} / {pct(r['fake_touched'])}",
                         f"{pct(r.get(f'real_{k1}'))} ({pct(r.get(f'real_lo_{k1}'))}-{pct(r.get(f'real_hi_{k1}'))}), "
                         f"n={int(r[f'n_{k1}']):,}", pct(r.get(f"fake_{k1}")),
                         f"{pts(r.get(f'diff_{k1}'))} ({pts(r.get(f'diff_lo_{k1}'))} to {pts(r.get(f'diff_hi_{k1}'))})",
                         f"{r['q']:.2f}",
                         "n/a" if n is None else f"{pct(n.get(f'fake_{k1}'))}; {pts(n.get(f'diff_{k1}'))} "
                                                 f"({pts(n.get(f'diff_lo_{k1}'))} to {pts(n.get(f'diff_hi_{k1}'))})",
                         f"{pct(r.get(f'real_{k2}'))} / {pct(r.get(f'fake_{k2}'))} ({pts(r.get(f'diff_{k2}'))})"])
    w(alerts.md_table(["Ticker", "Line", "Lines", "Touched: real / random", f"Held +/-{x1:g} ATR, real (95%)",
                       "Random fakes", "Difference (95%)", "q", "Near fakes held; difference (95%)",
                       f"+/-{x2:g} ATR: real / random (diff)"], rows) + "\n")
    w("All three tickers together, by when the line was drawn and by side (for reading only, not part of the "
      f"test count; +/-{x1:g} ATR):\n")
    st, po, ne = res["by_origin"]
    rows = []
    for name in TYPES:
        for origin in ORIGINS:
            label = f"{name}, {origin}"
            cells = []
            for side in ("both", "support", "resistance"):
                r = find(po, level=label, side=side)
                cells.append("n/a" if r is None or pd.isna(r.get(f"real_{k1}")) else
                             f"{pct(r[f'real_{k1}'])} vs {pct(r[f'fake_{k1}'])} ({pts(r[f'diff_{k1}'])}), "
                             f"n={int(r[f'n_{k1}']):,}")
            n = find(ne, ticker="all", level=label)
            cells.append("n/a" if n is None or pd.isna(n.get(f"diff_{k1}")) else
                         f"{pts(n[f'diff_{k1}'])} ({pts(n[f'diff_lo_{k1}'])} to {pts(n[f'diff_hi_{k1}'])})")
            rows.append([label, *cells])
    w(alerts.md_table(["Line", "Held: real vs random fakes (diff)", "As support", "As resistance",
                       "Against near fakes (95%)"], rows) + "\n")

    w("## Part 2: is the time between peaks and valleys regular?\n")
    w("A leg is the time from one swing to the next (valley to peak = up-leg). Real sessions are compared with "
      f"{res['null_reps']} random-direction copies of each session: every 5-minute bar keeps its size and shape "
      "relative to the previous close but is flipped up or down at random, so the copies have the same volatility "
      "pattern (busy open, quiet lunch) and no memory. If real swings keep better time than the copies, there is a "
      "rhythm beyond chance. *Spread* is the standard deviation of log leg durations (lower = more regular); "
      "*consecutive-leg correlation* asks whether a long leg tends to be followed by a long one.\n")
    t = res["tests"]
    q = t[t["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(t)} timing tests qualified (q < {lv.FDR:g})" + (": " + "; ".join(
        f"{r.ticker} {STAT_LABELS[r.stat].lower()} ({num(r.real)} vs {num(r.null_mean)} in the copies)"
        for r in q.itertuples()) if len(q) else ".") + "\n")
    tm = res["timing"]
    for tk in res["tickers"]:
        w(f"### {tk}, design period, R = {R:g} ATR\n")
        rows = []
        for stat, label in STAT_LABELS.items():
            r = find(tm, ticker=tk, period="design", r=R, stat=stat)
            if r is None:
                continue
            qv = find(t, ticker=tk, stat=stat)
            digits = 1 if stat in ("legs_per_session", "mean", "median", "q1", "q3", "up_median",
                                   "down_median") else 3
            rows.append([label, num(r["real"], digits),
                         f"{num(r['null_mean'], digits)} ({num(r['null_lo'], digits)} to {num(r['null_hi'], digits)})",
                         num(r["p"], 3), "" if qv is None else num(qv["q"], 3)])
        w(alerts.md_table(["Measure", "Real", "Random-direction copies (95% range)", "p", "q (tests)"], rows) + "\n")
    w("Other zigzag sizes (for reading only): real vs the copies' average.\n")
    rows = []
    for tk in res["tickers"]:
        for r_ in R_CHECKS:
            cells = []
            for stat in ("legs_per_session", "mean", "spread", "lag1"):
                r = find(tm, ticker=tk, period="design", r=r_, stat=stat)
                cells.append("n/a" if r is None else f"{num(r['real'])} vs {num(r['null_mean'])}")
            rows.append([tk, f"{r_:g}", *cells])
    w(alerts.md_table(["Ticker", "R (ATR)", "Legs per session", "Average leg (min)", "Spread",
                       "Consecutive-leg correlation"], rows) + "\n")
    if res["final"]:
        w("## Holdout: did the qualifiers hold up?\n")
        h, ht = res["holdout"], res["holdout_timing"]
        if (h is None or h.empty) and (ht is None or ht.empty):
            w("No qualifiers to test.\n")
        if h is not None and len(h):
            rows = [[r.ticker, r.level, pts(r.design_diff), f"{pts(getattr(r, f'diff_{k1}'))} "
                     f"({pts(getattr(r, f'diff_lo_{k1}'))} to {pts(getattr(r, f'diff_hi_{k1}'))})",
                     f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"] for r in h.itertuples()]
            w(alerts.md_table(["Ticker", "Line", "Design diff", "Holdout diff (95%)", "Holdout q", "Holds up"],
                              rows) + "\n")
        if ht is not None and len(ht):
            rows = [[r.ticker, STAT_LABELS[r.stat], f"{num(r.design_real)} vs {num(r.design_null)}",
                     f"{num(r.real)} vs {num(r.null_mean)}", f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"]
                    for r in ht.itertuples()]
            w(alerts.md_table(["Ticker", "Measure", "Design: real vs copies", "Holdout: real vs copies",
                               "Holdout q", "Holds up"], rows) + "\n")
    w("## Notes\n")
    w("- A swing is known only at the close of the 5-minute bar that confirms it; lines are never drawn earlier, and "
      "never before 11:30. Leg durations are measured between the swings' extremes, which is what a chart shows "
      "in hindsight.")
    w("- Near fakes can land on another real swing line of the same session (swing lines crowd together), which "
      "pulls the two held rates together.")
    w("- The random-direction copies keep each bar's size, so they keep the time-of-day volatility pattern and "
      "volatile versus calm days. They remove any tendency of moves to continue or reverse, which is the only "
      "thing that could make turns predictable in time.")
    w("- Charts: `swings.py --chart DATE` draws a session's 5-minute candles with its zigzag (minutes per leg) and "
      "swing lines.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    """Part 1 differences (left) and, for the first ticker, the leg duration shares and turn hazard against the
    random-direction copies (right)."""
    d, nr = res["design"], res["near"]
    k1 = lv.KEYS[0]
    fig = plt.figure(figsize=(10, 4.8), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.13, 0.17, 0.27, 0.64])
    labels, y = [], 0.0
    for tk, color in zip(res["tickers"], lv.COLORS):
        for name in TYPES:
            for table, control, fill in ((d, "random", color), (nr, "near", SURFACE)):
                r = find(table, ticker=tk, level=name)
                if r is None or pd.isna(r.get(f"diff_{k1}")):
                    continue
                m, lo, hi = r[f"diff_{k1}"], r[f"diff_lo_{k1}"], r[f"diff_hi_{k1}"]
                dy = 0.15 if control == "random" else -0.15
                ax.errorbar([m], [y + dy], xerr=[[m - lo], [hi - m]], fmt="o", ms=4, mfc=fill, color=color,
                            ecolor=color, elinewidth=1.1, capsize=2)
            labels.append((y, f"{tk} {name}"))
            y -= 1
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Held % real minus fake, points (95%)", color=INK_2, fontsize=7)
    ax.set_title("Swing lines: filled = random fakes, hollow = near", color=INK, fontsize=8, loc="left")

    tk = res["tickers"][0]
    e = res["extras"][tk]
    minutes = np.arange(1, MAX_AGE + 1) * 5
    for k, (key, title, ylabel) in enumerate((
            ("shares", f"{tk} leg durations", "Share of legs"),
            ("hazard", f"{tk}: chance the leg ends now", "Share of running legs ending at this age"))):
        ax = sds._axes(fig, [0.5 + k * 0.26, 0.17, 0.2, 0.64])
        real, null = e[key][:MAX_AGE], e[f"null_{key}"][:, :MAX_AGE]
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(minutes, lo * 100, hi * 100, color=GRID, lw=0, label="random-direction copies (95%)")
        ax.plot(minutes, null.mean(axis=0) * 100, color=MUTED, lw=1)
        ax.plot(minutes, real * 100, color=SERIES[0], lw=1.4, marker="o", ms=2.5, label="real")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlabel("Leg length, minutes" if key == "shares" else "Minutes since the last swing", color=INK_2,
                      fontsize=7)
        ax.set_ylabel(ylabel + ", %", color=INK_2, fontsize=7)
        ax.set_title(title, color=INK, fontsize=8, loc="left")
        if k == 0:
            ax.legend(frameon=False, fontsize=6.5, labelcolor=INK_2, loc="upper right")
    fig.text(0.02, 0.97, "Swings on a 5-minute chart", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Zigzag of {R:g} daily ATR, design period 2021-2024. Left: do swing lines hold better "
             "than fake lines? Right: real swing timing against random-direction copies of the same days.",
             color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- charts

def plot_session(ticker, day, sessions, rth, atr, path):
    """A session's 5-minute candles with its zigzag (minutes per leg) and swing lines from the time each is drawn,
    marking each line's first touch and whether it held or broke."""
    days = pd.DatetimeIndex([day])
    g = grid(rth, sessions, days)
    sw = zigzag(g["high"], g["low"], R * atr[day])
    real = swing_levels(sw, days, sessions, rth, atr)
    ev = lv.measure(rth, real, pd.DataFrame({"atr": atr})) if len(real) else real
    o = sessions.loc[day, "open"]
    n = int(last_bars(g)[0]) + 1
    fig = plt.figure(figsize=(10, 5.9), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.06, 0.2, 0.7, 0.66])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    bars = pd.DataFrame({k: g[k][0, :n] for k in ("open", "high", "low", "close")})
    have = bars["high"].notna().to_numpy()
    lv._candles(ax, np.flatnonzero(have), bars[have], INK_2)
    ax.plot(sw["bar"], sw["price"], color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=5)
    for a, b in zip(sw.iloc[:-1].itertuples(), sw.iloc[1:].itertuples()):
        ax.annotate(f"{(b.bar - a.bar) * 5}m", ((a.bar + b.bar) / 2, (a.price + b.price) / 2), xytext=(3, 0),
                    textcoords="offset points", color=INK_2, fontsize=6.5, va="center", zorder=7)
    lo_y, hi_y = np.nanmin(g["low"][0]), np.nanmax(g["high"][0])
    pad = (hi_y - lo_y) * 0.06
    lo_y, hi_y = lo_y - pad, hi_y + pad
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlim(-1, n + 1)
    first = DRAW_FROM / BAR
    ax.axvline(first - 0.5, color=BASELINE, lw=0.9, zorder=1)
    ax.text(first - 0.3, hi_y, "lines start 11:30", color=INK_2, fontsize=7, va="top")
    col = lv.race_col(lv.SIZES[0])
    rows = ev.sort_values("price", kind="stable") if len(ev) else ev
    for r, y in zip(rows.itertuples(), lv.spread(rows["price"].to_numpy(float), (hi_y - lo_y) * 0.034)):
        color = SERIES[1] if r.level == TYPES[0] else SERIES[0]
        x0 = (r.start - o) / BAR - 0.5
        ax.plot([x0, n], [r.price, r.price], color=color, lw=0.9, zorder=2)
        if not r.touched:
            what = "not touched"
        else:
            out = getattr(r, col)
            what = lv.OUTCOME.get(out, "unclear")
            mx = r.touch_minute // 5
            if out == 1.0:
                ax.plot(mx, r.price, "^" if r.side > 0 else "v", ms=7, color=TARGET, mec=SURFACE, mew=0.8, zorder=6)
            elif out == -1.0:
                ax.plot(mx, r.price, "X", ms=7, color=INK, mec=SURFACE, mew=0.8, zorder=6)
            else:
                ax.plot(mx, r.price, "o", ms=6, mfc=SURFACE, color=MUTED, mew=1.2, zorder=6)
        when = (o + r.swing_bar * BAR).tz_convert(NY).strftime("%H:%M")
        ax.annotate(lv.tex(f"{r.level} {r.price:.2f} ({when}) · {what}"), (n, r.price), xytext=(n + 3, y),
                    textcoords="data", color=color, fontsize=6.5, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.6) if abs(y - r.price) > 1e-9 else None)
    ticks = np.arange(0, n + 1, 12)
    ax.set_xticks(ticks, [(o + t * BAR).tz_convert(NY).strftime("%H:%M") for t in ticks], fontsize=7)
    fig.text(0.02, 0.975, f"{ticker} {day:%A %Y-%m-%d}: 5-minute swings", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.935, lv.tex(f"Zigzag of {R:g} daily ATR (${R * atr[day]:.2f}); the numbers are minutes from one "
             "swing to the next. A line starts when its swing is confirmed, but not before 11:30; markers show its "
             f"first touch and whether price then moved +/-{lv.SIZES[0]:g} ATR (${lv.SIZES[0] * atr[day]:.2f}) back "
             "away (held) or through it (broke) first."), color=INK_2, fontsize=7.5, va="top", wrap=True)
    handles = [Line2D([], [], color=MUTED, ls=(0, (4, 2)), label="zigzag"),
               Line2D([], [], color=SERIES[1], label="swing high line"),
               Line2D([], [], color=SERIES[0], label="swing low line"),
               Line2D([], [], ls="", marker="^", color=TARGET, label="held"),
               Line2D([], [], ls="", marker="X", color=INK, label="broke"),
               Line2D([], [], ls="", marker="o", mfc=SURFACE, color=MUTED, label="neither / unclear")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.04), ncol=6, frameon=False, fontsize=7,
               labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_charts(ticker, minutes, sessions, days, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rth, _ = features.regular_session_minutes(minutes, sessions)
    daily = features.daily_features(rth, sessions, atr_period=lv.ATR)
    atr = daily[f"prev_atr_{lv.ATR}"].where(daily["usable"])
    paths = []
    for day in days:
        if not np.isfinite(atr.get(day, np.nan)):
            print(f"skipped {day:%Y-%m-%d}: no data or no ATR yet")
            continue
        path = out / f"{ticker}_{day:%Y-%m-%d}.png"
        plot_session(ticker, day, sessions, rth, atr, path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="5-minute swings: swing lines as support/resistance, and swing timing.")
    p.add_argument("--out", default="output/swings")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read the 2025+ holdout (run once)")
    p.add_argument("--null-reps", type=int, default=NULL_REPS, help="random-direction copies per session")
    p.add_argument("--chart", nargs="+", metavar="DATE",
                   help="only draw these sessions (YYYY-MM-DD or START:END) for --chart-ticker")
    p.add_argument("--chart-ticker", default="SPY")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    out = Path(args.out)
    if args.chart:
        minutes, _ = gr.load_minutes(args.chart_ticker, sessions, args.cache_dir, args.refresh)
        paths = write_charts(args.chart_ticker, minutes, sessions, lv.chart_days(args.chart, sessions), out / "charts")
        print(f"wrote {len(paths)} chart(s) to {out / 'charts'}")
        return
    with timed(timings, "data"):
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    res = run_study(data, sessions, final=args.final_test, timings=timings, null_reps=args.null_reps)
    out.mkdir(parents=True, exist_ok=True)
    res["events"].to_parquet(out / "events.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["design"].to_csv(out / "design.csv", index=False)
    res["near"].to_csv(out / "near.csv", index=False)
    res["timing"].to_csv(out / "timing.csv", index=False)
    if res["final"]:
        for key in ("holdout", "holdout_timing"):
            if res[key] is not None:
                res[key].to_csv(out / f"{key}.csv", index=False)
    plot(res, out / "swings.png")
    (out / "report.md").write_text(render_report(res))
    print(f"swing-line tests: {len(res['design'])}, qualified: {int(res['design']['qualifies'].sum())}; "
          f"timing tests: {len(res['tests'])}, qualified: {int(res['tests']['qualifies'].sum())}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
