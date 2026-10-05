"""Support and resistance: do widely watched price levels hold more often than chance? A signal study on Massive
minute bars (no options, no P&L), plus charts that draw the levels on any session.

  uv run --env-file .env python levels.py --out output/levels                        # design period
  uv run --env-file .env python levels.py --final-test --out output/levels           # adds the holdout, once
  uv run --env-file .env python levels.py --chart 2024-08-05 2024-03-04:2024-03-08 --out output/levels   # charts only

Price touches some line every day, so bounces on a chart prove little. Each real level is compared with fake ones
at similar distances from the open on the same sessions, measured the same way.

Fixed before any result was seen:
- Tickers SPY (primary), QQQ, IWM. Levels, all known by the 9:30 ET open (LEVELS): yesterday's high, low and close
  (regular hours); last week's high and low (the previous calendar week's regular hours); today's pre-market high
  and low (4:00 ET to the open, at least 30 minute bars); yesterday's volume point of control and value-area high
  and low (auction_reclaim.previous_session_levels: the bar-approximated profile, 48 bins, 70% value area); the
  nearest whole dollar and the nearest multiple of $5 below and above the open.
- A level below the session's opening price is support, one above it resistance. Its touch is the first
  regular-hours minute that reaches it (low <= support, high >= resistance), starting at least 30 minutes before
  the close. Levels within 0.05 ATR of the open are skipped. ATR = daily ATR(14) through yesterday. Sessions with
  under 80% of their minutes are skipped.
- Outcome, a race from the level: it held if price later moved X back away from it (to support + X) before X
  through it (support - X), by the close; X = 0.1 ATR (primary) or 0.25 ATR. A break within the touching minute
  counts (price had reached the level first); a hold there does not, since the order of its high and low is
  unknown. A later minute that does both is left out; neither by the close is "neither".
- Fake levels: for each real level and session, 10 on the same session and side, at distances from the open (in
  ATR) drawn at random from the same level's distances on other sessions (same ticker, side and period), measured
  with the same rules.
- Test: held % at real levels minus held % at fake ones, with a standard error from 2,000 bootstrap resamples of
  sessions (events on the same day move together). A ticker x level test at X = 0.1 ATR qualifies with BH q < 0.10
  across the 36 design tests and a difference of the same sign at 0.25 ATR. Design to 2024-12-31; the holdout
  (2025-01-01 on) is read only with --final-test, for the qualifiers (one-sided in the design direction, BH across
  them).

Added after the first results (a check, not part of the test count): the random fakes match the real distances only
on average, so a level whose distance tracks the day (yesterday's high sits close to the open on gap-up days) is
touched on a different mix of days than its fakes. The "near" fakes sit 0.2, 0.3 and 0.4 ATR above and below each
real level on the same session (same side of the open, at least 0.05 ATR from it), so the days match exactly.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
import strategies.auction_reclaim as ar  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM")
LEVELS = {
    "prev_high": "yesterday's high", "prev_low": "yesterday's low", "prev_close": "yesterday's close",
    "week_high": "last week's high", "week_low": "last week's low",
    "pre_high": "pre-market high", "pre_low": "pre-market low",
    "poc": "yesterday's volume POC", "vah": "yesterday's value-area high", "val": "yesterday's value-area low",
    "dollar": "whole dollar", "five": "multiple of $5",
}
ROUND = {"dollar": 1.0, "five": 5.0}       # the nearest multiple below and above the open
PRE_START = pd.Timedelta(hours=4)           # pre-market from 4:00 ET
PRE_MIN_BARS = 30
PROFILE_BINS = 48
ATR = 14
SIZES = (0.1, 0.25)                         # race size X, in daily ATR; the first is primary
MIN_DIST = 0.05                             # ATR: levels closer to the open are skipped
TOUCH_END = pd.Timedelta(minutes=30)        # a touch starts at least this long before the close
FAKES = 10
NEAR = (0.2, 0.3, 0.4)                      # check added after the first results: fakes this far either side (ATR)
REPS = 2000
SEED = 7
MIN_EVENTS = 20                             # resolved touches of real levels needed for a test
HOLDOUT_START = pd.Timestamp("2025-01-01")
FDR = 0.10
COLORS = (SERIES[0], SERIES[1], INK_2, MUTED, TARGET)


def key(x):
    """Column suffix for race size x: 0.1 -> "10" (hundredths of ATR)."""
    return str(round(x * 100))


def race_col(x):
    return f"race_{key(x)}"


KEYS = [key(x) for x in SIZES]


# ---------------------------------------------------------------- levels

def pre_market(minutes, sessions):
    """Per session: the high, low and number of minute bars from 4:00 ET up to the regular open."""
    m = minutes.drop_duplicates("ts", keep="last")
    ts = m["ts"].dt.tz_convert("UTC").dt.as_unit("ns")
    day = ts.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    start = (sessions.index + PRE_START).tz_localize(NY).tz_convert("UTC").as_unit("ns")
    bounds = pd.DataFrame({"pre_start": start, "pre_end": sessions["open"].dt.as_unit("ns")}, index=sessions.index)
    m = m.assign(ts=ts, session=day).join(bounds, on="session", how="inner")
    m = m[(m["ts"] >= m["pre_start"]) & (m["ts"] < m["pre_end"])]
    g = m.groupby("session")
    return pd.DataFrame({"pre_high": g["high"].max(), "pre_low": g["low"].min(),
                         "pre_bars": g.size()}).reindex(sessions.index)


def level_table(rth, minutes, sessions):
    """One row per session: the regular-hours open, the daily ATR(14) through yesterday and every level's price,
    all known by the open. A level is NaN when its source session(s) lack data; an older session is never
    substituted. Round levels have a _below and an _above column."""
    daily = features.daily_features(rth, sessions, atr_period=ATR)
    ok = daily["usable"]
    t = pd.DataFrame(index=sessions.index)
    t["open"] = daily["open"].where(ok)
    t["atr"] = daily[f"prev_atr_{ATR}"]
    for k in ("high", "low", "close"):
        t[f"prev_{k}"] = daily[k].where(ok).shift(1)
    week = pd.Series(sessions.index.to_period("W"), index=sessions.index)
    hl = daily[["high", "low"]].copy()
    hl.loc[~ok] = np.nan
    weekly = hl.groupby(week).agg({"high": "max", "low": "min"}).shift(1)  # each week's row: the week before
    t["week_high"], t["week_low"] = week.map(weekly["high"]), week.map(weekly["low"])
    pre = pre_market(minutes, sessions)
    enough = pre["pre_bars"] >= PRE_MIN_BARS
    t["pre_high"], t["pre_low"] = pre["pre_high"].where(enough), pre["pre_low"].where(enough)
    t["pre_bars"] = pre["pre_bars"].fillna(0).astype(int)
    prof = ar.previous_session_levels(rth, daily, sessions, (PROFILE_BINS,))
    for k in ("poc", "vah", "val"):
        t[k] = prof[f"{k}_{PROFILE_BINS}"]
    o = t["open"].to_numpy(float)
    for name, step in ROUND.items():
        t[f"{name}_below"] = (np.ceil(o / step) - 1) * step
        t[f"{name}_above"] = (np.floor(o / step) + 1) * step
    return t


def real_levels(table, min_dist=MIN_DIST):
    """Long form: one row per session and level at least min_dist ATR from the open, with its side (+1 support,
    below the open; -1 resistance, above it), signed distance from the open in ATR, and period."""
    frames = []
    for name in LEVELS:
        for col in ([f"{name}_below", f"{name}_above"] if name in ROUND else [name]):
            frames.append(pd.DataFrame({"session": table.index, "level": name, "price": table[col].to_numpy(float),
                                        "dist": ((table[col] - table["open"]) / table["atr"]).to_numpy(float)}))
    lv = pd.concat(frames, ignore_index=True)
    lv = lv[np.isfinite(lv["dist"]) & (lv["dist"].abs() >= min_dist)].reset_index(drop=True)
    lv["side"] = np.where(lv["dist"] < 0, 1, -1)
    lv["period"] = np.where(lv["session"] >= HOLDOUT_START, "holdout", "design")
    lv["fake"], lv["control"] = False, "real"
    return lv


def fake_levels(real, table, k=FAKES, seed=SEED):
    """k fake levels per real one: the same session and side, at a distance from the open (in ATR) drawn at random
    from the same level's distances on other sessions with the same side and period."""
    rng = np.random.default_rng(seed)
    frames = []
    for (name, side, period), g in real.groupby(["level", "side", "period"], sort=True):
        n = len(g)
        if n < 2:
            continue
        idx = rng.integers(0, n - 1, size=(n, k))
        idx += idx >= np.arange(n)[:, None]  # skip the row itself: each session appears once per group
        dist = g["dist"].to_numpy()[idx].ravel()
        days = np.repeat(g["session"].to_numpy(), k)
        o, atr = table["open"].loc[days].to_numpy(float), table["atr"].loc[days].to_numpy(float)
        frames.append(pd.DataFrame({"session": days, "level": name, "price": o + dist * atr, "dist": dist,
                                    "side": side, "period": period, "fake": True, "control": "random"}))
    return pd.concat(frames, ignore_index=True) if frames else real.iloc[:0]


def near_levels(real, table, offsets=NEAR):
    """Fake levels next to each real one on the same session, `offsets` ATR above and below it; kept only on the
    same side of the open and at least MIN_DIST from it."""
    f = pd.concat([real.assign(dist=real["dist"] + sgn * o) for o in offsets for sgn in (1, -1)], ignore_index=True)
    f = f[((f["dist"] < 0) == (f["side"] > 0)) & (f["dist"].abs() >= MIN_DIST)].reset_index(drop=True)
    o, atr = table["open"].loc[f["session"]].to_numpy(float), table["atr"].loc[f["session"]].to_numpy(float)
    return f.assign(price=o + f["dist"].to_numpy() * atr, fake=True, control="near")


# ---------------------------------------------------------------- touches and races

def races(high, low, n_touch, price, side, widths, start=None):
    """First touch and race outcomes on one session's minute bars (in time order) for levels `price` with `side`
    (+1 support, -1 resistance); only the first n_touch bars may hold the touch, and none before the level's
    `start` index (when given). Returns (index of the touching bar or -1, [outcomes per width]): 1 held, -1 broke,
    0 neither by the close, NaN untouched or both in one later minute. A break inside the touching bar counts; a
    hold there does not."""
    n = len(high)
    sup = side[:, None] > 0
    reach = np.where(sup, low[None, :] <= price[:, None], high[None, :] >= price[:, None])
    reach[:, n_touch:] = False
    if start is not None:
        reach &= np.arange(n)[None, :] >= np.asarray(start)[:, None]
    touched = reach.any(axis=1)
    first = np.where(touched, reach.argmax(axis=1), -1)
    j = np.arange(n)[None, :]
    out = []
    for w in widths:
        through = np.where(sup, low[None, :] <= (price - w)[:, None], high[None, :] >= (price + w)[:, None])
        away = np.where(sup, high[None, :] >= (price + w)[:, None], low[None, :] <= (price - w)[:, None])
        through &= j >= first[:, None]
        away &= j > first[:, None]
        ft = np.where(through.any(axis=1), through.argmax(axis=1), n)
        fa = np.where(away.any(axis=1), away.argmax(axis=1), n)
        res = np.where(fa < ft, 1.0, np.where(ft < fa, -1.0, np.where(ft == n, 0.0, np.nan)))
        res[~touched] = np.nan
        out.append(res)
    return first, out


def measure(rth, levels, table, sizes=SIZES):
    """Adds each level's first touch (minutes after the open, by the touching bar's start) and race outcomes. With a
    `start` column (UTC times), a level can be touched only from its first minute at or after that time."""
    n = len(levels)
    touched, minute = np.zeros(n, bool), np.full(n, np.nan)
    res = {race_col(x): np.full(n, np.nan) for x in sizes}
    hi, lo = rth["high"].to_numpy(float), rth["low"].to_numpy(float)
    t, op, cl = (_ns(rth[k]) for k in ("ts", "session_open", "session_close"))
    bars_by_day = rth.groupby("session").indices
    price, side = levels["price"].to_numpy(float), levels["side"].to_numpy()
    start = _ns(levels["start"]) if "start" in levels else None
    atr = table["atr"]
    for day, rows in levels.groupby("session").indices.items():
        bars = bars_by_day.get(day)
        a = atr.get(day, np.nan)
        if bars is None or not np.isfinite(a):
            continue
        n_touch = int((t[bars] < cl[bars[0]] - TOUCH_END.value).sum())
        first, outs = races(hi[bars], lo[bars], n_touch, price[rows], side[rows], [x * a for x in sizes],
                            None if start is None else np.searchsorted(t[bars], start[rows]))
        hit = first >= 0
        touched[rows] = hit
        minute[rows[hit]] = (t[bars][first[hit]] - op[bars[0]]) / 60e9
        for x, o in zip(sizes, outs):
            res[race_col(x)][rows] = o
    return levels.assign(touched=touched, touch_minute=minute, **res)


def ticker_events(minutes, sessions, final):
    """(level table, real and fake levels with touches and outcomes, availability per level) for one ticker.
    Without `final`, holdout sessions are not measured at all."""
    rth, _ = features.regular_session_minutes(minutes, sessions)
    table = level_table(rth, minutes, sessions)
    every = real_levels(table, min_dist=0.0)
    avail = every.groupby("level").agg(level_days=("dist", "size"),
                                       too_close=("dist", lambda d: int((d.abs() < MIN_DIST).sum())))
    lv = every[every["dist"].abs() >= MIN_DIST]
    if not final:
        lv = lv[lv["period"] == "design"]
    lv = pd.concat([lv, fake_levels(lv, table), near_levels(lv, table)], ignore_index=True)
    return table, measure(rth, lv, table), avail


# ---------------------------------------------------------------- statistics

def boot_weights(n, reps=REPS, seed=SEED):
    """(reps, n): how many times each of n sessions is drawn in each bootstrap resample."""
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    flat = (np.arange(reps)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=reps * n).reshape(reps, n).astype(float)


def held_counts(ev, col, days):
    """Per session in `days`: (held, resolved) counts of one race column."""
    r = ev[col].to_numpy(float)
    pos = days.get_indexer(ev["session"])
    held = np.bincount(pos, weights=(r == 1).astype(float), minlength=len(days))
    resolved = np.bincount(pos, weights=np.isin(r, (1.0, -1.0)).astype(float), minlength=len(days))
    return held, resolved


def compare(real, fake, col, days, w):
    """Held % at real and at fake levels (touched ones), the difference in points, its 95% interval from the
    session bootstrap `w` (boot_weights over `days`), and p-values (two-sided, and for real > fake)."""
    rh, rn = held_counts(real, col, days)
    fh, fn = held_counts(fake, col, days)
    out = {"n": int(rn.sum()), "fake_n": int(fn.sum())}
    if rn.sum() < MIN_EVENTS or fn.sum() == 0:
        return out
    real_pct, fake_pct = 100 * rh.sum() / rn.sum(), 100 * fh.sum() / fn.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = 100 * (w @ rh) / (w @ rn) - 100 * (w @ fh) / (w @ fn)
    se = np.nanstd(boot, ddof=1)
    diff = real_pct - fake_pct
    z = diff / se if se > 0 else np.nan
    lo, hi = alerts.wilson(int(rh.sum()), int(rn.sum()))
    return {**out, "real": real_pct, "real_lo": lo, "real_hi": hi, "fake": fake_pct, "diff": diff,
            "diff_lo": diff - 1.96 * se, "diff_hi": diff + 1.96 * se,
            "p": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan, "p_up": norm.sf(z) if np.isfinite(z) else np.nan}


def level_stats(g, days, w, control="random"):
    real, fake = g[~g["fake"]], g[g["control"] == control]
    out = {"level_days": len(real), "touched": real["touched"].mean() * 100,
           "fake_touched": fake["touched"].mean() * 100}
    for x, k in zip(SIZES, KEYS):
        c = race_col(x)
        rt = real[real["touched"]]
        out.update({f"{name}_{k}": v for name, v in compare(rt, fake[fake["touched"]], c, days, w).items()})
        out[f"neither_{k}"] = (rt[c] == 0).mean() * 100 if len(rt) else np.nan
    return out


def statistics(events, sessions, periods, reps=REPS, seed=SEED):
    """Per ticker x level x period, and pooled over tickers (also by side): touches, held % at real and fake
    levels for each race size, and the difference with its bootstrap interval. Every comparison in a period
    resamples the same sessions. Also the near-fake check, design period only (ticker "all" = pooled)."""
    period_of = np.where(sessions.index >= HOLDOUT_START, "holdout", "design")
    rows, pooled, near = [], [], []
    for period in periods:
        days = sessions.index[period_of == period]
        w = boot_weights(len(days), reps, seed)
        ev = events[events["period"] == period]
        for (tk, name), g in ev.groupby(["ticker", "level"], sort=False):
            rows.append({"ticker": tk, "level": name, "period": period, **level_stats(g, days, w)})
        for name, g in ev.groupby("level", sort=False):
            pooled.append({"level": name, "period": period, "side": "both", **level_stats(g, days, w)})
            for side, gs in g.groupby("side"):
                pooled.append({"level": name, "period": period, "side": "support" if side > 0 else "resistance",
                               **level_stats(gs, days, w)})
        if period == "design":
            for (tk, name), g in ev.groupby(["ticker", "level"], sort=False):
                near.append({"ticker": tk, "level": name, **level_stats(g, days, w, "near")})
            for name, g in ev.groupby("level", sort=False):
                near.append({"ticker": "all", "level": name, **level_stats(g, days, w, "near")})
    return pd.DataFrame(rows), pd.DataFrame(pooled), pd.DataFrame(near)


def run_study(minutes_by_ticker, sessions, *, final=False, timings=None, reps=REPS):
    """minutes_by_ticker: {ticker: minute bars including pre-market}. No file I/O."""
    timings = {} if timings is None else timings
    events, tables, avail = [], {}, {}
    with timed(timings, "levels and touches"):
        for tk, minutes in minutes_by_ticker.items():
            tables[tk], ev, avail[tk] = ticker_events(minutes, sessions, final)
            events.append(ev.assign(ticker=tk))
        events = pd.concat(events, ignore_index=True)
    with timed(timings, "statistics"):
        periods = ["design", "holdout"] if final else ["design"]
        stats, pooled, near = statistics(events, sessions, periods, reps)
        p1, k1, k2 = f"p_{KEYS[0]}", f"diff_{KEYS[0]}", f"diff_{KEYS[1]}"
        design = stats[stats["period"] == "design"].copy()
        design["q"] = mr.bh_qvalues(design[p1].fillna(1).to_numpy())
        design["qualifies"] = (design["q"] < FDR) & (np.sign(design[k1]) == np.sign(design[k2]))
        held = None
        if final:
            keys = ["ticker", "level"]
            q = design.loc[design["qualifies"], keys + [k1]].rename(columns={k1: "design_diff"})
            held = stats[stats["period"] == "holdout"].merge(q, on=keys)
            if len(held):
                up = held[f"p_up_{KEYS[0]}"]
                held["holdout_q"] = mr.bh_qvalues(np.where(held["design_diff"] > 0, up, 1 - up))
                held["holds_up"] = held["holdout_q"] < FDR
    atr = {tk: float(t.loc[t.index < HOLDOUT_START, "atr"].median()) for tk, t in tables.items()}
    return {"events": events, "stats": stats, "design": design, "pooled": pooled, "near": near, "holdout": held,
            "final": final,
            "tickers": list(minutes_by_ticker), "availability": avail, "median_atr": atr,
            "first": sessions.index[0], "last": sessions.index[-1]}


# ---------------------------------------------------------------- report

def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def pts(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1f}"


def find(table, **match):
    m = table
    for k, v in match.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


def render_report(res):
    d, pooled = res["design"], res["pooled"]
    k1, k2 = KEYS
    x1, x2 = SIZES
    lines = []
    w = lines.append
    w("# Support and resistance: do key levels hold?\n")
    w("For each level known before the 9:30 ET open, the first time price reaches it during regular hours, then a "
      f"race from the level: it **held** if price moved back away from it by X before going through it by X (X = "
      f"{x1:g} or {x2:g} of the daily ATR). The same is measured at **fake levels**: {FAKES} per real level and "
      "session, on the same session and side of the open, at distances drawn from the same level's distances on "
      "other days. *Difference* = held % at real levels minus held % at fake ones, in percentage points, with a "
      "95% interval from resampling sessions. Design period to 2024-12-31"
      + (", holdout 2025-01-01 on." if res["final"] else " (holdout not read).") + "\n")
    q = d[d["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(d)} design tests qualified (q < {FDR:g} at +/-{x1:g} ATR and the same "
      f"sign at +/-{x2:g} ATR)" + (": " + "; ".join(f"{r.ticker} {LEVELS[r.level]} ({pts(r[f'diff_{k1}'])} pts)"
                                                     for _, r in q.iterrows()) if len(q) else ".") + "\n")
    w("A negative difference means the real level broke *more* often than fake ones: price tended to run through "
      "it rather than bounce. Fake levels hold a little under half the time because a break within the touching "
      "minute counts and a hold there does not; real levels carry the same handicap, so only the difference "
      "matters.\n")
    for tk in res["tickers"]:
        w(f"## {tk}, design period\n")
        rows = []
        for name in LEVELS:
            r = find(d, ticker=tk, level=name)
            if r is None:
                continue
            rows.append([LEVELS[name], f"{int(r['level_days']):,}", f"{pct(r['touched'])} / {pct(r['fake_touched'])}",
                         f"{pct(r.get(f'real_{k1}'))} ({pct(r.get(f'real_lo_{k1}'))}-{pct(r.get(f'real_hi_{k1}'))}), "
                         f"n={int(r[f'n_{k1}']):,}", pct(r.get(f"fake_{k1}")),
                         f"{pts(r.get(f'diff_{k1}'))} ({pts(r.get(f'diff_lo_{k1}'))} to {pts(r.get(f'diff_hi_{k1}'))})",
                         f"{r['q']:.2f}",
                         f"{pct(r.get(f'real_{k2}'))} / {pct(r.get(f'fake_{k2}'))} ({pts(r.get(f'diff_{k2}'))})"])
        w(alerts.md_table(["Level", "Level-days", "Touched: real / fake", f"Held +/-{x1:g} ATR, real (95%)", "Fake",
                           "Difference, pts (95%)", "q", f"Held +/-{x2:g} ATR: real / fake (diff)"], rows) + "\n")
    w("## All tickers together, by side (design period)\n")
    w(f"At +/-{x1:g} ATR; support = the level was below the open, resistance = above it. Not part of the test "
      "count; for reading only.\n")
    rows = []
    pd_ = pooled[pooled["period"] == "design"]
    for name in LEVELS:
        cells = []
        for side in ("both", "support", "resistance"):
            r = find(pd_, level=name, side=side)
            cells.append("n/a" if r is None or pd.isna(r.get(f"real_{k1}")) else
                         f"{pct(r[f'real_{k1}'])} vs {pct(r[f'fake_{k1}'])} ({pts(r[f'diff_{k1}'])}; "
                         f"{pts(r[f'diff_lo_{k1}'])} to {pts(r[f'diff_hi_{k1}'])}), n={int(r[f'n_{k1}']):,}")
        rows.append([LEVELS[name], *cells])
    w(alerts.md_table(["Level", "Held: real vs fake (diff, 95%)", "As support", "As resistance"], rows) + "\n")
    w("## Check added after the first results: fake levels next to the real one\n")
    w(f"The random fakes match the real levels' distances only on average, so a level whose distance tracks the day "
      "(yesterday's high sits close to the open on gap-up days) is touched on a different mix of days than its "
      f"fakes. Here the fakes sit {', '.join(f'{o:g}' for o in NEAR)} ATR above and below each real level on the "
      "same session (same side of the open), so the days match exactly. A strong real effect can leak into "
      f"neighbours beyond the level, so these differences lean small. Held +/-{x1:g} ATR, real minus near, points "
      "(95%).\n")
    nr = res["near"]
    rows = []
    for name in LEVELS:
        cells = []
        for tk in [*res["tickers"], "all"]:
            r = find(nr, ticker=tk, level=name)
            cells.append("n/a" if r is None or pd.isna(r.get(f"diff_{k1}")) else
                         f"{pts(r[f'diff_{k1}'])} ({pts(r[f'diff_lo_{k1}'])} to {pts(r[f'diff_hi_{k1}'])})")
        r = find(nr, ticker="all", level=name)
        cells.append("n/a" if r is None or pd.isna(r.get(f"real_{k1}")) else
                     f"{pct(r[f'real_{k1}'])} vs {pct(r[f'fake_{k1}'])}")
        rows.append([LEVELS[name], *cells])
    w(alerts.md_table(["Level", *res["tickers"], "All together", "All: real vs near held"], rows) + "\n")
    if res["final"]:
        w("## Holdout: did the qualifiers hold up?\n")
        h = res["holdout"]
        if h is None or h.empty:
            w("No qualifiers to test.\n")
        else:
            rows = [[r.ticker, LEVELS[r.level], pts(r.design_diff),
                     f"{pts(getattr(r, f'diff_{k1}'))} ({pts(getattr(r, f'diff_lo_{k1}'))} to "
                     f"{pts(getattr(r, f'diff_hi_{k1}'))}), n={int(getattr(r, f'n_{k1}')):,}",
                     f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"] for r in h.itertuples()]
            w(alerts.md_table(["Ticker", "Level", "Design diff", "Holdout diff (95%)", "Holdout q", "Holds up"],
                              rows) + "\n")
    w("## Notes\n")
    sizes = ", ".join(f"{tk} ${a:.2f} (+/-{x1:g} ATR = ${x1 * a:.2f}, +/-{x2:g} = ${x2 * a:.2f})"
                      for tk, a in res["median_atr"].items())
    w(f"- Median daily ATR(14) in the design period: {sizes}.")
    for tk, a in res["availability"].items():
        miss = ", ".join(f"{LEVELS[n]} {int(r.too_close):,}" for n, r in a.iterrows() if r.too_close)
        w(f"- {tk}: level-days per level {int(a['level_days'].min()):,}-{int(a['level_days'].max()):,} (whole "
          f"dollar and $5 count two a day); skipped within {MIN_DIST:g} ATR of the open: {miss or 'none'}.")
    w("- Touched = share of level-days on which price reached the level by 30 minutes before the close. Fake "
      "levels match the real ones' distances only on average, so touch rates are a rough guide, not a test.")
    w("- A fake level can land near a real one by chance; that would pull the two held rates together, so "
      "differences are, if anything, understated.")
    w("- Volume levels use a bar-approximated profile: each minute's volume spread evenly over its range, not "
      "traded volume at price.")
    w("- Charts: `levels.py --chart DATE` (or START:END) draws a session's levels with yesterday and the "
      "pre-market for reference, marking each level's first touch and what happened.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    d = res["design"]
    k1 = KEYS[0]
    names = list(LEVELS)
    fig = plt.figure(figsize=(8, 5.2), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.3, 0.15, 0.66, 0.69])
    y = np.arange(len(names))[::-1].astype(float)
    n_tk = len(res["tickers"])
    for j, (tk, color) in enumerate(zip(res["tickers"], COLORS)):
        rows = [find(d, ticker=tk, level=name) for name in names]
        get = lambda c: np.array([np.nan if r is None else r.get(c, np.nan) for r in rows], float)  # noqa: E731
        m, lo, hi = get(f"diff_{k1}"), get(f"diff_lo_{k1}"), get(f"diff_hi_{k1}")
        ax.errorbar(m, y + ((n_tk - 1) / 2 - j) * 0.22, xerr=[m - lo, hi - m], fmt="o", ms=4, color=color,
                    ecolor=color, elinewidth=1.2, capsize=2, label=tk)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y, [LEVELS[n] for n in names], fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Held % at real levels minus at fake levels, points (95%)", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, "Do key levels hold more often than fake ones?", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"First touch each session, race of +/-{SIZES[0]:g} daily ATR, design period 2021-2024. "
             "Right of zero = held more often than a fake level.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=n_tk, frameon=False, fontsize=7.5,
               labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- charts

STYLE = {  # level -> (colour, dashes, width)
    "prev_high": (INK, "-", 1.0), "prev_low": (INK, "-", 1.0), "prev_close": (INK, (0, (5, 2)), 0.9),
    "week_high": (INK_2, (0, (6, 2, 1, 2)), 1.0), "week_low": (INK_2, (0, (6, 2, 1, 2)), 1.0),
    "pre_high": (SERIES[0], (0, (1, 1.5)), 1.4), "pre_low": (SERIES[0], (0, (1, 1.5)), 1.4),
    "poc": (SERIES[1], (0, (3, 2)), 1.2), "vah": (SERIES[1], (0, (3, 2)), 0.8), "val": (SERIES[1], (0, (3, 2)), 0.8),
    "dollar": (MUTED, "-", 0.5), "five": (MUTED, "-", 1.2),
}
LEGEND = (("yesterday", "prev_high"), ("last week", "week_high"), ("pre-market", "pre_high"),
          ("volume profile", "poc"), ("whole $ / $5", "five"))
OUTCOME = {1.0: "held", -1.0: "broke", 0.0: "neither by the close"}
GAP = 3     # chart slots between segments
PRE_SLOT = pd.Timedelta(minutes=15)
BAR = pd.Timedelta(minutes=5)


def chart_days(specs, sessions):
    """Sessions named by --chart: dates (2024-08-05) or inclusive ranges (2024-08-01:2024-08-09)."""
    days = []
    for s in specs:
        a, _, b = s.partition(":")
        lo, hi = pd.Timestamp(a), pd.Timestamp(b or a)
        days += list(sessions.index[(sessions.index >= lo) & (sessions.index <= hi)])
    return sorted(set(days))


def chart_data(minutes, rth, sessions, day):
    """5-minute bars of the previous session and of `day`, and 15-minute pre-market bars of `day`."""
    i = sessions.index.get_loc(day)
    prev = sessions.index[i - 1] if i > 0 else None

    def five(d):
        g = rth[rth["session"] == d]
        return features.resample_bars(g, 5, min_coverage=0.0)[0] if len(g) else None

    start = (day + PRE_START).tz_localize(NY).tz_convert("UTC")
    m = minutes[(minutes["ts"] >= start) & (minutes["ts"] < sessions.loc[day, "open"])]
    slot = ((m["ts"] - start) // PRE_SLOT).to_numpy()
    pre = m.groupby(slot).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                              close=("close", "last"))
    return prev, (five(prev) if prev is not None else None), pre, five(day)


def tex(text):
    """Escape $ so matplotlib does not read a pair of them as math."""
    return text.replace("$", r"\$")


def spread(prices, gap):
    """Label heights for ascending `prices`, at least `gap` apart: crowded labels are spaced evenly around the
    mean of their levels."""
    blocks = []  # [sum of prices, count]
    for p in prices:
        blocks.append([p, 1])
        while len(blocks) > 1:
            (s1, n1), (s2, n2) = blocks[-2], blocks[-1]
            if (s2 / n2 - (n2 - 1) * gap / 2) - (s1 / n1 + (n1 - 1) * gap / 2) >= gap:
                break
            blocks[-2:] = [[s1 + s2, n1 + n2]]
    return [s / n + (k - (n - 1) / 2) * gap for s, n in blocks for k in range(n)]


def _candles(ax, x, bars, color):
    for xi, b in zip(x, bars.itertuples()):
        ax.plot([xi, xi], [b.low, b.high], color=color, lw=0.6, zorder=3)
        ax.add_patch(plt.Rectangle((xi - 0.35, min(b.open, b.close)), 0.7, max(abs(b.close - b.open), 1e-9),
                                   facecolor=SURFACE if b.close >= b.open else color, edgecolor=color, lw=0.6,
                                   zorder=4))


def plot_session(ticker, day, sessions, data, levels, atr, path):
    """One session's 5-minute candles with every level, after yesterday's session and today's pre-market (grey),
    and each level's first touch today with its race outcome at the primary size."""
    prev, prev_bars, pre, today = data
    fig = plt.figure(figsize=(10, 5.9), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.06, 0.2, 0.68, 0.66])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    segments, ticks, labels, low, high = [], [], [], [], []
    off = 0.0

    def add(bars, x, color, name, start, length, tick_every, step):
        nonlocal off
        if bars is not None and len(bars):
            _candles(ax, off + x, bars, color)
            low.append(bars["low"].min())
            high.append(bars["high"].max())
        segments.append((off, off + length, name))
        for t in range(0, int(length) + 1, tick_every):
            ticks.append(off + t)
            labels.append((start + t * step).tz_convert(NY).strftime("%H:%M"))
        off += length + GAP

    if prev is not None:
        o = sessions.loc[prev, "open"]
        x = ((prev_bars["bar_start"] - o) / BAR).to_numpy() if prev_bars is not None else None
        add(prev_bars, x, MUTED, f"yesterday {prev:%a %m-%d}", o, (sessions.loc[prev, "close"] - o) / BAR, 18, BAR)
    pre_start = (day + PRE_START).tz_localize(NY).tz_convert("UTC")
    add(pre, pre.index.to_numpy(float), MUTED, "pre-market", pre_start, (sessions.loc[day, "open"] - pre_start) / PRE_SLOT,
        8, PRE_SLOT)
    o = sessions.loc[day, "open"]
    today_off = off
    add(today, ((today["bar_start"] - o) / BAR).to_numpy() if today is not None else None, INK_2, f"{day:%a %m-%d}",
        o, (sessions.loc[day, "close"] - o) / BAR, 12, BAR)
    right = off - GAP
    lo_y, hi_y = min(low), max(high)
    pad = (hi_y - lo_y) * 0.06
    lo_y, hi_y = lo_y - pad, hi_y + pad
    ax.set_xlim(-2, right + 2)
    ax.set_ylim(lo_y, hi_y)
    for a, b, name in segments:
        ax.text((a + b) / 2, hi_y, name, color=INK_2, fontsize=7.5, ha="center", va="bottom")
        if a > 0:
            ax.axvline(a - GAP / 2, color=BASELINE, lw=0.8, zorder=1)
    ax.set_xticks(ticks, labels, fontsize=6.5)

    width = SIZES[0] * atr
    col = race_col(SIZES[0])
    shown = levels[(levels["price"] >= lo_y) & (levels["price"] <= hi_y)].sort_values("price", kind="stable")
    off_chart = [f"{LEVELS[r.level]} {r.price:.2f}" for r in levels.sort_values("price").itertuples()
                 if not lo_y <= r.price <= hi_y]
    for r, y in zip(shown.itertuples(), spread(shown["price"].to_numpy(float), (hi_y - lo_y) * 0.034)):
        name = LEVELS[r.level]
        color, dashes, lw = STYLE[r.level]
        ax.plot([-2, right + 2], [r.price, r.price], color=color, ls=dashes, lw=lw, zorder=2)
        if abs(r.dist) < MIN_DIST:
            what = "at the open, not tested"
        elif not r.touched:
            what = "not touched"
        else:
            out = getattr(r, col)
            what = OUTCOME.get(out, "unclear")
            mx = today_off + r.touch_minute // 5
            if out == 1.0:
                ax.plot(mx, r.price, "^" if r.side > 0 else "v", ms=7, color=TARGET, mec=SURFACE, mew=0.8, zorder=6)
            elif out == -1.0:
                ax.plot(mx, r.price, "X", ms=7, color=SERIES[1], mec=SURFACE, mew=0.8, zorder=6)
            else:
                ax.plot(mx, r.price, "o", ms=6, mfc=SURFACE, color=MUTED, mew=1.2, zorder=6)
        ax.annotate(tex(f"{name} {r.price:.2f} · {what}"), (right + 2, r.price), xytext=(right + 5, y),
                    textcoords="data",
                    color=color if color != MUTED else INK_2, fontsize=6.5, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.6) if abs(y - r.price) > 1e-9 else None)

    fig.text(0.02, 0.975, f"{ticker} {day:%A %Y-%m-%d}: key levels", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.935, tex(f"Every line is known before the 9:30 ET open. Markers: each level's first touch "
             f"today, and whether price then moved +/-{SIZES[0]:g} ATR (${width:.2f}) back away (held) or through it "
             "(broke) first."), color=INK_2, fontsize=7.5, va="top")
    handles = [Line2D([], [], color=STYLE[k][0], ls=STYLE[k][1], lw=max(STYLE[k][2], 1.0), label=tex(label))
               for label, k in LEGEND]
    handles += [Line2D([], [], ls="", marker="^", color=TARGET, label="held"),
                Line2D([], [], ls="", marker="X", color=SERIES[1], label="broke"),
                Line2D([], [], ls="", marker="o", mfc=SURFACE, color=MUTED, label="neither / unclear")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.06), ncol=8, frameon=False, fontsize=7,
               labelcolor=INK_2, handlelength=2.6)
    note = "Grey candles: yesterday's session and today's pre-market (15-minute bars), for reference."
    if off_chart:
        note += " Off the chart: " + ", ".join(off_chart) + "."
    fig.text(0.02, 0.02, tex(note), color=MUTED, fontsize=7, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_charts(ticker, minutes, sessions, days, out_dir):
    """A chart and the level list for each session in `days`; returns the chart paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rth, _ = features.regular_session_minutes(minutes, sessions)
    table = level_table(rth, minutes, sessions)
    lv = real_levels(table, min_dist=0.0)
    ev = measure(rth, lv[lv["session"].isin(days)].reset_index(drop=True), table)
    ev.loc[ev["dist"].abs() < MIN_DIST, [race_col(x) for x in SIZES]] = np.nan
    paths = []
    for day in days:
        if not np.isfinite(table["atr"].get(day, np.nan)) or day not in set(rth["session"]):
            print(f"skipped {day:%Y-%m-%d}: no data or no ATR yet")
            continue
        path = out / f"{ticker}_{day:%Y-%m-%d}.png"
        plot_session(ticker, day, sessions, chart_data(minutes, rth, sessions, day), ev[ev["session"] == day],
                     table.loc[day, "atr"], path)
        paths.append(path)
    ev.assign(label=ev["level"].map(LEVELS)).to_csv(out / f"{ticker}_levels.csv", index=False)
    return paths


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Support and resistance levels: do they hold more often than chance?")
    p.add_argument("--out", default="output/levels")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read the 2025+ holdout (run once)")
    p.add_argument("--chart", nargs="+", metavar="DATE",
                   help="only draw the levels on these sessions (YYYY-MM-DD or START:END), for --chart-ticker")
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
        days = chart_days(args.chart, sessions)
        minutes, _ = gr.load_minutes(args.chart_ticker, sessions, args.cache_dir, args.refresh)
        paths = write_charts(args.chart_ticker, minutes, sessions, days, out / "charts")
        print(f"wrote {len(paths)} chart(s) to {out / 'charts'}")
        return
    with timed(timings, "data"):
        data = {tk: gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0] for tk in args.tickers}
    res = run_study(data, sessions, final=args.final_test, timings=timings)
    out.mkdir(parents=True, exist_ok=True)
    res["events"].to_parquet(out / "events.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["design"].to_csv(out / "design.csv", index=False)
    res["pooled"].to_csv(out / "pooled.csv", index=False)
    res["near"].to_csv(out / "near.csv", index=False)
    if res["final"] and res["holdout"] is not None:
        res["holdout"].to_csv(out / "holdout.csv", index=False)
    plot(res, out / "levels.png")
    (out / "report.md").write_text(render_report(res))
    q = res["design"][res["design"]["qualifies"]]
    print(f"design tests: {len(res['design'])}, qualified: {len(q)}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
