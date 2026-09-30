"""Auction reclaim: a failed auction outside prior value, a reclaim, then a confirmed retest.

This is our own deterministic research approximation, inspired by the failed-auction
reversion model in Fabio Valentini's published Auction Market playbook
(https://www.chartfanatics.com/strategies/auction-market-strategy). It is not his method.
Every number here is a research default, not a verified rule of his, and nothing here
reproduces or implies his performance.

What minute OHLCV cannot show: order flow. Relative volume measures total-volume intensity
and the candle filter is a shape rule. Neither measures aggressive buyers or sellers,
absorption or cumulative delta. The previous-session profile is a
bar_approximated_volume_profile (see volume_profile.py), not traded volume at price.

Long rules, per ticker and session (the short side mirrors them):
- Frozen at the open: VAL / POC / VAH from the previous XNYS session's complete minutes.
  A = daily ATR(14) through the previous session, b = 0.02 A and d = 0.03 A.
- Balance: at least 3 of the 4 complete, consecutive same-session 5-minute bars just
  before the excursion close inside [VAL, VAH].
- Excursion: a close < VAL - b when the previous close was not (a fresh break). The
  lowest low is tracked from this bar through the reclaim bar.
- Reclaim: within the next 6 bars, the first close in (VAL + b, POC) with close > open.
  If any high reaches POC first, the setup expires. The reclaim bar itself never signals.
- Retest: within the next 3 bars, the first bar that overlaps [VAL - d, VAL + d] (or
  the frozen reclaim LVN), closes in (VAL + b, POC) and passes the confirmation, VWAP
  and reward/risk filters. Invalidation is checked first, on every bar after the
  reclaim including the trigger bar: a low below the tracked low, a high at or above
  POC, or a close < VAL - b.
- Signal bars complete between 09:50 and 11:30 ET inclusive. Each session allows one
  candidate per side and one active setup at a time; a missing or incomplete bar resets.

The "loose" rule set (RULES) keeps every rule above except five: signals may complete
until 15:30 ET, the retest window is 6 bars, the candle needs a body >= 30% of its range
with the close in the top (long) or bottom (short) 40%, and reward/risk >= 1.0. It was
fixed after seeing only 2024 signal counts, never any outcomes.
"""

import numpy as np
import pandas as pd

import volume_profile as vp
from features import MINUTE

BAR_MINUTES = 5
ATR_PERIOD = 14
BUFFER_ATR, RETEST_ATR, STOP_ATR, VWAP_SLOPE_ATR = 0.02, 0.03, 0.01, 0.10
BALANCE_BARS, BALANCE_MIN_INSIDE = 4, 3
RECLAIM_BARS = 6
SIGNAL_FIRST = pd.Timedelta(minutes=20)  # earliest signal-bar completion: 09:50 ET
RVOL_MIN, RVOL_SESSIONS, RVOL_MIN_OBS = 1.20, 20, 10
VWAP_LOOKBACK = pd.Timedelta(minutes=15)
LVN_BINS, LVN_MIN_MINUTES = 16, 10
LOCATIONS = ("value_edge", "reclaim_lvn")
# Named rule sets. signal_last = latest signal-bar completion after the open (120 min = 11:30 ET, 360 = 15:30 ET).
RULES = {
    "baseline": {"signal_last": pd.Timedelta(minutes=120), "retest_bars": 3, "body_min": 0.50,
                 "close_location_min": 0.75, "min_reward_risk": 1.25},
    "loose": {"signal_last": pd.Timedelta(minutes=360), "retest_bars": 6, "body_min": 0.30,
              "close_location_min": 0.60, "min_reward_risk": 1.00},
}
BASELINE = {"rules": "baseline", "location": "value_edge", "profile_bins": 48, "rvol_filter": True}
SIDES = {"long": 1, "short": -1}
PATTERNS = {"long": (), "short": ()}  # one summary row per side
VENDOR_VWAP, APPROX_VWAP = "massive_minute_vwap", "hlc3_approximation"

SESSION_COUNTS = ("sessions", "eligible_sessions", "no_bars", "profile_unavailable", "atr_unavailable")
SIDE_COUNTS = ("excursions", "reclaim_poc_first", "reclaim_expired", "reclaims", "lvn_no_node", "invalid_extreme",
               "invalid_poc", "invalid_close", "retest_expired", "missing_bar_reset", "window_closed")
UTC = "datetime64[ns, UTC]"
TRIGGER_COLUMNS = {
    "side": "str", "side_sign": int, "profile_method": "str", "prev_session": "datetime64[ns]",
    "val": float, "poc": float, "vah": float, f"prev_atr_{ATR_PERIOD}": float, "buffer_b": float, "retest_tol_d": float,
    "excursion_slot": int, "excursion_end": UTC, "excursion_close": float, "extreme_slot": int,
    "excursion_extreme": float, "reclaim_slot": int, "reclaim_end": UTC, "reclaim_close": float, "signal_slot": int,
    "location_lo": float, "location_hi": float, "signal_open": float, "signal_high": float, "signal_low": float,
    "body_frac": float, "close_location": float, "rvol": float, "rvol_base": float, "vwap": float,
    "vwap_15m_ago": float, "vwap_change_15m": float, "vwap_method": "str", "frozen_stop": float,
    "frozen_target": float, "signal_reward_risk": float,
}
IDLE, RECLAIM, RETEST = 0, 1, 2


def _ns(times):
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def config_label(rules, location, profile_bins, rvol_filter):
    """'baseline', or the settings that differ from it."""
    diffs = ([rules] if rules != BASELINE["rules"] else []) + (
        [location] if location != BASELINE["location"] else []) + (
        [f"bins={profile_bins}"] if profile_bins != BASELINE["profile_bins"] else []) + (
        [] if rvol_filter else ["rvol_filter off"])
    return " ".join(diffs) or "baseline"


def make_configs(location, profile_bins, rvol_filter, compare=False, rules="baseline"):
    """One configuration, or the five fixed diagnostic comparisons for one rule set (its own base first).

    The comparisons are the base and four variants that each change one setting: not a grid.
    """
    base = {**BASELINE, "rules": rules}
    variants = ([base, {**base, "location": "reclaim_lvn"}, {**base, "rvol_filter": False},
                 {**base, "profile_bins": 32}, {**base, "profile_bins": 64}] if compare else
                [{"rules": rules, "location": location, "profile_bins": profile_bins, "rvol_filter": rvol_filter}])
    return [{"label": config_label(**v), "params": dict(v)} for v in variants]


def session_vwap(minutes):
    """Cumulative session VWAP as of each minute's END, reset at every regular-session open.

    Two running sums are kept per session: Massive's minute `vwap` x volume, and HLC3 x
    volume. `vwap_vendor_ok` stays True while every positive-volume minute so far has a
    vendor VWAP. From the first one without it, only the HLC3 series is valid, and it
    covers the whole observed prefix. Each value depends only on its own prefix, so a
    later switch never revises an earlier value.
    """
    vol = minutes["volume"].to_numpy(float)
    vendor = minutes["vwap"].to_numpy(float) if "vwap" in minutes else np.full(len(minutes), np.nan)
    hlc3 = minutes[["high", "low", "close"]].to_numpy(float).sum(axis=1) / 3
    traded = np.isfinite(vol) & (vol > 0)
    row_ok = (np.isfinite(vol) & (vol >= 0)) & (~traded | np.isfinite(hlc3))
    has_vendor = ~traded | (np.isfinite(vendor) & (vendor > 0))
    v = np.where(traded, vol, 0.0)
    frame = pd.DataFrame({"v": v, "pv": np.where(traded & has_vendor, vendor * v, 0.0),
                          "ph": np.where(traded & row_ok, hlc3 * v, 0.0), "vendor_ok": has_vendor.astype(np.int8),
                          "row_ok": row_ok.astype(np.int8)}, index=minutes.index)
    g = frame.groupby(minutes["session"].to_numpy(), sort=False)
    cum = g[["v", "pv", "ph"]].cumsum()
    ok = g[["vendor_ok", "row_ok"]].cummin().astype(bool)
    valid = ok["row_ok"] & (cum["v"] > 0)
    out = pd.DataFrame(index=minutes.index)
    out["vwap_vendor_cum"] = (cum["pv"] / cum["v"]).where(valid & ok["vendor_ok"])
    out["vwap_hlc3_cum"] = (cum["ph"] / cum["v"]).where(valid)
    out["vwap_vendor_ok"] = ok["vendor_ok"]
    out["session_vwap"] = out["vwap_vendor_cum"].where(out["vwap_vendor_ok"], out["vwap_hlc3_cum"])
    return out


def bar_vwap(bars, minutes):
    """Session VWAP at each bar's completion T and exactly 15 minutes earlier, with its method.

    Each value is read from the minute that ends at that time. If that minute is missing or
    belongs to another session, the value is NaN. Both values use the method valid at T, so
    an approximated T is never compared with a vendor value.
    """
    t = pd.Index(_ns(minutes["ts"]))
    end = _ns(bars["bar_end"])
    minute_ns = MINUTE.value
    same = minutes["session"].to_numpy()
    sessions = bars["session"].to_numpy()

    def locate(times):
        pos = t.get_indexer(times)
        return np.where((pos >= 0) & (same[np.maximum(pos, 0)] == sessions), pos, -1)

    now, lag = locate(end - minute_ns), locate(end - minute_ns - VWAP_LOOKBACK.value)
    vendor_ok = minutes["vwap_vendor_ok"].to_numpy(bool)[np.maximum(now, 0)] & (now >= 0)
    vendor, hlc3 = minutes["vwap_vendor_cum"].to_numpy(float), minutes["vwap_hlc3_cum"].to_numpy(float)

    def pick(pos):
        return np.where(pos >= 0, np.where(vendor_ok, vendor[np.maximum(pos, 0)], hlc3[np.maximum(pos, 0)]), np.nan)

    method = np.where(now >= 0, np.where(vendor_ok, VENDOR_VWAP, APPROX_VWAP), None).astype(object)
    return pd.DataFrame({"vwap": pick(now), "vwap_15m_ago": pick(lag), "vwap_method": method}, index=bars.index)


def rvol_baseline(bars, sessions):
    """Median volume for the same session-relative 5-minute slot over the 20 preceding XNYS sessions.

    Today is always excluded. Only complete bars count as observations, at least 10 are
    required, and a non-positive median gives NaN.
    """
    if bars.empty:
        return np.array([], dtype=float)
    done = bars[bars["complete"]]
    table = done.pivot(index="session", columns="slot", values="volume").reindex(
        index=sessions.index, columns=range(int(bars["slot"].max()) + 1))
    base = table.rolling(RVOL_SESSIONS, min_periods=RVOL_MIN_OBS).median().shift(1).to_numpy()
    values = base[sessions.index.get_indexer(bars["session"]), bars["slot"].to_numpy()]
    return np.where(values > 0, values, np.nan)


def _valid_ohlcv(o, h, l, c, v):
    finite = np.isfinite(o).all() and np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()
    return bool(finite and np.isfinite(v).all() and (l > 0).all() and (v >= 0).all()
                and (l <= np.minimum(o, c)).all() and (h >= np.maximum(o, c)).all())


def previous_session_levels(minutes, daily, sessions, bin_counts):
    """One frozen profile per study session, built from the immediately preceding XNYS session.

    That previous session must have every expected regular-session minute, valid OHLCV,
    positive total volume and a nonzero range. Otherwise the levels are NaN and
    `profile_status` says why. An older session is never substituted.
    """
    day = pd.DataFrame(index=sessions.index)
    day["in_study"] = sessions["in_study"]
    day["prev_session"] = pd.Series(sessions.index, index=sessions.index).shift(1)
    day["prev_atr"] = daily[f"prev_atr_{ATR_PERIOD}"]
    status = np.full(len(day), "warm_up", dtype=object)
    levels = {f"{k}_{n}": np.full(len(day), np.nan) for n in bin_counts for k in ("val", "poc", "vah")}
    complete = (daily["n_minutes"] == daily["expected_minutes"]) & (daily["expected_minutes"] > 0)
    rows = minutes.groupby("session").indices
    o, h, l, c, v = (minutes[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    for i, prev in enumerate(day["prev_session"]):
        if not day["in_study"].iloc[i]:
            continue
        if pd.isna(prev):
            status[i] = "no_previous_session"
            continue
        if not complete.loc[prev] or prev not in rows:
            status[i] = "previous_session_incomplete"
            continue
        p = rows[prev]
        if not _valid_ohlcv(o[p], h[p], l[p], c[p], v[p]):
            status[i] = "previous_session_invalid"
            continue
        status[i] = "ok"
        for n in bin_counts:
            profile = vp.bar_profile(l[p], h[p], v[p], n)
            val, poc, vah = vp.value_area(profile[1], profile[0]) if profile else (np.nan,) * 3
            if not val < poc < vah:
                status[i] = "previous_session_invalid"
                break
            levels[f"val_{n}"][i], levels[f"poc_{n}"][i], levels[f"vah_{n}"][i] = val, poc, vah
        if status[i] != "ok":
            for values in levels.values():
                values[i] = np.nan
    day["profile_status"] = status
    return day.assign(**levels)


def add_features(bars, minutes, daily, sessions, bin_counts):
    """Inputs shared by every variant, computed once: (bars, minutes, day).

    `minutes` gains the per-minute session VWAP columns. `bars` gains `session_open`,
    `complete`, `slot`, VWAP at completion and 15 minutes earlier, relative volume, and the
    frozen levels for each bin count. `day` has one row per session, holding those levels,
    `profile_status` and the frozen ATR.
    """
    minutes = minutes.join(session_vwap(minutes))
    b = bars.join(sessions["open"].rename("session_open"), on="session")
    size = BAR_MINUTES * MINUTE
    b["complete"] = ((b["n_minutes"] == BAR_MINUTES) & (b["bar_end"] - b["bar_start"] == size)).to_numpy()
    b["slot"] = ((b["bar_start"] - b["session_open"]) // size).astype(int)
    b = b.join(bar_vwap(b, minutes))
    b["rvol_base"] = rvol_baseline(b, sessions)
    b["rvol"] = b["volume"] / b["rvol_base"]
    day = previous_session_levels(minutes, daily, sessions, sorted(set(bin_counts)))
    b = b.join(day.drop(columns=["in_study", "prev_atr"]), on="session")
    return b, minutes, day


def reclaim_lvn(minutes, first, last, sgn, val, vah, reclaim_close):
    """[low, high] of the reclaim-move low-volume node nearest the reclaimed value edge, or None.

    The 16-bin profile covers the minutes in [first, last), from the start of the 5-minute bar
    that set the excursion extreme through the reclaim bar's completion. Every one of those
    minutes must be present, and there must be at least 10. The node's center must lie inside
    prior value and below the reclaim close for a long (above it for a short). Ties choose the
    lower price.
    """
    t, low, high, vol = minutes
    a, z = np.searchsorted(t, first), np.searchsorted(t, last)
    if z - a < LVN_MIN_MINUTES or z - a != (last - first) // MINUTE.value:
        return None
    profile = vp.bar_profile(low[a:z], high[a:z], vol[a:z], LVN_BINS)
    if profile is None:
        return None
    edges, bins = profile
    centers = (edges[:-1] + edges[1:]) / 2
    edge = val if sgn > 0 else vah
    nodes = [k for k in vp.low_volume_nodes(bins)
             if val <= centers[k] <= vah and sgn * (reclaim_close - centers[k]) > 0]
    if not nodes:
        return None
    k = min(nodes, key=lambda k: (abs(centers[k] - edge), centers[k]))
    return edges[k], edges[k + 1]


def candidate_snapshot(f, i, sgn, levels, exc, extreme, extreme_at, rec, band, stop):
    """Frozen feature and level snapshot for one emitted signal. It runs only when a signal fires."""
    val, poc, vah, A, prev = levels
    o, h, l, c = (f[k].iat[i] for k in ("open", "high", "low", "close"))
    risk, reward = sgn * (c - stop), sgn * (poc - c)
    return {"side": "long" if sgn > 0 else "short", "side_sign": sgn, "profile_method": vp.PROFILE_METHOD,
            "prev_session": prev, "val": val, "poc": poc, "vah": vah,
            f"prev_atr_{ATR_PERIOD}": A, "buffer_b": BUFFER_ATR * A, "retest_tol_d": RETEST_ATR * A,
            "excursion_slot": f["slot"].iat[exc], "excursion_end": f["bar_end"].iat[exc],
            "excursion_close": f["close"].iat[exc], "extreme_slot": f["slot"].iat[extreme_at],
            "excursion_extreme": extreme, "reclaim_slot": f["slot"].iat[rec], "reclaim_end": f["bar_end"].iat[rec],
            "reclaim_close": f["close"].iat[rec], "signal_slot": f["slot"].iat[i], "location_lo": band[0],
            "location_hi": band[1], "signal_open": o, "signal_high": h, "signal_low": l,
            "body_frac": abs(c - o) / (h - l), "close_location": (c - l if sgn > 0 else h - c) / (h - l),
            "rvol": f["rvol"].iat[i], "rvol_base": f["rvol_base"].iat[i], "vwap": f["vwap"].iat[i],
            "vwap_15m_ago": f["vwap_15m_ago"].iat[i], "vwap_change_15m": f["vwap"].iat[i] - f["vwap_15m_ago"].iat[i],
            "vwap_method": f["vwap_method"].iat[i], "frozen_stop": stop, "frozen_target": poc,
            "signal_reward_risk": reward / risk}


def auction_reclaim(features, minutes, day, rules="baseline", location="value_edge", profile_bins=48,
                    rvol_filter=True):
    """Scan each study session once with a small state machine and return (mask, triggers, funnel).

    `mask` flags signal bars. `triggers` has one row per signal, indexed by bar position,
    and its rows are built only when a signal fires. `funnel` holds per-session setup
    counts: aggregates only, never per-bar records.
    """
    f = features
    rule = RULES[rules]
    last_after_open, retest_bars = rule["signal_last"].value, rule["retest_bars"]
    body_min, close_location_min, min_reward_risk = (rule[k] for k in ("body_min", "close_location_min",
                                                                        "min_reward_risk"))
    o, h, l, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    start, end = _ns(f["bar_start"]), _ns(f["bar_end"])
    open_ns = _ns(f["session_open"])
    complete = f["complete"].to_numpy(bool)
    rvol, vwap, lag = (f[k].to_numpy(float) for k in ("rvol", "vwap", "vwap_15m_ago"))
    lvn = location == "reclaim_lvn"
    minute_arrays = ((_ns(minutes["ts"]), *(minutes[k].to_numpy(float) for k in ("low", "high", "volume")))
                     if lvn else None)
    sess = f["session"].to_numpy()
    cuts = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1], True])
    ranges = {pd.Timestamp(sess[a]): (a, z) for a, z in zip(cuts[:-1], cuts[1:])}

    study = day[day["in_study"]]
    counts = np.zeros((len(study), len(SESSION_COUNTS) + 2 * len(SIDE_COUNTS)), dtype=int)
    col = {k: j for j, k in enumerate(SESSION_COUNTS)}
    col |= {(s, k): len(SESSION_COUNTS) + n * len(SIDE_COUNTS) + j
            for n, s in enumerate(SIDES.values()) for j, k in enumerate(SIDE_COUNTS)}
    mask = np.zeros(len(f), dtype=bool)
    positions, snapshots = [], []

    def qualifies(i, sgn, poc, edge, A, b, band, stop):
        rng = h[i] - l[i]
        if not (rng > 0 and l[i] <= band[1] and h[i] >= band[0]):
            return False
        if not (sgn * (c[i] - edge) > b and sgn * (poc - c[i]) > 0):
            return False
        if lvn and not (c[i] > band[1] if sgn > 0 else c[i] < band[0]):
            return False
        if not (sgn * (c[i] - o[i]) > 0 and abs(c[i] - o[i]) / rng >= body_min
                and (c[i] - l[i] if sgn > 0 else h[i] - c[i]) / rng >= close_location_min):
            return False
        if rvol_filter and not rvol[i] >= RVOL_MIN:
            return False
        if not (sgn * (vwap[i] - c[i]) > 0 and sgn * (vwap[i] - lag[i]) >= -VWAP_SLOPE_ATR * A):
            return False
        risk, reward = sgn * (c[i] - stop), sgn * (poc - c[i])
        return risk > 0 and reward > 0 and reward / risk >= min_reward_risk

    def scan(i0, i1, levels, cnt):
        val, poc, vah, A, _ = levels
        b, d = BUFFER_ATR * A, RETEST_ATR * A
        state, sgn, run = IDLE, 0, 0
        done = {1: False, -1: False}
        for i in range(i0, i1):
            if end[i] > open_ns[i] + last_after_open:  # no signal can follow: the setup ends with the window
                if state != IDLE:
                    cnt[col[sgn, "window_closed"]] += 1
                return
            # run = complete, contiguous bars ending here; a setup needs every bar since it started.
            run = run + 1 if complete[i] and i > i0 and run > 0 and start[i] == end[i - 1] else int(complete[i])
            if state != IDLE and run < 2:
                cnt[col[sgn, "missing_bar_reset"]] += 1
                state = IDLE
                continue
            if state == IDLE:
                if run <= BALANCE_BARS:
                    continue
                for s in (1, -1):
                    edge = val if s > 0 else vah
                    if s * (edge - c[i]) > b:  # an excursion close beyond the edge (both sides can't hold)
                        before = c[i - BALANCE_BARS:i]
                        if (not done[s] and not s * (edge - c[i - 1]) > b
                                and np.count_nonzero((before >= val) & (before <= vah)) >= BALANCE_MIN_INSIDE):
                            state, sgn, exc, extreme, extreme_at = RECLAIM, s, i, (l[i] if s > 0 else h[i]), i
                            cnt[col[s, "excursions"]] += 1
                        break
                continue
            edge = val if sgn > 0 else vah
            beyond_extreme = sgn * (extreme - (l[i] if sgn > 0 else h[i])) > 0
            touches_poc = h[i] >= poc if sgn > 0 else l[i] <= poc
            if state == RECLAIM:
                if touches_poc:
                    cnt[col[sgn, "reclaim_poc_first"]] += 1
                    state = IDLE
                    continue
                if beyond_extreme:  # strictly beyond: ties keep the earliest extreme bar
                    extreme, extreme_at = (l[i] if sgn > 0 else h[i]), i
                if sgn * (c[i] - edge) > b and sgn * (poc - c[i]) > 0 and sgn * (c[i] - o[i]) > 0:
                    cnt[col[sgn, "reclaims"]] += 1
                    state, rec = RETEST, i
                    band = (reclaim_lvn(minute_arrays, start[extreme_at], end[i], sgn, val, vah, c[i]) if lvn
                            else (edge - d, edge + d))
                    if band is None:  # never substitute the value edge
                        cnt[col[sgn, "lvn_no_node"]] += 1
                        state = IDLE
                elif i - exc >= RECLAIM_BARS:
                    cnt[col[sgn, "reclaim_expired"]] += 1
                    state = IDLE
                continue
            # RETEST: invalidation first (intrabar order is unknown), then the earliest qualifying bar.
            reason = ("invalid_extreme" if beyond_extreme else "invalid_poc" if touches_poc
                      else "invalid_close" if sgn * (edge - c[i]) > b else None)
            if reason:
                cnt[col[sgn, reason]] += 1
                state = IDLE
                continue
            stop = extreme - sgn * STOP_ATR * A
            if end[i] >= open_ns[i] + SIGNAL_FIRST.value and qualifies(i, sgn, poc, edge, A, b, band, stop):
                mask[i] = True
                positions.append(i)
                snapshots.append(candidate_snapshot(f, i, sgn, levels, exc, extreme, extreme_at, rec, band, stop))
                done[sgn] = True
                state = IDLE
            elif i - rec >= retest_bars:
                cnt[col[sgn, "retest_expired"]] += 1
                state = IDLE

    for r, (session, s) in enumerate(study.iterrows()):
        cnt = counts[r]
        cnt[col["sessions"]] = 1
        levels = (s[f"val_{profile_bins}"], s[f"poc_{profile_bins}"], s[f"vah_{profile_bins}"], s["prev_atr"],
                  s["prev_session"])
        span = ranges.get(session)
        cnt[col["no_bars"]] = span is None
        cnt[col["profile_unavailable"]] = not np.isfinite(levels[0])
        cnt[col["atr_unavailable"]] = not levels[3] > 0
        if span is None or not np.isfinite(levels[0]) or not levels[3] > 0:
            continue
        cnt[col["eligible_sessions"]] = 1
        scan(*span, levels, cnt)

    triggers = pd.DataFrame(snapshots, index=pd.Index(positions, dtype=int),
                            columns=list(TRIGGER_COLUMNS)).astype(TRIGGER_COLUMNS)
    names = list(SESSION_COUNTS) + [f"{side}_{k}" for side in SIDES for k in SIDE_COUNTS]
    return mask, triggers, pd.DataFrame(counts, index=study.index, columns=names)


def funnel_counts(funnel):
    """Setup counts for one segment, per side; the session counts repeat on both sides."""
    common = {k: int(funnel[k].sum()) for k in SESSION_COUNTS}
    return {side: {**common, **{k: int(funnel[f"{side}_{k}"].sum()) for k in SIDE_COUNTS}} for side in SIDES}
