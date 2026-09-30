"""Forward outcomes, computed only for triggered candidates.

This is a signal study, not a fill model. The reference price is the completed
trigger bar's close, and the clock starts when that bar completes (time T).

- Returns are directional: side s = +1 (long, the default) or -1 (short) multiplies
  the raw return, so a positive value always means price moved the signal's way.
- Return at h minutes: s x (close of the one-minute bar that ends at T+h, which
  starts at T+h-1min, / reference close - 1), in percent. h is elapsed minutes, found
  by timestamp lookup, not "the next h rows".
- MFE / MAE: the same signed return applied to every minute high and low in
  [T, T+60), with zero included, so MFE >= 0 >= MAE. The trigger bar's own minutes
  (before T) are excluded.
- An outcome is available only if every minute in [T, T+h) is present and T+h is no
  later than the session close or the evaluation segment end. Otherwise that value
  is NaN; the candidate row and its other outcomes are kept.

barrier_exits() is a separate, idealized fixed stop/target comparison for strategies
that define those levels; it is not portfolio accounting or an execution model.
"""

import numpy as np
import pandas as pd

HORIZONS = (10, 30, 60, 120)
EXCURSION_MINUTES = 60
OUTCOME_COLUMNS = [f"fwd_ret_{h}m_pct" for h in HORIZONS] + [
    f"mfe_{EXCURSION_MINUTES}m_pct", f"mae_{EXCURSION_MINUTES}m_pct"]
NS_PER_MINUTE = 60 * 10**9


def _ns(times):
    """UTC epoch nanoseconds (int64 array) for tz-aware timestamps."""
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def forward_outcomes(minutes, signal_time, ref_close, session_close, segment_end, side=None):
    """One row of outcomes per candidate, in the order given.

    minutes: regular-session one-minute bars sorted by unique `ts` (bar start, UTC).
    signal_time / ref_close / session_close: per-candidate arrays (trigger bar end,
    trigger bar close, that session's close). segment_end: latest time any outcome
    may reach, e.g. the last session close before a split date. side: per-candidate
    +1 / -1; None means all long.
    """
    t = _ns(minutes["ts"])
    close = minutes["close"].to_numpy(float)
    high = minutes["high"].to_numpy(float)
    low = minutes["low"].to_numpy(float)
    sig = _ns(signal_time)
    ref = np.asarray(ref_close, dtype=float)
    s = np.ones(len(sig)) if side is None else np.asarray(side, dtype=float)
    limit = np.minimum(_ns(session_close), pd.Timestamp(segment_end).as_unit("ns").value)
    first = np.searchsorted(t, sig)  # first minute bar starting at or after T

    def window(h):
        end = sig + h * NS_PER_MINUTE
        stop = np.searchsorted(t, end)  # minute bars in [T, T+h) are first .. stop-1
        return stop, (end <= limit) & (stop - first == h)

    out = {}
    for h in HORIZONS:
        stop, ok = window(h)
        ret = np.full(len(sig), np.nan)
        ret[ok] = s[ok] * (close[stop[ok] - 1] / ref[ok] - 1) * 100
        out[f"fwd_ret_{h}m_pct"] = ret

    stop, ok = window(EXCURSION_MINUTES)
    mfe = np.full(len(sig), np.nan)
    mae = np.full(len(sig), np.nan)
    for i in np.flatnonzero(ok):
        signed = s[i] * (np.array([high[first[i]:stop[i]].max(), low[first[i]:stop[i]].min()]) / ref[i] - 1) * 100
        mfe[i] = max(0.0, signed.max())
        mae[i] = min(0.0, signed.min())
    out[f"mfe_{EXCURSION_MINUTES}m_pct"] = mfe
    out[f"mae_{EXCURSION_MINUTES}m_pct"] = mae
    return pd.DataFrame(out, columns=OUTCOME_COLUMNS)


# Illustrative friction per side, in basis points: 0 = gross. A sensitivity assumption,
# not calibrated transaction costs; borrow costs for shorts are not included.
COST_BPS = (0, 1)
BARRIER_COLUMNS = {
    "entry_time": "datetime64[ns, UTC]", "entry_status": "str", "entry_price": float, "stop_price": float,
    "target_price": float, "exit_reason": "str", "exit_time": "datetime64[ns, UTC]", "exit_price": float,
    "ambiguous": bool, "holding_minutes": float,
    **{k: float for b in COST_BPS for k in (f"barrier_ret_{b}bp_pct", f"barrier_r_{b}bp")},
}


def barrier_exits(minutes, entry_time, side, stop, target, session_close, segment_end, cost_bps=COST_BPS,
                  min_reward_risk=None):
    """Idealized fixed stop/target path for each already-triggered candidate, in the order given.

    - Entry: the open of the minute bar starting at entry_time, which must exist.
      entry_status is "unavailable" otherwise, and "invalid" unless stop < entry <
      target (long) or target < entry < stop (short) and, when min_reward_risk is given,
      |target - entry| / |entry - stop| >= min_reward_risk. The candidate is kept either way.
    - Minutes are inspected in time order from the entry minute. A later minute that
      opens beyond the stop exits at that open ("stop_gap"); one that opens beyond the
      target exits at the target, conservatively. Otherwise a touch (low <= stop,
      high >= target for longs, mirrored for shorts) exits at that level. Both levels
      inside one minute cannot be ordered: ambiguous, counted as the stop.
    - No touch by the session close: exit at the last minute's close ("close").
      A missing minute before any exit, or a segment end before the close, leaves the
      path "unresolved"; nothing after a gap is used. An exit observed before a gap stands.
    - Return: s x (exit - entry), less cost = rate x (entry + exit) per share, as a
      percent of entry and as R (multiples of the initial |entry - stop|).
    """
    t = _ns(minutes["ts"])
    op, hi, lo, cl = (minutes[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    start = _ns(entry_time)
    s = np.asarray(side, dtype=float)
    stop, target = np.asarray(stop, dtype=float), np.asarray(target, dtype=float)
    close_ns = _ns(session_close)
    limit = np.minimum(close_ns, pd.Timestamp(segment_end).as_unit("ns").value)
    n = len(start)
    status, reason = np.full(n, None, dtype=object), np.full(n, None, dtype=object)
    entry, exit_price, exit_ns = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.iinfo(np.int64).min)
    ambiguous = np.zeros(n, dtype=bool)

    for i in range(n):
        j = np.searchsorted(t, start[i])
        if start[i] >= limit[i] or j == len(t) or t[j] != start[i]:
            status[i] = "unavailable"
            continue
        entry[i] = op[j]
        reward, risk = s[i] * (target[i] - entry[i]), s[i] * (entry[i] - stop[i])
        if not (reward > 0 and risk > 0) or (min_reward_risk is not None and reward / risk < min_reward_risk):
            status[i] = "invalid"
            continue
        status[i] = "ok"
        expected = (limit[i] - start[i]) // NS_PER_MINUTE  # minutes in [entry, limit)
        k = np.arange(j, min(j + expected, len(t)))
        seen = t[k] == start[i] + NS_PER_MINUTE * np.arange(len(k))
        k = k[:len(k) if seen.all() else np.argmin(seen)]  # contiguous observed path only
        adverse, favorable = (lo[k], hi[k]) if s[i] > 0 else (hi[k], lo[k])
        hit_stop = s[i] * (adverse - stop[i]) <= 0
        hit_target = s[i] * (favorable - target[i]) >= 0
        events = np.flatnonzero(hit_stop | hit_target)
        if events.size:
            e = events[0]
            opened_beyond_stop = e > 0 and s[i] * (op[k[e]] - stop[i]) <= 0
            opened_beyond_target = e > 0 and s[i] * (op[k[e]] - target[i]) >= 0
            if opened_beyond_stop:
                reason[i], exit_price[i], exit_ns[i] = "stop_gap", op[k[e]], t[k[e]]
            elif opened_beyond_target:
                reason[i], exit_price[i], exit_ns[i] = "target", target[i], t[k[e]]
            else:
                ambiguous[i] = hit_stop[e] and hit_target[e]
                reason[i] = "stop" if hit_stop[e] else "target"
                exit_price[i] = stop[i] if hit_stop[e] else target[i]
                exit_ns[i] = t[k[e]] + NS_PER_MINUTE  # touched at some point inside this minute
        elif len(k) == expected and limit[i] == close_ns[i]:
            reason[i], exit_price[i], exit_ns[i] = "close", cl[k[-1]], limit[i]
        else:
            reason[i] = "unresolved"

    done = ~np.isnan(exit_price)
    holding = np.full(n, np.nan)
    holding[done] = (exit_ns[done] - start[done]) / NS_PER_MINUTE
    out = {"entry_time": pd.DatetimeIndex(entry_time).as_unit("ns"), "entry_status": status, "entry_price": entry,
           "stop_price": stop, "target_price": target, "exit_reason": reason,
           "exit_time": pd.to_datetime(exit_ns, utc=True).as_unit("ns"),  # int64 min -> NaT
           "exit_price": exit_price, "ambiguous": ambiguous, "holding_minutes": holding}
    for b in cost_bps:
        net = np.full(n, np.nan)
        net[done] = s[done] * (exit_price[done] - entry[done]) - b / 1e4 * (entry[done] + exit_price[done])
        out[f"barrier_ret_{b}bp_pct"] = net / entry * 100
        out[f"barrier_r_{b}bp"] = net / np.abs(entry - stop)  # > 0 whenever an exit exists
    return pd.DataFrame(out, columns=list(BARRIER_COLUMNS))
