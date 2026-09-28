"""Forward outcomes, computed only for triggered candidates.

This is a signal study, not a fill model. The reference price is the completed
trigger bar's close, and the clock starts when that bar completes (time T).

- Return at h minutes: close of the one-minute bar that ends at T+h (it starts at
  T+h-1min) versus the reference close, in percent. h is elapsed minutes, found by
  timestamp lookup, not "the next h rows".
- MFE / MAE: highest high / lowest low of the minute bars in [T, T+60) versus the
  reference close, in percent, clipped so MFE >= 0 >= MAE. The trigger bar's own
  minutes (before T) are excluded.
- An outcome is available only if every minute in [T, T+h) is present and T+h is no
  later than the session close or the evaluation segment end. Otherwise that value
  is NaN; the candidate row and its other outcomes are kept.
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


def forward_outcomes(minutes, signal_time, ref_close, session_close, segment_end):
    """One row of outcomes per candidate, in the order given.

    minutes: regular-session one-minute bars sorted by unique `ts` (bar start, UTC).
    signal_time / ref_close / session_close: per-candidate arrays (trigger bar end,
    trigger bar close, that session's close). segment_end: latest time any outcome
    may reach, e.g. the last session close before a split date.
    """
    t = _ns(minutes["ts"])
    close = minutes["close"].to_numpy(float)
    high = minutes["high"].to_numpy(float)
    low = minutes["low"].to_numpy(float)
    sig = _ns(signal_time)
    ref = np.asarray(ref_close, dtype=float)
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
        ret[ok] = (close[stop[ok] - 1] / ref[ok] - 1) * 100
        out[f"fwd_ret_{h}m_pct"] = ret

    stop, ok = window(EXCURSION_MINUTES)
    mfe = np.full(len(sig), np.nan)
    mae = np.full(len(sig), np.nan)
    for i in np.flatnonzero(ok):
        mfe[i] = max(0.0, (high[first[i]:stop[i]].max() / ref[i] - 1) * 100)
        mae[i] = min(0.0, (low[first[i]:stop[i]].min() / ref[i] - 1) * 100)
    out[f"mfe_{EXCURSION_MINUTES}m_pct"] = mfe
    out[f"mae_{EXCURSION_MINUTES}m_pct"] = mae
    return pd.DataFrame(out, columns=OUTCOME_COLUMNS)
