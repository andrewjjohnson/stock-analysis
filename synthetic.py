"""SYNTHETIC minute bars for tests and the offline demo only.

Nothing here is market data, and run.py never falls back to it: a failed Massive
request is an error, not a reason to substitute this.
"""

import numpy as np
import pandas as pd


def session_minute_starts(sessions):
    """UTC start of every regular-session minute (holidays and early closes respected)."""
    ranges = [pd.date_range(o, c, freq="1min", inclusive="left") for o, c in zip(sessions["open"], sessions["close"])]
    return ranges[0].append(ranges[1:]).as_unit("ns")


def bars_from_closes(ts, closes):
    """Minute OHLCV from a close path: open = previous close, high/low = max/min(open, close)."""
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"ts": ts, "open": opens, "high": np.maximum(opens, closes),
                         "low": np.minimum(opens, closes), "close": closes, "volume": 1000.0,
                         "vwap": (opens + closes) / 2, "transactions": 10})


def random_walk_minutes(sessions, seed=0, start_price=100.0, drift=3e-6, vol=4e-4, gap_vol=4e-3,
                        drop_fraction=5e-4, drop_sessions=1):
    """Seeded random walk with overnight gaps, small wicks and a few missing minutes
    (plus `drop_sessions` whole sessions removed) so coverage reporting shows up."""
    rng = np.random.default_rng(seed)
    ts = session_minute_starts(sessions)
    steps = rng.normal(drift, vol, len(ts))
    new_session = np.r_[True, np.diff(ts.tz_convert("America/New_York").normalize().asi8) != 0]
    steps[new_session] += rng.normal(0, gap_vol, new_session.sum())
    bars = bars_from_closes(ts, start_price * np.exp(np.cumsum(steps)))
    wick = np.abs(rng.normal(0, vol / 2, (2, len(bars))))
    bars["high"] *= 1 + wick[0]
    bars["low"] *= 1 - wick[1]
    keep = rng.random(len(bars)) >= drop_fraction
    session = ts.tz_convert("America/New_York").normalize().tz_localize(None)
    for day in rng.choice(sessions.index[sessions["in_study"]], drop_sessions, replace=False):
        keep &= session != day
    return bars[keep].reset_index(drop=True)
