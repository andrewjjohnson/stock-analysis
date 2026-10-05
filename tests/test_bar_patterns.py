"""bar_patterns.py on SYNTHETIC bars: matching a candle pattern, the higher-low / lower-high condition, session
boundaries, and measuring a fade in the short direction."""

import numpy as np
import pandas as pd
import pytest

import bar_patterns as bp
import red_bars as rb


def bars(opens, closes, lows=None, highs=None, sessions=None):
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    n = len(o)
    return pd.DataFrame({"session": pd.to_datetime(sessions or ["2024-01-02"] * n), "open": o, "close": c,
                         "low": np.minimum(o, c) - 0.1 if lows is None else np.asarray(lows, float),
                         "high": np.maximum(o, c) + 0.1 if highs is None else np.asarray(highs, float)})


# red, red, red, green, red: opens/closes for each bar
RRRGR_O = [10.0, 9.8, 9.6, 9.4, 9.7]
RRRGR_C = [9.8, 9.6, 9.4, 9.7, 9.5]


def test_the_pattern_fires_on_its_fifth_bar_and_the_higher_low_check_uses_the_red_bars_lows():
    b = bars([10.5] + RRRGR_O, [10.6] + RRRGR_C)  # a green bar first, then the pattern
    assert np.flatnonzero(bp.pattern_mask(b, "RRRGR", 1, False, True)).tolist() == [5]
    lows = [10.4, 9.7, 9.5, 9.3, 9.35, 9.4]  # the last bar's low (9.4) stays above the red bars' lowest (9.3)
    assert bp.pattern_mask(bars([10.5] + RRRGR_O, [10.6] + RRRGR_C, lows=lows), "RRRGR", 1, True, True)[5]
    lows[5] = 9.25  # a new low: no longer a higher low
    assert not bp.pattern_mask(bars([10.5] + RRRGR_O, [10.6] + RRRGR_C, lows=lows), "RRRGR", 1, True, True).any()


def test_a_pattern_across_two_sessions_counts_only_when_allowed():
    sessions = ["2024-01-02"] * 3 + ["2024-01-03"] * 2
    b = bars(RRRGR_O, RRRGR_C, sessions=sessions)
    assert not bp.pattern_mask(b, "RRRGR", 1, False, True).any()  # 5- and 15-minute bars: inside one session
    assert bp.pattern_mask(b, "RRRGR", 1, False, False)[4]  # hourly bars may cross the night


def test_a_fade_is_measured_in_the_short_direction():
    g = pd.DataFrame({"session": pd.to_datetime(["2024-01-02"] * 4), "race": [1.0, -1.0, -1.0, 0.0],
                      **{f"ret_{k}": [-6.0, 2.0, 2.0, 2.0] for k in rb.HORIZONS}})
    s = rb.summarize(g, sig=[True, False, False, False], side=-1.0)
    # The faded bar fell 6 bps (+6 the short way); every bar averaged 0 bps, so the excess is +6.
    assert s["avg_3"] == pytest.approx(6.0) and s["excess_3"] == pytest.approx(6.0) and s["race"] == 0.0
