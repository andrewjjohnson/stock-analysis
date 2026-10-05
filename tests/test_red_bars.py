"""red_bars.py on SYNTHETIC bars: runs of red candles and down closes, session restarts, forward moves (inside a
session or across nights), the race, and the excess over every bar."""

import numpy as np
import pandas as pd
import pytest

import red_bars as rb


def bars(opens, closes, sessions, highs=None, lows=None):
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"session": pd.to_datetime(sessions), "open": o, "close": c,
                         "high": np.maximum(o, c) if highs is None else highs,
                         "low": np.minimum(o, c) if lows is None else lows})


def test_runs_count_red_candles_or_down_closes_and_can_restart_each_session():
    b = bars([10, 10, 10, 10, 10, 10], [9, 9, 9, 11, 9, 9], ["2024-01-02"] * 4 + ["2024-01-03"] * 2)
    assert rb.red_runs(b, "red candles", False).tolist() == [1, 2, 3, 0, 1, 2]
    b2 = bars([10] * 6, [9, 8, 7, 6, 5, 4], ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
    assert rb.red_runs(b2, "down closes", False).tolist() == [0, 1, 2, 3, 4, 5]
    # Restarting each session: the first bar's change would include the overnight gap, so it does not count.
    assert rb.red_runs(b2, "down closes", True).tolist() == [0, 1, 2, 0, 1, 2]


def test_forward_moves_stay_inside_the_session_when_asked():
    c = [100, 101, 102, 103, 104, 105]
    b = bars(c, c, ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
    across = rb.forward(b, False, horizons=(1, 3))
    inside = rb.forward(b, True, horizons=(1, 3))
    assert across["ret_3"].iloc[0] == pytest.approx((103 / 100 - 1) * 1e4)  # crosses the night
    assert np.isnan(inside["ret_3"].iloc[0]) and inside["ret_1"].iloc[0] == pytest.approx(100.0)


def test_race_needs_one_atr_up_before_one_atr_down():
    n = 30
    c = np.full(n, 100.0)
    highs, lows = c + 0.5, c - 0.5  # a 1.0 range: ATR(14) = 1.0
    highs[21] = 101.5  # two bars after bar 19: +1 ATR comes first
    lows[25] = 98.0
    b = bars(c, c, ["2024-01-02"] * n, highs=highs, lows=lows)
    f = rb.forward(b, False, horizons=(1,), race_bars=5)
    assert f["race"].iloc[19] == 1.0
    assert f["race"].iloc[22] == -1.0  # from bar 22 the low at bar 25 comes first
    assert f["race"].iloc[15] == 0.0  # neither within 5 bars
    assert np.isnan(f["race"].iloc[10])  # no 14-bar ATR yet


def test_excess_is_measured_against_every_bar():
    g = pd.DataFrame({"session": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]),
                      "run": [3, 0, 4, 0], "race": [1.0, -1.0, 1.0, 0.0],
                      **{f"ret_{k}": [10.0, -2.0, 6.0, 2.0] for k in rb.HORIZONS}})
    s = rb.summarize(g)
    assert s["signals"] == 2 and s["avg_3"] == pytest.approx(8.0) and s["base_3"] == pytest.approx(4.0)
    assert s["excess_3"] == pytest.approx(4.0) and s["race"] == 100.0
