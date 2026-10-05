"""breakout_flow.py on SYNTHETIC one-second bars: the pressure measure (classing, carrying flat seconds, the window,
the sign for shorts, nothing from the fill second on), the clustered with-minus-against test, and the separate
volume cache of the one-second loader."""

import pickle

import numpy as np
import pandas as pd
import pytest

import breakout_check as bc
import breakout_flow as bf


def secs(rows, t0=0):
    """rows: (close, volume) per second from t0 (ms); open = previous close."""
    out, prev = [], rows[0][0]
    for k, (c, v) in enumerate(rows):
        out.append((t0 + k * 1000, prev, max(prev, c), min(prev, c), c, v))
        prev = c
    return np.array(out, float)


def test_pressure_classes_each_second_by_its_change_and_carries_flat_seconds():
    # second 0 only sets the price; 1 up (100), 2 flat -> still buying (50), 3 down (30), 4 flat -> selling (20)
    s = secs([(10.0, 999), (10.1, 100), (10.1, 50), (10.0, 30), (10.0, 20)])
    p, n, move = bf.pressure(s, 1000, 5000, +1, min_seconds=1)
    assert n == 4 and p == pytest.approx((150 - 50) / 200)
    assert move == pytest.approx((10.0 / 10.0 - 1) * 1e4)  # from second 0's close to second 4's close
    short, _, _ = bf.pressure(s, 1000, 5000, -1, min_seconds=1)
    assert short == pytest.approx(-p)


def test_pressure_ignores_the_fill_second_and_later_and_needs_enough_seconds():
    s = secs([(10.0, 1), (10.1, 100), (10.2, 100), (9.0, 10_000), (8.0, 10_000)])
    before, _, _ = bf.pressure(s, 1000, 3000, +1, min_seconds=1)  # the fill second starts at 3000
    assert before == pytest.approx(1.0)
    assert np.isnan(bf.pressure(s, 1000, 3000, +1, min_seconds=3)[0])
    first_flat = secs([(10.0, 1), (10.0, 500), (10.1, 100)])  # no change yet in second 1: not counted
    assert bf.pressure(first_flat, 1000, 3000, +1, min_seconds=1)[0] == pytest.approx(1.0)


def test_with_minus_against_matches_the_difference_in_means():
    t = pd.DataFrame({"gross": [10.0, 14.0, 6.0, -4.0, 0.0, -2.0], "pressure_5": [0.5, 0.2, 0.1, -0.3, 0.0, np.nan],
                      "session": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06",
                                                 "2022-01-07", "2022-01-10"])})
    diff, se, p = bf.split_test(t, 5, cost=1)
    assert diff == pytest.approx(np.mean([10, 14, 6]) - np.mean([-4, 0]))  # the no-measure trade is left out
    assert se > 0 and 0 < p < 0.5
    beta, se2 = bf.cluster_ols(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 1.0, 1.0]), [1, 2, 3, 4])
    assert beta.tolist() == pytest.approx([1.5, 2.0]) and np.all(se2 > 0)


def test_volume_bars_are_cached_apart_from_the_price_bars(tmp_path):
    folder = tmp_path / "seconds"
    folder.mkdir()
    key = ("TSLA", 0, 999)
    (folder / "TSLA.pkl").write_bytes(pickle.dumps({key: np.ones((1, 5))}))
    (folder / "TSLA_volume.pkl").write_bytes(pickle.dumps({key: np.ones((1, 6))}))
    assert bc.second_loader(tmp_path)(*key).shape == (1, 5)
    assert bc.second_loader(tmp_path, volume=True)(*key).shape == (1, 6)
