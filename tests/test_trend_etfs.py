"""trend_etfs.py on SYNTHETIC prices: trend signals from the past only, risk weights, next-day returns and costs,
the SPY dip-buying positions and the performance measures."""

import numpy as np
import pandas as pd
import pytest

import trend_etfs as te


def test_signals_use_only_prices_up_to_the_session_and_ignore_later_shocks():
    days = pd.bdate_range("2023-01-02", periods=150)
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 150)))}, index=days)
    for rule, n in te.RULES.values():
        base = te.trend_signal(p, rule, n)
        shocked = p.copy()
        shocked.iloc[120:] *= 3
        after = te.trend_signal(shocked, rule, n)
        pd.testing.assert_frame_equal(base.iloc[:120], after.iloc[:120])
        assert base.iloc[: n - 1].isna().all().all()
    mom = te.trend_signal(p, "mom", 21)
    assert mom.iloc[30, 0] == np.sign(p.iloc[30, 0] / p.iloc[9, 0] - 1)


def test_risk_weights_target_ten_percent_and_cap_at_two():
    days = pd.bdate_range("2023-01-02", periods=80)
    r = pd.DataFrame({"calm": np.tile([0.001, -0.001], 40), "wild": np.tile([0.03, -0.03], 40)}, index=days)
    w = te.risk_weights(r).iloc[-1]
    assert w["calm"] == 2.0  # 1.6% a year would need 6x; capped
    assert w["wild"] == pytest.approx(0.10 / (r["wild"].iloc[-60:].std() * np.sqrt(252)))


def test_a_position_earns_the_next_sessions_move_and_pays_for_changes():
    days = pd.bdate_range("2023-01-02", periods=4)
    pos = pd.DataFrame({"A": [1.0, 1.0, -1.0, -1.0]}, index=days)
    ret = pd.DataFrame({"A": [0.0, 0.02, 0.01, -0.03]}, index=days)
    r = te.sleeve_returns(pos, ret, cost_bps=10)["A"]
    # Day 2: held +1 from day 1 -> +2%, less opening it from flat on day 1 (1 unit x 10 bps). Day 3: still +1 ->
    # +1%. Day 4: held -1 -> +3%, less the flip from +1 to -1 decided on day 3 (2 units x 10 bps).
    assert r.iloc[1] == pytest.approx(0.02 - 0.001)
    assert r.iloc[2] == pytest.approx(0.01)
    assert r.iloc[3] == pytest.approx(0.03 - 0.002)


def test_dip_buying_enters_after_three_down_days_and_leaves_on_the_first_up_day_or_after_five():
    f = pd.DataFrame({"streak": [-1, -2, -3, -4, 1, -1, -2, -3, -4, -5, -6, -7, -8.0],
                      "ret_1d": [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1.0]})
    pos = te.dip_positions(f).tolist()
    # In from the 3rd down day until the up day (index 4); in again from index 7 for five sessions (out at 12,
    # where the streak is still -3 or lower, so it re-enters at once).
    assert pos == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_performance_measures():
    days = pd.bdate_range("2023-01-02", periods=4)
    r = pd.Series([0.10, -0.20, 0.05, 0.0], index=days)
    p = te.perf(r)
    assert p["max_dd"] == pytest.approx(0.88 / 1.10 - 1)  # 1.10 -> 0.88
    assert p["y2023"] == pytest.approx(1.10 * 0.80 * 1.05 - 1)
