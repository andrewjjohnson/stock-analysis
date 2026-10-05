"""meanrev.py on SYNTHETIC minutes: the 15:50 snapshot, no lookahead, dividend adjustment, forward
outcomes, the rule simulator, the walk-forward purge and the statistics helpers."""

import numpy as np
import pandas as pd
import pytest

import features
import meanrev as mr
import synthetic

NY = "America/New_York"


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def minutes_for(sessions, seed=0):
    return synthetic.random_walk_minutes(sessions, seed=seed, drop_fraction=0, drop_sessions=0)


def test_snapshot_is_the_price_known_ten_minutes_before_the_close_including_early_closes():
    # 2024-11-29 closes early at 13:00 ET, so its decision time is 12:50.
    sessions = features.trading_sessions("2024-11-27", "2024-11-29", warmup_sessions=0)
    m = minutes_for(sessions)
    d, _ = mr.daily_table(m, sessions)
    bars = m.set_index("ts")
    for day, decision in (("2024-11-27", "15:50"), ("2024-11-29", "12:50")):
        last_known = et(f"{day} {decision}") - pd.Timedelta(minutes=1)  # the bar that ends at the decision time
        row = d.loc[pd.Timestamp(day)]
        assert row["snap"] == bars.loc[last_known, "close"] and row["snap_age_min"] == 0
        before = bars[(bars.index >= et(f"{day} 09:30")) & (bars.index <= last_known)]
        after = bars[(bars.index > last_known) & (bars.index < et(f"{day} {'16:00' if day.endswith('27') else '13:00'}"))]
        assert row["snap_high"] == before["high"].max() and row["snap_low"] == before["low"].min()
        assert row["post_high"] == after["high"].max() and row["post_low"] == after["low"].min()
        assert row["close"] == after["close"].iloc[-1]


def test_features_ignore_everything_after_the_decision_time():
    sessions = features.trading_sessions("2024-01-02", "2024-12-31", warmup_sessions=0)
    m = minutes_for(sessions, seed=3)
    base = mr.ticker_features(mr.daily_table(m, sessions)[0])
    day = pd.Timestamp("2024-09-10")
    cut = et("2024-09-10 15:50")
    shocked = m.copy()
    later = shocked["ts"] >= cut  # the last 10 minutes of the day and every later session
    shocked.loc[later, ["open", "high", "low", "close"]] *= 1.5
    after = mr.ticker_features(mr.daily_table(shocked, sessions)[0])
    upto = base.index <= day
    pd.testing.assert_frame_equal(base[upto], after[upto])
    assert not base.loc[base.index > day].equals(after.loc[after.index > day])  # the shock did reach later days


def test_dividends_back_adjust_earlier_prices_so_returns_include_the_payout():
    days = pd.DatetimeIndex(["2024-03-14", "2024-03-15", "2024-03-18"])
    d = pd.DataFrame({c: [100.0, 99.0, 99.5] for c in mr.PRICES}, index=days)
    adj, applied = mr.adjust_dividends(d, pd.DataFrame({"ex_date": [pd.Timestamp("2024-03-15")],
                                                        "cash_amount": [2.0]}))
    assert applied["factor"].iloc[0] == pytest.approx(0.98)
    assert adj.loc["2024-03-14", "close"] == pytest.approx(98.0) and adj.loc["2024-03-15", "close"] == 99.0
    # 100 -> 99 plus a $2 dividend is a +1% total return, which the adjusted prices now show.
    assert adj.loc["2024-03-15", "close"] / adj.loc["2024-03-14", "close"] - 1 == pytest.approx(99 / 98 - 1)


def test_forward_outcomes_run_from_snapshot_to_snapshot_and_track_the_path_between():
    days = pd.bdate_range("2024-01-08", periods=6)
    d = pd.DataFrame({"snap": [100, 101, 102, 103, 104, 105.0], "snap_low": [99, 100, 98, 102, 103, 104.0],
                      "snap_high": [101, 102, 103, 104, 105, 106.0], "low": [98, 97, 96, 101, 50, 103.0],
                      "high": [102, 103, 104, 110, 106, 107.0], "post_low": [99.5, 100.5, 101.5, 102.5, 103.5, 104],
                      "post_high": [100.5, 101.5, 102.5, 103.5, 104.5, 105]}, index=days)
    o = mr.forward_outcomes(d, last_day=days[4], horizons=(1, 3))
    assert o["fwd_1"].iloc[0] == pytest.approx(1.0)
    # Day 0 -> day 3: path = day 0 after 15:50 (99.5), full days 1-2 (lows 97, 96), day 3 up to 15:50 (102).
    assert o["low_3"].iloc[0] == pytest.approx(-4.0)
    assert o["high_3"].iloc[0] == pytest.approx(4.0)  # day 3's full high (110) comes after 15:50: excluded
    assert np.isnan(o["fwd_3"].iloc[2]) and np.isnan(o["fwd_1"].iloc[4])  # windows past last_day are NaN
    # Day 3 -> 4: day 3 after 15:50 (102.5) and day 4 up to 15:50 (103); day 4's later low of 50 is excluded.
    assert o["low_1"].iloc[3] == pytest.approx((102.5 / 103 - 1) * 100)


def test_rules_hold_one_position_at_a_time_and_drop_trades_past_the_period():
    days = pd.bdate_range("2024-01-08", periods=8)
    f = pd.DataFrame({"sig": [1, 1, 0, 0, 1, 0, 0, 1], "out": [0, 0, 0, 1, 0, 0, 0, 0]}, index=days, dtype=float)
    snap = pd.Series([100, 101, 102, 104, 103, 102, 101, 100.0], index=days)
    t = mr.run_rule(f, snap, lambda f: f["sig"] == 1, lambda f: f["out"] == 1, 2, 1, days[-1])
    # Enter day 0, exit day 2 (max 2 sessions); re-enter? day 2 has no signal; day 4 entry -> day 6 (max hold).
    assert t[["sessions"]].to_numpy().ravel().tolist() == [2, 2]
    assert t["ret"].tolist() == pytest.approx([2.0, (101 / 103 - 1) * 100])
    short = mr.run_rule(f, snap, lambda f: f["sig"] == 1, lambda f: f["out"] == 1, 5, -1, days[-1])
    assert short["exit"].iloc[0] == days[3] and short["ret"].iloc[0] == pytest.approx(-4.0)  # exit signal on day 3


def test_walk_forward_trains_only_on_rows_whose_outcome_window_closed(monkeypatch):
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(X[:, 0].copy())
            return self

        def predict_proba(self, X):
            return np.c_[np.full(len(X), 0.5), np.full(len(X), 0.5)]

    monkeypatch.setitem(mr.MODELS, "logistic", Recorder)
    n, h = 700, 5
    pos = np.arange(n) * 1 + (np.arange(n) >= 400) * 3  # a 3-session gap with no rows
    X = np.c_[pos, np.random.default_rng(0).normal(size=n)]
    y = (np.random.default_rng(1).random(n) > 0.5).astype(float)
    proba, block, rec, _ = mr.walk_forward(X, y, pos + h, pos, 300, int(pos[-1]) + 1, ["logistic"], h)
    starts = [int(pos[block == b].min()) for b in np.unique(block[block >= 0])]
    assert len(seen) == len(starts) >= 3
    for trained, first in zip(seen, starts):
        assert trained.max() + h < first  # every training label ended before the test block began


def test_statistics_helpers():
    # Sorted p: 0.01, 0.03, 0.04, 0.5 -> p * 4 / rank = 0.04, 0.06, 0.0533, 0.5 -> running minimum from the top.
    assert mr.bh_qvalues([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    idx = mr.block_indices(50, 20, 0, block=10)
    assert idx.shape == (20, 50)
    assert (np.diff(idx[:, :10], axis=1) % 50 == 1).all()  # each block is 10 consecutive sessions (wrapping)

    events = pd.DataFrame([
        {"event": "a", "side": "dip", "horizon": 5, "regime": "all", "n": 40, "excess_lo": 0.1, "excess_hi": 0.9},
        {"event": "b", "side": "dip", "horizon": 5, "regime": "all", "n": 40, "excess_lo": 0.3, "excess_hi": 0.5},
        {"event": "c", "side": "dip", "horizon": 5, "regime": "all", "n": 10, "excess_lo": 0.9, "excess_hi": 2.0},
        {"event": "d", "side": "rally", "horizon": 5, "regime": "all", "n": 50, "excess_lo": -0.8, "excess_hi": -0.2},
        {"event": "e", "side": "rally", "horizon": 5, "regime": "all", "n": 50, "excess_lo": -0.5, "excess_hi": 0.1},
    ])
    ml = pd.DataFrame([{"horizon": 5, "model": "logistic", "period": "design", "auc": 0.55},
                       {"horizon": 5, "model": "boosted_trees", "period": "design", "auc": 0.52}])
    # b has the higher lower bound among events with enough samples (c has too few); d's upper bound is lowest.
    assert mr.select_primary(events, ml) == {"dip": "b", "rally": "d", "ml": "logistic"}
