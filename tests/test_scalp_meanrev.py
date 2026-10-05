"""scalp_meanrev.py on SYNTHETIC data: runs of bars, fresh signals, the session VWAP, forward moves and the
bracket, clustered errors, the per-round selection and the round-2 context columns."""

import numpy as np
import pandas as pd
import pytest

import features
import scalp_meanrev as sm
import synthetic

NY = "America/New_York"


def test_runs_count_same_direction_closes_and_reset_on_a_flat_or_new_session():
    assert sm.run_lengths(np.array([0, 1, 1, -1, -1, -1, 0, 1.0])).tolist() == [0, 1, 2, -1, -2, -3, 0, 1]


def test_a_signal_fires_on_the_first_bar_of_each_run_and_again_in_a_new_session():
    cond = np.array([True, True, False, True, True])
    sess = np.array(["a", "a", "a", "a", "b"])
    assert sm.first_of_run(cond, sess).tolist() == [True, False, False, True, True]


def one_session_minutes(closes, day="2024-03-05"):
    sessions = features.trading_sessions(day, day, warmup_sessions=0)
    ts = synthetic.session_minute_starts(sessions)[:len(closes)]
    m = synthetic.bars_from_closes(ts, closes)
    rth, _ = features.regular_session_minutes(m, sessions)
    return rth, sessions


def test_session_vwap_and_its_band_match_a_direct_computation():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.05, 60))
    rth, sessions = one_session_minutes(closes)
    rth["volume"] = rng.integers(100, 1000, len(rth)).astype(float)
    bar_end = pd.Series([sessions["open"].iloc[0] + pd.Timedelta(minutes=k) for k in (5, 30)])
    vwap, sd = sm.session_vwap(rth, bar_end)
    for k, minutes in enumerate((5, 30)):
        m = rth.iloc[:minutes]
        w = m["volume"] / m["volume"].sum()
        expect = (w * m["vwap"]).sum()
        assert vwap[k] == pytest.approx(expect)
        assert sd[k] == pytest.approx(np.sqrt((w * (m["vwap"] - expect) ** 2).sum()))


def test_forward_moves_and_the_bracket_follow_the_minutes_and_need_every_minute():
    # Flat at 100 for 10 minutes (the signal bar ends at minute 10), then up 0.05 a minute.
    closes = np.r_[np.full(10, 100.0), 100 + 0.05 * np.arange(1, 81)]
    rth, sessions = one_session_minutes(closes)
    open_ = sessions["open"].iloc[0]
    bars = pd.DataFrame({"bar_end": [open_ + pd.Timedelta(minutes=10)], "close": [100.0], "atr": [0.4],
                         "session_close": [sessions["close"].iloc[0]]})
    out = sm.bar_outcomes(rth, bars).iloc[0]
    assert out["ret_5"] == pytest.approx((closes[14] / 100 - 1) * 1e4)
    assert out["ret_15"] == pytest.approx((closes[24] / 100 - 1) * 1e4)
    assert out["race"] == 1  # 0.4 up comes first; it never falls 0.4
    gappy = rth.drop(index=20)  # a missing minute inside the 15- and 30-minute windows
    out = sm.bar_outcomes(gappy.reset_index(drop=True), bars).iloc[0]
    assert not np.isnan(out["ret_10"]) and np.isnan(out["ret_15"]) and np.isnan(out["race"])
    late = bars.assign(bar_end=[open_ + pd.Timedelta(minutes=88)])  # the data (90 minutes) ends inside the window
    assert np.isnan(sm.bar_outcomes(rth, late).iloc[0]["ret_5"])
    closing = bars.assign(session_close=[open_ + pd.Timedelta(minutes=12)])  # the session closes inside the window
    assert np.isnan(sm.bar_outcomes(rth, closing).iloc[0]["ret_5"])


def test_clustered_standard_error_matches_a_hand_computation():
    m, se = sm.cluster_mean([1, 2, 3, 10], ["a", "a", "b", "b"])
    # Residuals -3, -2, -1, 6 sum to -5 and 5 by day: sqrt((25 + 25) * 2 / 1) / 4.
    assert m == 4 and se == pytest.approx(2.5)


def test_selection_corrects_within_each_round_and_needs_events_excess_and_cost():
    rows = []
    for rnd in ("1", "2"):
        for k, p in enumerate([1e-6, 0.5, 0.9]):
            rows.append({"round": rnd, "who": "SPY", "pooled": False, "signal": f"s{k}", "side": "long",
                         "period": "design", "events": 400, "excess": 2.0, "avg": 2.0, "cost": 1.0, "p": p})
    stats = pd.DataFrame(rows)
    stats.loc[(stats["round"] == "2") & (stats["signal"] == "s0"), "avg"] = 0.5  # below the cost
    d = sm.select(stats)
    assert d.loc[d["round"] == "1", "qualifies"].tolist() == [True, False, False]
    assert not d.loc[d["round"] == "2", "qualifies"].any()
    assert d.loc[(d["round"] == "1") & (d["signal"] == "s0"), "q"].iloc[0] == pytest.approx(3e-6)  # 3 tests, not 6


def test_round_two_context_uses_only_earlier_sessions():
    sessions = features.trading_sessions("2024-01-02", "2024-03-28", warmup_sessions=0)
    minutes = synthetic.random_walk_minutes(sessions, seed=4, drop_fraction=0, drop_sessions=0)
    rth, bars, daily, _ = features.build_features(minutes, sessions, bar_minutes=5, ema_periods=(9, 20, 50))
    b = sm.add_context(sm.add_indicators(bars, rth), daily, design_end=pd.Timestamp("2025-01-01"))
    day = sessions.index[30]
    runs = sm.run_lengths(np.sign(daily["close"].diff()).fillna(0).to_numpy())
    row = b[b["session"] == day].iloc[3]
    assert row["daily_streak_prev"] == runs[29]
    assert row["gap_today"] == pytest.approx((daily.loc[day, "open"] / daily.loc[day, "prev_close"] - 1) * 100)
    slot = row["bar_start"].tz_convert(NY).strftime("%H:%M")
    same = b[(b["bar_start"].dt.tz_convert(NY).dt.strftime("%H:%M") == slot) & (b["session"] < day)]
    assert row["rvol"] == pytest.approx(row["volume"] / same["volume"].tail(20).mean())
