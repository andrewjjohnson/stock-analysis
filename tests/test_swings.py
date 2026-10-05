"""swings.py on SYNTHETIC bars: the zigzag, legs and the turn hazard, the random-direction copies, when a swing line
starts and which side it is on, the fakes, no lookahead, and the holdout gate."""

import numpy as np
import pandas as pd
import pytest

import features
import levels as lv
import swings as sg


def make_minutes(sessions, seed=0, step=0.05):
    """A random walk over each regular session (no extended hours)."""
    stamps = [pd.date_range(o, c - pd.Timedelta(minutes=1), freq="min") for o, c in zip(sessions["open"],
                                                                                      sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, step, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


#                0     1     2     3     4     5     6      7     8     9
HIGH = np.array([10.5, 11.0, 11.6, 11.4, 11.0, 10.6, 10.95, 11.5, 11.2, 10.6])
LOW = np.array([10.0, 10.4, 11.0, 10.7, 10.5, 9.9, 10.2, 10.0, 10.8, 10.4])


def test_zigzag_turns_need_a_full_reversal_that_the_extreme_bar_itself_cannot_give():
    sw = sg.zigzag(np.vstack([HIGH, HIGH]), np.vstack([LOW, LOW]), np.array([1.0, 5.0]))
    assert (sw["row"] == 0).all()  # a 5.0 reversal never happens
    got = sw[["kind", "price", "bar", "confirmed"]].to_numpy().tolist()
    # Bar 1 reaches 10.0 + 1: the valley at bar 0. Bar 4 is 1 below the 11.6 high of bar 2: a peak. Bar 6: a valley
    # at bar 5. Bar 7 sets a new high (11.5) and also dips 1 below it, in unknown order, so only bar 9 confirms it.
    assert got == [[-1, 10.0, 0, 1], [1, 11.6, 2, 4], [-1, 9.9, 5, 6], [1, 11.5, 7, 9]]
    done, unfinished = sg.legs(sw, np.array([9, 9]))
    assert done["bars"].tolist() == [2, 3, 2] and done["kind"].tolist() == [1, -1, 1]
    assert unfinished["age"].tolist() == [2]


def test_hazard_counts_legs_still_running_and_mc_p_is_one_or_two_sided():
    done = pd.DataFrame({"bars": [1, 2, 2, 3]})
    unfinished = pd.DataFrame({"age": [1]})
    assert sg.hazard(done, unfinished, max_age=3).tolist() == pytest.approx([1 / 5, 2 / 3, 1.0])
    null = np.arange(100.0)
    assert sg.mc_p(200.0, null) == pytest.approx(1 / 101) and sg.mc_p(200.0, null, side=1) == pytest.approx(1 / 101)
    assert sg.mc_p(200.0, null, side=-1) == pytest.approx(1.0)


class AllUp:
    def choice(self, a, size):
        return np.ones(size)


def test_random_direction_copies_keep_each_bar_and_reproduce_the_session_when_nothing_flips():
    o = np.array([[100.0, 100.4, 100.1]])
    c = np.array([[100.4, 100.1, 100.9]])
    h, l = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.2
    g = {"open": o, "high": h, "low": l, "close": c}
    same_h, same_l = sg.flipped(g, 2, AllUp())
    assert np.allclose(same_h, np.vstack([h, h])) and np.allclose(same_l, np.vstack([l, l]))
    fh, fl = sg.flipped(g, 50, np.random.default_rng(0))
    assert np.allclose(fh - fl, np.tile(h - l, (50, 1)))  # every bar keeps its range
    assert not np.allclose(fh, np.tile(h, (50, 1)))


def test_a_swing_line_starts_at_its_confirmation_or_1130_with_its_side_from_the_close_before():
    sessions = features.trading_sessions("2024-01-02", "2024-01-03", warmup_sessions=0)
    m = make_minutes(sessions)
    m["close"] = 100 + 0.01 * (np.arange(len(m)) % 390)  # a known close every minute
    rth, _ = features.regular_session_minutes(m, sessions)
    days = sessions.index[:1]
    atr = pd.Series(2.0, index=sessions.index)
    sw = pd.DataFrame({"row": 0, "kind": [1, -1, 1, -1], "price": [101.0, 102.0, 103.5, 101.2],
                       "bar": [3, 40, 70, 10], "confirmed": [5, 44, 73, 20]})
    got = sg.swing_levels(sw, days, sessions, rth, atr)
    o = sessions["open"].iloc[0]
    # Drawn at 10:00 -> starts 11:30, against the 11:29 close (101.19): a swing high below the price is support.
    # Drawn at 13:15 (bar 44 ends then) -> starts then, against the 13:14 close (102.24). Bar 73 ends at 15:40,
    # too late to touch; the 101.20 line sits 0.005 ATR from its 11:29 reference: both dropped.
    assert got["price"].tolist() == [101.0, 102.0]
    assert got["start"].tolist() == [o + pd.Timedelta(hours=2), o + pd.Timedelta(minutes=225)]
    assert got["ref"].tolist() == pytest.approx([101.19, 102.24])
    assert got["side"].tolist() == [1, 1] and got["origin"].tolist() == list(sg.ORIGINS)
    assert got["level"].tolist() == ["swing high", "swing low"]


def test_random_fakes_never_take_their_own_sessions_distances_and_near_fakes_stay_on_their_side():
    days = pd.date_range("2024-01-02", periods=3, freq="B")
    real = pd.DataFrame({"session": np.repeat(days, 2), "level": "swing low", "origin": sg.ORIGINS[0],
                         "dist": [-0.11, -0.12, -0.21, -0.22, -0.31, -0.32], "side": 1, "period": "design",
                         "ref": 100.0, "atr": 2.0, "fake": False, "control": "real"})
    fakes = sg.random_fakes(real, k=20, seed=3)
    home = dict(zip(real["dist"], real["session"]))
    assert len(fakes) == 120 and (fakes["session"] != fakes["dist"].map(home)).all()
    assert np.allclose(fakes["price"], 100 + 2 * fakes["dist"])
    near = sg.near_fakes(real.iloc[:1], offsets=(0.2,))
    assert near["dist"].tolist() == pytest.approx([-0.31])  # -0.11 + 0.2 would cross to the other side


def test_changing_later_minutes_never_changes_earlier_swings_lines_or_outcomes():
    sessions = features.trading_sessions("2024-01-02", "2024-02-15", warmup_sessions=0)
    m = make_minutes(sessions, seed=4)
    cut = pd.Timestamp("2024-02-06 13:00", tz=lv.NY)
    m2 = m.copy()
    m2.loc[m2["ts"] >= cut, ["open", "high", "low", "close"]] += 1.5

    def lines(minutes):
        rth, atr, days, g = sg.ticker_data(minutes, sessions, ["design"])
        sw = sg.zigzag(g["high"], g["low"], sg.R * atr.reindex(days).to_numpy(float))
        real = sg.swing_levels(sw, days, sessions, rth, atr)
        return lv.measure(rth, real, pd.DataFrame({"atr": atr})), sw.assign(day=days[sw["row"].astype(int)])

    (a, swa), (b, swb) = lines(m), lines(m2)
    early = [x[x["start"] < cut].reset_index(drop=True) for x in (a, b)]
    pd.testing.assert_frame_equal(early[0][["session", "price", "start", "side"]],
                                  early[1][["session", "price", "start", "side"]])
    before = [x[x["session"] < "2024-02-06"].reset_index(drop=True) for x in (a, b)]
    pd.testing.assert_frame_equal(before[0], before[1])
    sw_early = [s[s["day"] < "2024-02-06"].reset_index(drop=True) for s in (swa, swb)]
    pd.testing.assert_frame_equal(sw_early[0], sw_early[1])


def test_holdout_sessions_are_read_only_with_the_final_test():
    sessions = features.trading_sessions("2024-10-01", "2025-02-28", warmup_sessions=0)
    m = make_minutes(sessions, seed=5)
    res = sg.run_study({"SPY": m}, sessions, reps=50, null_reps=4)
    assert res["events"]["session"].max() < lv.HOLDOUT_START and set(res["timing"]["period"]) == {"design"}
    assert len(res["design"]) == 2 and len(res["tests"]) == 2
    final = sg.run_study({"SPY": m}, sessions, final=True, reps=50, null_reps=4)
    assert final["events"]["session"].max() >= lv.HOLDOUT_START
    assert set(final["timing"]["period"]) == {"design", "holdout"}
