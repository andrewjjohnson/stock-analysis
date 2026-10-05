"""levels.py on SYNTHETIC minutes: where each level comes from (never later data), the first touch and the race,
fake levels, the held-rate comparison, and the holdout gate."""

import numpy as np
import pandas as pd
import pytest

import features
import levels as lv


def make_minutes(sessions, seed=0, pre=40, post=30):
    """A random walk with `pre` pre-market minutes before each open and `post` after-hours minutes."""
    stamps = [pd.date_range(o - pd.Timedelta(minutes=pre), c + pd.Timedelta(minutes=post - 1), freq="min")
              for o, c in zip(sessions["open"], sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.05, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


def at(sessions, day, hhmm):
    return (pd.Timestamp(day) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))).tz_localize(lv.NY)


def table_for(minutes, sessions):
    rth, _ = features.regular_session_minutes(minutes, sessions)
    return lv.level_table(rth, minutes, sessions)


def test_levels_use_yesterdays_regular_hours_last_week_and_todays_pre_market_only():
    sessions = features.trading_sessions("2024-01-02", "2024-02-29", warmup_sessions=0)
    m = make_minutes(sessions)
    day, prev = pd.Timestamp("2024-02-07"), pd.Timestamp("2024-02-06")
    m.loc[m["ts"] == at(sessions, prev, "16:05"), "high"] = 200.0   # yesterday's after-hours: not a level
    m.loc[m["ts"] == at(sessions, day, "09:00"), "high"] = 150.0    # today's pre-market high
    m.loc[m["ts"] == at(sessions, day, "10:00"), "high"] = 120.0    # today's own regular hours: tomorrow's level
    m.loc[m["ts"] == at(sessions, day, "09:30"), "open"] = 100.0    # today's open, for the round levels
    t = table_for(m, sessions)
    rth, _ = features.regular_session_minutes(m, sessions)
    y = rth[rth["session"] == prev]
    assert t.loc[day, "prev_high"] == y["high"].max() < 200
    assert (t.loc[day, "prev_low"], t.loc[day, "prev_close"]) == (y["low"].min(), y["close"].iloc[-1])
    assert t.loc[day, "pre_high"] == 150.0 and t.loc[day, "prev_high"] < 120
    assert t.loc[pd.Timestamp("2024-02-08"), "prev_high"] == 120.0
    last_week = rth[(rth["session"] >= "2024-01-29") & (rth["session"] <= "2024-02-02")]
    assert t.loc[day, "week_high"] == last_week["high"].max() == t.loc[pd.Timestamp("2024-02-05"), "week_high"]
    assert t.loc[day, "week_low"] == last_week["low"].min()
    assert t.loc[day, ["dollar_below", "dollar_above", "five_below", "five_above"]].tolist() == [99, 101, 95, 105]
    assert np.isnan(t["atr"].iloc[14]) and np.isfinite(t["atr"].iloc[15])  # ATR(14) through yesterday


def test_pre_market_levels_need_thirty_bars_and_round_levels_bracket_the_open():
    sessions = features.trading_sessions("2024-01-02", "2024-01-31", warmup_sessions=0)
    m = make_minutes(sessions, pre=20)
    t = table_for(m, sessions)
    assert t["pre_high"].isna().all() and (t["pre_bars"] == 20).all()
    o = t["open"]
    assert ((t["dollar_below"] < o) & (o < t["dollar_above"]) & (t["dollar_above"] - t["dollar_below"] <= 2)).all()
    assert ((t["five_below"] % 5 == 0) & (t["five_below"] < o) & (o < t["five_above"])).all()


def test_changing_later_minutes_never_changes_earlier_levels_or_outcomes():
    sessions = features.trading_sessions("2024-01-02", "2024-02-29", warmup_sessions=0)
    m = make_minutes(sessions, seed=3)
    cut = at(sessions, "2024-02-12", "12:00")
    m2 = m.copy()
    later = m2["ts"] >= cut
    m2.loc[later, ["open", "high", "low", "close"]] *= 1.03
    a = lv.ticker_events(m, sessions, final=False)
    b = lv.ticker_events(m2, sessions, final=False)
    assert a[0].loc[:"2024-02-12"].equals(b[0].loc[:"2024-02-12"])  # levels are known at the open
    real = [x[1][~x[1]["fake"]] for x in (a, b)]
    early = [r[r["session"] < "2024-02-12"].reset_index(drop=True) for r in real]
    pd.testing.assert_frame_equal(early[0], early[1])


def test_touch_and_race():
    #               0      1      2      3      4      5      6      7
    high = np.array([101.0, 100.7, 100.3, 100.6, 100.4, 101.3, 102.1, 101.6])
    low = np.array([100.6, 100.1, 99.8, 100.0, 99.2, 100.9, 101.0, 101.4])
    price = np.array([100.0, 99.5, 99.7, 102.0, 103.0])
    side = np.array([1, 1, 1, -1, -1])
    first, (res,) = lv.races(high, low, 8, price, side, [0.5])
    assert first.tolist() == [2, 4, 4, 6, -1]
    # 100: back to 100.5 at bar 3 -> held. 99.5: never to 99.0, back to 100.0 at bar 5 -> held.
    # 99.7: through to 99.2 inside its touching bar -> broke, although bar 5 then rallies.
    # 102 (resistance): back down to 101.5 at bar 7 -> held. 103: never touched.
    assert res[:4].tolist() == [1.0, 1.0, -1.0, 1.0] and np.isnan(res[4])
    first, (res,) = lv.races(high, low, 2, price[:1], side[:1], [0.5])  # the touch must start in the window
    assert first[0] == -1 and np.isnan(res[0])
    # A later bar reaching both sides is left out; nothing either way by the close is "neither".
    _, (both,) = lv.races(np.array([101.0, 100.4, 100.6]), np.array([100.5, 99.9, 99.4]), 3, np.array([100.0]),
                          np.array([1]), [0.5])
    _, (none,) = lv.races(np.array([101.0, 100.4, 100.2]), np.array([100.5, 99.9, 99.7]), 3, np.array([100.0]),
                          np.array([1]), [0.5])
    assert np.isnan(both[0]) and none[0] == 0.0


def test_fake_levels_reuse_other_sessions_distances_on_the_same_side():
    days = pd.date_range("2024-01-02", periods=5, freq="B")
    real = pd.concat([pd.DataFrame({"session": days, "level": "prev_high", "dist": [0.1, 0.2, 0.3, 0.4, 0.5],
                                    "side": -1}),
                      pd.DataFrame({"session": days, "level": "prev_low", "dist": [-0.1, -0.2, -0.3, -0.4, -0.5],
                                    "side": 1})], ignore_index=True).assign(period="design", fake=False)
    table = pd.DataFrame({"open": 100.0, "atr": 2.0}, index=days)
    fakes = lv.fake_levels(real, table, k=10, seed=1)
    assert len(fakes) == 100 and fakes["fake"].all()
    merged = fakes.merge(real, on=["session", "level"], suffixes=("", "_real"))
    assert (merged["side"] == merged["side_real"]).all() and (merged["dist"] != merged["dist_real"]).all()
    assert set(fakes.loc[fakes["level"] == "prev_low", "dist"]) <= {-0.1, -0.2, -0.3, -0.4, -0.5}
    assert np.allclose(fakes["price"], 100 + 2.0 * fakes["dist"])
    pd.testing.assert_frame_equal(fakes, lv.fake_levels(real, table, k=10, seed=1))


def test_near_fakes_sit_either_side_of_the_real_level_without_crossing_the_open():
    days = pd.date_range("2024-01-02", periods=2, freq="B")
    real = pd.DataFrame({"session": days, "level": "prev_high", "dist": [0.26, 0.6], "side": -1, "period": "design",
                         "fake": False, "control": "real"})
    table = pd.DataFrame({"open": 100.0, "atr": 2.0}, index=days)
    near = lv.near_levels(real, table, offsets=(0.2, 0.3))
    first = sorted(near.loc[near["session"] == days[0], "dist"].round(10))
    second = sorted(near.loc[near["session"] == days[1], "dist"].round(10))
    assert first == [0.06, 0.46, 0.56] and second == [0.3, 0.4, 0.8, 0.9]  # 0.26 - 0.3 would cross the open
    assert (near["control"] == "near").all() and np.allclose(near["price"], 100 + 2.0 * near["dist"])


def test_compare_counts_held_against_broke_only():
    days = pd.date_range("2024-01-02", periods=30, freq="B")
    real = pd.DataFrame({"session": days, "race_10": [1.0] * 20 + [-1.0] * 10})
    real = pd.concat([real, pd.DataFrame({"session": days[:5], "race_10": [0.0, 0.0, np.nan, np.nan, 0.0]})])
    fake = pd.DataFrame({"session": np.repeat(days, 2), "race_10": [1.0, -1.0] * 30})
    w = lv.boot_weights(len(days), reps=200, seed=0)
    c = lv.compare(real, fake, "race_10", days, w)
    assert c["n"] == 30 and c["real"] == pytest.approx(200 / 3) and c["fake"] == pytest.approx(50.0)
    assert c["diff"] == pytest.approx(200 / 3 - 50) and c["diff_lo"] < c["diff"] < c["diff_hi"]
    few = lv.compare(real.iloc[:15], fake, "race_10", days, w)  # 15 resolved touches: too few for a test
    assert few["n"] == 15 and "diff" not in few
    assert (w.sum(axis=1) == len(days)).all()


def test_holdout_sessions_are_measured_only_with_the_final_test():
    sessions = features.trading_sessions("2024-10-01", "2025-02-28", warmup_sessions=0)
    m = make_minutes(sessions, seed=5)
    res = lv.run_study({"SPY": m}, sessions, reps=50)
    assert res["events"]["session"].max() < lv.HOLDOUT_START and set(res["stats"]["period"]) == {"design"}
    assert len(res["design"]) == len(lv.LEVELS) and res["holdout"] is None
    final = lv.run_study({"SPY": m}, sessions, final=True, reps=50)
    assert final["events"]["session"].max() >= lv.HOLDOUT_START
    assert set(final["stats"]["period"]) == {"design", "holdout"}
