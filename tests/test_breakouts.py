"""breakouts.py on SYNTHETIC bars: stop-order fills at the level, targets, stops (worst case inside a minute, gaps
through the stop), exits at the close, which levels each rule trades, costs, and the holdout gate."""

import numpy as np
import pandas as pd
import pytest

import breakouts as bo
import features
import levels as lv


def bars(rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return o, h, l, c


def run(rows, price, side, stop_w=0.5, target_w=0.5, n_touch=None):
    o, h, l, c = bars(rows)
    return bo.simulate(o, h, l, c, len(o) if n_touch is None else n_touch, np.array(price, float),
                       np.array(side), stop_w, target_w)


ROWS = [(99.0, 99.4, 98.8, 99.3),
        (99.3, 100.2, 99.6, 100.1),   # reaches 100 from below: a long fills at 100.0
        (100.1, 100.6, 100.0, 100.5),  # 100.5: the target; also reaches 100.6, whose stop (100.1) it hits too
        (100.5, 100.7, 99.0, 99.2),
        (99.2, 99.4, 98.9, 99.0)]


def test_a_long_breakout_fills_at_the_level_and_the_entry_minute_counts_its_stop_first():
    s = run(ROWS, [100.0, 100.6], [-1, -1])
    assert s["entry"].tolist() == [1, 2] and s["fill"].tolist() == [100.0, 100.6]
    assert s["reason"].tolist() == [1, -1] and s["exit"].tolist() == pytest.approx([100.5, 100.1])
    # A short below support, touched by a minute that opened through it, fills at that open (99.0, not 99.1).
    short = run(ROWS, [99.1], [1])
    assert short["fill"][0] == 99.0 and short["reason"][0] == -1 and short["exit"][0] == pytest.approx(99.6)


def test_stops_that_gap_fill_at_the_open_and_a_minute_with_both_counts_as_stopped():
    gap = run([(99.7, 100.2, 99.7, 100.1), (99.4, 99.5, 99.2, 99.3)], [100.0], [-1], stop_w=0.4, target_w=0.4)
    assert gap["exit"][0] == pytest.approx(99.4)  # the stop was 99.6, but the minute opened at 99.4
    both = run([(99.7, 100.1, 99.7, 100.0), (100.0, 100.6, 99.4, 100.0)], [100.0], [-1])
    assert both["reason"][0] == -1 and both["exit"][0] == pytest.approx(99.5)


def test_without_a_stop_or_target_the_trade_ends_at_the_last_close_and_untouched_levels_do_not_trade():
    rows = [(99.7, 100.1, 99.7, 100.0), (100.0, 100.2, 99.8, 100.1)]
    s = run(rows, [100.0, 101.0], [-1, -1])
    assert s["reason"][0] == 0 and s["exit"][0] == 100.1 and s["exit_bar"][0] == 1
    assert s["entry"][1] == -1 and np.isnan(s["fill"][1])
    held = run(rows, [100.0], [-1], target_w=None)
    assert held["exit"][0] == 100.1
    late = run(rows, [100.0], [-1], n_touch=0)  # the touch must start inside the window
    assert late["entry"][0] == -1


def test_each_rule_trades_through_its_level_from_the_side_the_session_opened_on():
    real = pd.DataFrame({"level": ["prev_high", "prev_high", "prev_close", "prev_close", "prev_low", "prev_low",
                                   "week_high", "week_low", "pre_high"],
                         "side": [-1, 1, -1, 1, 1, -1, -1, 1, -1]})
    got = bo.rule_levels(real)
    assert sorted(zip(got["level"], got["side"], got["rule"])) == sorted([
        ("prev_high", -1, "yesterday's high"), ("prev_close", -1, "yesterday's close"),
        ("prev_close", 1, "yesterday's close"), ("prev_low", 1, "yesterday's low (check)"),
        ("week_high", -1, "last week's high"), ("week_low", 1, "last week's low (check)")])


def test_costs_come_off_both_sides():
    t = pd.DataFrame({"gross": [10.0, 10.0, -4.0, 6.0], "reason": [1, 1, -1, 0],
                      "session": pd.date_range("2024-01-02", periods=4, freq="B")})
    s = bo.trade_stats(t, 1, years=2.0)
    assert s["mean"] == pytest.approx(3.5) and s["win"] == 75.0 and s["pct_year"] == pytest.approx(0.07)


def make_minutes(sessions, seed=0, pre=40):
    stamps = [pd.date_range(o - pd.Timedelta(minutes=pre), c - pd.Timedelta(minutes=1), freq="min")
              for o, c in zip(sessions["open"], sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.05, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


def test_holdout_sessions_are_traded_only_with_the_final_test():
    sessions = features.trading_sessions("2024-10-01", "2025-02-28", warmup_sessions=0)
    m = make_minutes(sessions, seed=2)
    res = bo.run_study({"SPY": m}, sessions)
    assert res["trades"]["session"].max() < lv.HOLDOUT_START and res["holdout"] is None
    assert set(res["tests"]["rule"]) <= set(bo.TESTED)
    final = bo.run_study({"SPY": m}, sessions, final=True)
    assert final["trades"]["session"].max() >= lv.HOLDOUT_START


def test_the_best_case_ignores_only_the_fill_minutes_stop_touch():
    rows = [(99.7, 100.6, 99.4, 100.3)]  # fills at 100, dips to 99.4 and rises to 100.6 in the same minute
    worst, best = run(rows, [100.0], [-1]), bo.simulate(*bars(rows), 1, np.array([100.0]), np.array([-1]), 0.5,
                                                        0.5, best=True)
    assert worst["exit"][0] == pytest.approx(99.5) and best["exit"][0] == pytest.approx(100.5)
    closed_below = [(99.7, 100.2, 99.3, 99.4)]  # ended beyond the stop after filling at 100: certainly stopped
    cb = bo.simulate(*bars(closed_below), 1, np.array([100.0]), np.array([-1]), 0.5, 0.5, best=True)
    assert cb["reason"][0] == -1 and cb["exit"][0] == pytest.approx(99.5) and cb["exit_bar"][0] == 0
    later = [(99.7, 100.2, 99.4, 100.0), (100.0, 100.1, 99.3, 99.4)]  # a stop touch in a later minute still counts
    b = bo.simulate(*bars(later), 2, np.array([100.0]), np.array([-1]), 0.5, 0.5, best=True)
    assert b["reason"][0] == -1 and b["exit"][0] == pytest.approx(99.5) and b["exit_bar"][0] == 1
