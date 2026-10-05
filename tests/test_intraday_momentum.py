"""intraday_momentum.py on SYNTHETIC minutes: prices at fixed times, the morning / half-hour / last-half-hour moves
(dividend and early close included), the signals and the statistics."""

import numpy as np
import pandas as pd
import pytest

import features
import intraday_momentum as im
import synthetic


def minutes(sessions, closes_by_day):
    ts = synthetic.session_minute_starts(sessions)
    day = ts.tz_convert("America/New_York").tz_localize(None).normalize()
    frames = [synthetic.bars_from_closes(ts[day == d], closes_by_day[str(d.date())]) for d in sessions.index]
    rth, _ = features.regular_session_minutes(pd.concat(frames, ignore_index=True), sessions)
    return rth


def test_moves_use_the_bars_ending_at_10_00_and_30_and_60_minutes_before_the_close():
    # Thanksgiving (2024-11-28) is closed and 2024-11-29 closes early (13:00), so its "last half hour" is 12:30-13:00.
    sessions = features.trading_sessions("2024-11-27", "2024-11-29", warmup_sessions=0)
    day1 = np.full(390, 100.0)
    day3 = np.r_[np.full(30, 101.0), np.full(150, 102.0), np.full(30, 103.0)]  # 210 minutes, early close
    rth = minutes(sessions, {"2024-11-27": day1, "2024-11-29": day3})
    divs = pd.DataFrame({"ex_date": [pd.Timestamp("2024-11-29")], "cash_amount": [0.5]})
    d = im.day_table(rth, sessions, divs)
    row = d.loc["2024-11-29"]
    assert row["p_10"] == 101.0 and row["p_m60"] == 102.0 and row["p_m30"] == 102.0 and row["p_close"] == 103.0
    assert row["r_morning"] == pytest.approx((101 / (100 - 0.5) - 1) * 1e4)  # the payout is not a move
    assert row["r_last"] == pytest.approx((103 / 102 - 1) * 1e4) and row["r_late"] == pytest.approx(0.0)


def test_a_missing_bar_leaves_the_price_unknown():
    sessions = features.trading_sessions("2024-03-05", "2024-03-05", warmup_sessions=0)
    rth = minutes(sessions, {"2024-03-05": np.full(390, 50.0)})
    rth = rth[rth["ts"] != sessions["open"].iloc[0] + pd.Timedelta(minutes=29)]
    assert np.isnan(im.price_at(rth, sessions["open"] + pd.Timedelta(minutes=30))[0])


def test_signals_trade_with_the_morning_the_half_hour_or_both():
    d = pd.DataFrame({"r_morning": [10.0, -5, 3, 0], "r_late": [2.0, 4, -1, 6], "r_last": [8.0, -6, 5, 7]})
    assert im.signal_outcomes(d, "morning").tolist()[:3] == [8.0, 6.0, 5.0] and np.isnan(
        im.signal_outcomes(d, "morning").iloc[3])
    assert im.signal_outcomes(d, "late").tolist() == [8.0, -6.0, -5.0, 7.0]
    both = im.signal_outcomes(d, "both")
    assert both.iloc[0] == 8.0 and both.iloc[1:].isna().all()  # only the first day agrees


def test_statistics_and_slope():
    s = im.stats(pd.Series([1.0, 2.0, 3.0, -1.0]))
    assert s["days"] == 4 and s["avg"] == pytest.approx(1.25) and s["hit"] == 75.0
    assert s["net"] == pytest.approx(1.25 - im.COST_BPS)
    x = np.linspace(-50, 50, 41)
    noise = np.tile([0.5, -0.5], 21)[:41]
    b, t = im.slope(pd.DataFrame({"r_morning": x, "r_last": 0.1 * x + noise}))
    assert b == pytest.approx(0.1, abs=0.01) and t > 10
