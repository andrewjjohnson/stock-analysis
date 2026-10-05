"""green_goose.py on SYNTHETIC data: the direction rules and their priority, the 15:50 entry and next-morning prices
(with a dividend), Black-76 delta and theta, the contract choice, and both versions' option exits."""

import math

import numpy as np
import pandas as pd
import pytest

import call_spreads as cs
import features
import green_goose as gg

QUIET = (50, 20, 30, 25)  # RSI(2), ADX, +DI, -DI: ADX below both DI lines, RSI above both


def test_rsi_extremes_then_the_candle():
    assert gg.decide((90, 20, 30, 25), QUIET, -1.0) == (-1, "RSI(2) above 85")
    assert gg.decide((10, 20, 30, 25), (12, 20, 30, 25), 1.0) == (1, "RSI(2) below 15")
    assert gg.decide((50, 20, 30, 25), QUIET, 0.4) == (1, "candle up")
    assert gg.decide((50, 20, 30, 25), QUIET, -0.4) == (-1, "candle down")


def test_overrides_in_priority_and_the_adx_veto():
    # ADX moves between the DI lines (it was below both yesterday); -DI on top -> puts, even with RSI(2) < 15.
    assert gg.decide((10, 27, 25, 30), QUIET, 1.0) == (-1, "ADX entered the DI zone")
    # RSI(2) falls from above the zone (yesterday 50 > 30) into it (28 <= 30) -> calls.
    assert gg.decide((28, 20, 30, 25), QUIET, -1.0) == (1, "RSI(2) stabbed the zone from above")
    # RSI(2) rises from below the zone (yesterday 20 < 25) through it -> puts.
    assert gg.decide((40, 20, 30, 25), (20, 20, 30, 25), 1.0) == (-1, "RSI(2) stabbed the zone from below")
    # Above both lines yesterday, below both today: straight through, still a stab from above -> calls.
    assert gg.decide((20, 20, 30, 25), QUIET, -1.0) == (1, "RSI(2) stabbed the zone from above")
    assert gg.decide((90, 65, 30, 25), QUIET, 1.0) == (0, "ADX above 60")


def make_minutes(sessions, seed=0):
    stamps = [pd.date_range(o, c - pd.Timedelta(minutes=1), freq="min") for o, c in zip(sessions["open"],
                                                                                      sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.arange(len(ts)) * 0.001
    return pd.DataFrame({"ts": ts, "open": close - 0.0005, "high": close + 0.01, "low": close - 0.01,
                         "close": close, "volume": 1000.0})


def test_morning_moves_use_the_1550_price_less_a_dividend_paid_the_next_morning():
    sessions = features.trading_sessions("2024-03-04", "2024-03-06", warmup_sessions=0)
    m = make_minutes(sessions)
    rth, _ = features.regular_session_minutes(m, sessions)
    divs = pd.DataFrame({"ex_date": [pd.Timestamp("2024-03-05")], "cash_amount": [0.5]})
    mv = gg.morning_moves(rth, sessions, divs)
    day1 = rth[rth["session"] == "2024-03-04"]
    assert mv.loc["2024-03-04", "entry"] == pytest.approx(day1["close"].iloc[379] - 0.5)  # the 15:49 bar, less $0.50
    day2 = rth[rth["session"] == "2024-03-05"]
    assert mv.loc["2024-03-04", "open"] == day2["open"].iloc[0]
    assert mv.loc["2024-03-04", "9:35"] == day2["close"].iloc[4]  # the 9:34 bar's close
    assert mv.loc["2024-03-05", "entry"] == pytest.approx(day2["close"].iloc[379])  # no dividend on 03-06


def test_black76_delta_and_theta():
    delta, theta = gg.b76(600.0, 600.0, 0.15, 7 / 365)
    assert 0.5 < delta < 0.51 and theta == pytest.approx(-600 * 0.3989 * 0.15 / (2 * math.sqrt(7 / 365)) / 365,
                                                          rel=1e-3)


T0 = pd.Timestamp("2025-03-13 13:30", tz="UTC")  # 9:30 ET


def bars(rows, start=T0):
    return pd.DataFrame([(start + pd.Timedelta(minutes=k), o, h, l, c) for k, o, h, l, c in rows],
                        columns=["ts", "open", "high", "low", "close"])


def test_version_two_sells_at_935_and_version_one_follows_the_opening_price():
    o_ns = T0.value
    b = bars([(0, 2.0, 2.1, 1.9, 2.0), (4, 2.2, 2.3, 2.1, 2.25), (10, 2.4, 2.5, 2.3, 2.4)])
    assert gg.exit_v2(b, o_ns) == 2.25
    close_ns = (T0 + pd.Timedelta(hours=6, minutes=30)).value
    eod = bars([(0, 0.5, 0.5, 0.4, 0.45), (389, 0.6, 0.6, 0.55, 0.58)])
    assert gg.exit_v1(eod, o_ns, close_ns, paid=1.0) == (0.58, "down over 40%: held to the close")
    cut = bars([(0, 0.8, 0.85, 0.75, 0.8), (5, 0.9, 0.9, 0.85, 0.88)])
    assert gg.exit_v1(cut, o_ns, close_ns, paid=1.0)[0] == 0.8  # down 20% at the open: out
    trail = bars([(0, 1.1, 1.3, 1.1, 1.25), (1, 1.25, 1.26, 1.15, 1.2)])  # peak 1.3 -> stop 1.17 hit in minute 1
    assert gg.exit_v1(trail, o_ns, close_ns, paid=1.0)[0] == pytest.approx(1.17)
    big = bars([(0, 2.5, 2.6, 2.5, 2.55), (1, 2.5, 2.5, 2.2, 2.3)])  # +150%: 55% at 2.5, rest stopped at 2.34
    px, how = gg.exit_v1(big, o_ns, close_ns, paid=1.0)
    assert px == pytest.approx(0.55 * 2.5 + 0.45 * 0.9 * 2.6) and how.startswith("up 100%+")


def test_the_contract_is_the_strike_in_the_delta_band():
    sessions = features.trading_sessions("2025-03-10", "2025-03-21", warmup_sessions=0)
    day, expiry = pd.Timestamp("2025-03-12"), pd.Timestamp("2025-03-17")
    dec = sessions.loc[day, "close"] - gg.DECISION
    tau = (sessions.loc[expiry, "close"] - dec).total_seconds() / (365 * 86400)
    F, sigma = 560.3, 0.16

    def load(tk, d):  # Black-76 prices for every strike, one bar at 15:49
        k = int(tk[-8:]) / 1000
        call = float(cs.b76_call(F, k, sigma, tau))
        px = call if tk[-9] == "C" else call - F + k
        return pd.DataFrame({"ts": [dec - pd.Timedelta(minutes=1)], "open": [px], "high": [px], "low": [px],
                             "close": [px]})

    pick, why = gg.choose_contract(day, "C", 560.2, expiry, sessions, load)
    assert why == "ok" and pick[1] == 560 and 0.47 <= pick[2] <= 0.53 and pick[3] < -0.12
