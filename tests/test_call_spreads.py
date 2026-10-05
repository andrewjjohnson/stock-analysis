"""call_spreads.py on SYNTHETIC data: Black-76 implied volatility and delta, choosing the expiry, last prints,
call-spread settlement and the exit rules."""

import numpy as np
import pandas as pd
import pytest

import call_spreads as cs
import dip_spreads as ds
import features
import synthetic

NY = "America/New_York"


def test_implied_volatility_and_delta_round_trip():
    F, tau = 600.0, 40 / 365.25
    for K, sigma in ((600.0, 0.15), (620.0, 0.12), (585.0, 0.20)):
        price = cs.b76_call(F, K, sigma, tau)
        assert cs.implied_vol(price, F, K, tau) == pytest.approx(sigma, abs=1e-6)
    K = cs.strike_for_delta(F, 0.15, tau, 0.30)
    assert K > F and cs.call_delta(F, K, 0.15, tau) == pytest.approx(0.30, abs=1e-9)
    assert np.isnan(cs.implied_vol(0.5, F, 580.0, tau))  # below intrinsic value: no volatility fits


def test_expiry_is_the_weekly_closest_to_the_target_and_beyond_21_days():
    day = pd.Timestamp("2025-03-03")  # a Monday
    expiries = [pd.Timestamp("2025-03-21"), pd.Timestamp("2025-03-28"), pd.Timestamp("2025-04-04"),
                pd.Timestamp("2025-04-11"), pd.Timestamp("2025-04-17"), pd.Timestamp("2025-04-25")]
    assert cs.pick_expiry(day, expiries, pd.Timestamp("2026-01-01"), 30) == pd.Timestamp("2025-04-04")  # 32 days
    assert cs.pick_expiry(day, expiries, pd.Timestamp("2026-01-01"), 45) == pd.Timestamp("2025-04-17")  # 45 days
    assert cs.pick_expiry(day, expiries[:1], pd.Timestamp("2026-01-01"), 30) is None  # 18 days: too close
    assert cs.pick_expiry(day, expiries, pd.Timestamp("2025-04-01"), 45) == pd.Timestamp("2025-03-28")  # data end


def test_last_print_is_the_last_trade_inside_the_window():
    ts = pd.date_range("2025-03-03 19:50", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"ts": ts, "close": [1.0, 1.1, 1.2, 1.3, 1.4]})
    start, end = ts[1].value, ts[3].value
    assert cs.last_print(df, start, end) == 1.2  # bars starting before `end`
    assert np.isnan(cs.last_print(df, ts[4].value + 60 * 10**9, ts[4].value + 120 * 10**9))


def leg(sessions, value, overrides=None):
    ts = synthetic.session_minute_starts(sessions)
    stamp = ts.tz_convert(NY).strftime("%Y-%m-%d %H:%M")
    v = np.array([(overrides or {}).get(x, value) for x in stamp])
    return pd.DataFrame({"ts": ts, "open": v, "high": v, "low": v, "close": v, "volume": 1.0})


def call_trade(expiry_close, short_over=None):
    sessions = features.trading_sessions("2025-03-03", "2025-03-07", warmup_sessions=0)
    days = sessions.index
    raw = pd.DataFrame({"snap": 600.0, "close": [600.0, 601, 602, 603, expiry_close]}, index=days)
    bars = {"short": leg(sessions, 2.00, short_over), "long": leg(sessions, 0.80)}
    trade = {"day": days[0], "expiry": days[-1], "width": 5.0, "short_strike": 610.0, "long_strike": 615.0, "right": "C",
             "short_ticker": "S", "long_ticker": "L"}
    tr = ds.price_spread(trade, "2025-03-03", sessions, raw, lambda t, a, b: bars["short" if t == "S" else "long"])
    return tr, sessions


def test_a_call_spread_settles_on_how_far_spy_closed_above_the_short_strike():
    tr, _ = call_trade(612.0)
    assert tr["status"] == "ok" and tr["credit_traded"] == pytest.approx(1.20)
    assert tr["settle"] == pytest.approx(2.0)  # 612 - 610
    assert call_trade(630.0)[0]["settle"] == pytest.approx(5.0)  # capped at the width
    assert call_trade(605.0)[0]["settle"] == 0.0


def test_exit_rules_take_profit_stop_days_and_expiry():
    # Spread 1.20 at entry; worth 0.50 on 03-04 at 11:00 (short 1.30), 3.80 on 03-05 at 11:00 (short 4.60).
    over = {"2025-03-04 11:00": 1.30, "2025-03-05 11:00": 4.60}
    tr, sessions = call_trade(600.0, over)
    pnl, reason, day = cs.simulate(tr, {"tp": 50}, 0.0, sessions)
    assert reason == "take profit" and pnl == pytest.approx(0.60 * 100) and day == sessions.index[1]
    stop_only = call_trade(600.0, {"2025-03-05 11:00": 4.60})[0]
    pnl, reason, day = cs.simulate(stop_only, {"tp": 50, "stop": 2.0}, 0.0, sessions)
    assert reason == "stop" and pnl == pytest.approx((1.20 - 3.80) * 100)  # 3.80 >= 3 x 1.20 = 3.60
    pnl, reason, _ = cs.simulate(tr, {}, 0.0, sessions)
    assert reason == "expiry" and pnl == pytest.approx(120.0)  # SPY 600 at expiry: worthless, keep the credit
    tr["expiry"] = pd.Timestamp("2025-03-28")  # pretend the expiry is 25 days out: 21 days left from 03-07
    pnl, reason, day = cs.simulate(tr, {"days": 21}, 0.0, sessions)
    assert reason == "21 days" and day == sessions.index[4] and pnl == pytest.approx(0.0)  # closed at 1.20


def test_a_working_order_can_fill_the_next_morning_but_not_after_10_30():
    sessions = features.trading_sessions("2025-03-03", "2025-03-07", warmup_sessions=0)
    days = sessions.index
    raw = pd.DataFrame({"snap": 600.0, "close": 600.0}, index=days)
    quiet = [f"2025-03-03 {h:02d}:{m:02d}" for h in (15,) for m in range(50, 60)]
    ts = synthetic.session_minute_starts(sessions)
    stamp = ts.tz_convert(NY).strftime("%Y-%m-%d %H:%M")
    long_leg = leg(sessions, 0.80)
    long_leg = long_leg[~np.isin(stamp, quiet)].reset_index(drop=True)  # the long leg is silent after 15:50 on day 1

    def trade(until):
        t = {"day": days[0], "expiry": days[-1], "width": 5.0, "short_strike": 610.0, "long_strike": 615.0,
             "right": "C", "short_ticker": "S", "long_ticker": "L", "entry_until": until}
        bars = {"S": leg(sessions, 2.00), "L": long_leg}
        return ds.price_spread(t, "2025-03-03", sessions, raw, lambda k, a, b: bars[k])

    next_open = sessions.loc[days[1], "open"]
    tr = trade((next_open + pd.Timedelta(minutes=60)).value)
    assert tr["status"] == "ok" and pd.Timestamp(tr["entry_time"], tz="UTC") == next_open  # 9:30 the next day
    assert trade(None)["status"].startswith("no entry")  # without a working order: only 15:50 to the close
