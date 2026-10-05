"""dip_spreads.py on SYNTHETIC option bars: strikes, the 15:50 entry, a path across sessions (with an
early close), take profit vs settlement, one-at-a-time trading and the excess statistic."""

import numpy as np
import pandas as pd
import pytest

import dip_spreads as ds
import features
import synthetic

NY = "America/New_York"


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def leg(sessions, value, overrides=None, skip=()):
    """Minute bars over `sessions` with open = close = value, except {'YYYY-MM-DD HH:MM': value} overrides."""
    ts = synthetic.session_minute_starts(sessions)
    stamp = ts.tz_convert(NY).strftime("%Y-%m-%d %H:%M")
    v = np.array([(overrides or {}).get(x, value) for x in stamp])
    df = pd.DataFrame({"ts": ts, "open": v, "high": v, "low": v, "close": v, "volume": 1.0})
    return df[~np.isin(stamp, list(skip))].reset_index(drop=True)


def setup(short_over=None, long_over=None, long_skip=(), expiry_close=750.0):
    sessions = features.trading_sessions("2024-11-26", "2024-11-29", warmup_sessions=0)  # 11-29 closes at 13:00
    days = sessions.index
    raw = pd.DataFrame({"snap": [760.50, 755.0, 758.0], "close": [760.0, 756.0, expiry_close]}, index=days)
    bars = {"short": leg(sessions, 1.20, short_over), "long": leg(sessions, 0.40, long_over, long_skip)}
    load = lambda ticker, start, end: bars["short" if ticker.endswith("752000") else "long"]  # noqa: E731
    trade = ds.spread_trade(0, days, sessions, raw, 2, 1.0, 5, load)
    return trade, days


def test_strikes_sit_at_or_below_the_target_distance():
    assert ds.put_strikes(760.50, 1.0, 5) == (752, 747)  # 1% below 760.50 = 752.895 -> 752
    assert ds.put_strikes(760.50, 0.0, 1) == (760, 759)
    assert ds.put_strikes(752.00, 0.0, 5) == (752, 747)  # exactly on a strike: that strike
    assert ds.put_strikes(760.50, ds.ITM, 1) == (761, 760)  # just in the money, as in the alert spreads
    assert ds.put_strikes(760.50, 0.5, 1) == (756, 755)  # 0.5% below 760.50 = 756.70 -> 756


def test_entry_waits_for_both_legs_from_1550_and_the_path_runs_to_the_expiry_close():
    trade, days = setup(long_skip=("2024-11-26 15:50", "2024-11-26 15:51"))
    assert trade["status"] == "ok" and trade["short_ticker"] == "O:SPY241129P00752000"
    assert pd.Timestamp(trade["entry_time"], tz="UTC") == et("2024-11-26 15:52")  # first minute both legs printed
    assert trade["credit_traded"] == pytest.approx(0.80)
    # Watch minutes: 15:52-15:59 on 11-26, the full 11-27, and 11-29 up to its 13:00 early close.
    assert len(trade["watch"]) == 8 + 390 + 210
    assert pd.Timestamp(trade["watch_days"][-1]) == days[-1]
    assert trade["settle"] == pytest.approx(2.0)  # SPY closed at 750 on expiry: 752 - 750, within the $5 width


def test_take_profit_fills_at_its_level_otherwise_the_spread_settles_at_intrinsic_value():
    trade, days = setup(short_over={"2024-11-27 11:00": 0.75}, expiry_close=760.0)  # spread 0.35 at 11:00
    pnl, reason, exit_day, credit = ds.simulate(trade, 50, 0.0, 0.0)
    assert (reason, exit_day, pnl) == ("take profit", days[1], pytest.approx((0.80 - 0.40) * 100))
    pnl, reason, exit_day, _ = ds.simulate(trade, "expiry", 0.0, 0.0)
    assert (reason, exit_day, pnl) == ("expiry", days[-1], pytest.approx(80.0))  # expired worthless: keep it all
    # With $0.01 slippage the 50% level (0.39) needs a print at or below 0.37: 0.35 still qualifies.
    pnl, reason, _, credit = ds.simulate(trade, 50, 0.01, 0.0)
    assert reason == "take profit" and credit == pytest.approx(0.78) and pnl == pytest.approx(39.0)
    loser, _ = setup(expiry_close=700.0)
    assert ds.simulate(loser, "expiry", 0.0, 0.0)[0] == pytest.approx((0.80 - 5.0) * 100)  # max loss


def test_one_at_a_time_skips_signals_while_a_spread_is_open():
    days = pd.bdate_range("2024-03-04", periods=6)
    rows = pd.DataFrame({"pnl": [10, 20, 30, 40, 50, 60.0], "max_loss": 100.0,
                         "exit_day": [days[2], days[3], days[4], days[5], days[5], days[5]]}, index=days)
    taken = ds.one_at_a_time(days, [True, True, True, True, False, True], rows)
    assert taken["pnl"].tolist() == [10, 40]  # day 0 open until day 2; next allowed day 3, open until day 5


def test_signal_stats_measure_the_excess_over_every_day():
    pnl = np.array([10.0, -50, 20, 30, -10, 40, 5, 5, 5, 5])
    mask = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0], bool)
    out = ds.signal_stats(mask, pnl, np.full(10, 100.0), np.arange(10)[None, :].repeat(5, 0))
    assert out["n"] == 4 and out["mean"] == pytest.approx(25.0) and out["base_mean"] == pytest.approx(6.0)
    assert out["excess"] == pytest.approx(19.0) and out["win"] == 100 and out["ror"] == pytest.approx(25.0)


def test_breakeven_stop_arms_at_40pct_profit_and_exits_at_the_close_that_crosses_back():
    days = pd.bdate_range("2024-03-04", periods=4)

    def trade(watch, settle=3.0):
        return {"credit_traded": 1.0, "width": 5, "watch": np.array(watch, float), "settle": settle,
                "expiry": days[-1], "watch_days": np.array(days[[0, 1, 1, 2, 3, 3][:len(watch)]],
                                                           dtype="datetime64[ns]")}

    # Profit reaches 45% (worth 0.55), then the spread jumps to 1.04: the stop fills at 1.04, a $4 loss,
    # instead of riding to the $300 loss at expiry.
    path = trade([0.90, 0.55, 0.70, 1.04, 2.0, 3.0])
    pnl, reason, day, _ = ds.simulate(path, "expiry", 0.0, 0.0)
    assert (reason, day) == ("expiry", days[-1]) and pnl == pytest.approx(-200.0)
    pnl, reason, day, _ = ds.simulate(path, "expiry", 0.0, 0.0, breakeven_at=40)
    assert (reason, day) == ("breakeven stop", days[2]) and pnl == pytest.approx(-4.0)
    assert ds.simulate(path, 80, 0.0, 0.0, breakeven_at=40)[1] == "breakeven stop"  # 80% (0.20) never reached
    # Never armed (best was 0.65, a 35% profit): the stop does nothing.
    assert ds.simulate(trade([0.65, 1.2, 2.0]), "expiry", 0.0, 0.0, 40)[1] == "expiry"
    # The take profit comes first when it is reached before the spread crosses back.
    pnl, reason, _, _ = ds.simulate(trade([0.55, 0.18, 1.1]), 80, 0.0, 0.0, 40)
    assert reason == "take profit" and pnl == pytest.approx(80.0)
    # Armed, never crossed back: held to expiry as usual (expired worthless here).
    pnl, reason, _, _ = ds.simulate(trade([0.50, 0.40], settle=0.0), "expiry", 0.0, 0.0, 40)
    assert reason == "expiry" and pnl == pytest.approx(100.0)


def ns(text):
    return et(text).value


def test_first_up_day_exit_closes_at_the_first_both_legs_minute_from_its_time():
    # Spread 0.80 at entry (11-26); on 11-27 the short leg is 0.75 from 15:50, so the spread is worth 0.35 then.
    trade, days = setup(short_over={f"2024-11-27 15:{m:02d}": 0.75 for m in range(50, 60)})
    pnl, reason, day, _ = ds.simulate(trade, "expiry", 0.0, 0.0, exit_at=ns("2024-11-27 15:50"))
    assert (reason, day) == ("first up day", days[1]) and pnl == pytest.approx((0.80 - 0.35) * 100)
    pnl, _, _, _ = ds.simulate(trade, "expiry", 0.01, 0.0, exit_at=ns("2024-11-27 15:50"))
    assert pnl == pytest.approx((0.78 - 0.35 - 0.02) * 100)  # slippage on the way in and out
    # A take profit reached earlier still wins.
    early, _ = setup(short_over={"2024-11-27 11:00": 0.75})
    assert ds.simulate(early, 50, 0.0, 0.0, exit_at=ns("2024-11-27 15:50"))[1] == "take profit"
    # No both-legs print from 15:50 to the close: the exit waits for the next one (the next morning).
    quiet, _ = setup(long_skip=tuple(f"2024-11-27 15:{m:02d}" for m in range(50, 60)))
    pnl, reason, day, _ = ds.simulate(quiet, "expiry", 0.0, 0.0, exit_at=ns("2024-11-27 15:50"))
    assert reason == "first up day, next print" and day == days[2]
    # An exit time after the expiry: the spread settles as usual.
    assert ds.simulate(trade, "expiry", 0.0, 0.0, exit_at=ns("2024-12-02 15:50"))[1] == "expiry"


def test_first_up_exits_wait_for_an_up_day_or_the_fifth_session():
    sessions = features.trading_sessions("2024-03-04", "2024-03-15", warmup_sessions=0)
    days = sessions.index
    feats = pd.DataFrame({"ret_1d": [-1, -1, 0.5, -1, -1, -1, -1, -1, -1, -1.0]}, index=days)
    exits = ds.first_up_exits(feats, days, sessions)
    decision = lambda d: (sessions.loc[d, "close"] - pd.Timedelta(minutes=10)).value  # noqa: E731
    assert exits[days[0]] == decision(days[2])  # the first later session that is up
    assert exits[days[3]] == decision(days[8])  # no up day: the 5th session after entry
    assert days[9] not in exits  # nothing after the last session
