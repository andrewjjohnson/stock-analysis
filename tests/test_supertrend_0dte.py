"""supertrend_0dte.py on SYNTHETIC option bars: the contract for a share trade, the option price at a time (the
bar itself, the next one, a recent earlier one, none), which bar field each exit uses, and the P&L arithmetic."""

import numpy as np
import pandas as pd
import pytest

import alert_spreads as asp
import supertrend_0dte as so

T0 = pd.Timestamp("2025-03-12 14:30", tz="UTC")  # 10:30 ET


def bars(minutes, opens, closes):
    return pd.DataFrame({"ts": [T0 + pd.Timedelta(minutes=k) for k in minutes], "open": opens, "close": closes})


def test_the_contract_is_the_same_day_call_or_put_at_the_nearest_dollar_strike():
    day = pd.Timestamp("2025-03-12")
    assert so.contract(day, 1, 562.49) == "O:SPY250312C00562000"
    assert so.contract(day, -1, 562.50) == "O:SPY250312P00563000"


def test_the_price_at_a_time_falls_back_to_the_next_bar_then_a_recent_earlier_one():
    b = bars([0, 1, 5], [1.00, 1.10, 1.50], [1.05, 1.20, 1.55])
    assert so.price_at(b, T0 + pd.Timedelta(minutes=1), "close") == 1.20   # its own bar
    assert so.price_at(b, T0 + pd.Timedelta(minutes=3), "open") == 1.50    # the next bar, 2 minutes later
    assert so.price_at(b, T0 + pd.Timedelta(minutes=2), "open") == 1.20    # none within 2: the last close before
    assert so.price_at(b, T0 + pd.Timedelta(minutes=12), "open") is None   # the last bar is 7 minutes stale
    assert so.price_at(b.iloc[:0], T0, "open") is None


def test_inside_minute_exits_use_the_bars_close_and_bar_close_exits_the_next_open():
    b = bars(range(10), np.arange(10) * 0.1 + 1.0, np.arange(10) * 0.1 + 1.05)
    trades = pd.DataFrame({"day": pd.Timestamp("2025-03-12"), "dir": [1, -1], "fill": [562.2, 562.2],
                           "entry_time": [T0, T0], "exit_time": [T0 + pd.Timedelta(minutes=4)] * 2,
                           "reason": ["target", "wave"]})
    o = so.option_trades(trades, lambda tk, day: b)
    assert o["opt_entry"].tolist() == [1.0, 1.0]
    assert o["opt_exit"].tolist() == pytest.approx([1.45, 1.40])  # close of minute 4 vs open of minute 4
    o["day"] = trades["day"]
    st = so.stats(o.assign(entry_time=trades["entry_time"]), 0.02, years=1.0)
    assert st["mean"] == pytest.approx(100 * ((0.45 - 0.04) + (0.40 - 0.04)) / 2)


def test_trades_outside_the_plan_are_counted_not_priced():
    trades = pd.DataFrame({"day": pd.Timestamp("2024-09-30"), "dir": [1], "fill": [570.0], "entry_time": [T0],
                           "exit_time": [T0], "reason": ["time"]})

    def denied(tk, day):
        raise asp.NotInPlan(tk)

    assert so.option_trades(trades, denied)["opt_status"].tolist() == ["outside the data plan"]
