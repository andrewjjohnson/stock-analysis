"""close_momentum.py on SYNTHETIC minute bars: the day's move to 15:30 uses only prices up to 15:30 (no lookahead),
trades go with the move and the big-minus-small difference is measured on the same footing."""

import numpy as np
import pandas as pd

import close_momentum as cm
import features


def minutes_for(sessions, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for day, r in sessions.iterrows():
        ts = pd.date_range(r["open"], r["close"] - pd.Timedelta(minutes=1), freq="1min")
        steps = rng.normal(0, 0.0008, len(ts))
        p = price * np.exp(np.cumsum(steps))
        rows.append(pd.DataFrame({"ts": ts, "open": p, "high": p * 1.0002, "low": p * 0.9998, "close": p,
                                  "volume": 1000.0}))
        price = p[-1]
    return pd.concat(rows, ignore_index=True)


def test_the_move_to_1530_and_the_trade_use_only_prices_up_to_their_times():
    sessions = features.trading_sessions("2024-03-01", "2024-04-30", warmup_sessions=0)
    m = minutes_for(sessions)
    rth, _ = features.regular_session_minutes(m, sessions)
    divs = pd.DataFrame(columns=["ex_date", "cash_amount"])
    d = cm.day_table(rth, sessions, divs)
    day = sessions.index[30]
    late = m.copy()
    after = late["ts"] >= sessions.loc[day, "close"] - pd.Timedelta(minutes=30)
    late.loc[after & (late["ts"] < sessions.loc[day, "close"]), ["open", "high", "low", "close"]] *= 1.05
    d2 = cm.day_table(features.regular_session_minutes(late, sessions)[0], sessions, divs)
    for col in ("move_m30", "sd20", "prev", "p_m30"):
        assert np.isclose(d.at[day, col], d2.at[day, col], equal_nan=True)
    assert d2.at[day, "last30"] > d.at[day, "last30"]


def test_trades_go_with_the_move_and_only_on_big_days():
    days = pd.bdate_range("2024-01-02", periods=4)
    d = pd.DataFrame({"move_m30": [150.0, -120.0, 40.0, -300.0], "last30": [5.0, 5.0, 9.0, -8.0],
                      "sd20": [100.0] * 4}, index=days)
    t = cm.trades(d, "QQQ")
    assert list(t["gross"]) == [5.0, -5.0, 8.0]
    assert list(cm.trades(d, "QQQ", lo=200.0)["gross"]) == [8.0]
    assert list(cm.trades(d, "QQQ", units=1.5)["gross"]) == [5.0, 8.0]   # 150 bps is exactly 1.5 sd


def test_big_minus_small_difference_is_the_difference_of_means():
    s = pd.bdate_range("2024-01-02", periods=6)
    a = pd.DataFrame({"session": s[:3], "gross": [10.0, 12.0, 14.0]})
    b = pd.DataFrame({"session": s[3:], "gross": [1.0, 2.0, 3.0]})
    r = cm.difference(a, b)
    assert np.isclose(r["mean"], 10.0) and r["lo"] < 10.0 < r["hi"]
