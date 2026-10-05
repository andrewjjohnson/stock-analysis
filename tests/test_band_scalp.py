"""band_scalp.py on SYNTHETIC data: setups (pierce, re-entry, lapse, sessions, wick), indicators that later prices
cannot change, order-flow flags, the trade simulation, and the previous session's volume-profile levels."""

import numpy as np
import pandas as pd
import pytest

import band_scalp as bs
import features


def bar_frame(lows, highs, closes, lower, upper, rsi, sessions=None):
    n = len(lows)
    return pd.DataFrame({"session": sessions if sessions is not None else [pd.Timestamp("2024-03-04")] * n,
                         "low": lows, "high": highs, "close": closes, "lower": lower, "upper": upper, "rsi": rsi})


def test_a_pierce_with_rsi_then_a_close_back_inside_makes_one_setup():
    #            0     1     2     3     4     5
    lows = [100.0, 98.0, 97.5, 99.2, 99.5, 99.8]
    closes = [100.5, 98.5, 98.0, 99.6, 100.0, 100.2]
    b = bar_frame(lows, [c + 0.5 for c in closes], closes, [99.0] * 6, [102.0] * 6, [50, 25, 22, 35, 40, 45])
    st = bs.find_setups(b)
    assert len(st) == 1  # bar 2 pierces again but belongs to the same setup
    r = st.iloc[0]
    assert (r.side, r.p, r.c, r.w, r.wick) == (1, 1, 3, 2, 97.5)
    b2 = b.assign(rsi=[50, 35, 35, 35, 40, 45])  # RSI never at or below 30
    assert bs.find_setups(b2).empty


def test_a_setup_lapses_after_six_bars_and_never_crosses_sessions():
    n = 9
    lows = [98.0] * n
    closes = [98.5] * 8 + [99.5]  # closes outside for 7 bars after the pierce, back inside on bar 8
    b = bar_frame(lows, [c + 0.5 for c in closes], closes, [99.0] * n, [102.0] * n, [25] * n)
    st = bs.find_setups(b)
    assert len(st) == 1 and st.iloc[0].p == 7 and st.iloc[0].c == 8  # bar 0's setup lapsed; bar 7 starts a new one
    days = [pd.Timestamp("2024-03-04")] * 2 + [pd.Timestamp("2024-03-05")] * 2
    b3 = bar_frame([98.0, 98.0, 100.0, 100.0], [99.0] * 4, [98.5, 98.5, 100.0, 100.0], [99.0] * 4, [102.0] * 4,
                   [25, 25, 50, 50], sessions=days)
    assert bs.find_setups(b3).empty  # the close back inside comes the next session


def test_the_short_mirrors_the_long():
    highs = [100.0, 102.5, 101.0]
    closes = [99.5, 102.2, 100.8]
    b = bar_frame([c - 0.5 for c in closes], highs, closes, [97.0] * 3, [102.0] * 3, [50, 75, 60])
    r = bs.find_setups(b).iloc[0]
    assert (r.side, r.p, r.c, r.wick) == (-1, 1, 2, 102.5)


def make_minutes(sessions, seed=0):
    stamps = [pd.date_range(o, c - pd.Timedelta(minutes=1), freq="min") for o, c in zip(sessions["open"],
                                                                                      sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.0007, len(ts))))
    return pd.DataFrame({"ts": ts, "open": close * 0.9999, "high": close * 1.0005, "low": close * 0.9995,
                         "close": close, "volume": 1000.0})


def test_bands_rsi_and_the_hourly_anchor_ignore_later_prices():
    sessions = features.trading_sessions("2024-01-02", "2024-03-28", warmup_sessions=0)
    m = make_minutes(sessions, 1)
    rth, _ = features.regular_session_minutes(m, sessions)
    cut = rth["ts"].iloc[len(rth) - 2000]
    b1 = bs.five_minute_bars(rth)
    rth2 = rth.copy()
    later = rth2["ts"] >= cut
    rth2.loc[later, ["open", "high", "low", "close"]] *= 1.2
    b2 = bs.five_minute_bars(rth2)
    early = b1["bar_end"] <= cut
    assert b1.loc[early, "anchor"].notna().sum() > 100
    cols = ["upper", "middle", "lower", "rsi", "width", "expansion", "anchor"]
    pd.testing.assert_frame_equal(b1.loc[early, cols], b2.loc[early, cols])


def test_bar_deltas_and_the_order_flow_flags():
    # seconds 0..5: close 10, 10.1 (buy 100), 10.1 (still buying, 50), 10.0 (sell 30), 9.9 (sell 20), 10.0 (buy 5)
    sec = np.array([(k * 1000, 0, 0, 0, c, v) for k, (c, v) in
                    enumerate([(10.0, 999), (10.1, 100), (10.1, 50), (10.0, 30), (9.9, 20), (10.0, 5)])], float)
    d = bs.bar_deltas(sec, np.array([0, 3000, 6000]), open_ms=0)  # the first second only sets the price
    assert d.tolist() == pytest.approx([150.0, -45.0])
    # long: lookback bars 0-1 (earlier low 98.5 at bar 1), pierce bar 2, wick bar 3 (97.0), entry bar 4
    lo = np.array([99.0, 98.5, 98.0, 97.0, 98.8])
    hi = lo + 1.0
    cl = np.array([99.5, 99.0, 98.2, 97.8, 99.5])  # the wick bar closes in the upper half of its range
    flags = lambda deltas: bs.cvd_flags(1, lo, hi, cl, np.array(deltas, float), p=2, c=4, w=3)  # noqa: E731
    assert flags([-50, -80, 40, 10, 20]) == (True, False, False)     # lower low, cumulative delta -80 above -130
    assert flags([-50, -80, -40, -60, 5]) == (False, True, False)    # sellers hit the wick bar, it held its upper half
    assert flags([-50, -80, -40, -60, -90])[2]                       # the entry bar sells harder than the pierce bar


def minute_frame(day, rows):
    """rows: (minute after 9:30, open, high, low, close)."""
    t0 = pd.Timestamp(f"{day} 09:30", tz=features.NY).tz_convert("UTC")
    return pd.DataFrame({"ts": [t0 + pd.Timedelta(minutes=k) for k, *_ in rows], "session": pd.Timestamp(day),
                         "open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
                         "close": [r[4] for r in rows]})


def setup_row(day, side, wick, middle, entry_minute):
    t0 = pd.Timestamp(f"{day} 09:30", tz=features.NY).tz_convert("UTC")
    return {"side": side, "wick": wick, "middle": middle, "session": pd.Timestamp(day),
            "entry_bar_end": t0 + pd.Timedelta(minutes=entry_minute)}


def test_trades_hit_the_target_the_stop_or_time_out():
    day = "2024-03-05"
    rth = minute_frame(day, [(0, 100.0, 100.1, 99.9, 100.0), (1, 100.0, 100.2, 99.95, 100.1),
                             (2, 100.1, 100.6, 100.0, 100.5), (3, 100.5, 100.6, 100.4, 100.5)])
    win = bs.simulate(pd.DataFrame([setup_row(day, 1, 99.5, 100.5, 1)]), rth).iloc[0]
    assert win.exit_reason == "target" and win.entry == 100.0 and win.exit == 100.5
    assert win.gross == pytest.approx(50.0) and win.stop == pytest.approx(99.5 * (1 - bs.STOP_BUFFER))
    both = minute_frame(day, [(0, 100.0, 100.6, 99.4, 100.0)])  # one minute reaches the stop and the target
    assert bs.simulate(pd.DataFrame([setup_row(day, 1, 99.5, 100.5, 0)]), both).iloc[0].exit_reason == "stop"
    flat = minute_frame(day, [(k, 100.0, 100.1, 99.9, 100.0 + k / 1000) for k in range(60)])
    t = bs.simulate(pd.DataFrame([setup_row(day, 1, 99.0, 101.0, 0)]), flat).iloc[0]
    assert t.exit_reason == "time or close" and t.exit_time == flat["ts"].iloc[49]  # 50 minutes, out at the close
    beyond = pd.DataFrame([setup_row(day, 1, 99.0, 99.8, 0)])  # entry 100.0 is already above the target
    assert bs.simulate(beyond, flat).empty
    short = bs.simulate(pd.DataFrame([setup_row(day, -1, 100.4, 99.95, 0)]), flat).iloc[0]
    assert short.exit_reason == "target" and short.gross == pytest.approx(5.0)


def test_one_position_at_a_time():
    day = "2024-03-05"
    flat = minute_frame(day, [(k, 100.0, 100.1, 99.9, 100.0) for k in range(60)])
    two = pd.DataFrame([setup_row(day, 1, 99.0, 101.0, 0), setup_row(day, 1, 99.0, 101.0, 10)])
    assert len(bs.simulate(two, flat)) == 1  # the second setup comes while the first is still open


def test_levels_need_80_percent_of_the_previous_session():
    sessions = features.trading_sessions("2024-03-04", "2024-03-06", warmup_sessions=0)
    m = make_minutes(sessions, 2)
    day1 = m["ts"] < sessions["close"].iloc[0]
    sparse = m[~day1 | (np.arange(len(m)) % 10 != 0)]  # day 1 keeps 90% of its minutes
    rth, _ = features.regular_session_minutes(sparse, sessions)
    lev = bs.profile_levels(rth, sessions)
    assert lev.loc["2024-03-05", ["val", "poc", "vah"]].notna().all()
    assert lev.loc["2024-03-05", "val"] < lev.loc["2024-03-05", "poc"] < lev.loc["2024-03-05", "vah"]
    thin = m[~day1 | (np.arange(len(m)) % 2 == 0)]  # day 1 keeps half its minutes
    rth2, _ = features.regular_session_minutes(thin, sessions)
    assert bs.profile_levels(rth2, sessions).loc["2024-03-05", ["poc", "vah", "val"]].isna().all()
