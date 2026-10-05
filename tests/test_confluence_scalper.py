"""confluence_scalper.py on SYNTHETIC data: ChrisMoody's SMA signal line, LazyBear's Bollinger width quirk, the
k-means of SuperTrend AI, its trend on a rising then falling series, the swing structure, exit B, and no lookahead."""

import numpy as np
import pandas as pd
import pytest
import talib

import confluence_scalper as cs
import features
import kalman_supertrend as ks


def test_the_macd_signal_is_an_sma_and_the_squeeze_uses_the_keltner_multiplier_for_bollinger():
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, 200))
    macd, sig, hist = cs.cm_macd(close)
    assert np.allclose(sig[60:], talib.SMA(talib.EMA(close, 12) - talib.EMA(close, 26), 9)[60:])
    high, low = close + 0.2, close - 0.2
    val, on = cs.squeeze_momentum(high, low, close)
    dev = 1.5 * talib.STDDEV(close, 20, 1)
    rng = talib.SMA(talib.TRANGE(high, low, close), 20)
    assert (on[40:] == (dev[40:] < 1.5 * rng[40:])).all()  # Bollinger (1.5 sd) inside Keltner (1.5 ranges)


def test_kmeans_puts_the_best_scores_in_the_last_cluster():
    groups, labels = cs.kmeans3([0, 0.1, 0.05, 1.0, 1.1, 0.9, 5.0, 5.2, 4.8], [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    assert sorted(labels[2]) == [4, 4.5, 5] and sorted(labels[0]) == [1, 1.5, 2]


def test_supertrend_ai_follows_a_rise_then_a_fall():
    up = np.linspace(100, 120, 150)
    down = np.linspace(120, 95, 150)
    close = np.r_[up, down] + np.random.default_rng(2).normal(0, 0.05, 300)
    os_, ts, tf, perf = cs.supertrend_ai(close + 0.1, close - 0.1, close)
    assert os_[120:150].all() and not os_[-30:].any()
    assert np.nanmin(tf) >= 1.0 and np.nanmax(tf) <= 5.0


def test_swing_structure_turns_bullish_after_a_close_above_an_intermediate_high():
    # short-term highs at bars 2 (105), 5 (108) and 8 (106): bar 5 is an intermediate high, known at bar 9.
    high = np.array([100, 103, 105, 102, 104, 108, 103, 104, 106, 101, 107, 110], float)
    low, close = high - 2, high - 1
    st = cs.swing_structure(high, low, close)
    assert st[:11].tolist() == [0] * 11 and st[11] == 1  # 109 closes above 108


def test_exit_b_holds_until_the_trend_turns():
    paths = [[(100, 100.1, 99.9, 100)] * 5] * 4
    bars, m = __import__("tests.test_kalman_supertrend", fromlist=["frames"]).frames(paths)
    ctx = ks.context(bars, m)
    tr = cs.trend_trade_from(0, 1, ctx, np.array([1, 1, -1, -1]))
    assert tr["reason"] == "trend turned" and tr["exit_bar"] == 2


def make_minutes(sessions, seed=0):
    stamps = [pd.date_range(o - pd.Timedelta(hours=5, minutes=30), c + pd.Timedelta(hours=3, minutes=59), freq="min")
              for o, c in zip(sessions["open"], sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.04, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


def test_later_minutes_never_change_earlier_signals():
    sessions = features.trading_sessions("2024-01-02", "2024-01-10", warmup_sessions=0)
    m = make_minutes(sessions, seed=6)
    cut = pd.Timestamp("2024-01-08 12:00", tz="America/New_York")
    m2 = m.copy()
    m2.loc[m2["ts"] >= cut, ["open", "high", "low", "close"]] += 1.5
    b1 = cs.indicators(ks.five_minute_bars(m, sessions)[0])
    b2 = cs.indicators(ks.five_minute_bars(m2, sessions)[0])
    early = (b1["bar"] < cut - pd.Timedelta(minutes=5)).to_numpy()
    for col in ("sig", "st_sig", "st_dir", "structure"):
        assert (b1[col].to_numpy()[early] == b2[col].to_numpy()[early]).all()
