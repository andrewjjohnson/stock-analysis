"""kalman_supertrend.py on SYNTHETIC bars: the Kalman filter's steady state, the SuperTrend flip, Pine's RMA seed,
every exit of a trade (target, stop gapped through, breakeven, opposite flag, time, regular close), flags in the
same direction while a trade is open, and no lookahead."""

import numpy as np
import pandas as pd
import pytest

import features
import kalman_supertrend as ks


def test_the_kalman_filter_settles_into_a_nine_bar_ema():
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.3, 300))
    est = ks.kalman(close, 0.01, 0.2)
    ema = est[50] + np.zeros(1)
    for c in close[51:]:
        ema = 0.8 * ema + 0.2 * c  # gain 0.2 = an EMA with alpha 0.2 (2 / (9 + 1))
    assert est[-1] == pytest.approx(ema[0], rel=1e-6)


def test_supertrend_flips_down_when_the_close_breaks_the_ratcheted_band():
    close = np.array([10, 11, 12, 13, 12, 10.5, 9.5, 9.0])
    up, dn, trend = ks.supertrend(close, close, np.ones(8), 2.0)
    assert up[:5].tolist() == [8, 9, 10, 11, 11]  # ratchets up, never down while the close stays above
    assert trend.tolist() == [1, 1, 1, 1, 1, -1, -1, -1]  # 10.5 closes below the 11 band


def test_rma_is_seeded_with_the_first_full_window_mean():
    out = ks.rma(np.array([np.nan, 1, 2, 3, 4, 5]), 3)
    assert np.isnan(out[:3]).all() and out[3:].tolist() == pytest.approx([2.0, 8 / 3, (16 / 3 + 5) / 3])


def frames(paths, atr=1.0, sig=None, ob=None, last_rth=None):
    """Bars of five one-minute (open, high, low, close) rows each, with fixed ATR and flags."""
    rows = [r for bar in paths for r in bar]
    ts = pd.date_range("2024-01-02 15:00", periods=len(rows), freq="min", tz="UTC")
    m = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(ts=ts)
    n = len(paths)
    bars = pd.DataFrame({"bar": ts[::5], "day": pd.Timestamp("2024-01-02"),
                         "high": [max(r[1] for r in b) for b in paths], "low": [min(r[2] for r in b) for b in paths],
                         "close": [b[-1][3] for b in paths], "atr": atr,
                         "sig": np.zeros(n, int) if sig is None else np.array(sig),
                         "ob": np.zeros(n, bool) if ob is None else np.array(ob), "os": np.zeros(n, bool),
                         "last_rth": np.zeros(n, bool) if last_rth is None else np.array(last_rth),
                         "m_lo": np.arange(n) * 5, "m_hi": np.arange(n) * 5 + 5, "rth": True})
    return bars, m


def flat(px, n=5):
    return [(px, px + 0.05, px - 0.05, px)] * n


def test_target_and_a_stop_gapped_through():
    # Signal bar closes at 100 (ATR 1): stop 98.5, target 102. Fill at the next minute's open + 1 tick.
    bars, m = frames([flat(100), flat(100) + [], [(100, 102.2, 99.9, 102)] * 5])
    tr = ks.trade_from(0, 1, ks.context(bars, m))
    assert tr["fill"] == pytest.approx(100.01) and tr["reason"] == "target" and tr["exit"] == 102
    bars, m = frames([flat(100), [(98.0, 98.1, 97.9, 98.0)] * 5])  # opens below the 98.5 stop
    tr = ks.trade_from(0, 1, ks.context(bars, m))
    assert tr["reason"] == "stop" and tr["exit"] == pytest.approx(97.99)  # the open, less a tick


def test_breakeven_opposite_flag_time_and_regular_close():
    up = [(100, 101.1, 99.9, 100.8)] * 5  # reaches +1 ATR: from the next bar the stop is the signal close (100)
    dip = [(100.5, 100.6, 99.95, 100.2)] * 5
    bars, m = frames([flat(100), up, dip])
    tr = ks.trade_from(0, 1, ks.context(bars, m))
    assert tr["reason"] == "breakeven" and tr["exit"] == pytest.approx(99.99)
    bars, m = frames([flat(100), flat(100.3), flat(100.4), flat(100.5)], sig=[1, 0, -1, 0])
    tr = ks.trade_from(0, 1, ks.context(bars, m))
    assert tr["reason"] == "opposite flag" and tr["exit_bar"] == 2 and tr["exit"] == pytest.approx(100.49)
    bars, m = frames([flat(100)] * 15)
    assert ks.trade_from(0, 1, ks.context(bars, m))["exit_bar"] == 12  # the time stop: 12 bars after the signal
    bars, m = frames([flat(100), flat(100.2), flat(100.3)], last_rth=[False, True, False])
    tr = ks.trade_from(0, 1, ks.context(bars, m), flat_by_close=True)
    assert tr["reason"] == "regular close" and tr["exit"] == pytest.approx(100.19)


def test_a_flag_in_the_same_direction_is_ignored_while_the_trade_is_open():
    bars, m = frames([flat(100), flat(100.1), flat(100.2), flat(100.3), flat(100.4), flat(100.5)],
                     sig=[1, 0, 1, 0, -1, 0])
    t = ks.strategy(bars, m)
    assert t["dir"].tolist() == [1, -1]  # bar 2's BUY is ignored; bar 4's SELL reverses
    assert t["reason"].iloc[0] == "opposite flag"


def make_minutes(sessions, seed=0):
    stamps = [pd.date_range(o - pd.Timedelta(hours=5, minutes=30), c + pd.Timedelta(hours=3, minutes=59), freq="min")
              for o, c in zip(sessions["open"], sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.04, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


def test_later_minutes_never_change_earlier_signals_or_trades():
    sessions = features.trading_sessions("2024-01-02", "2024-01-12", warmup_sessions=0)
    m = make_minutes(sessions, seed=4)
    cut = pd.Timestamp("2024-01-09 12:00", tz="America/New_York")
    m2 = m.copy()
    m2.loc[m2["ts"] >= cut, ["open", "high", "low", "close"]] += 2.0

    def run(minutes):
        bars, mm = ks.five_minute_bars(minutes, sessions)
        bars = ks.indicators(bars)
        return bars, ks.strategy(bars, mm)

    (b1, t1), (b2, t2) = run(m), run(m2)
    early = b1["bar"] < cut - pd.Timedelta(minutes=5)
    assert (b1.loc[early, "sig"].to_numpy() == b2.loc[early, "sig"].to_numpy()).all()
    done = [t[t["exit_time"] < cut].reset_index(drop=True) for t in (t1, t2)]
    pd.testing.assert_frame_equal(done[0], done[1])
    assert len(done[0]) > 0
