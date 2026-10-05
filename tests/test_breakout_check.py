"""breakout_check.py on SYNTHETIC bars: the second-by-second replay (a dip before the fill doesn't count, one after
it does; targets; gaps through the stop; later minutes), the option contract, option prices near a time, the
per-contract statistics, and the period gate."""

import numpy as np
import pandas as pd
import pytest

import breakout_check as bc
import features


def secs(start_ms, rows):
    """One-second bars [(offset s, open, high, low, close)] from start_ms."""
    return np.array([(start_ms + 1000 * k, o, h, l, c) for k, o, h, l, c in rows], float).reshape(-1, 5)


MIN = [0, 60_000, 120_000]


def run(seconds, high, low, close, price=100.0, d=1, stop=99.5, target=100.5):
    return bc.replay(np.array(MIN), np.array(high), np.array(low), np.array(close), 0, price, d, stop, target,
                     lambda m: seconds[m], 179_999)


def test_a_dip_before_the_fill_does_not_stop_the_trade_but_one_after_it_does():
    # Minute 0 dips to 99.4 first (second 1), then reaches the level at second 3 and later the target.
    before = {0: secs(0, [(1, 99.8, 99.8, 99.4, 99.5), (3, 99.9, 100.1, 99.9, 100.0), (9, 100.2, 100.6, 100.2, 100.5)])}
    fill, fill_ms, exit_, exit_ms, reason = run(before, [100.6], [99.4], [100.5])
    assert (fill, fill_ms, exit_, exit_ms, reason) == (100.0, 3000, 100.5, 9000, 1)
    # Same minute range, but the dip comes after the fill: stopped.
    after = {0: secs(0, [(1, 99.8, 100.1, 99.8, 100.0), (5, 99.9, 99.9, 99.4, 99.5), (9, 100.2, 100.6, 100.2, 100.5)])}
    assert run(after, [100.6], [99.4], [100.5])[2:] == (99.5, 5000, -1)


def test_later_minutes_are_read_by_the_second_and_a_gap_through_the_stop_fills_at_the_open():
    s = {0: secs(0, [(2, 99.9, 100.1, 99.9, 100.0)]),                       # fills at 100, nothing else
         120_000: secs(120_000, [(4, 99.3, 99.3, 99.2, 99.2)])}             # minute 2 opens below the stop
    out = run(s, [100.1, 100.3, 99.3], [99.9, 99.9, 99.2], [100.0, 100.1, 99.2])
    assert out == (100.0, 2000, 99.3, 124_000, -1)
    none = run({0: secs(0, [(2, 99.9, 99.95, 99.9, 99.9)])}, [100.1], [99.9], [100.0])
    assert none is None  # the seconds never reach the level the minute bar shows


def test_a_trade_reaching_neither_ends_at_the_last_close():
    s = {0: secs(0, [(2, 99.9, 100.1, 99.9, 100.0)])}
    assert run(s, [100.1, 100.2, 100.2], [99.9, 99.8, 99.8], [100.0, 100.1, 100.15])[2:] == (100.15, 179_999, 0)


def test_the_option_is_the_first_expiry_after_the_day_at_the_strike_nearest_the_level():
    chain = {pd.Timestamp("2025-01-03"): np.array([395.0, 400.0, 405.0]),
             pd.Timestamp("2025-01-10"): np.array([390.0, 400.0, 410.0])}
    assert bc.option_leg(pd.Timestamp("2025-01-03"), 1, 401.0, chain) == (pd.Timestamp("2025-01-10"), "C", 400.0)
    assert bc.option_leg(pd.Timestamp("2025-01-02"), -1, 403.0, chain) == (pd.Timestamp("2025-01-03"), "P", 405.0)
    assert bc.option_leg(pd.Timestamp("2025-01-10"), 1, 400.0, chain) is None


def test_option_prices_come_from_the_first_trade_after_a_time_or_the_last_before_it():
    sec = secs(0, [(10, 4.0, 4.1, 4.0, 4.05), (50, 4.2, 4.2, 4.1, 4.15)])
    assert bc.first_trade(sec, 12_000) == 4.2 and bc.first_trade(sec, 51_000) is None
    assert bc.last_trade(sec, 49_000) == 4.05 and bc.last_trade(sec, 200_000) is None
    o = pd.DataFrame({"opt_entry": [4.0, 5.0, 3.0], "opt_exit": [4.5, 4.8, 3.6], "fill_ms": [1, 2, 3],
                      "session": pd.date_range("2025-01-02", periods=3, freq="B")})
    st = bc.option_stats(o, 0.05, years=1.0)
    # (0.5 - 0.1) * 100 = 40, (-0.2 - 0.1) * 100 = -30, (0.6 - 0.1) * 100 = 50
    assert st["mean"] == pytest.approx(20.0) and st["win"] == pytest.approx(200 / 3)
    assert st["drawdown"] == pytest.approx(30.0)
    assert st["premium"] == pytest.approx(100 * (4.0 + 5.0 + 3.0) / 3 + 5)


def make_minutes(sessions, seed=0, pre=40):
    stamps = [pd.date_range(o - pd.Timedelta(minutes=pre), c - pd.Timedelta(minutes=1), freq="min")
              for o, c in zip(sessions["open"], sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.05, len(ts)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": ts, "open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close, "volume": 1000.0})


def test_the_dry_run_never_reads_2025_and_the_check_reads_only_2025_on():
    sessions = features.trading_sessions("2024-10-01", "2025-02-28", warmup_sessions=0)
    m = make_minutes(sessions, seed=3)
    rth, _ = features.regular_session_minutes(m, sessions)
    by_ms = {int(t.value // 1_000_000): r for t, r in zip(rth["ts"], rth.itertuples())}

    def seconds_of(ticker, start, end):  # the minute's path: open, high, low, close at seconds 0/15/30/45
        r = by_ms.get(int(start))
        if r is None or ticker.startswith("O:"):
            return np.zeros((0, 5))
        return secs(int(start), [(0, r.open, r.open, r.open, r.open), (15, r.high, r.high, r.high, r.high),
                                 (30, r.low, r.low, r.low, r.low), (45, r.close, r.close, r.close, r.close)])

    dry = bc.run_check(m, sessions, {}, seconds_of, fake_sample=20)
    assert dry["shares"]["session"].max() < pd.Timestamp("2025-01-01") and len(dry["shares"])
    check = bc.run_check(m, sessions, {}, seconds_of, final=True, fake_sample=20)
    assert check["shares"]["session"].min() >= pd.Timestamp("2025-01-01")
    assert set(check["options"]["opt_status"]) <= {"no listed expiry"}
