"""Candidate-only processing: zero triggers, and one known trigger with exact outcomes."""

import numpy as np
import pandas as pd

import features
import outcomes
import run
from synthetic import bars_from_closes, session_minute_starts

NY = "America/New_York"
EXPECTED_COLUMNS = [
    "strategy", "config", "fast", "slow", "daily_filter", "segment", "session", "bar_start", "signal_time",
    "ref_close", "fast_ema", "slow_ema", "prev_session_close", "prev_session_ema50", *outcomes.OUTCOME_COLUMNS,
]


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def three_sessions():
    # warm-up 2024-03-04, study 2024-03-05 and 2024-03-06
    return features.trading_sessions("2024-03-05", "2024-03-06", warmup_sessions=1)


def test_zero_triggers_never_build_rows_or_outcomes(monkeypatch):
    sessions = three_sessions()
    ts = session_minute_starts(sessions)
    minutes = bars_from_closes(ts, 100 - 0.001 * np.arange(len(ts)))  # steady decline: fast never crosses above slow

    def forbidden(*args, **kwargs):
        raise AssertionError("called without any candidates")

    monkeypatch.setattr(outcomes, "forward_outcomes", forbidden)
    monkeypatch.setattr(run, "build_candidate_rows", forbidden)
    result = run.run_study(minutes, sessions, fasts=[3], slows=[6], daily_filter=False)

    assert result["candidates"].empty
    assert list(result["candidates"].columns) == EXPECTED_COLUMNS
    summary = result["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_candidates"] == 0 and row["n_30m"] == 0 and row["unavailable_30m"] == 0
    assert np.isnan(row["mean_30m_pct"]) and np.isnan(row["frac_pos_30m"]) and np.isnan(row["mean_mfe_60m_pct"])


def known_trigger_minutes(sessions, trigger_start, trigger_end):
    """Steady decline, a jump inside one 5-minute bar, then +0.1% per elapsed minute."""
    ts = session_minute_starts(sessions)
    closes = 100 - 0.001 * np.arange(len(ts))
    closes[(ts >= trigger_start) & (ts < trigger_end)] = 101.0
    after = ts >= trigger_end
    elapsed_at_bar_end = ((ts[after] - trigger_end) / pd.Timedelta(minutes=1)).to_numpy() + 1
    closes[after] = 101.0 * (1 + 0.001 * elapsed_at_bar_end)
    return bars_from_closes(ts, closes)


def test_known_trigger_timestamp_and_outcomes_only_for_triggers(monkeypatch):
    sessions = three_sessions()
    start, end = et("2024-03-06 10:20"), et("2024-03-06 10:25")
    minutes = known_trigger_minutes(sessions, start, end)
    evaluated = []
    real = outcomes.forward_outcomes

    def spy(minutes, signal_time, *args):
        evaluated.extend(signal_time)
        return real(minutes, signal_time, *args)

    monkeypatch.setattr(outcomes, "forward_outcomes", spy)
    result = run.run_study(minutes, sessions, fasts=[3], slows=[6], daily_filter=False)

    assert evaluated == [end]  # outcomes computed for the trigger and nothing else
    cands = result["candidates"]
    assert len(cands) == 1
    c = cands.iloc[0]
    assert c["bar_start"] == start and c["signal_time"] == end  # signal known when the bar completes
    assert c["ref_close"] == 101.0 and c["fast_ema"] > c["slow_ema"]
    expected = {10: 1.0, 30: 3.0, 60: 6.0, 120: 12.0}  # price at T+h = 101 * (1 + 0.001 h)
    for h, pct in expected.items():
        assert np.isclose(c[f"fwd_ret_{h}m_pct"], pct)
    assert np.isclose(c["mfe_60m_pct"], 6.0)
    assert c["mae_60m_pct"] == 0.0  # never below the reference close: zero, not a spurious negative
    assert result["summary"].iloc[0]["n_candidates"] == 1


def test_trigger_skipped_when_daily_feature_unavailable():
    sessions = three_sessions()  # far fewer than 50 sessions: no daily EMA50
    minutes = known_trigger_minutes(sessions, et("2024-03-06 10:20"), et("2024-03-06 10:25"))
    result = run.run_study(minutes, sessions, fasts=[3], slows=[6], daily_filter=True)
    assert result["candidates"].empty
    assert result["info"]["study_sessions_missing_daily"] == 2  # reported, not fabricated
