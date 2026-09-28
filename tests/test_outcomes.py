"""Unavailable outcomes (missing minutes, session close, split boundaries) never drop candidates."""

import numpy as np
import pandas as pd

import features
import run
import synthetic
from outcomes import HORIZONS, forward_outcomes
from synthetic import bars_from_closes, session_minute_starts

NY = "America/New_York"


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def availability(frame):
    return {col: frame[col].notna().tolist() for col in frame.columns}


def test_missing_minutes_close_and_segment_end_give_nan_but_keep_rows():
    # 2024-11-27 is a normal session; 2024-11-29 closes early at 13:00 ET (from the exchange calendar).
    sessions = features.trading_sessions("2024-11-27", "2024-11-29", warmup_sessions=0)
    ts = session_minute_starts(sessions)
    minutes = bars_from_closes(ts, 100 * (1 + 1e-4 * np.arange(len(ts))))
    minutes = minutes[minutes["ts"] != et("2024-11-27 13:20")].reset_index(drop=True)  # one missing minute
    close_at = minutes.set_index("ts")["close"]

    def evaluate(times, segment_end):
        times = pd.Series([et(t) for t in times])
        ref = [close_at[t - pd.Timedelta(minutes=1)] for t in times]  # close of the bar ending at T
        closes = [sessions.loc[t.tz_convert(NY).tz_localize(None).normalize(), "close"] for t in times]
        return times, ref, forward_outcomes(minutes, times, ref, pd.Series(closes), segment_end)

    signal = ["2024-11-27 10:00",   # control: everything available
              "2024-11-27 13:00",   # 13:20 missing: only 10m survives
              "2024-11-27 15:15",   # close at 16:00: 10m, 30m survive
              "2024-11-29 12:30"]   # early close at 13:00: 10m, 30m survive
    times, ref, out = evaluate(signal, sessions["close"].iloc[-1])
    assert len(out) == len(signal)
    assert availability(out) == {
        "fwd_ret_10m_pct": [True, True, True, True],
        "fwd_ret_30m_pct": [True, False, True, True],
        "fwd_ret_60m_pct": [True, False, False, False],
        "fwd_ret_120m_pct": [True, False, False, False],
        "mfe_60m_pct": [True, False, False, False],
        "mae_60m_pct": [True, False, False, False],
    }
    for h in HORIZONS:  # explicit timestamp lookup: price at T+h is the close of the bar ending then
        expected = (close_at[times[0] + pd.Timedelta(minutes=h - 1)] / ref[0] - 1) * 100
        assert np.isclose(out.loc[0, f"fwd_ret_{h}m_pct"], expected)

    # A segment (split) boundary inside the session cuts horizons exactly like a close.
    _, _, out = evaluate(["2024-11-27 10:00", "2024-11-27 12:00"], et("2024-11-27 12:45"))
    assert availability(out)["fwd_ret_120m_pct"] == [True, False]
    assert availability(out)["fwd_ret_30m_pct"] == [True, True]
    assert availability(out)["fwd_ret_60m_pct"] == [True, False]
    assert availability(out)["mfe_60m_pct"] == [True, False]


def test_split_selects_on_earlier_segment_and_outcomes_stay_inside_segments():
    sessions = features.trading_sessions("2024-03-04", "2024-03-15", warmup_sessions=5)
    minutes = synthetic.random_walk_minutes(sessions, seed=11, drop_sessions=0)
    kwargs = dict(fasts=[3, 5], slows=[8, 13], daily_filter=False, split_date="2024-03-11")
    result = run.run_study(minutes, sessions, min_labeled=5, **kwargs)
    summary, cands = result["summary"], result["candidates"]

    earlier = summary[summary["segment"] == "earlier"]
    qualified = earlier[earlier["n_30m"] >= 5]
    assert len(earlier) == 4 and len(qualified) > 0
    assert result["selected"] == qualified.sort_values("mean_30m_pct", ascending=False)["config"].iloc[0]
    later = summary[summary["segment"] == "later"]
    assert later["config"].tolist() == [result["selected"]]  # only the pick is evaluated later

    split = pd.Timestamp("2024-03-11", tz=NY)
    early, late = cands[cands["segment"] == "earlier"], cands[cands["segment"] == "later"]
    assert (early["signal_time"] < split).all() and (late["signal_time"] > split).all()
    assert set(late["config"]) == {result["selected"]}
    last_early_close = sessions.loc["2024-03-08", "close"]
    for h in HORIZONS:  # earlier outcomes finish strictly before the split
        ok = early[f"fwd_ret_{h}m_pct"].notna()
        assert (early.loc[ok, "signal_time"] + pd.Timedelta(minutes=h) <= last_early_close).all()
    afternoon = early[early["signal_time"].dt.tz_convert(NY).dt.time > pd.Timestamp("14:00").time()]
    assert len(afternoon) > 0 and afternoon["fwd_ret_120m_pct"].isna().all()  # kept, outcome NaN

    none = run.run_study(minutes, sessions, min_labeled=10**6, **kwargs)
    assert none["selected"] is None  # no silent winner
    assert set(none["summary"]["segment"]) == {"earlier"}
    assert (none["candidates"]["segment"] == "earlier").all()
