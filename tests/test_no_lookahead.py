"""Changing future prices must not change earlier features or triggers."""

import numpy as np
import pandas as pd

import features
import synthetic
from strategies.spy_ema import spy_ema

NY = "America/New_York"
COLUMNS = ["close", "ema_5", "ema_20", "prev_close", "prev_ema_50"]


def features_and_triggers(minutes, sessions):
    _, bars, _, _ = features.build_features(minutes, sessions, 5, (5, 20), 50, 0.8)
    mask, _ = spy_ema(bars, fast=5, slow=20, daily_filter=True)
    return bars, mask


def test_future_prices_cannot_change_earlier_features_or_triggers():
    sessions = features.trading_sessions("2024-05-20", "2024-05-24", warmup_sessions=60)  # enough for daily EMA50
    minutes = synthetic.random_walk_minutes(sessions, seed=3, drop_sessions=0)
    cut = pd.Timestamp("2024-05-22 12:02", tz=NY).tz_convert("UTC")  # mid-bar, mid-session
    changed = minutes.copy()
    later = changed["ts"] >= cut
    changed.loc[later, ["open", "high", "low", "close"]] *= np.linspace(0.9, 1.3, later.sum())[:, None]

    before, mask_before = features_and_triggers(minutes, sessions)
    after, mask_after = features_and_triggers(changed, sessions)
    # Intraday EMAs run continuously across sessions (no morning reset), so even each
    # session's first bars have values.
    assert before.loc[before["in_study"], ["ema_5", "ema_20"]].notna().all().all()

    assert before["bar_start"].equals(after["bar_start"])
    known = (before["bar_end"] <= cut).to_numpy()  # bars completed by the cut
    pd.testing.assert_frame_equal(before.loc[known, COLUMNS], after.loc[known, COLUMNS], check_exact=True)
    assert mask_before[known].any()
    assert np.array_equal(mask_before[known], mask_after[known])

    # Daily features all day on 2024-05-22 come from 2024-05-21, even for bars after the cut...
    day = (before["session"] == "2024-05-22").to_numpy()
    daily = ["prev_close", "prev_ema_50"]
    pd.testing.assert_frame_equal(before.loc[day, daily], after.loc[day, daily], check_exact=True)
    # ...while the altered 2024-05-22 close reaches the next session, and later intraday values move.
    next_day = (before["session"] == "2024-05-23").to_numpy()
    assert not np.allclose(before.loc[next_day, "prev_close"], after.loc[next_day, "prev_close"])
    assert not np.allclose(before.loc[~known, "ema_20"], after.loc[~known, "ema_20"])
