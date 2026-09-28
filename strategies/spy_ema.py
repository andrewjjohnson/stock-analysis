"""Example strategy: intraday EMA crossover with an optional daily trend filter.

Demonstration defaults only; not a reproduction of any Quant Forge strategy.

A strategy is a plain function. It receives the feature table from
features.build_features (completed, adequately covered bars; no outcome data) and
returns a boolean trigger mask aligned with those rows, plus the feature arrays
worth keeping for each triggered row.
"""

import numpy as np

DAILY_EMA_PERIOD = 50


def spy_ema(features, fast=9, slow=21, daily_filter=True):
    """Long candidate on the completed bar where EMA(fast) crosses above EMA(slow).

    Cross = previous usable bar has fast <= slow and this bar has fast > slow, so it
    fires once per cross, not on every bar that stays above. With daily_filter, the
    previous completed session's close must also be above its daily EMA 50.
    Rows missing any required value never trigger.
    """
    fast_ema = features[f"ema_{fast}"].to_numpy()
    slow_ema = features[f"ema_{slow}"].to_numpy()
    fast_prev = features[f"ema_{fast}"].shift(1).to_numpy()
    slow_prev = features[f"ema_{slow}"].shift(1).to_numpy()
    ready = ~(np.isnan(fast_ema) | np.isnan(slow_ema) | np.isnan(fast_prev) | np.isnan(slow_prev))
    mask = ready & (fast_prev <= slow_prev) & (fast_ema > slow_ema)

    daily_close = features["prev_close"].to_numpy()
    daily_ema = features[f"prev_ema_{DAILY_EMA_PERIOD}"].to_numpy()
    if daily_filter:
        mask &= ~(np.isnan(daily_close) | np.isnan(daily_ema)) & (daily_close > daily_ema)

    keep = {
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
        "prev_session_close": daily_close,
        f"prev_session_ema{DAILY_EMA_PERIOD}": daily_ema,
    }
    return mask, keep
