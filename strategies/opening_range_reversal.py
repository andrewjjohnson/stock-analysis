"""Opening-range reversal: fade an unusually large first 15 minutes on completed 5-minute bars.

A deterministic research adaptation of the setup presented in
https://www.youtube.com/watch?v=XFtayhPIdEs, not a replay or an endorsement of it. The
numeric definitions, completed-candle signal, first-signal limit and fixed barrier
levels are this project's engineering assumptions.

Baseline, per ticker and session; times are relative to that session's XNYS open:
- Opening range: minutes [open, open+15), all 15 present; known at open+15.
- Size gate: opening high - low >= threshold x daily ATR(14) as of the previous
  session's close (threshold 0.25 baseline).
- Direction: opening close < opening open -> long only; > -> short only; equal -> skip.
- Signal bar: a complete (5 of 5 minutes) 5-minute bar anchored to the open that
  starts at or after open+15 and completes strictly before open+90.
- Pattern (installed TA-Lib): long CDLHAMMER > 0 or CDLENGULFING > 0; short
  CDLSHOOTINGSTAR < 0 or CDLENGULFING < 0. All three compare with the previous candle,
  so that bar must also be complete, contiguous and in the same session.
- Outside: long low and close strictly below the opening low; short high and close
  strictly above the opening high. The bar may straddle the boundary.
- Only the earliest qualifying bar per session; no trend, volume or other filter.
"""

import numpy as np
import pandas as pd
import talib

from features import MINUTE

BAR_MINUTES = 5
OPENING_MINUTES = 15
WINDOW_MINUTES = 90
ATR_PERIOD = 14
BASELINE_THRESHOLD = 0.25
SIDES = {"long": 1, "short": -1}
# Mutually exclusive pattern groups per side: together they add up to the side total.
PATTERNS = {"long": ("hammer", "engulfing", "hammer+engulfing"),
            "short": ("shooting_star", "engulfing", "shooting_star+engulfing")}
OPENING_COLUMNS = ["or_open", "or_high", "or_low", "or_close"]


def config_label(threshold):
    """Unique per threshold: two decimals only when they are exact, since labels key selection."""
    text = f"{threshold:.2f}" if float(f"{threshold:.2f}") == threshold else repr(threshold)
    return f"threshold={text}" + (" (baseline)" if threshold == BASELINE_THRESHOLD else "")


def make_configs(thresholds):
    """One configuration per size-gate threshold; every other definition is fixed."""
    return [{"label": config_label(t), "params": {"threshold": t}} for t in dict.fromkeys(map(float, thresholds))]


def opening_ranges(rth, sessions):
    """Opening OHLC per session from its first OPENING_MINUTES regular-session minutes.

    One row per session. Values are NaN unless every opening minute is present.
    """
    first = rth[rth["ts"] < rth["session_open"] + pd.Timedelta(minutes=OPENING_MINUTES)]
    o = (first.groupby("session")
         .agg(or_open=("open", "first"), or_high=("high", "max"), or_low=("low", "min"),
              or_close=("close", "last"), or_minutes=("ts", "size"))
         .reindex(sessions.index))
    o["or_complete"] = o["or_minutes"] == OPENING_MINUTES
    o.loc[~o["or_complete"], OPENING_COLUMNS] = np.nan
    o["session_open"] = sessions["open"]
    return o


def opening_gate(eligible, or_range, prev_atr, threshold):
    """Size gate, shared by the trigger mask and the session counts."""
    return eligible & (or_range >= threshold * prev_atr)


def add_features(bars, rth, daily, sessions):
    """Opening values, TA-Lib patterns and the setup mask, computed once for every threshold.

    Returns (bars with extra columns, one row per session). The pattern functions run
    over the continuous history of usable bars, like the intraday EMAs, so their
    trailing candle averages are warm at the open. Nothing here reads past a bar's end.
    """
    opening = opening_ranges(rth, sessions)
    opening["prev_atr"] = daily[f"prev_atr_{ATR_PERIOD}"]
    opening["or_range"] = opening["or_high"] - opening["or_low"]
    opening["range_atr_ratio"] = opening["or_range"] / opening["prev_atr"]
    opening["side_sign"] = np.select([opening["or_close"] < opening["or_open"],
                                      opening["or_close"] > opening["or_open"]], [1, -1], 0)
    opening["eligible"] = opening["or_complete"] & (opening["prev_atr"] > 0)

    b = bars.join(opening.drop(columns=["or_minutes", "or_complete"]), on="session")
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    side = b["side_sign"].to_numpy()
    long_day, short_day = side == 1, side == -1
    engulfing = talib.CDLENGULFING(o, h, l, c)
    b["hammer"] = long_day & (talib.CDLHAMMER(o, h, l, c) > 0)
    b["shooting_star"] = short_day & (talib.CDLSHOOTINGSTAR(o, h, l, c) < 0)
    b["engulfing"] = (long_day & (engulfing > 0)) | (short_day & (engulfing < 0))
    single = np.where(b["hammer"], "hammer", np.where(b["shooting_star"], "shooting_star", ""))
    b["pattern"] = np.where(b["engulfing"], np.where(single != "", np.char.add(single, "+engulfing"), "engulfing"),
                            single).astype(object)
    b["side"] = np.where(long_day, "long", np.where(short_day, "short", "")).astype(object)

    complete = (b["n_minutes"] == (b["bar_end"] - b["bar_start"]) / MINUTE).to_numpy()
    follows = ((b["bar_start"] == b["bar_end"].shift(1)) & (b["session"] == b["session"].shift(1))).to_numpy()
    pair_complete = complete & np.r_[False, complete[:-1]] & follows
    in_window = ((b["bar_start"] >= b["session_open"] + pd.Timedelta(minutes=OPENING_MINUTES))
                 & (b["bar_end"] < b["session_open"] + pd.Timedelta(minutes=WINDOW_MINUTES))).to_numpy()
    outside = ((long_day & (l < b["or_low"].to_numpy()) & (c < b["or_low"].to_numpy()))
               | (short_day & (h > b["or_high"].to_numpy()) & (c > b["or_high"].to_numpy())))
    pattern = (b["hammer"] | b["shooting_star"] | b["engulfing"]).to_numpy()
    b["setup"] = b["eligible"].to_numpy() & in_window & pair_complete & outside & pattern
    return b, opening


def opening_range_reversal(features, threshold=BASELINE_THRESHOLD):
    """Trigger mask: the earliest qualifying 5-minute bar per session, plus the fields to keep.

    `features` is the bar table from add_features. The first qualifying bar is chosen
    by time alone, so later bars can never replace or precede it.
    """
    f = features
    gate = opening_gate(f["eligible"], f["or_range"], f["prev_atr"], threshold).to_numpy()
    qualifying = np.flatnonzero(f["setup"].to_numpy() & gate)
    _, first = np.unique(f["session"].to_numpy()[qualifying], return_index=True)
    mask = np.zeros(len(f), dtype=bool)
    mask[qualifying[first]] = True

    keep = {"side": f["side"].to_numpy(), "session_open": f["session_open"].array,
            "opening_end": (f["session_open"] + pd.Timedelta(minutes=OPENING_MINUTES)).array,
            **{k: f[k].to_numpy() for k in OPENING_COLUMNS}, "or_range": f["or_range"].to_numpy(),
            f"prev_atr_{ATR_PERIOD}": f["prev_atr"].to_numpy(), "range_atr_ratio": f["range_atr_ratio"].to_numpy(),
            **{k: f[k].to_numpy() for k in ("hammer", "shooting_star", "engulfing", "pattern")},
            "signal_open": f["open"].to_numpy(), "signal_high": f["high"].to_numpy(), "signal_low": f["low"].to_numpy()}
    return mask, keep


def opening_counts(opening, threshold):
    """Session counts for one segment and threshold, per side (no per-bar or outcome work)."""
    passes = opening_gate(opening["eligible"], opening["or_range"], opening["prev_atr"], threshold)
    common = {"sessions": len(opening), "or_incomplete": int((~opening["or_complete"]).sum()),
              "atr_unavailable": int(opening["prev_atr"].isna().sum()),
              "eligible_sessions": int(opening["eligible"].sum()),
              "flat_openings_skipped": int((passes & (opening["side_sign"] == 0)).sum())}
    return {side: {**common, "qualifying_openings": int((passes & (opening["side_sign"] == sign)).sum())}
            for side, sign in SIDES.items()}


def barrier_levels(bars, idx):
    """Fixed stop and target for triggered bars only (idx = positions in `bars`).

    Stop: the signal bar's low (long) or high (short); for an engulfing signal, the
    extreme across both candles. Target: the opposite opening-range boundary.
    """
    b, prev = bars.iloc[idx], bars.iloc[np.asarray(idx) - 1]  # previous bar is contiguous by construction
    engulfing = b["engulfing"].to_numpy()
    low = np.where(engulfing, np.minimum(b["low"].to_numpy(), prev["low"].to_numpy()), b["low"].to_numpy())
    high = np.where(engulfing, np.maximum(b["high"].to_numpy(), prev["high"].to_numpy()), b["high"].to_numpy())
    long_ = b["side_sign"].to_numpy() > 0
    return np.where(long_, low, high), np.where(long_, b["or_high"].to_numpy(), b["or_low"].to_numpy())
