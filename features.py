"""Sessions, regular-hours minute bars, session-anchored resampling, TA-Lib features.

Conventions used throughout the project:
- Timestamps are timezone-aware UTC; America/New_York is used only to name sessions.
- XNYS sessions, holidays and early closes come from exchange_calendars.
- `ts` / `bar_start` is when a bar opens and `bar_end` is when it completes. A bar's
  close/high/low are usable at `bar_end`, never at `bar_start`.
- Only regular-session minutes are used. Missing minutes are reported, never filled.
"""

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import talib

NY = "America/New_York"
MINUTE = pd.Timedelta(minutes=1)


def trading_sessions(start, end, warmup_sessions=120):
    """XNYS sessions from `warmup_sessions` sessions before `start` through `end`.

    Indexed by session date, with UTC `open`/`close` (early closes included) and an
    `in_study` flag that is False for warm-up sessions.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cal = xcals.get_calendar("XNYS", start=start - pd.Timedelta(days=2 * warmup_sessions + 30),
                             end=end + pd.Timedelta(days=30))
    first = cal.date_to_session(start, direction="next")
    last = cal.date_to_session(end, direction="previous")
    if first > last:
        raise ValueError(f"No XNYS sessions between {start.date()} and {end.date()}.")
    sessions = cal.schedule.loc[cal.session_offset(first, -warmup_sessions):last, ["open", "close"]].copy()
    sessions["in_study"] = sessions.index >= first
    return sessions


def regular_session_minutes(minutes, sessions):
    """Sorted, de-duplicated minute bars that start inside an XNYS regular session.

    Adds `session` plus that session's `session_open`/`session_close`.
    """
    m = minutes.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    counts = {"minutes_in": len(minutes), "duplicates_dropped": len(minutes) - len(m)}
    ts = m["ts"].dt.tz_convert("UTC").dt.as_unit("ns")
    m = m.assign(ts=ts, session=ts.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize())
    bounds = sessions[["open", "close"]].rename(columns={"open": "session_open", "close": "session_close"})
    m = m.join(bounds, on="session", how="inner")
    m = m[(m["ts"] >= m["session_open"]) & (m["ts"] < m["session_close"])].reset_index(drop=True)
    counts["outside_regular_hours_dropped"] = counts["minutes_in"] - counts["duplicates_dropped"] - len(m)
    return m, counts


def resample_bars(rth, bar_minutes=5, min_coverage=0.8):
    """Intraday bars anchored to each session's open; returns (usable_bars, n_dropped).

    A bar ends at bar_start + size, or at the session close if that comes first. It
    is usable only if at least `min_coverage` of its expected minutes are present.
    """
    size = pd.Timedelta(minutes=bar_minutes)
    bar_start = rth["session_open"] + ((rth["ts"] - rth["session_open"]) // size) * size
    bars = (
        rth.assign(bar_start=bar_start)
        .groupby("bar_start", sort=True)
        .agg(session=("session", "first"), session_close=("session_close", "first"),
             open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), n_minutes=("ts", "size"))
        .reset_index()
    )
    end = bars["bar_start"] + size
    bars.insert(1, "bar_end", end.where(end <= bars["session_close"], bars["session_close"]))
    bars["coverage"] = bars["n_minutes"] / ((bars["bar_end"] - bars["bar_start"]) / MINUTE)
    usable = bars["coverage"] >= min_coverage
    return bars[usable].reset_index(drop=True), int((~usable).sum())


def daily_features(rth, sessions, min_coverage=0.8, ema_period=50, atr_period=14):
    """Regular-session daily bars from the same minutes, plus prior-session features.

    Row D holds session D's own bar and, in `prev_close` / `prev_ema_{p}` /
    `prev_atr_{p}`, the values from the previous XNYS session: the daily information
    usable throughout session D. Sessions below `min_coverage` are left out of the EMA
    and ATR (whose true range then uses the last usable close) and give NaN to the next day.
    """
    d = (
        rth.groupby("session")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), n_minutes=("ts", "size"))
        .reindex(sessions.index)
    )
    d["n_minutes"] = d["n_minutes"].fillna(0).astype(int)
    d["expected_minutes"] = ((sessions["close"] - sessions["open"]) / MINUTE).astype(int)
    d["usable"] = d["n_minutes"] >= min_coverage * d["expected_minutes"]
    ema = np.full(len(d), np.nan)
    usable = d["usable"].to_numpy()
    ema[usable] = talib.EMA(d["close"].to_numpy(float)[usable], timeperiod=ema_period)
    d[f"ema_{ema_period}"] = ema
    atr = np.full(len(d), np.nan)
    high, low, close = (d[k].to_numpy(float)[usable] for k in ("high", "low", "close"))
    atr[usable] = talib.ATR(high, low, close, timeperiod=atr_period)
    d[f"atr_{atr_period}"] = atr
    d["prev_close"] = d["close"].where(d["usable"]).shift(1)
    d[f"prev_ema_{ema_period}"] = d[f"ema_{ema_period}"].shift(1)
    d[f"prev_atr_{atr_period}"] = d[f"atr_{atr_period}"].shift(1)
    return d


def build_features(minutes, sessions, bar_minutes=5, ema_periods=(9, 21), daily_ema_period=50, min_coverage=0.8,
                   atr_period=14):
    """Everything a strategy may look at, computed over warm-up + study history.

    Returns (rth_minutes, bars, daily, info). `bars` holds usable bars only, with one
    `ema_{p}` column per distinct period (TA-Lib, continuous across sessions, never
    reset overnight) and the previous session's daily features. Indicators are
    computed on the full history first; the `in_study` flag does the trimming.
    """
    rth, info = regular_session_minutes(minutes, sessions)
    bars, n_dropped = resample_bars(rth, bar_minutes, min_coverage)
    close = bars["close"].to_numpy(float)
    for p in sorted(set(ema_periods)):
        bars[f"ema_{p}"] = talib.EMA(close, timeperiod=p)
    daily = daily_features(rth, sessions, min_coverage, daily_ema_period, atr_period)
    bars = bars.join(daily[["prev_close", f"prev_ema_{daily_ema_period}", f"prev_atr_{atr_period}"]], on="session")
    bars = bars.join(sessions["in_study"], on="session")

    in_study = sessions["in_study"]
    have, expected = daily["n_minutes"], daily["expected_minutes"]
    study_bars = bars[bars["in_study"]]
    info.update(
        sessions_total=len(sessions),
        sessions_study=int(in_study.sum()),
        sessions_without_data=[str(d.date()) for d in daily.index[have == 0]],
        sessions_partial=[str(d.date()) for d in daily.index[(have > 0) & (have < expected)]],
        rth_minutes_expected=int(expected.sum()),
        rth_minutes_present=int(have.sum()),
        bars_usable=len(bars),
        bars_dropped_low_coverage=n_dropped,
        study_bars=len(study_bars),
        study_bars_missing={f"ema_{p}": int(study_bars[f"ema_{p}"].isna().sum()) for p in sorted(set(ema_periods))},
        study_sessions_missing_daily=int(daily.loc[in_study, f"prev_ema_{daily_ema_period}"].isna().sum()),
        study_sessions_missing_atr=int(daily.loc[in_study, f"prev_atr_{atr_period}"].isna().sum()),
    )
    return rth, bars, daily, info
