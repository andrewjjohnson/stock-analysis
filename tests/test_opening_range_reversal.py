"""Opening-range reversal: real TA-Lib patterns on hand-built candles, causal timing, signed
outcomes, zero-trigger bypass, and the fixed barrier comparison.

Prices sit on a 1/64 grid so every comparison is exact. Warm-up sessions repeat one
pattern with daily open = close = 100, high 101 and low 99, so the daily ATR(14) that
the first study session sees is exactly 2.0 and 0.25 x ATR is exactly 0.5.
"""

import numpy as np
import pandas as pd
import pytest
import talib

import features
import outcomes
import run
import strategies.opening_range_reversal as orr
import synthetic

NY = "America/New_York"
ORR = "opening_range_reversal"
MIN = pd.Timedelta(minutes=1)
BULL, BEAR = (100.0, 100.25, 99.875, 100.125), (100.125, 100.25, 99.875, 100.0)

# Long day: the first 15 minutes fall from 100 to 99.625 (opening high 100.125, low 99.5, range 0.625).
LONG_OPENING = [(100.0, 100.125, 99.75, 99.875), (99.875, 100.0, 99.625, 99.75), (99.75, 99.875, 99.5, 99.625)]
HAMMER_DAY = LONG_OPENING + [(99.625, 99.625, 99.375, 99.375),      # 09:45 outside, no pattern
                             (99.375, 99.375, 99.125, 99.125),      # 09:50 outside, no pattern
                             (99.125, 99.15625, 98.875, 99.15625)]  # 09:55-10:00 hammer
# Short day: the first 15 minutes rise from 100 to 100.375 (opening high 100.5, low 99.875).
SHORT_OPENING = [BULL, (100.125, 100.375, 100.0, 100.25), (100.25, 100.5, 100.125, 100.375)]
ENGULF_DAY = SHORT_OPENING + [(100.375, 100.75, 100.375, 100.75),       # 09:45 outside, white
                              (100.75, 100.875, 100.6875, 100.8125),    # 09:50 small white
                              (100.8125, 100.9375, 100.625, 100.6875)]  # 09:55-10:00 bearish engulfing
STAR_DAY = SHORT_OPENING + [(100.375, 100.75, 100.375, 100.75),         # 09:45 outside, white
                            (100.8125, 101.0625, 100.78125, 100.78125)]  # 09:50-09:55 shooting star
DOJI = (99.25, 99.3125, 99.1875, 99.25)  # outside the long opening range, no pattern
BLACK = (99.25, 99.3125, 99.125, 99.1875)
WHITE_ENGULF = (99.1875, 99.3125, 99.125, 99.28125)  # bullish engulfing of BLACK


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def background(n):
    bars = [BULL if k % 2 == 0 else BEAR for k in range(n)]
    bars[20] = (100.0, 101.0, 99.875, 100.125)  # daily high 101
    bars[40] = (100.0, 100.25, 99.0, 100.125)   # daily low 99
    return bars


def bar_minutes(bar):
    """Five minutes whose 5-minute bar is exactly (o, h, l, c): low first on white bars, high first on black."""
    o, h, l, c = bar
    if c >= o:
        return [(o, o, l, l), (l, h, l, h), (h, h, c, c), (c, c, c, c), (c, c, c, c)]
    return [(o, h, o, h), (h, h, l, l), (l, c, l, c), (c, c, c, c), (c, c, c, c)]


def build(sessions, designed=None, drop=()):
    """Minute bars: background everywhere, `designed` 5-minute bars from the open of named sessions."""
    rows = []
    for day, s in sessions.iterrows():
        bars = background(int((s["close"] - s["open"]) / (5 * MIN)))
        custom = (designed or {}).get(str(day.date()), [])
        bars[:len(custom)] = custom
        for k, bar in enumerate(bars):
            for m, (o, h, l, c) in enumerate(bar_minutes(bar)):
                rows.append((s["open"] + (5 * k + m) * MIN, o, h, l, c))
    minutes = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).assign(
        volume=1000.0, vwap=100.0, transactions=10)
    minutes["ts"] = minutes["ts"].dt.as_unit("ns")
    return minutes[~minutes["ts"].isin([et(t) for t in drop])].reset_index(drop=True)


def one_session():
    return features.trading_sessions("2024-03-08", "2024-03-08", warmup_sessions=20)


def three_sessions():  # 2024-03-11 is the first session after the switch to daylight time
    return features.trading_sessions("2024-03-08", "2024-03-12", warmup_sessions=20)


def study(minutes, sessions, **kwargs):
    return run.run_study(minutes, sessions, strategy=ORR, **kwargs)


def signal_times(result):
    return [t.tz_convert(NY).strftime("%Y-%m-%d %H:%M") for t in result["candidates"]["signal_time"]]


def test_installed_talib_pattern_conventions():
    background_bars = [BULL if k % 2 == 0 else BEAR for k in range(12)]

    def last(fn, *extra):
        o, h, l, c = (np.array(x) for x in zip(*background_bars, *extra))
        return int(getattr(talib, fn)(o, h, l, c)[-1])

    down, up = (100.0, 100.0, 99.75, 99.75), (100.0, 100.25, 100.0, 100.25)
    assert last("CDLHAMMER", down, (99.75, 99.78125, 99.25, 99.78125)) == 100
    assert last("CDLSHOOTINGSTAR", up, (100.3125, 100.75, 100.28125, 100.28125)) == -100
    # TA-Lib's shooting star needs the body to gap above the previous body; touching is not enough.
    assert last("CDLSHOOTINGSTAR", up, (100.28125, 100.75, 100.25, 100.25)) == 0
    # The same upper-wick shape after a decline is the BULLISH inverted hammer, which we do not use.
    assert last("CDLINVERTEDHAMMER", down, (99.71875, 100.25, 99.6875, 99.6875)) == 100
    assert last("CDLENGULFING", down, (99.6875, 100.125, 99.625, 100.0625)) == 100
    assert last("CDLENGULFING", down, (99.75, 100.125, 99.625, 100.0625)) == 80  # open == previous close
    assert last("CDLENGULFING", (99.75, 100.0, 99.75, 100.0), (100.0625, 100.125, 99.5, 99.625)) == -100
    assert last("CDLENGULFING", (99.75, 100.0, 99.75, 100.0), (100.0, 100.125, 99.5, 99.625)) == -80


def test_known_long_and_short_candidates():
    sessions = three_sessions()
    minutes = build(sessions, {"2024-03-08": HAMMER_DAY, "2024-03-11": ENGULF_DAY, "2024-03-12": STAR_DAY})
    result = study(minutes, sessions, barriers=True)
    c = result["candidates"].set_index(result["candidates"]["session"].dt.strftime("%Y-%m-%d"))

    assert signal_times(result) == ["2024-03-08 10:00", "2024-03-11 10:00", "2024-03-12 09:55"]
    assert c["signal_time"].tolist() == [pd.Timestamp("2024-03-08 15:00", tz="UTC"),   # EST
                                         pd.Timestamp("2024-03-11 14:00", tz="UTC"),   # EDT
                                         pd.Timestamp("2024-03-12 13:55", tz="UTC")]
    assert c["side"].tolist() == ["long", "short", "short"]
    assert c["pattern"].tolist() == ["hammer", "engulfing", "shooting_star"]
    long_, short = c.loc["2024-03-08"], c.loc["2024-03-11"]
    assert (long_["or_open"], long_["or_high"], long_["or_low"], long_["or_close"]) == (100.0, 100.125, 99.5, 99.625)
    assert long_["opening_end"] == et("2024-03-08 09:45")
    assert long_["prev_atr_14"] == 2.0 and long_["or_range"] == 0.625 and long_["range_atr_ratio"] == 0.3125
    assert (long_["signal_open"], long_["signal_high"], long_["signal_low"], long_["ref_close"]) == \
        (99.125, 99.15625, 98.875, 99.15625)
    assert long_["hammer"] and not long_["engulfing"] and not long_["shooting_star"]
    assert long_["stop_price"] == 98.875 and long_["target_price"] == 100.125
    # Engulfing: the stop covers both candles (the signal high 100.9375 is above the previous 100.875).
    assert short["engulfing"] and short["stop_price"] == 100.9375 and short["target_price"] == 99.875
    assert c.loc["2024-03-12", "stop_price"] == 101.0625
    assert (c["entry_time"] == c["signal_time"]).all()

    # Session funnel: all three study sessions eligible; each has one qualifying opening.
    rows = result["summary"].set_index(["side", "pattern"])
    assert rows.loc[("long", "all"), "eligible_sessions"] == 3
    assert rows.loc[("long", "all"), "qualifying_openings"] == 1 and rows.loc[("short", "all"), "qualifying_openings"] == 2
    assert rows.loc[("short", "engulfing"), "n_candidates"] + rows.loc[("short", "shooting_star"), "n_candidates"] == 2
    assert np.isnan(rows.loc[("short", "engulfing"), "qualifying_openings"])  # counts only on side rows


def test_threshold_boundary_uses_greater_or_equal():
    sessions = one_session()
    minutes = build(sessions, {"2024-03-08": HAMMER_DAY})
    # Opening range 0.625 vs prior ATR 2.0: 0.3125 x 2.0 is exactly 0.625; one 1/64 step higher fails.
    result = study(minutes, sessions, thresholds=[0.3125, 0.328125])
    assert result["candidates"]["config"].tolist() == ["threshold=0.31"]
    summary = result["summary"].set_index(["threshold", "side", "pattern"])
    assert summary.loc[(0.3125, "long", "all"), "qualifying_openings"] == 1
    assert summary.loc[(0.328125, "long", "all"), "qualifying_openings"] == 0


@pytest.mark.parametrize("name, bars, drop, expected", [
    ("hammer on the first eligible bar", LONG_OPENING + [(99.46875, 99.484375, 99.25, 99.484375)], (), "09:50"),
    ("engulfing completing 10:55", LONG_OPENING + [DOJI] * 12 + [BLACK, WHITE_ENGULF], (), "10:55"),
    # The bearish engulfing at 10:50-10:55 is ignored on a long day; the 10:55-11:00 bar is too late.
    ("engulfing completing 11:00", LONG_OPENING + [DOJI] * 13 + [BLACK, WHITE_ENGULF], (), None),
    ("close on the opening low", LONG_OPENING + [DOJI, DOJI, (99.4375, 99.46875, 99.375, 99.40625),
                                                 (99.40625, 99.5, 99.375, 99.5)], (), None),
    ("candle straddling the opening low", LONG_OPENING + [DOJI, DOJI, (99.4375, 99.46875, 99.375, 99.40625),
                                                          (99.40625, 99.625, 99.375, 99.484375)], (), "10:05"),
    ("missing minute in the signal bar", LONG_OPENING + [DOJI] * 12 + [BLACK, WHITE_ENGULF], ["2024-03-08 10:53"], None),
    ("missing minute in the paired bar", LONG_OPENING + [DOJI] * 12 + [BLACK, WHITE_ENGULF], ["2024-03-08 10:48"], None),
    ("missing opening minute", LONG_OPENING + [DOJI] * 12 + [BLACK, WHITE_ENGULF], ["2024-03-08 09:31"], None),
    ("flat opening", [LONG_OPENING[0], LONG_OPENING[1], (99.75, 100.0, 99.625, 100.0)]
                     + [DOJI] * 12 + [BLACK, WHITE_ENGULF], (), None),
])
def test_signal_window_outside_and_completeness(name, bars, drop, expected):
    sessions = one_session()
    result = study(build(sessions, {"2024-03-08": bars}, drop), sessions)
    assert signal_times(result) == ([] if expected is None else [f"2024-03-08 {expected}"]), name
    counts = result["summary"].set_index("side").loc["long"].iloc[0]
    assert counts["or_incomplete"] == (1 if name == "missing opening minute" else 0)
    assert counts["flat_openings_skipped"] == (1 if name == "flat opening" else 0)


def test_zero_triggers_build_no_rows_or_outcomes(monkeypatch, tmp_path):
    sessions = three_sessions()
    minutes = build(sessions)  # background only: opening range 0.375 < 0.25 x 2.0

    def forbidden(*args, **kwargs):
        raise AssertionError("called without any candidates")

    for module, name in ((outcomes, "forward_outcomes"), (outcomes, "barrier_exits"), (orr, "barrier_levels"),
                         (run, "build_candidate_rows")):
        monkeypatch.setattr(module, name, forbidden)
    args = run.parse_args(["--strategy", ORR, "--barriers", "--start", "2024-03-08", "--end", "2024-03-12",
                           "--warmup-sessions", "20", "--out", str(tmp_path)])
    result = run.execute(args, minutes, sessions, "SYNTHETIC test fixture", {})

    cands = result["candidates"]
    assert cands.empty and {"ticker", "side", "pattern", "or_low", "fwd_ret_30m_pct", "exit_reason",
                            "barrier_r_1bp"} <= set(cands.columns)
    side_rows = result["summary"][result["summary"]["pattern"] == "all"]
    assert side_rows["eligible_sessions"].tolist() == [3, 3]
    assert (side_rows["qualifying_openings"] == 0).all() and (side_rows["n_candidates"] == 0).all()
    assert side_rows[["mean_30m_pct", "frac_pos_30m", "mean_r_0bp", "frac_pos_r_1bp"]].isna().all().all()
    assert result["candidate_charts"] == []
    assert pd.read_parquet(tmp_path / "candidates.parquet").empty


def test_later_data_cannot_change_earlier_triggers_and_atr_is_prior_only():
    sessions = three_sessions()
    designed = {"2024-03-08": HAMMER_DAY, "2024-03-11": ENGULF_DAY, "2024-03-12": STAR_DAY}
    minutes = build(sessions, designed)
    cut = et("2024-03-08 10:02")
    changed = minutes.copy()
    later = changed["ts"] >= cut
    changed.loc[later, ["open", "high", "low", "close"]] *= np.linspace(0.97, 1.05, later.sum())[:, None]
    # Also add a later, "better" candidate on the same session: it must not displace the first.
    after = build(sessions, {**designed, "2024-03-08": HAMMER_DAY + [DOJI] * 4 + [BLACK, WHITE_ENGULF]})

    base = study(minutes, sessions)
    for variant in (study(changed, sessions), study(after, sessions)):
        first = variant["candidates"].iloc[0]
        assert first["signal_time"] == et("2024-03-08 10:00") and first["pattern"] == "hammer"
        known = (base["bars"]["bar_end"] <= cut).to_numpy()
        cols = ["open", "high", "low", "close", "or_low", "or_high", "prev_atr", "hammer", "engulfing", "setup"]
        pd.testing.assert_frame_equal(base["bars"].loc[known, cols], variant["bars"].loc[known, cols])

    # Prior-session ATR: today's range never reaches today's value; it reaches tomorrow's.
    day = lambda r, d: r["bars"].loc[r["bars"]["session"] == d, "prev_atr"].iloc[0]  # noqa: E731
    assert day(base, "2024-03-08") == day(study(changed, sessions), "2024-03-08") == 2.0
    assert day(base, "2024-03-11") != day(study(changed, sessions), "2024-03-11")
    # ...and it equals TA-Lib ATR(14) on regular-session daily bars through the previous session only.
    daily = (minutes.assign(session=minutes["ts"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize())
             .groupby("session").agg(high=("high", "max"), low=("low", "min"), close=("close", "last")))
    for d in ("2024-03-11", "2024-03-12"):
        prior = daily[daily.index < d]
        expected = talib.ATR(*(prior[k].to_numpy() for k in ("high", "low", "close")), timeperiod=14)[-1]
        assert np.isclose(day(base, d), expected)


def test_short_returns_are_signed_with_exact_horizons_and_unavailable_kept():
    sessions = one_session()
    s = sessions.iloc[-1]
    ts = pd.date_range(s["open"], s["close"], freq="1min", inclusive="left").as_unit("ns")
    close = 100 - 0.01 * np.arange(len(ts))  # falls 0.01 per minute
    minutes = pd.DataFrame({"ts": ts, "open": close + 0.01, "high": close + 0.02, "low": close - 0.03, "close": close})
    minutes = minutes[minutes["ts"] != et("2024-03-08 12:00")].reset_index(drop=True)  # one missing minute
    t = pd.Series([et("2024-03-08 10:00"), et("2024-03-08 11:40"), et("2024-03-08 15:30")])
    ref = [100 - 0.01 * ((x - s["open"]) / MIN - 1) for x in t]  # close of the minute ending at T
    close_at = pd.Series([s["close"]] * 3)

    long_ = outcomes.forward_outcomes(minutes, t, ref, close_at, s["close"], side=[1, 1, 1])
    short = outcomes.forward_outcomes(minutes, t, ref, close_at, s["close"], side=[-1, -1, -1])
    for h in outcomes.HORIZONS:
        pd.testing.assert_series_equal(short[f"fwd_ret_{h}m_pct"], -long_[f"fwd_ret_{h}m_pct"], check_names=False)
    assert np.isclose(short.loc[0, "fwd_ret_30m_pct"], -((ref[0] - 0.30) / ref[0] - 1) * 100)  # exact T+30 close
    assert short.loc[0, "fwd_ret_30m_pct"] > 0  # price fell: favourable for a short
    pd.testing.assert_series_equal(short["mfe_60m_pct"], -long_["mae_60m_pct"], check_names=False)
    pd.testing.assert_series_equal(short["mae_60m_pct"], -long_["mfe_60m_pct"], check_names=False)
    assert (short["mfe_60m_pct"].dropna() >= 0).all() and (short["mae_60m_pct"].dropna() <= 0).all()
    # 11:40 + 30 min crosses the missing 12:00 minute; 15:30 + 60 min crosses the close. Rows are kept.
    assert len(short) == 3
    assert short["fwd_ret_10m_pct"].notna().all()
    assert short["fwd_ret_30m_pct"].isna().tolist() == [False, True, False]
    assert short["fwd_ret_60m_pct"].isna().tolist() == [False, True, True]


def barrier_case(changes, side=1, stop=99.0, target=101.0, drop=(), segment_end=None, entry="10:00"):
    """One barrier path on a flat-100 session with minute overrides {"HH:MM": (o, h, l, c)}."""
    s = one_session().iloc[-1]
    ts = pd.date_range(s["open"], s["close"], freq="1min", inclusive="left").as_unit("ns")
    minutes = pd.DataFrame({"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    for hhmm, ohlc in changes.items():
        minutes.loc[minutes["ts"] == et(f"2024-03-08 {hhmm}"), ["open", "high", "low", "close"]] = ohlc
    minutes = minutes[~minutes["ts"].isin([et(f"2024-03-08 {d}") for d in drop])].reset_index(drop=True)
    out = outcomes.barrier_exits(minutes, pd.Series([et(f"2024-03-08 {entry}")]), [side], [stop], [target],
                                 pd.Series([s["close"]]), s["close"] if segment_end is None else et(segment_end))
    return out.iloc[0]


def test_barrier_exits_gaps_ambiguity_geometry_and_missing_data():
    r = barrier_case({"10:03": (100.5, 101.0, 100.4, 100.9)})
    assert (r["exit_reason"], r["exit_price"], r["barrier_r_0bp"]) == ("target", 101.0, 1.0)
    assert r["exit_time"] == et("2024-03-08 10:04") and r["holding_minutes"] == 4
    assert np.isclose(r["barrier_r_1bp"], (1.0 - 1e-4 * (100 + 101)) / 1.0)
    assert np.isclose(r["barrier_ret_1bp_pct"], (1.0 - 1e-4 * (100 + 101)) / 100 * 100)

    r = barrier_case({"10:05": (98.5, 98.6, 98.4, 98.5)})  # opens through the stop: exit at that open
    assert (r["exit_reason"], r["exit_price"], r["barrier_r_0bp"], r["exit_time"]) == \
        ("stop_gap", 98.5, -1.5, et("2024-03-08 10:05"))
    r = barrier_case({"10:05": (101.5, 101.6, 101.4, 101.5)})  # opens through the target: exit AT the target
    assert (r["exit_reason"], r["exit_price"], r["barrier_r_0bp"]) == ("target", 101.0, 1.0)
    r = barrier_case({"10:04": (100.0, 101.2, 98.8, 100.0)})  # both inside one minute: stop first, flagged
    assert (r["exit_reason"], r["exit_price"], r["ambiguous"]) == ("stop", 99.0, True)
    r = barrier_case({"10:00": (100.0, 100.1, 98.9, 99.5)})  # stop touched inside the entry minute itself
    assert (r["exit_reason"], r["exit_time"], r["ambiguous"]) == ("stop", et("2024-03-08 10:01"), False)
    r = barrier_case({"10:07": (101.5, 101.6, 101.4, 101.5)}, side=-1, stop=101.0, target=99.0)  # short
    assert (r["exit_reason"], r["exit_price"], r["barrier_r_0bp"]) == ("stop_gap", 101.5, -1.5)
    assert np.isclose(r["barrier_ret_0bp_pct"], -1.5)

    r = barrier_case({"10:00": (98.9, 99.0, 98.8, 98.9)})  # entry already beyond the stop
    assert r["entry_status"] == "invalid" and r["entry_price"] == 98.9 and pd.isna(r["exit_reason"])
    assert np.isnan(r["barrier_r_0bp"]) and pd.isna(r["exit_time"])
    assert barrier_case({"10:00": (101.0, 101.0, 101.0, 101.0)})["entry_status"] == "invalid"  # entry == target
    assert barrier_case({}, drop=["10:00"])["entry_status"] == "unavailable"

    r = barrier_case({"10:06": (100.5, 101.0, 100.5, 101.0)}, drop=["10:03"])  # gap before the touch
    assert r["entry_status"] == "ok" and r["exit_reason"] == "unresolved" and np.isnan(r["barrier_r_0bp"])
    r = barrier_case({"10:02": (100.5, 101.0, 100.5, 101.0)}, drop=["10:05"])  # exit observed before the gap
    assert r["exit_reason"] == "target"
    r = barrier_case({})  # nothing touched: the regular-session close
    assert (r["exit_reason"], r["exit_price"], r["exit_time"]) == ("close", 100.0, et("2024-03-08 16:00"))
    assert r["holding_minutes"] == 360 and r["barrier_r_0bp"] == 0.0
    assert barrier_case({}, segment_end="2024-03-08 14:00")["exit_reason"] == "unresolved"  # no invented close


def test_invalid_first_signal_is_kept_not_replaced():
    sessions = one_session()
    # Entry minute opens above the target (opening high 100.125); a valid engulfing completes at 10:15.
    bars = HAMMER_DAY + [(100.25, 100.375, 100.1875, 100.3125), BLACK, WHITE_ENGULF]
    result = study(build(sessions, {"2024-03-08": bars}), sessions, barriers=True)
    later = result["bars"]["bar_end"] == et("2024-03-08 10:15")
    assert result["bars"].loc[later, "setup"].item()  # the later bar qualifies on its own...
    cands = result["candidates"]
    assert signal_times(result) == ["2024-03-08 10:00"]  # ...but the first signal stands
    assert cands.iloc[0]["entry_status"] == "invalid" and cands.iloc[0]["entry_price"] == 100.25
    row = result["summary"].set_index(["side", "pattern"]).loc[("long", "all")]
    assert (row["n_candidates"], row["n_entry_invalid"], row["n_entry_ok"]) == (1, 1, 0)
    assert np.isnan(row["mean_r_0bp"])


def test_split_selects_each_side_on_earlier_rows_only(tmp_path):
    sessions = features.trading_sessions("2024-01-02", "2024-12-31", warmup_sessions=120)
    minutes = synthetic.random_walk_minutes(sessions, seed=7)
    args = run.parse_args(["--strategy", ORR, "--barriers", "--threshold", "0.20", "0.25", "0.30", "--start",
                           "2024-01-02", "--end", "2024-12-31", "--split-date", "2024-07-01", "--min-labeled", "5",
                           "--out", str(tmp_path)])
    result = run.execute(args, minutes, sessions, "SYNTHETIC test fixture", {})
    summary, cands = result["summary"], result["candidates"]
    earlier = summary[(summary["segment"] == "earlier") & (summary["pattern"] == "all")]
    for side, pick in result["selected"].items():
        rows = earlier[(earlier["side"] == side) & (earlier["n_30m"] >= 5)]
        assert pick == rows.sort_values("mean_30m_pct", ascending=False)["config"].iloc[0]
    later = cands[cands["segment"] == "later"]
    assert set(zip(later["side"], later["config"])) == set(result["selected"].items())  # each side's own pick only
    last_early_close = sessions.loc["2024-06-28", "close"]
    early = cands[cands["segment"] == "earlier"]
    assert (early["exit_time"].dropna() <= last_early_close).all()  # earlier labels never reach past the split
    assert (early["signal_time"] + pd.Timedelta(minutes=120))[early["fwd_ret_120m_pct"].notna()].le(last_early_close).all()
