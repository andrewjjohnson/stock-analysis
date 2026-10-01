"""VWAP + Sauce: anchored VWAP bands, the Setup A state machine, the optional modules, fills,
one-position rule, symmetry, no lookahead and the CLI outputs.

Most tests hand-build 2-minute bars with VWAP fixed at 100 and sigma at 1, so the bands are
L2 98, L1.5 98.5, L1 99, U1 101, U1.5 101.5 and U2 102, and FAST/SLOW are given directly.
"""

import numpy as np
import pandas as pd
import pytest

import features
import sauce
import strategies.vwap_sauce as vs
import synthetic

NY = "America/New_York"
BAR = pd.Timedelta(minutes=2)

# Setup A on the lower band: (close, FAST, SLOW), optionally (open, high, low, close) as a 4th item.
# FAST closes below L2 = 98 at bar 1 (arm). SLOW - L2 shrinks to 0.20 and flattens: the 5-bar
# slope is -0.022 sigma/bar at bar 7 (not flat) and -0.004 at bar 8 (parallel). FAST closes back
# above L2 at bar 9 (signal); entry at bar 10's open 98.3; bar 11's high touches L1.5 = 98.5.
LONG = [
    (98.9, 98.8, 99.2),
    (97.9, 97.9, 98.8),
    (97.5, 97.6, 98.5),
    (97.3, 97.5, 98.3),  # low 97.2 = setup_extreme
    (97.6, 97.6, 98.22),
    (97.8, 97.7, 98.2),
    (97.9, 97.8, 98.2),
    (98.0, 97.9, 98.2),
    (98.1, 97.95, 98.2),
    (98.25, 98.1, 98.21),
    (98.35, 98.2, 98.22, (98.3, 98.4, 98.2, 98.35)),
    (98.55, 98.3, 98.23, (98.35, 98.6, 98.3, 98.55)),
]
FILLER = (98.3, 98.35, 98.24)  # FAST inside, high 98.4 below L1.5, low far above any stop


def bars_for(rows, day="2024-03-04", vwap=100.0, sigma=1.0, filler=3):
    """One session of hand-built 2-minute bars from 09:30 ET; `filler` quiet bars end it."""
    rows = list(rows) + [FILLER] * filler
    open_ = pd.Timestamp(f"{day} 09:30", tz=NY).tz_convert("UTC")
    out = []
    for k, r in enumerate(rows):
        o, h, l, c = r[3] if len(r) > 3 else (r[0], r[0] + 0.1, r[0] - 0.1, r[0])
        out.append({"session": pd.Timestamp(day), "bar_start": open_ + k * BAR, "bar_end": open_ + (k + 1) * BAR,
                    "open": o, "high": h, "low": l, "close": c, "fast": r[1], "slow": r[2]})
    b = pd.DataFrame(out)
    b["vwap"], b["sigma"], b["in_study"] = vwap, sigma, True
    return b


def with_row(rows, k, row):
    rows = list(rows)
    rows[k] = row
    return rows


def mirror(bars):
    """Reflect every price around VWAP 100: the lower-band long becomes an upper-band short."""
    m = bars.copy()
    for k in ("open", "close", "fast", "slow"):
        m[k] = 200 - bars[k]
    m["high"], m["low"] = 200 - bars["low"], 200 - bars["high"]
    return m


def trades(log):
    return log[log["trade_no"].notna()].reset_index(drop=True)


def et(ts):
    return ts.tz_convert(NY).strftime("%H:%M")


def test_setup_a_long_arms_goes_parallel_enters_next_open_and_exits_at_the_moving_target():
    log = vs.simulate(bars_for(LONG))
    assert len(log) == 1
    r = log.iloc[0]
    assert (r["setup"], r["band"], r["side"], r["outcome"]) == ("A", "lower", "long", "exited")
    assert et(r["arm_time"]) == "09:34" and r["setup_extreme"] == pytest.approx(97.2)
    assert et(r["slow_parallel_time"]) == "09:48"  # bar 8, not bar 7: the slope must flatten first
    assert pd.isna(r["invalidation_time"])
    assert et(r["signal_time"]) == "09:50" and et(r["entry_time"]) == "09:50"  # bar 9 close -> bar 10 open
    assert (r["entry_type"], r["entry_price"], r["stop_price"]) == ("standard", 98.3, pytest.approx(97.2))
    assert (r["target_level"], r["target_price_at_entry"]) == ("L1.5", pytest.approx(98.5))
    assert (r["exit_reason"], et(r["exit_time"]), r["exit_price"]) == ("target", "09:54", pytest.approx(98.5))
    assert r["pnl_usd"] == pytest.approx(0.2) and r["pnl_sigma"] == pytest.approx(0.2)
    assert r["bars_held"] == 2 and not r["ambiguous"]


def test_upper_band_is_the_exact_mirror_of_the_lower_band():
    long, short = vs.simulate(bars_for(LONG)), vs.simulate(mirror(bars_for(LONG)))
    assert (short["band"].iloc[0], short["side"].iloc[0], short["target_level"].iloc[0]) == ("upper", "short", "U1.5")
    assert short["setup_extreme"].iloc[0] == pytest.approx(200 - long["setup_extreme"].iloc[0])
    same = ["arm_time", "slow_parallel_time", "signal_time", "entry_time", "exit_time", "exit_reason", "pnl_usd",
            "pnl_sigma", "bars_held", "outcome"]
    pd.testing.assert_frame_equal(long[same], short[same], check_exact=False)


def test_slow_crossing_l2_invalidates_and_the_side_rearms_only_after_fast_comes_back_inside():
    rows = with_row(LONG, 5, (97.8, 97.7, 97.9))  # SLOW closes below L2 = 98 at bar 5
    rows = with_row(rows, 10, (98.35, 97.9, 98.22))  # FAST back outside after closing inside at bar 9
    log = vs.simulate(bars_for(rows))
    first, second = log.iloc[0], log.iloc[1]
    assert (first["outcome"], first["invalidation_reason"]) == ("invalidated", "slow_crossed_band")
    assert et(first["invalidation_time"]) == "09:42" and first["n_trades"] == 0
    # Bars 6-8 have FAST outside but may not re-arm; FAST closes inside at bar 9, outside again at bar 10.
    assert et(second["arm_time"]) == "09:52"


def test_slow_parallel_requirement_can_be_switched_off():
    rows = [(c, f, s + 1.0) for c, f, s, *rest in LONG]  # SLOW never comes near L2
    rows = [r + tuple(x[3:]) for r, x in zip(rows, LONG)]
    required = vs.simulate(bars_for(rows))
    assert required["outcome"].tolist() == ["no_parallel"] and trades(required).empty
    optional = trades(vs.simulate(bars_for(rows), require_slow_parallel=False))
    assert len(optional) == 1 and pd.isna(optional["slow_parallel_time"].iloc[0])
    assert et(optional["entry_time"].iloc[0]) == "09:50"


def test_unlatched_slow_parallel_must_hold_on_the_entry_bar():
    rows = with_row(LONG, 9, (98.25, 98.1, 98.3))  # SLOW lifts 0.3 sigma off L2 on the signal bar: not parallel now
    latched = vs.simulate(bars_for(rows))
    assert et(latched["slow_parallel_time"].iloc[0]) == "09:48" and len(trades(latched)) == 1
    unlatched = vs.simulate(bars_for(rows), latch_slow_parallel=False)
    assert unlatched["outcome"].tolist() == ["no_parallel"] and trades(unlatched).empty
    assert len(trades(vs.simulate(bars_for(LONG), latch_slow_parallel=False))) == 1  # still parallel at bar 9


def test_price_stop_structure_stop_ambiguity_gaps_and_time_stop():
    # Price stop touched and target touched in one bar: ambiguous, counted as the stop at setup_extreme.
    both = with_row(LONG, 11, (98.55, 98.3, 98.23, (98.35, 98.6, 97.1, 98.55)))
    t = trades(vs.simulate(bars_for(both))).iloc[0]
    assert (t["exit_reason"], t["exit_price"], t["ambiguous"]) == ("price_stop", pytest.approx(97.2), True)
    t = trades(vs.simulate(bars_for(both), ambiguous_fill="target")).iloc[0]  # the benefit of the doubt
    assert (t["exit_reason"], t["exit_price"], t["ambiguous"]) == ("target", pytest.approx(98.5), True)
    # A stop 0.2 sigma wider (97.0) survives that bar's low of 97.1, and the target fills cleanly.
    t = trades(vs.simulate(bars_for(both), stop_buffer_sigma=0.2)).iloc[0]
    assert (t["stop_price"], t["exit_reason"], t["ambiguous"]) == (pytest.approx(97.0), "target", False)
    # A bar opening below the stop fills at that open, at the bar start.
    gap = with_row(LONG, 11, (97.1, 98.3, 98.23, (97.0, 97.3, 96.9, 97.1)))
    t = trades(vs.simulate(bars_for(gap))).iloc[0]
    assert (t["exit_reason"], t["exit_price"], et(t["exit_time"]), t["bars_held"]) == ("price_stop", 97.0, "09:52", 1)
    # FAST closing back below L2 exits at the next bar's open.
    structure = with_row(LONG, 11, (98.3, 97.9, 98.23, (98.35, 98.45, 98.25, 98.3)))
    t = trades(vs.simulate(bars_for(structure))).iloc[0]
    assert (t["exit_reason"], et(t["exit_time"]), t["exit_price"], t["bars_held"]) == (
        "structure_stop", "09:54", 98.3, 2)
    assert trades(vs.simulate(bars_for(structure), structure_stop=False))["exit_reason"].iloc[0] == "time_stop"
    # An entry bar that opens beyond the target exits at that same open: no room, zero P&L.
    no_room = with_row(LONG, 10, (98.6, 98.2, 98.22, (98.6, 98.7, 98.5, 98.6)))
    t = trades(vs.simulate(bars_for(no_room))).iloc[0]
    assert (t["exit_reason"], t["pnl_usd"], t["bars_held"]) == ("target", 0.0, 0)
    assert t["exit_time"] == t["entry_time"]
    # Nothing touched: flat at the session's last bar close.
    quiet = with_row(LONG, 11, (98.3, 98.3, 98.23, (98.35, 98.45, 98.25, 98.3)))
    t = trades(vs.simulate(bars_for(quiet, filler=0))).iloc[0]
    assert (t["exit_reason"], t["exit_price"], et(t["exit_time"])) == ("time_stop", 98.3, "09:54")


def two_days(first, second):
    return pd.concat([bars_for(first, "2024-03-04", filler=0), bars_for(second, "2024-03-05", filler=0)],
                     ignore_index=True)


def test_sessions_are_independent_unless_the_time_stop_is_off():
    # A signal on a session's last bar (bar 9) never fills at the next session's open; the setup ends.
    log = vs.simulate(two_days(LONG[:10], [FILLER] * 3))
    assert trades(log).empty and log["outcome"].tolist() == ["session_end"]
    # A B cross compares FAST with the previous bar of the same session only.
    assert vs.simulate(two_days([(100.0, 99.9, 99.0)] * 3, [(100.2, 100.1, 99.0)] * 3),
                       enable_vwap_to_vwap=True).empty
    # With the time stop off, a position survives the close and meets its target the next morning.
    quiet = with_row(LONG, 11, (98.3, 98.3, 98.23, (98.35, 98.45, 98.25, 98.3)))
    b = two_days(quiet, [(98.55, 98.4, 98.25, (98.3, 98.6, 98.25, 98.55)), FILLER])
    on = trades(vs.simulate(b)).iloc[0]
    assert (on["exit_reason"], on["exit_time"].tz_convert(NY)) == ("time_stop", pd.Timestamp("2024-03-04 09:54", tz=NY))
    off = trades(vs.simulate(b, time_stop=False)).iloc[0]
    assert (off["exit_reason"], off["exit_price"], off["bars_held"]) == ("target", pytest.approx(98.5), 3)
    assert off["exit_time"].tz_convert(NY) == pd.Timestamp("2024-03-05 09:32", tz=NY)


def test_target_level_is_the_one_known_at_the_previous_close():
    b = bars_for(LONG)
    b.loc[11, "vwap"] = 99.5  # bar 11's own L1.5 is 98.0, below its open 98.35...
    b.loc[11, ["high", "close"]] = 98.45, 98.4  # ...but the level known when it opened (98.5) is not touched
    t = trades(vs.simulate(b)).iloc[0]
    # Bar 12 is judged against bar 11's 98.0: it opens beyond it and exits at that open, not at bar 11's.
    assert (t["exit_reason"], t["exit_price"], et(t["exit_time"]), t["bars_held"]) == ("target", 98.3, "09:54", 2)


def test_fade_entry_needs_a_green_day_and_extended_target_uses_l1():
    rows = LONG + [(98.4, 98.3, 98.24), (97.8, 97.9, 98.23), (97.9, 97.85, 98.23), (98.0, 97.9, 98.23)]
    log = trades(vs.simulate(bars_for(rows), allow_fade_entry=True))
    # The first setup was parallel at bar 8 with FAST outside, but the day had no P&L yet: standard entry.
    assert log["entry_type"].tolist() == ["standard", "fade"]
    assert et(log["signal_time"].iloc[1]) == "09:58"  # arm bar 13 is already parallel and the day is green
    assert log["pnl_usd"].iloc[0] > 0
    default = trades(vs.simulate(bars_for(rows)))  # without fades the second setup waits for FAST to come back
    assert default["entry_type"].tolist() == ["standard", "standard"] and et(default["signal_time"].iloc[1]) == "10:04"
    ext = trades(vs.simulate(bars_for(LONG), allow_extended_target=True)).iloc[0]
    assert (ext["target_level"], ext["target_price_at_entry"], ext["exit_reason"]) == ("L1", 99.0, "time_stop")
    far = trades(vs.simulate(bars_for(LONG), target_level=0)).iloc[0]
    assert (far["target_level"], far["target_price_at_entry"]) == ("VWAP", 100.0)


def test_reentry_after_a_structure_stop_on_a_pullback_to_fast():
    rows = with_row(LONG, 11, (98.3, 97.9, 98.23, (98.35, 98.45, 98.25, 98.3)))  # structure stop -> exit bar 12
    rows = rows + [(98.1, 97.95, 98.23, (98.0, 98.2, 97.9, 98.1))]  # bar 12: low <= FAST < close
    log = trades(vs.simulate(bars_for(rows), allow_reentry=True))
    assert log["entry_type"].tolist() == ["standard", "reentry"] and log["instance"].nunique() == 1
    assert log["trade_no"].tolist() == [1, 2] and et(log["entry_time"].iloc[1]) == "09:56"
    assert trades(vs.simulate(bars_for(rows), allow_reentry=True, max_reentries=0))["trade_no"].tolist() == [1]
    assert len(trades(vs.simulate(bars_for(rows)))) == 1


def test_continuation_trades_off_fast_and_hands_off_to_the_reversal_at_the_same_open():
    rows = with_row(LONG, 6, (97.7, 97.8, 98.2))  # bar 5 closed above FAST; bar 6 closes back below it
    # The stop is the pullback bar's high (97.9). Bar 7 opens at 98.0, already above it: out at that open.
    stopped = trades(vs.simulate(bars_for(rows), enable_continuation=True)).query("setup == 'A_cont'").iloc[0]
    assert (stopped["setup"], stopped["stop_price"], stopped["exit_reason"], stopped["exit_price"]) == (
        "A_cont", pytest.approx(97.9), "price_stop", 98.0)
    assert stopped["bars_held"] == 0
    rows = with_row(rows, 5, (97.8, 97.7, 98.2, (97.6, 98.4, 97.55, 97.8)))  # a pullback bar with a high of 98.4
    log = trades(vs.simulate(bars_for(rows), enable_continuation=True))
    cont, rev = log[log["setup"] == "A_cont"].iloc[0], log[log["setup"] == "A"].iloc[0]
    assert (cont["band"], cont["side"], et(cont["pullback_time"]), et(cont["entry_time"])) == (
        "lower", "short", "09:42", "09:44")
    assert pd.isna(cont["target_level"]) and cont["stop_price"] == 98.4
    assert len(trades(vs.simulate(bars_for(rows), enable_continuation=True, price_stop=False))) == 2
    # FAST closes back above L2 at bar 9: the short exits and the long enters at bar 10's open.
    assert (cont["exit_reason"], et(cont["exit_time"]), cont["exit_price"]) == ("structure_stop", "09:50", 98.3)
    assert (et(rev["entry_time"]), rev["entry_price"]) == ("09:50", 98.3)
    assert cont["pnl_usd"] == pytest.approx(98.0 - 98.3)  # short from bar 7's open 98.0


def test_vwap_to_vwap_bias_pullback_target_head_fake_and_center_tag():
    rows = [(100.0, 99.9, 99.0), (100.2, 100.1, 99.0), (100.4, 100.2, 99.0, (100.3, 100.45, 100.15, 100.4)),
            (100.6, 100.4, 99.1), (101.0, 100.7, 99.1, (100.8, 101.1, 100.7, 101.0))]
    log = vs.simulate(bars_for(rows, filler=0), enable_vwap_to_vwap=True)
    t = trades(log).iloc[0]
    assert (t["setup"], t["side"], t["level_crossed"], t["center_cross"]) == ("B", "long", "VWAP", True)
    assert (et(t["arm_time"]), et(t["signal_time"]), et(t["entry_time"]), t["entry_price"]) == (
        "09:34", "09:36", "09:36", 100.6)
    assert (t["target_level"], t["exit_reason"], t["exit_price"]) == ("U1", "target", 101.0)
    # A head fake: FAST closes back below VWAP before any pullback entry. No trade, and a down bias starts.
    fake = with_row(rows, 2, (99.9, 99.95, 99.0))
    log = vs.simulate(bars_for(fake, filler=0), enable_vwap_to_vwap=True)
    first = log.iloc[0]
    assert (first["outcome"], first["invalidation_reason"], et(first["invalidation_time"])) == (
        "head_fake", "fast_back_across_level", "09:36")
    assert (log.iloc[1]["side"], log.iloc[1]["level_crossed"]) == ("short", "VWAP")
    assert vs.summarize(log, 1)[6]["n_head_fake"] >= 1  # B, all, both sides


def test_one_position_at_a_time_blocks_other_entries():
    # Bar 9 (FAST back above L2) is also a B cross of L2; B's first pullback (bar 10) comes while the
    # Setup A long that bar 9 triggered is open, so it is blocked and never trades.
    log = vs.simulate(bars_for(LONG), enable_vwap_to_vwap=True)
    b = log[(log["setup"] == "B") & (log["level_crossed"] == "L2")].iloc[0]
    assert (b["outcome"], b["n_blocked"], b["n_trades"]) == ("blocked", 1, 0)
    assert trades(log)["setup"].tolist() == ["A"]


def test_zero_setups_give_an_empty_log_and_nan_statistics():
    flat = vs.simulate(bars_for([(100.0, 100.0, 100.0)] * 20))
    assert flat.empty and list(flat.columns) == list(vs.LOG_COLUMNS)
    a = vs.summarize(flat, 3)[0]
    assert (a["setups"], a["trades"], a["setups_per_month"]) == (0, 0, 0.0)
    assert np.isnan(a["win_rate"]) and np.isnan(a["expectancy_usd"]) and np.isnan(a["max_drawdown_usd"])


def test_trade_statistics():
    log = pd.DataFrame({"entry_time": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
                        "pnl_usd": [1.0, -2.0, 0.5, -1.0, 3.0], "pnl_sigma": [0.5, -1.0, 0.25, -0.5, 1.5],
                        "exit_reason": ["target"] * 5, "entry_type": ["standard"] * 5,
                        "ambiguous": pd.array([False] * 5, dtype="boolean"), "bars_held": [3] * 5})
    s = vs.trade_stats(log)
    assert (s["trades"], s["wins"], s["losses"], s["win_rate"]) == (5, 3, 2, pytest.approx(0.6))
    assert (s["avg_win_usd"], s["avg_loss_usd"], s["expectancy_usd"]) == pytest.approx((1.5, -1.5, 0.3))
    assert (s["max_drawdown_usd"], s["max_drawdown_sigma"]) == pytest.approx((2.5, 1.25))  # peak 1.0 -> trough -1.5


def test_rolling_line_matches_polyfit_and_needs_a_contiguous_session_run():
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    runs = np.array([1, 2, 3, 4, 5, 1, 2, 3])  # a new session starts at row 5
    slope, last = vs.rolling_line(y, 4, runs)
    b, a = np.polyfit(np.arange(4), y[1:5], 1)
    assert slope[4] == pytest.approx(b) and last[4] == pytest.approx(a + 3 * b)
    assert np.isnan(slope[:3]).all() and np.isnan(slope[5:]).all()


def minute_sessions():
    sessions = features.trading_sessions("2024-03-04", "2024-03-15", warmup_sessions=4)
    minutes = synthetic.random_walk_minutes(sessions, seed=5, drop_sessions=0)
    minutes["volume"] = np.random.default_rng(1).integers(100, 5000, len(minutes)).astype(float)
    return sessions, minutes


@pytest.mark.parametrize("lookback", [3, 4])
def test_anchored_vwap_and_sigma_match_a_direct_computation(lookback):
    sessions, minutes = minute_sessions()
    _, bars, _, _ = features.build_features(minutes, sessions, 2, (8, 48), min_coverage=0.5)
    vwap, sigma = vs.anchored_vwap(bars, sessions, lookback)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    days = list(sessions.index)
    assert np.isnan(vwap[(bars["session"] < days[lookback - 1]).to_numpy()]).all()  # too few earlier sessions
    for i in np.random.default_rng(2).choice(np.flatnonzero(bars["session"] >= days[lookback - 1]), 25):
        window = days[days.index(bars["session"].iloc[i]) - lookback + 1]
        w = (bars["session"] >= window) & (bars["bar_end"] <= bars["bar_end"].iloc[i])
        v = bars.loc[w, "volume"]
        mean = (tp[w] * v).sum() / v.sum()
        assert vwap[i] == pytest.approx(mean, rel=1e-12)
        assert sigma[i] == pytest.approx(np.sqrt((v * (tp[w] - mean) ** 2).sum() / v.sum()), rel=1e-9)


def test_a_session_without_data_blanks_every_window_that_contains_it():
    sessions, minutes = minute_sessions()
    days = list(sessions.index)
    local = minutes["ts"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    _, bars, _, _ = features.build_features(minutes[local != days[6]], sessions, 2, (8, 48), min_coverage=0.5)
    vwap, _ = vs.anchored_vwap(bars, sessions, 3)
    blank = bars["session"].isin(days[7:9]).to_numpy()  # the two sessions whose window holds days[6]
    assert np.isnan(vwap[blank]).all() and not np.isnan(vwap[(bars["session"] >= days[9]).to_numpy()]).any()


def study(minutes, sessions, **params):
    _, bars, _, _ = features.build_features(minutes, sessions, 2, (8, 48), min_coverage=0.5)
    b = vs.add_indicators(bars, sessions, 3)
    return b, vs.simulate(b, **params)


def test_future_prices_cannot_change_earlier_indicators_or_trades():
    sessions = features.trading_sessions("2024-01-02", "2024-03-28", warmup_sessions=5)
    minutes = synthetic.random_walk_minutes(sessions, seed=11, drop_sessions=0)
    cut = pd.Timestamp("2024-03-06 12:31", tz=NY).tz_convert("UTC")  # mid-session, mid-bar
    changed = minutes.copy()
    later = changed["ts"] >= cut
    changed.loc[later, ["open", "high", "low", "close"]] *= np.linspace(0.95, 1.2, later.sum())[:, None]
    changed.loc[later, "volume"] *= 7
    params = {"enable_continuation": True, "enable_vwap_to_vwap": True, "allow_reentry": True}
    before, log_before = study(minutes, sessions, **params)
    after, log_after = study(changed, sessions, **params)
    known = (before["bar_end"] <= cut).to_numpy()
    cols = ["close", "fast", "slow", "vwap", "sigma"]
    pd.testing.assert_frame_equal(before.loc[known, cols], after.loc[known, cols], check_exact=True)
    assert not np.allclose(before.loc[~known, "vwap"], after.loc[~known, "vwap"])
    earlier = lambda log: log[log["session"] < pd.Timestamp("2024-03-06")].reset_index(drop=True)  # noqa: E731
    assert (earlier(log_before)["setup"] == "A").any() and earlier(log_before)["trade_no"].notna().sum() > 10
    pd.testing.assert_frame_equal(earlier(log_before), earlier(log_after))


def test_cli_writes_the_log_summary_and_charts(tmp_path):
    sessions = features.trading_sessions("2024-01-02", "2024-02-29", warmup_sessions=10)
    minutes = synthetic.random_walk_minutes(sessions, seed=7)
    args = sauce.parse_args(["--ticker", "SYNTHETIC", "--start", "2024-01-02", "--end", "2024-02-29", "--compare",
                             "--charts", "2", "--out", str(tmp_path), "--warmup-sessions", "10"])
    results = sauce.execute(args, minutes, sessions, "SYNTHETIC test", {})
    assert results["summary"]["config"].unique().tolist() == [
        "baseline", "vwap_lookback_sessions=4", "require_slow_parallel=off",
        "vwap_lookback_sessions=4 require_slow_parallel=off"]
    assert set(results["summary"]["setup"]) == {"A"}  # the optional modules are off by default
    log = pd.read_csv(tmp_path / "setup_log.csv")
    assert list(log.columns) == ["ticker", "config", *vs.LOG_COLUMNS]
    assert len(list(tmp_path.glob("session_*.png"))) == 2
    for name in ("summary.csv", "monthly.csv", "settings.json"):
        assert (tmp_path / name).exists()
    with pytest.raises(SystemExit):
        sauce.parse_args(["--start", "2024-01-02", "--end", "2024-02-29", "--compare", "--vwap-lookback-sessions", "4"])


def test_random_baseline_matches_each_setup_a_trade_on_session_side_and_bracket():
    sessions = features.trading_sessions("2024-01-02", "2024-03-28", warmup_sessions=5)
    minutes = synthetic.random_walk_minutes(sessions, seed=11, drop_sessions=0)
    configs = sauce.make_configs({})
    r = sauce.run_sauce(minutes, sessions, configs)
    log, bars = r["log"], r["bars"][3]
    e = sauce.baseline_entries(log, bars, r["minutes"], draws=20, seed=4)
    setup, flipped, rand = (e[e["group"] == g].reset_index(drop=True) for g in ("setup", "flipped", "random"))
    n = len(setup)
    assert n > 10 and len(flipped) == n and len(rand) == 20 * n
    pd.testing.assert_frame_equal(e, sauce.baseline_entries(log, bars, r["minutes"], draws=20, seed=4))  # seeded
    t = log[(log["setup"] == "A") & log["trade_no"].notna()].set_index("entry_time")
    real = t.loc[setup["entry_time"]]
    assert np.allclose(setup["target"], real["target_price_at_entry"]) and np.allclose(setup["stop"], real["stop_price"])
    assert (flipped["side"] == -setup["side"]).all() and (flipped["entry_time"] == setup["entry_time"]).all()
    # Each random entry: the same session and side as its trade, never the session's first bar, and the same
    # target and stop distances in sigma, with sigma read at the previous bar's close.
    trade = setup.loc[rand["trade"]].reset_index(drop=True)
    day = lambda ts: ts.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()  # noqa: E731
    assert (day(rand["entry_time"]) == day(trade["entry_time"])).all() and (rand["side"] == trade["side"]).all()
    pos = pd.Index(bars["bar_start"]).get_indexer(rand["entry_time"])
    assert (bars["session"].to_numpy()[pos - 1] == bars["session"].to_numpy()[pos]).all()
    assert np.allclose(rand["sigma"], bars["sigma"].to_numpy()[pos - 1])
    for col in ("target", "stop"):
        assert np.allclose((rand[col] - rand["open"]) / rand["sigma"], (trade[col] - trade["open"]) / trade["sigma"])
    rows, pct, (low, high) = sauce.entry_baseline(log, bars, r["minutes"], sessions["close"].iloc[-1], draws=20)
    assert [x["group"] for x in rows] == list(sauce.BASELINE_GROUPS) and 0 <= pct <= 100 and low <= high
    assert rows[1]["trades"] <= n and rows[2]["trades"] <= 20 * n
