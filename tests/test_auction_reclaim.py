"""auction_reclaim: bar-approximated profile, frozen levels, the reclaim/retest state machine,
causal VWAP and relative volume, candidate-only processing and the fixed barrier path.

Every background session uses one designed minute layout. Its bar-approximated profile
(48 bins of 0.125 over 96..102, a triangle of volume around 99.0625 and ties that expand
lower first) has VAL 98.25, POC 99.0625 and VAH 99.75. Its daily ATR(14) is exactly 6.0,
so b = 0.12, d = 0.18 and the stop offset 0.01 A = 0.06. The vendor minute `vwap` is
(high + low) / 2, which differs from HLC3, so the two VWAP methods can be told apart.
"""

import numpy as np
import pandas as pd
import pytest

import features
import outcomes
import report
import run
import strategies.auction_reclaim as ar
import synthetic
import volume_profile as vp

NY = "America/New_York"
AR = "auction_reclaim"
MIN = pd.Timedelta(minutes=1)
VAL, POC, VAH, ATR = 98.25, 99.0625, 99.75, 6.0


def background():
    """390 (o, h, l, c, volume) minutes: 22 full-range minutes plus a triangle of point minutes, every
    5-minute close inside value. Bin k (center 96 + (k + 0.5) / 8) holds 1000 x (12 - |k - 24|)."""
    center = lambda k: 96 + (k + 0.5) * 0.125  # noqa: E731
    in_value = [k for k in range(18, 30) for _ in range(16)]
    outside = [k for k in (*range(13, 18), *range(30, 36)) for _ in range(16)]
    closes, free = ["R"] * 22 + in_value[:56], in_value[56:] + outside
    rows = []
    for j in range(78):
        for k in free[4 * j:4 * j + 4] + [closes[j]]:
            rows.append((99.0625, 102.0, 96.0, 99.0625, 480.0) if k == "R"
                        else (center(k),) * 4 + ((12 - abs(k - 24)) * 62.5,))
    return rows


BACKGROUND = background()
SLOT_VOLUME = [sum(m[4] for m in BACKGROUND[5 * j:5 * j + 5]) for j in range(78)]

LONG_IN = (98.75, 98.8125, 98.6875, 98.75)  # doji inside value: never a reclaim or a retest
EXC = (98.75, 98.78125, 98.0625, 98.09375)  # close < VAL - b = 98.13; low 98.0625
OUT = (98.09375, 98.1875, 98.078125, 98.109375)  # still outside, no reclaim
RECLAIM = (98.109375, 98.5, 98.09375, 98.453125)  # close in (VAL + b, POC), white
RETEST = (98.34375, 98.421875, 98.328125, 98.40625, 2.0)  # 5th value: volume multiple of the slot's usual volume
LONG = [LONG_IN] * 4 + [EXC, OUT, RECLAIM, RETEST]  # signal known 10:10 ET
SHORT_IN = (99.25, 99.3125, 99.1875, 99.25)
SHORT = [SHORT_IN] * 4 + [(99.25, 99.9375, 99.21875, 99.90625),  # close > VAH + b = 99.87; high 99.9375
                          (99.90625, 99.921875, 99.8125, 99.890625),
                          (99.890625, 99.90625, 99.5, 99.546875),  # reclaim: close in (POC, VAH - b), black
                          (99.65625, 99.671875, 99.578125, 99.59375, 2.0)]  # retest, signal known 10:10 ET


def minutes_of(bar, slot):
    """Five minutes whose 5-minute bar is exactly (o, h, l, c): low first on white bars, high first on black.
    A list is taken as explicit (o, h, l, c, volume) minutes."""
    if isinstance(bar, list):
        return bar
    o, h, l, c, mult = bar if len(bar) == 5 else (*bar, 1.0)
    path = ([(o, o, l, l), (l, h, l, h), (h, h, c, c), (c, c, c, c), (c, c, c, c)] if c >= o else
            [(o, h, o, h), (h, h, l, l), (l, c, l, c), (c, c, c, c), (c, c, c, c)])
    return [(*m, SLOT_VOLUME[slot] / 5 * mult) for m in path]


def build(sessions, designed=None, drop=(), volume_scale=None):
    """Background minutes everywhere; a designed day uses its bars from the open, then repeats its first bar."""
    rows = []
    for n, (day, s) in enumerate(sessions.iterrows()):
        minutes = list(BACKGROUND)
        bars = (designed or {}).get(str(day.date()))
        if bars is not None:
            for j in range(78):
                minutes[5 * j:5 * j + 5] = minutes_of(bars[j] if j < len(bars) else bars[0], j)
        scale = volume_scale(n) if volume_scale else 1.0
        rows += [(s["open"] + m * MIN, o, h, l, c, v * scale) for m, (o, h, l, c, v) in enumerate(minutes)]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].dt.as_unit("ns")
    df["vwap"] = (df["high"] + df["low"]) / 2
    df["transactions"] = 10
    return df[~df["ts"].isin([et(t) for t in drop])].reset_index(drop=True)


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def one_session():
    return features.trading_sessions("2024-03-08", "2024-03-08", warmup_sessions=20)


def study(minutes, sessions, **kwargs):
    return run.run_study(minutes, sessions, strategy=AR, **kwargs)


def signal_times(result):
    return [t.tz_convert(NY).strftime("%H:%M") for t in result["candidates"]["signal_time"]]


def funnel(result, side="long", config="baseline"):
    s = result["summary"]
    return s[(s["side"] == side) & (s["config"] == config)].iloc[0]


# --- profile helper -------------------------------------------------------------------------------------------

def test_profile_allocation_edges_and_volume_conservation():
    edges = np.linspace(0, 4, 5)  # [0,1) [1,2) [2,3) [3,4]
    # Points at 0.5, on the interior edge 1.0 (goes right) and on the top edge 4.0 (last bin); a 0..4 bar
    # spreads 10 per bin; a 1.5..3.5 bar puts 15 / 30 / 15 by overlap length.
    bins = vp.allocate([0.5, 1.0, 4.0, 0.0, 1.5], [0.5, 1.0, 4.0, 4.0, 3.5], [10, 20, 30, 40, 60], edges)
    assert bins.tolist() == [20, 45, 40, 55]

    rng = np.random.default_rng(3)
    low = 100 + rng.normal(0, 1, 400).cumsum() * 0.05
    high = low + np.abs(rng.normal(0, 0.05, 400)) * (rng.random(400) > 0.2)  # some high == low
    volume = rng.integers(0, 5000, 400).astype(float)
    for n in (16, 32, 48, 64):
        edges, bins = vp.bar_profile(low, high, volume, n)
        assert len(bins) == n and np.isclose(bins.sum(), volume.sum(), rtol=1e-12)
        assert edges[0] == low.min() and edges[-1] == high.max()

    assert vp.bar_profile([1.0, 1.0], [1.0, 1.0], [5, 5], 8) is None  # no range
    assert vp.bar_profile([1.0, 2.0], [1.5, 2.5], [0, 0], 8) is None  # no volume
    assert vp.bar_profile([1.0, np.nan], [1.5, 2.5], [1, 1], 8) is None
    assert vp.bar_profile([1.0, 2.0], [1.5, 1.9], [1, 1], 8) is None  # high < low
    assert vp.bar_profile([1.0, 2.0], [1.5, 2.5], [1, -1], 8) is None


def test_poc_and_value_area_tie_rules():
    # POC tie between bins 1 and 4: the lower bin wins. Then the larger neighbour is added each time.
    assert vp.value_area([1, 5, 2, 2, 5, 1], np.arange(7.0)) == (1.0, 1.5, 5.0)
    # Equal neighbours: expand lower first (7 of 10 reached with bins 1..2).
    assert vp.value_area([0, 3, 4, 3, 0], np.arange(6.0)) == (1.0, 2.5, 3.0)
    # Lower edge exhausted: use the remaining side even though it is small.
    assert vp.value_area([5, 1, 3, 1], np.arange(5.0)) == (0.0, 0.5, 3.0)


def test_low_volume_node_rules():
    lvn = [1000, 0, 1000, 3000, 1045, 45, 45, 45, 45, 5, 45, 45, 45, 45, 1500, 1500]
    assert vp.low_volume_nodes(lvn) == [9]  # smoothed 25 < 35 on both sides, far below both peaks
    assert vp.low_volume_nodes([100.0] * 16) == []  # flat: no strict local minimum
    tie = [1000, 0, 1000, 3000, 1045, 45, 45, 45, 45, 5, 5, 45, 45, 45, 1500, 1500]
    assert vp.low_volume_nodes(tie) == []  # equal smoothed neighbours are not strictly lower
    edge_dip = [500, 5, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 5, 500]
    assert vp.low_volume_nodes(edge_dip) == []  # the first and last two bins are never candidates
    zero = [1000, 1000, 1000, 1000, 0, 1000, 1000, 1000]
    assert vp.low_volume_nodes(zero) == []  # a node needs positive raw volume


# --- known sequences ------------------------------------------------------------------------------------------

def test_known_long_and_short_with_frozen_levels_and_barrier_paths():
    sessions = one_session()
    long_path = LONG + [(98.40625, 99.125, 98.40625, 99.0625)]  # 10:11 trades through the target (POC)
    short_path = SHORT + [(99.59375, 99.59375, 99.0, 99.0)]  # 10:11 trades through the target
    for bars, side, stop, target, extreme, band, r in (
            (long_path, "long", 98.0025, POC, 98.0625, (VAL - 0.18, VAL + 0.18), 0.65625 / 0.40375),
            (short_path, "short", 99.9975, POC, 99.9375, (VAH - 0.18, VAH + 0.18), 0.53125 / 0.40375)):
        result = study(build(sessions, {"2024-03-08": bars}), sessions, barriers=True)
        assert signal_times(result) == ["10:10"], side
        c = result["candidates"].iloc[0]
        assert c["side"] == side and c["signal_time"] == et("2024-03-08 10:10")
        assert (c["val"], c["poc"], c["vah"], c["prev_atr_14"]) == (VAL, POC, VAH, ATR)
        assert c["profile_method"] == "bar_approximated_volume_profile" and c["profile_bins"] == 48
        assert c["prev_session"] == pd.Timestamp("2024-03-07") and c["location"] == "value_edge"
        assert np.isclose(c["buffer_b"], 0.12) and np.isclose(c["retest_tol_d"], 0.18)
        assert (c["excursion_slot"], c["extreme_slot"], c["reclaim_slot"], c["signal_slot"]) == (4, 4, 6, 7)
        assert c["excursion_end"] == et("2024-03-08 09:55") and c["reclaim_end"] == et("2024-03-08 10:05")
        assert c["excursion_extreme"] == extreme and np.isclose(c["frozen_stop"], stop)
        assert c["frozen_target"] == target and np.allclose((c["location_lo"], c["location_hi"]), band)
        assert np.isclose(c["rvol"], 2.0) and c["vwap_method"] == "massive_minute_vwap"
        assert np.isclose(c["signal_reward_risk"], r) and c["signal_reward_risk"] >= 1.25
        # Idealized path: entry at the 10:10 open, never the retest extreme; the target fills at the target.
        assert c["entry_time"] == et("2024-03-08 10:10") and c["entry_status"] == "ok"
        assert c["entry_price"] == c["ref_close"]  # the next bar opens at the signal close in this fixture
        assert (c["exit_reason"], c["exit_price"], c["exit_time"]) == ("target", POC, et("2024-03-08 10:12"))
        assert np.isclose(c["barrier_r_0bp"], r) and c["barrier_r_1bp"] < c["barrier_r_0bp"]
        # Directional returns: positive means price moved the signal's way, for shorts too.
        assert c["fwd_ret_10m_pct"] > 0 and c["mfe_60m_pct"] > 0
        row = funnel(result, side)
        assert (row["n_candidates"], row["excursions"], row["reclaims"], row["eligible_sessions"]) == (1, 1, 1, 1)


@pytest.mark.parametrize("name, bars, drop, expected, counter", [
    ("reclaim alone never signals", [LONG_IN] * 4 + [EXC, OUT, (*RECLAIM, 2.0), LONG_IN, LONG_IN, LONG_IN], (),
     None, "retest_expired"),
    ("retest on the third bar after the reclaim", [LONG_IN] * 4 + [EXC, OUT, RECLAIM, LONG_IN, LONG_IN, RETEST], (),
     "10:20", None),
    ("retest on the fourth bar has expired", [LONG_IN] * 4 + [EXC, OUT, RECLAIM, LONG_IN, LONG_IN, LONG_IN, RETEST],
     (), None, "retest_expired"),
    ("reclaim on the sixth bar after the excursion", [LONG_IN] * 4 + [EXC] + [LONG_IN] * 5 + [RECLAIM, RETEST], (),
     "10:30", None),
    ("reclaim on the seventh bar has expired", [LONG_IN] * 4 + [EXC] + [LONG_IN] * 6 + [RECLAIM, RETEST], (),
     None, "reclaim_expired"),
    ("POC reached before the reclaim", [LONG_IN] * 4 + [EXC, (98.1, POC, 98.1, 98.2), RECLAIM, RETEST], (),
     None, "reclaim_poc_first"),
    ("extreme breached after the reclaim", LONG[:7] + [(98.4, 98.4375, 98.03125, 98.40625), RETEST], (),
     None, "invalid_extreme"),
    ("POC touched after the reclaim", LONG[:7] + [(98.5, POC, 98.45, 98.6), RETEST], (), None, "invalid_poc"),
    ("close back beyond the excursion threshold", LONG[:7] + [(98.4, 98.4, 98.1, 98.109375), RETEST], (),
     None, "invalid_close"),
    ("trigger bar touching POC invalidates first", LONG[:7] + [(98.34375, POC, 98.328125, 98.40625, 2.0)], (),
     None, "invalid_poc"),
    ("balance: 2 of 4 closes inside value", [LONG_IN, (98.2, 98.21875, 98.1875, 98.1875)] * 2 + LONG[4:], (),
     None, None),
    ("balance: 3 of 4 is enough", [(98.2, 98.21875, 98.1875, 98.1875)] + [LONG_IN] * 3 + LONG[4:], (), "10:10", None),
    ("stale break: the previous close was already beyond", [LONG_IN] * 3 + [EXC, EXC] + LONG[5:], (), None, None),
    ("missing minute while waiting for the reclaim", LONG, ["2024-03-08 09:57"], None, "missing_bar_reset"),
    ("missing minute in the retest bar", LONG, ["2024-03-08 10:07"], None, "missing_bar_reset"),
    ("relative volume below 1.20", LONG[:7] + [(*RETEST[:4], 1.1)], (), None, None),
    ("retest completing 11:30 is inside the window", [LONG_IN] * 20 + LONG[4:], (), "11:30", None),
    ("retest completing 11:35 is outside it", [LONG_IN] * 21 + LONG[4:], (), None, "window_closed"),
])
def test_setup_sequence(name, bars, drop, expected, counter):
    sessions = one_session()
    result = study(build(sessions, {"2024-03-08": bars}, drop), sessions)
    assert signal_times(result) == ([] if expected is None else [expected]), name
    row = funnel(result)
    if counter:
        assert row[counter] == 1, name
    if name.startswith("balance: 2") or name.startswith("stale"):
        assert row["excursions"] == 0  # no setup ever started
    if name.startswith("reclaim alone"):
        assert row["reclaims"] == 1 and row["excursions"] == 1


def test_relative_volume_filter_can_be_disabled_and_needs_ten_prior_observations():
    sessions = one_session()
    low_rvol = build(sessions, {"2024-03-08": LONG[:7] + [(*RETEST[:4], 1.1)]})
    assert signal_times(study(low_rvol, sessions)) == []
    off = study(low_rvol, sessions, rvol_filter=False)
    assert signal_times(off) == ["10:10"] and np.isclose(off["candidates"].iloc[0]["rvol"], 1.1)
    assert off["candidates"].iloc[0]["config"] == "rvol_filter off"

    # Only 9 of the 20 preceding sessions have a complete 10:05 bar: no baseline, so no signal with the filter.
    minutes = build(sessions, {"2024-03-08": LONG}, drop=[f"{d.date()} 10:07" for d in sessions.index[:11]])
    result = study(minutes, sessions)
    signal_bar = result["bars"]["bar_end"] == et("2024-03-08 10:10")
    assert result["bars"].loc[signal_bar, "rvol_base"].isna().all() and signal_times(result) == []
    assert signal_times(study(minutes, sessions, rvol_filter=False)) == ["10:10"]


EXC_DEEP = (98.75, 98.78125, 97.9375, 98.09375)  # extreme 97.9375: the usual retest then has reward/risk 1.24


@pytest.mark.parametrize("name, bars, baseline, loose", [
    ("retest completing 13:10", [LONG_IN] * 40 + LONG[4:], None, "13:10"),
    ("retest on the fifth bar after the reclaim", LONG[:7] + [LONG_IN] * 4 + [RETEST], None, "10:30"),
    ("body 0.375 of the range", LONG[:7] + [(98.34375, 98.421875, 98.296875, 98.390625, 2.0)], None, "10:10"),
    ("reward/risk 1.24", LONG[:4] + [EXC_DEEP] + LONG[5:], None, "10:10"),
    ("relative volume 1.1 still fails", LONG[:7] + [(*RETEST[:4], 1.1)], None, None),
    ("seventh bar after the reclaim still expired", LONG[:7] + [LONG_IN] * 6 + [RETEST], None, None),
])
def test_loose_rules_widen_only_their_own_settings(name, bars, baseline, loose):
    sessions = one_session()
    minutes = build(sessions, {"2024-03-08": bars})
    assert signal_times(study(minutes, sessions)) == ([] if baseline is None else [baseline]), name
    result = study(minutes, sessions, rules="loose")
    assert signal_times(result) == ([] if loose is None else [loose]), name
    assert result["configs"] == ["loose"] and set(result["candidates"]["rules"]) <= {"loose"}


def test_loose_barrier_eligibility_uses_its_own_reward_risk_floor():
    sessions = one_session()
    bars = LONG[:4] + [EXC_DEEP] + LONG[5:] + [(98.40625, 98.421875, 98.40625, 98.421875)]  # opens at the close
    minutes = build(sessions, {"2024-03-08": bars})
    c = study(minutes, sessions, rules="loose", barriers=True)["candidates"].iloc[0]
    entry_rr = (c["frozen_target"] - c["entry_price"]) / (c["entry_price"] - c["frozen_stop"])
    assert c["entry_price"] == 98.40625 and 1.0 <= entry_rr < 1.25 and c["entry_status"] == "ok"


def test_one_candidate_per_side_and_one_active_setup():
    sessions = one_session()
    # The same long setup twice, then a short setup: one long (the first) and one short.
    bars = LONG + LONG + [SHORT_IN] * 4 + SHORT[4:]
    result = study(build(sessions, {"2024-03-08": bars}), sessions)
    assert result["candidates"]["side"].tolist() == ["long", "short"]
    assert signal_times(result) == ["10:10", "11:30"]
    assert funnel(result, "long")["excursions"] == 1  # the capped side starts no further setup


def test_reclaim_lvn_location_is_frozen_and_never_replaced_by_the_edge():
    sessions = one_session()
    exc = (98.75, 98.78125, 98.09375, 98.109375)
    extreme_bar = [(98.09375,) * 4 + (100,), (98.0,) * 4 + (100,), (98.0625,) * 4 + (100,),
                   (98.09375,) * 4 + (100,), (98.109375,) * 4 + (100,)]  # sets the extreme 98.0
    reclaim = [(98.125,) * 4 + (100,), (98.125, 98.28125, 98.125, 98.28125, 20),
               (98.28125, 98.4375, 98.125, 98.4375, 5), (98.4375, 98.4375, 98.3125, 98.3125, 16),
               (98.4375, 98.5, 98.4375, 98.453125, 300)]  # thin volume around 98.28-98.31
    lvn_retest = (98.3125, 98.421875, 98.296875, 98.40625, 2.0)  # overlaps the node, closes above it
    head = [LONG_IN] * 4 + [exc, extreme_bar, reclaim]

    hit = study(build(sessions, {"2024-03-08": head + [lvn_retest]}), sessions, compare=True)
    # The 32/64-bin profiles freeze different levels here (POC 98.906; VAL 96.0 from bin aliasing), so they stay out.
    assert set(hit["candidates"]["config"]) == {"baseline", "reclaim_lvn", "rvol_filter off"}
    c = hit["candidates"].set_index("config")
    assert c.loc["reclaim_lvn", "signal_time"] == c.loc["baseline", "signal_time"] == et("2024-03-08 10:10")
    assert (c.loc["reclaim_lvn", "location_lo"], c.loc["reclaim_lvn", "location_hi"]) == (98.28125, 98.3125)
    assert c.loc["reclaim_lvn", "extreme_slot"] == 5 and c.loc["reclaim_lvn", "excursion_extreme"] == 98.0
    assert np.allclose((c.loc["baseline", "location_lo"], c.loc["baseline", "location_hi"]), (VAL - 0.18, VAL + 0.18))

    # The edge retest does not reach the node: the edge variant signals, the LVN variant does not.
    miss = study(build(sessions, {"2024-03-08": head + [RETEST]}), sessions, compare=True)
    assert set(miss["candidates"]["config"]) == {"baseline", "rvol_filter off"}
    assert funnel(miss, config="reclaim_lvn")["retest_expired"] == 1

    # Every minute spans the same range: a flat profile has no node, so no setup (counted, not substituted).
    flat = [[(98.1, 98.5, 98.0, 98.1, 100)] * 4 + [(98.1, 98.5, 98.0, 98.109375, 100)],
            [(98.125, 98.5, 98.0, 98.3, 100)] * 4 + [(98.3, 98.5, 98.0, 98.453125, 100)]]
    none = study(build(sessions, {"2024-03-08": [LONG_IN] * 4 + [exc] + flat + [RETEST]}), sessions, compare=True)
    assert "reclaim_lvn" not in set(none["candidates"]["config"]) and "baseline" in set(none["candidates"]["config"])
    assert funnel(none, config="reclaim_lvn")["lvn_no_node"] == 1


def test_bin_variants_reuse_shared_features_and_outcomes(monkeypatch):
    sessions = one_session()
    calls = {"features": 0, "outcomes": []}
    real_features, real_outcomes = ar.add_features, outcomes.forward_outcomes

    def count_features(*args):
        calls["features"] += 1
        return real_features(*args)

    def count_outcomes(minutes, signal_time, *args):
        calls["outcomes"].append(len(signal_time))
        return real_outcomes(minutes, signal_time, *args)

    monkeypatch.setattr(ar, "add_features", count_features)
    monkeypatch.setattr(outcomes, "forward_outcomes", count_outcomes)
    result = study(build(sessions, {"2024-03-08": LONG}), sessions, compare=True)
    assert result["configs"] == ["baseline", "reclaim_lvn", "rvol_filter off", "bins=32", "bins=64"]
    assert calls["features"] == 1 and calls["outcomes"] == [1]  # one shared trigger bar, one evaluation
    bars = result["bars"]
    assert {"val_32", "val_48", "val_64"} <= set(bars.columns)


# --- causality, VWAP, relative volume, missing data -------------------------------------------------------

def test_later_data_cannot_change_earlier_features_levels_or_signals():
    sessions = features.trading_sessions("2024-03-08", "2024-03-11", warmup_sessions=20)
    minutes = build(sessions, {"2024-03-08": LONG})
    cut = et("2024-03-08 10:12")
    changed = minutes.copy()
    later = changed["ts"] >= cut
    changed.loc[later, ["open", "high", "low", "close", "vwap"]] *= np.linspace(0.98, 1.03, later.sum())[:, None]
    changed.loc[later, "volume"] *= 3
    # A second, "better" long setup later that session cannot displace or precede the first.
    after = build(sessions, {"2024-03-08": LONG + [LONG_IN] * 4 + LONG[4:]})

    base = study(minutes, sessions)
    cols = ["open", "high", "low", "close", "volume", "vwap", "vwap_15m_ago", "vwap_method", "rvol", "rvol_base",
            "val_48", "poc_48", "vah_48", "prev_atr_14"]
    first = ["signal_time", "side", "val", "poc", "vah", "excursion_extreme", "vwap", "rvol", "frozen_stop",
             "frozen_target"]
    for variant in (study(changed, sessions), study(after, sessions)):
        known = (base["bars"]["bar_end"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(base["bars"].loc[known, cols], variant["bars"].loc[known, cols])
        pd.testing.assert_series_equal(base["candidates"].iloc[0][first], variant["candidates"].iloc[0][first])
        today = (base["bars"]["session"] == "2024-03-08").to_numpy()  # levels frozen for the whole session
        pd.testing.assert_frame_equal(base["bars"].loc[today, ["val_48", "poc_48", "vah_48"]],
                                      variant["bars"].loc[today, ["val_48", "poc_48", "vah_48"]])
    # Today's changed minutes do reach tomorrow's frozen levels and ATR.
    moved = study(changed, sessions)["day"].loc["2024-03-11"]
    assert moved["poc_48"] != base["day"].loc["2024-03-11", "poc_48"]
    assert moved["prev_atr"] != base["day"].loc["2024-03-11", "prev_atr"]


def test_vwap_resets_each_open_and_approximation_never_revises_earlier_values():
    sessions = features.trading_sessions("2024-03-08", "2024-03-11", warmup_sessions=20)
    minutes = build(sessions, {"2024-03-08": LONG}, volume_scale=lambda n: 1 + n % 3)
    bars = study(minutes, sessions)["bars"].set_index("bar_end")
    for day in ("2024-03-08", "2024-03-11"):
        first = minutes[(minutes["ts"] >= et(f"{day} 09:30")) & (minutes["ts"] < et(f"{day} 09:35"))]
        expected = (first["vwap"] * first["volume"]).sum() / first["volume"].sum()
        assert np.isclose(bars.loc[et(f"{day} 09:35"), "vwap"], expected)  # only today's first minutes
        assert np.isnan(bars.loc[et(f"{day} 09:45"), "vwap_15m_ago"])  # would need 09:30 or earlier
        assert not np.isnan(bars.loc[et(f"{day} 09:50"), "vwap_15m_ago"])

    # A minute without Massive's vwap after the 10:10 signal switches later bars to HLC3, never earlier ones.
    base = study(minutes, sessions)
    gap = minutes.copy()
    gap.loc[gap["ts"] == et("2024-03-08 10:20"), "vwap"] = np.nan
    approx = study(gap, sessions)
    pd.testing.assert_frame_equal(base["candidates"], approx["candidates"])
    b = approx["bars"].set_index("bar_end")
    assert b.loc[et("2024-03-08 10:20"), "vwap_method"] == "massive_minute_vwap"
    assert b.loc[et("2024-03-08 10:25"), "vwap_method"] == "hlc3_approximation"
    assert b.loc[et("2024-03-11 09:35"), "vwap_method"] == "massive_minute_vwap"  # reset at the next open
    day = gap[(gap["ts"] >= et("2024-03-08 09:30")) & (gap["ts"] < et("2024-03-08 10:25"))]
    hlc3 = (day["high"] + day["low"] + day["close"]) / 3
    assert np.isclose(b.loc[et("2024-03-08 10:25"), "vwap"], (hlc3 * day["volume"]).sum() / day["volume"].sum())
    early = day[day["ts"] < et("2024-03-08 10:10")]  # the 15-minute-earlier value uses HLC3 too: no mixing
    assert np.isclose(b.loc[et("2024-03-08 10:25"), "vwap_15m_ago"],
                      (((early["high"] + early["low"] + early["close"]) / 3) * early["volume"]).sum()
                      / early["volume"].sum())

    # If the approximation starts before the signal, the signal's VWAP inputs are both HLC3.
    gap.loc[gap["ts"] == et("2024-03-08 09:31"), "vwap"] = np.nan
    c = study(gap, sessions)["candidates"]
    assert (c["vwap_method"] == "hlc3_approximation").all()


def test_relative_volume_uses_twenty_prior_sessions_and_excludes_today():
    sessions = one_session()
    scale = lambda n: 1 + (n * 7 % 11) / 10  # noqa: E731 - varying volume across sessions
    minutes = build(sessions, {"2024-03-08": LONG}, volume_scale=scale)
    louder = minutes.copy()
    today = louder["ts"] >= et("2024-03-08 09:30")
    louder.loc[today, "volume"] *= 5
    base, loud = study(minutes, sessions)["bars"], study(louder, sessions)["bars"]
    at = (base["bar_end"] == et("2024-03-08 10:10")).to_numpy()
    prior = [SLOT_VOLUME[7] * scale(n) for n in range(20)]  # slot 7 = the 10:05 bar in each earlier session
    assert np.isclose(base.loc[at, "rvol_base"].item(), np.median(prior))
    assert loud.loc[at, "rvol_base"].item() == base.loc[at, "rvol_base"].item()  # today's volume is excluded
    assert np.isclose(loud.loc[at, "rvol"].item(), 5 * base.loc[at, "rvol"].item())


def test_incomplete_previous_session_is_not_replaced_by_an_older_one():
    sessions = one_session()
    result = study(build(sessions, {"2024-03-08": LONG}, drop=["2024-03-07 12:00"]), sessions)
    day = result["day"].loc["2024-03-08"]
    assert day["profile_status"] == "previous_session_incomplete" and np.isnan(day["val_48"])
    assert result["candidates"].empty
    row = funnel(result)
    assert (row["profile_unavailable"], row["eligible_sessions"], row["excursions"]) == (1, 0, 0)


def test_zero_triggers_build_no_rows_outcomes_or_charts(monkeypatch, tmp_path):
    sessions = features.trading_sessions("2024-03-08", "2024-03-12", warmup_sessions=20)

    def forbidden(*args, **kwargs):
        raise AssertionError("called without any candidates")

    for module, name in ((outcomes, "forward_outcomes"), (outcomes, "barrier_exits"), (run, "build_candidate_rows"),
                         (ar, "candidate_snapshot"), (ar, "reclaim_lvn"), (report, "_plot_auction_candidate")):
        monkeypatch.setattr(module, name, forbidden)
    args = run.parse_args(["--strategy", AR, "--compare", "--barriers", "--start", "2024-03-08", "--end",
                           "2024-03-12", "--warmup-sessions", "20", "--out", str(tmp_path)])
    assert args.ticker == "QQQ"  # auction_reclaim's default ticker
    result = run.execute(args, build(sessions), sessions, "SYNTHETIC test fixture", {})

    cands = result["candidates"]
    assert cands.empty and {"ticker", "side", "val", "poc", "vah", "frozen_stop", "fwd_ret_30m_pct",
                            "exit_reason", "barrier_r_1bp"} <= set(cands.columns)
    summary = result["summary"]
    assert len(summary) == 10  # five variants x two sides
    assert (summary["eligible_sessions"] == 3).all() and (summary["excursions"] == 0).all()
    assert summary[["mean_30m_pct", "frac_pos_30m", "mean_r_0bp"]].isna().all().all()
    assert result["candidate_charts"] == [] and pd.read_csv(tmp_path / "stability.csv").empty
    assert pd.read_parquet(tmp_path / "candidates.parquet").empty


def test_entry_reward_risk_below_minimum_is_kept_but_ineligible():
    sessions = one_session()
    gap_up = LONG + [(98.625, 98.65625, 98.59375, 98.625)]  # entry 98.625: reward 0.4375 / risk 0.6225
    result = study(build(sessions, {"2024-03-08": gap_up}), sessions, barriers=True)
    c = result["candidates"]
    assert len(c) == 1 and c.iloc[0]["entry_status"] == "invalid" and c.iloc[0]["entry_price"] == 98.625
    assert pd.isna(c.iloc[0]["exit_reason"]) and np.isnan(c.iloc[0]["barrier_r_0bp"])
    assert not np.isnan(c.iloc[0]["fwd_ret_30m_pct"])  # the signal-response outcomes are still reported
    row = funnel(result)
    assert (row["n_candidates"], row["n_entry_invalid"], row["n_entry_ok"]) == (1, 1, 0)


def test_split_compares_variants_early_and_evaluates_only_the_fixed_baseline_later(tmp_path):
    sessions = features.trading_sessions("2024-01-02", "2024-06-28", warmup_sessions=30)
    minutes = synthetic.random_walk_minutes(sessions, seed=5)
    args = run.parse_args(["--strategy", AR, "--compare", "--barriers", "--start", "2024-01-02", "--end",
                           "2024-06-28", "--warmup-sessions", "30", "--split-date", "2024-04-01",
                           "--out", str(tmp_path)])
    result = run.execute(args, minutes, sessions, "SYNTHETIC test fixture", {})
    summary, cands = result["summary"], result["candidates"]
    assert result["selected"] == "baseline"
    assert set(summary.loc[summary["segment"] == "earlier", "config"]) == {c["label"] for c in ar.make_configs(
        "value_edge", 48, True, compare=True)}
    assert set(summary.loc[summary["segment"] == "later", "config"]) == {"baseline"}
    assert (summary.loc[summary["segment"] == "earlier", "role"] == "comparison").all()
    assert set(cands.loc[cands["segment"] == "later", "config"]) <= {"baseline"}
    last_early_close = sessions.loc["2024-03-28", "close"]
    early = cands[cands["segment"] == "earlier"]
    assert (early["exit_time"].dropna() <= last_early_close).all()


def test_cli_rejects_options_for_other_strategies():
    base = ["--start", "2024-03-08", "--end", "2024-03-08"]
    for extra in (["--compare"], ["--location", "reclaim_lvn"], ["--profile-bins", "32"], ["--no-rvol-filter"]):
        with pytest.raises(SystemExit):
            run.parse_args(base + extra)  # spy_ema
    with pytest.raises(SystemExit):
        run.parse_args(base + ["--strategy", AR, "--compare", "--profile-bins", "32"])
    with pytest.raises(SystemExit):
        run.parse_args(base + ["--rules", "loose"])  # spy_ema
    assert run.parse_args(base + ["--strategy", AR, "--barriers"]).ticker == "QQQ"
    assert [c["label"] for c in ar.make_configs("value_edge", 48, True, compare=True, rules="loose")] == [
        "loose", "loose reclaim_lvn", "loose rvol_filter off", "loose bins=32", "loose bins=64"]
    assert run.parse_args(base).ticker == "SPY"
