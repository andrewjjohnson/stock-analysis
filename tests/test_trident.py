"""trident.py on SYNTHETIC data: the FVG + doji + confirmation pattern and its mirror, the kill-zone window, the
outcome races in R, matched random entries, and a design run that cannot see data after 2024."""

import numpy as np
import pandas as pd

import features
import trident as td

DAY = pd.Timestamp("2024-03-04")


def bar_frame(rows, day=DAY, first="09:30"):
    """30-minute bars from (open, high, low, close) rows, starting at `first` ET, with every filter passing."""
    start = pd.Timestamp(f"{day.date()} {first}", tz=td.NY).tz_convert("UTC")
    starts = pd.Series([start + pd.Timedelta(minutes=30 * k) for k in range(len(rows))])
    b = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    b.insert(0, "session", day)
    b.insert(1, "bar_start", starts)
    b.insert(2, "bar_end", starts + pd.Timedelta(minutes=30))
    b["atr"] = 1.0
    b["tod"] = b["bar_end"].dt.tz_convert(td.NY).dt.strftime("%H:%M")
    for k in ("trend_long", "trend_short", "bias_long", "bias_short", "green_long", "green_short"):
        b[k] = True
    return b


LONG = [(100.0, 100.5, 99.5, 100.3),    # candle 1 (09:30)
        (100.3, 102.0, 100.2, 101.8),   # candle 2
        (101.8, 102.5, 101.5, 102.3),   # candle 3 closes 11:00: gap 100.5 -> 101.5, 50% level 101.0
        (102.0, 102.3, 100.9, 102.05),  # doji: body 0.05 of a 1.4 range, wick to 100.9, body above 101.0
        (102.0, 102.2, 101.8, 102.1),   # confirmation: closes below the doji's high
        (102.1, 102.4, 101.9, 102.2)]


def mirror(rows, axis=300.0):
    return [(axis - o, axis - lo, axis - h, axis - c) for o, h, lo, c in rows]


def test_the_long_pattern_and_its_measures():
    f = td.fvg_table(bar_frame(LONG))
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.side, r.i3, r.ce, r.gap, r.stop1, r.stop2, r.c5) == (1, 2, 101.0, 1.0, 99.5, 100.2, 102.1)
    assert r.wick4 and r.confirm and r.five_ok and abs(r.body4 - 0.05 / 1.4) < 1e-12


def test_the_short_mirrors_the_long():
    f = td.fvg_table(bar_frame(mirror(LONG)))
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.side, r.ce, r.gap, r.stop1, r.c5) == (-1, 199.0, 1.0, 200.5, 197.9)
    assert r.wick4 and r.confirm and abs(r.body4 - 0.05 / 1.4) < 1e-9


def test_confirmation_doji_body_and_window_rules():
    above = list(LONG)
    above[4] = (102.0, 102.6, 101.8, 102.5)  # closes above the doji's high
    assert not td.fvg_table(bar_frame(above)).iloc[0].confirm
    through = list(LONG)
    through[3] = (102.0, 102.3, 100.9, 100.95)  # the doji's body reaches below the 50% level
    assert not td.fvg_table(bar_frame(through)).iloc[0].wick4
    assert len(td.fvg_table(bar_frame(LONG, first="10:30"))) == 1  # candle 3 closes 12:00: the last slot
    assert td.fvg_table(bar_frame(LONG, first="11:00")).empty      # candle 3 closes 12:30: outside


def test_candles_must_be_consecutive_in_one_session():
    b = bar_frame(LONG)
    b.loc[1:, "session"] = DAY + pd.Timedelta(days=1)
    assert td.fvg_table(b).empty


def minute_frame(paths):
    """`m` for td.outcomes from per-session lists of (open, high, low, close) minutes."""
    sessions = pd.DatetimeIndex([DAY + pd.Timedelta(days=k) for k in range(len(paths))])
    rows = [(k, *bar) for k, path in enumerate(paths) for bar in path]
    ordv = np.array([r[0] for r in rows])
    arr = np.array([r[1:] for r in rows], float)
    return {"ts": np.arange(len(rows), dtype=np.int64), "op": arr[:, 0], "hi": arr[:, 1], "lo": arr[:, 2],
            "cl": arr[:, 3], "ordv": ordv, "ends": np.searchsorted(ordv, np.arange(len(paths)), side="right"),
            "ordinal": pd.Series(np.arange(len(paths)), index=sessions),
            "long_now": np.ones(len(paths)), "short_now": np.zeros(len(paths))}


def one_entry(price=100.0, stop=99.0, side=1):
    return pd.DataFrame({"side": [side], "session": [DAY], "ordinal": [0], "price": [price], "stop": [stop],
                         "atr": [1.0], "first": [0], "stop_first": [0], "entry_ns": [0], "tod": ["12:00"]})


def test_races_count_only_targets_before_the_stop():
    m = minute_frame([[(100.0, 100.5, 99.8, 100.4), (100.4, 101.2, 100.3, 101.0),  # +1R reached
                       (101.0, 101.1, 98.9, 99.0),                                  # stopped
                       (99.0, 104.0, 98.9, 103.5)]])                                # +3R after the stop: ignored
    r = td.outcomes(one_entry(), m).iloc[0]
    assert r.hit1 and not r.hit2 and not r.hit3 and r.stopped
    assert r.bracket == -1.0 and abs(r.mfe0 - 1.2) < 1e-12
    both = minute_frame([[(100.0, 102.5, 98.5, 100.0)]])  # one minute reaches +2R and the stop
    r = td.outcomes(one_entry(), both).iloc[0]
    assert not r.hit1 and r.stopped and r.bracket == -1.0 and r.mfe0 == 0.0


def test_a_gap_through_the_stop_exits_at_the_open_and_the_short_mirrors():
    m = minute_frame([[(100.0, 100.4, 99.6, 100.2)], [(98.0, 98.5, 97.5, 98.2)]])
    r = td.outcomes(one_entry(), m).iloc[0]
    assert r.bracket == -2.0 and r.swing == -2.0
    ms = minute_frame([[(100.0, 100.4, 99.6, 99.8)], [(102.0, 102.5, 101.5, 101.8)]])
    r = td.outcomes(one_entry(stop=101.0, side=-1), ms).iloc[0]
    assert r.bracket == -2.0 and abs(r.ret0 - 0.2) < 1e-9


def test_the_playbook_exit_waits_for_the_daily_emas_to_unstack():
    m = minute_frame([[(100.0, 100.5, 99.8, 100.4)], [(100.4, 101.0, 100.2, 100.9)], [(100.9, 101.6, 100.8, 101.5)]])
    m["long_now"] = np.array([1.0, np.nan, 0.0])  # an unusable day is skipped; the third day unstacks
    r = td.outcomes(one_entry(), m).iloc[0]
    assert r.swing_sessions == 2 and abs(r.swing - 1.5) < 1e-12


def synthetic_minutes(sessions, seed):
    stamps = [pd.date_range(o, c - pd.Timedelta(minutes=1), freq="min") for o, c in zip(sessions["open"],
                                                                                      sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.00002, 0.0012, len(ts))))
    wig = np.abs(rng.normal(0, 0.0008, len(ts)))
    return pd.DataFrame({"ts": ts, "open": close * (1 - wig / 2), "high": close * (1 + wig),
                         "low": close * (1 - wig), "close": close, "volume": 1000.0})


def test_design_run_ignores_2025_and_random_stops_match_in_atrs():
    sessions = features.trading_sessions("2024-09-03", "2025-02-28", warmup_sessions=0)
    data = {"AAA": synthetic_minutes(sessions, 1), "BBB": synthetic_minutes(sessions, 2)}
    res = td.run_study(data, sessions, draws=5)
    later = {k: v.copy() for k, v in data.items()}
    for v in later.values():
        late = v["ts"] >= pd.Timestamp("2025-01-01", tz="UTC")
        v.loc[late, ["open", "high", "low", "close"]] *= 1.3
    res2 = td.run_study(later, sessions, draws=5)
    pd.testing.assert_frame_equal(res["trades"], res2["trades"])
    t = res["trades"]
    assert len(t) and (t["session"] <= td.DESIGN_END).all() and t["truncated"].any()
    assert (t["row"] == "no trend filters").sum() >= (t["row"] == td.PRIMARY).sum()
    rnd = res["random"].merge(t[["pid", "price", "stop", "atr"]], on="pid", suffixes=("", "_t"))
    assert len(rnd)
    r_atr = rnd["side"] * (rnd["price_t"] - rnd["stop_t"]) / rnd["atr_t"]
    assert np.allclose(rnd["side"] * (rnd["price"] - rnd["stop"]) / rnd["atr"], r_atr)
    parent = t.set_index("pid")
    assert (rnd["tod"].to_numpy() == parent.loc[rnd["pid"], "tod"].to_numpy()).all()


LATE = LONG[:3] + [(101.6, 102.6, 100.9, 102.5),   # wicks into the 50% level but its body is 53% of its range
                   (102.5, 102.8, 102.2, 102.6),   # no wick into the gap
                   (102.6, 102.9, 100.95, 102.62),  # the doji, three candles after candle 3
                   (102.62, 102.8, 102.4, 102.7)]  # confirmation: closes below the doji's high


def test_the_doji_may_come_within_the_window_and_a_close_through_the_gap_voids_it():
    one = td.fvg_table(bar_frame(LATE)).iloc[0]
    assert one.doji_k == 1 and abs(one.body4 - 0.9 / 1.7) < 1e-12  # candle 4, kept for the body filter later
    r = td.fvg_table(bar_frame(LATE), doji_window=4).iloc[0]
    assert (r.doji_k, r.wick4, r.confirm, r.c5) == (3, True, True, 102.7)
    assert not td.fvg_table(bar_frame(LATE), doji_window=2).iloc[0].five_ok
    void = list(LATE)
    void[4] = (102.5, 102.6, 100.3, 100.4)  # closes below candle 1's high before the doji
    assert not td.fvg_table(bar_frame(void), doji_window=4).iloc[0].five_ok
