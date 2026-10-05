"""gap_recovery.py on SYNTHETIC minutes: the gap and its fill, recoveries, dividends, missing sessions, later
fills, the kind of gap, the summaries, and filling a renamed ticker's missing sessions."""

import numpy as np
import pandas as pd
import pytest

import features
import gap_recovery as gr
import meanrev as mr
import synthetic

NY = "America/New_York"


def minutes_for(sessions, paths):
    """Minute bars with open = previous close per session; paths: {day: closes per minute} (missing days: none)."""
    ts = synthetic.session_minute_starts(sessions)
    day = ts.tz_convert(NY).tz_localize(None).normalize()
    frames = []
    for d in sessions.index:
        if d.strftime("%Y-%m-%d") not in paths:
            continue
        closes = np.asarray(paths[d.strftime("%Y-%m-%d")], float)
        t = ts[day == d][:len(closes)]
        frames.append(synthetic.bars_from_closes(t, closes))
    return pd.concat(frames, ignore_index=True)


def path(start, points, n=390):
    """Piecewise-linear minute closes through (minute, price) points, the first bar opening at `start`."""
    xs, ys = zip(*points)
    return np.r_[start, np.interp(np.arange(1, n), xs, ys)]


def events_for(paths, dividends=None, **kw):
    sessions = features.trading_sessions("2025-03-03", "2025-03-07", warmup_sessions=0)
    m = minutes_for(sessions, paths)
    rth, _ = features.regular_session_minutes(m, sessions)
    daily, _ = mr.daily_table(m, sessions)
    divs = dividends if dividends is not None else pd.DataFrame(columns=["ex_date", "cash_amount"])
    return gr.gap_events("XYZ", rth, daily, divs, **kw), sessions


def test_a_filled_gap_records_when_it_filled_how_far_it_fell_first_and_where_it_closed():
    flat = np.full(390, 100.0)
    # Opens at 97 (-3%), falls to 96 by minute 10, back to 100.2 by minute 45, closes at 99.
    day2 = path(97.0, [(1, 97.0), (10, 96.0), (45, 100.2), (389, 99.0)])
    ev, _ = events_for({"2025-03-03": flat, "2025-03-04": day2})
    e = ev.iloc[0]
    assert e["gap"] == pytest.approx(-3.0) and e["size"] == "3-5%" and e["filled"]
    first_hit = int(np.argmax(day2 >= 100.0))  # bars_from_closes: high = max(open, close)
    assert e["fill_minutes"] == first_hit + 1
    assert e["further_drop"] == pytest.approx((96 / 97 - 1) * 100)
    assert e["best"] == pytest.approx((100.2 - 97) / 3 * 100)
    assert e["close_rec"] == pytest.approx((99 - 97) / 3 * 100)
    assert not e["extended"] and e["first"] == "fill" and e["days_to_fill"] == 0


def test_an_unfilled_gap_reports_its_best_recovery_and_later_fills():
    flat = np.full(390, 100.0)
    day2 = path(98.0, [(1, 98.0), (100, 99.0), (389, 98.5)])       # -2% gap, best 50% of it, closes 25% back
    day3 = path(98.5, [(1, 98.5), (200, 100.5), (389, 100.0)])    # fills the next day
    ev, _ = events_for({"2025-03-03": flat, "2025-03-04": day2, "2025-03-05": day3,
                        "2025-03-06": np.full(390, 100.0), "2025-03-07": np.full(390, 100.0)})
    e = ev.iloc[0]
    assert not e["filled"] and e["best"] == pytest.approx(50.0) and e["close_rec"] == pytest.approx(25.0)
    assert e["days_to_fill"] == 1 and bool(e["filled_1d"])
    assert np.isnan(e["filled_5d"])  # the data ends before 5 sessions after the gap day


def test_a_morning_dividend_is_not_counted_as_a_gap_and_a_missing_session_never_makes_one():
    flat = np.full(390, 100.0)
    open_low = path(98.9, [(1, 98.9), (389, 98.9)])
    divs = pd.DataFrame({"ex_date": [pd.Timestamp("2025-03-04")], "cash_amount": [1.2]})
    ev, _ = events_for({"2025-03-03": flat, "2025-03-04": open_low}, dividends=divs)
    assert ev.empty  # 98.9 vs 100 - 1.20 = 98.80 is no gap down
    ev, _ = events_for({"2025-03-03": flat, "2025-03-04": open_low})
    assert ev.iloc[0]["gap"] == pytest.approx(-1.1)
    ev, _ = events_for({"2025-03-03": flat, "2025-03-05": path(90.0, [(1, 90.0), (389, 90.0)])})
    assert ev.empty  # 03-04 has no data, so 03-05's open has no previous close to gap from


def test_gaps_are_labeled_earnings_with_the_market_or_the_stock_alone():
    flat = np.full(390, 100.0)
    down = path(98.0, [(1, 98.0), (389, 98.0)])
    paths = {"2025-03-03": flat, "2025-03-04": down, "2025-03-05": path(96.0, [(1, 96.0), (389, 96.0)]),
             "2025-03-06": path(94.0, [(1, 94.0), (389, 94.0)])}
    spy_gap = pd.Series({pd.Timestamp("2025-03-05"): -1.5, pd.Timestamp("2025-03-06"): -1.5})
    ev, _ = events_for(paths, spy_gap=spy_gap, earnings=pd.DatetimeIndex(["2025-03-06"]))
    assert ev["kind"].tolist() == ["stock alone", "with the market", "earnings"]


def test_summaries_count_fills_bins_and_the_race():
    ev = pd.DataFrame({"filled": [True, True, False, False], "fill_minutes": [20, 100, np.nan, np.nan],
                       "best": [110, 130, 30, 80], "close_rec": [120, 50, -10, 60], "further_drop": [-1, -2, -3, -1.0],
                       "further_drop_gaps": [0.5, 1, 1.5, 0.5], "extended": [False, True, True, False],
                       "first": ["fill", "extension", "extension", "neither"],
                       **{f"filled_{d}d": [True, True, False, np.nan] for d in gr.DAYS}})
    s = gr.summarize(ev)
    assert (s["gaps"], s["filled"], s["unfilled"]) == (4, 50.0, 2)
    assert s["by_10:00"] == 25.0 and s["by_11:30"] == 50.0
    assert s["best_25_50"] == 50.0 and s["best_75_100"] == 50.0  # the two unfilled gaps
    assert s["within_1d"] == pytest.approx(200 / 3) and s["within_1d_n"] == 3
    assert (s["race_n"], s["race_fill"]) == (3, pytest.approx(100 / 3))
    assert s["close_below_open"] == 25.0 and s["closed_filled"] == 25.0


def test_sessions_missing_under_a_renamed_ticker_come_from_its_old_ticker(tmp_path):
    sessions = features.trading_sessions("2025-03-03", "2025-03-07", warmup_sessions=0)
    first, last = "2025-03-03", "2025-03-07"
    flat = {d: np.full(390, 100.0 + k) for k, d in enumerate(["2025-03-03", "2025-03-04", "2025-03-07"])}
    old = {"2025-03-05": np.full(390, 50.0), "2025-03-06": np.full(390, 51.0)}
    minutes_for(sessions, flat).to_parquet(tmp_path / f"META_1min_{first}_{last}_splitadj.parquet")
    minutes_for(sessions, old).to_parquet(tmp_path / "FB_1min_2025-03-05_2025-03-06_splitadj.parquet")
    pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "cash_amount": pd.Series(dtype=float)}).to_parquet(
        tmp_path / f"META_cash_dividends_{first}_{last}.parquet")
    rth, daily, _, note = gr.load_ticker("META", sessions, tmp_path, refresh=False)
    assert daily["close"].tolist() == [100.0, 101.0, 50.0, 51.0, 102.0]
    assert "2 META sessions" in note and "FB" in note


def test_from_10_oclock_the_measures_start_at_the_10_oclock_price():
    flat = np.full(390, 100.0)
    # Opens at 97, is at 98 at 10:00 (up since the open), reaches 100.1 by minute 120.
    day2 = path(97.0, [(1, 97.0), (29, 98.0), (120, 100.1), (389, 99.0)])
    ev, _ = events_for({"2025-03-03": flat, "2025-03-04": day2})
    e = ev.iloc[0]
    assert e["price_10"] == pytest.approx(day2[29]) and not e["filled_by_10"]
    left = 100.0 - e["price_10"]
    assert e["filled_10"] and e["first_10"] == "fill"
    assert e["best_10"] == pytest.approx((100.1 - e["price_10"]) / left * 100)
    assert e["close_10"] == pytest.approx((99.0 - e["price_10"]) / left * 100)
    up = gr.FILTERS["Skip earnings, from 10:00, up since the open"][1]
    down = gr.FILTERS["Skip earnings, from 10:00, down since the open"][1]
    assert up(ev).tolist() == [True] and down(ev).tolist() == [False]


def test_gaps_filled_before_10_and_earnings_gaps_are_left_out_of_the_10_oclock_filters():
    flat = np.full(390, 100.0)
    early_fill = path(98.0, [(1, 98.0), (15, 100.5), (389, 100.0)])   # filled at minute 15
    no_fill = path(97.0, [(1, 97.0), (389, 96.0)])
    paths = {"2025-03-03": flat, "2025-03-04": early_fill, "2025-03-05": flat, "2025-03-06": no_fill}
    ev, _ = events_for(paths, earnings=pd.DatetimeIndex(["2025-03-06"]))
    assert ev["filled_by_10"].tolist() == [True, False]
    assert np.isnan(ev.iloc[0]["filled_10"])  # nothing left to measure from 10:00
    waited = gr.FILTERS["Skip earnings, from 10:00"][1]
    assert waited(ev).tolist() == [False, False]  # one filled before 10:00, the other is an earnings gap
    assert gr.FILTERS["Skip earnings, from the open"][1](ev).tolist() == [True, False]


def test_size_buckets_cover_small_gaps_and_the_filter_tables_group_the_rest():
    assert [gr.size_label(g) for g in (-0.3, -0.5, -0.99, -1.0, -4.9, -12.0)] == [
        "0.25-0.5%", "0.5-1%", "0.5-1%", "1-2%", "3-5%", "5%+"]
    assert gr.size_label(-0.2) is None
    assert gr.size_groups(1.0) == [("1-2%", ["1-2%"]), ("2-3%", ["2-3%"]), ("3%+", ["3-5%", "5%+"])]
    assert gr.size_groups(0.25) == [("0.25-0.5%", ["0.25-0.5%"]), ("0.5-1%", ["0.5-1%"]),
                                    ("1%+", ["1-2%", "2-3%", "3-5%", "5%+"])]


def test_with_small_gaps_the_market_counts_as_down_from_the_same_threshold():
    flat = np.full(390, 100.0)
    paths = {"2025-03-03": flat, "2025-03-04": path(99.6, [(1, 99.6), (389, 99.6)])}
    spy_gap = pd.Series({pd.Timestamp("2025-03-04"): -0.4})
    ev, _ = events_for(paths, spy_gap=spy_gap, min_gap=0.25, market_gap=0.25)
    assert ev["kind"].tolist() == ["with the market"] and ev["size"].tolist() == ["0.25-0.5%"]
