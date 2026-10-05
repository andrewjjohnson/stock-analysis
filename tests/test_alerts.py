"""alerts.py on SYNTHETIC minutes: reference-price alignment, horizon lookups across a holiday and
an early close, pending and unavailable outcomes, schedule checks, ex-dividend flags, statistics.

Calendar used throughout (exchange_calendars XNYS): 2024-11-27 is a full session, 11-28 is
Thanksgiving, 11-29 closes early at 13:00 ET, and 12-02 is the following Monday.
"""

import numpy as np
import pandas as pd
import pytest

import alerts
import features
import synthetic

NY = "America/New_York"
COLUMNS = ["date", "weekday", "strategy", "time_et", "direction", "action", "ticker", "ref_price", "score", "flag",
           "alerts_posted", "notes", "source_line"]


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def market(today):
    """Calendar through mid-December and a steadily rising minute path for sessions before `today`.
    Closes rise a cent a minute; open = close - 0.03, high = close + 0.02, low = close - 0.04 and
    vwap = close - 0.015, so every candidate ref_price convention gives a different price."""
    sessions = features.trading_sessions("2024-11-26", "2024-12-10", warmup_sessions=1)
    ts = synthetic.session_minute_starts(sessions[sessions.index < pd.Timestamp(today)])
    close = 100 + 0.01 * np.arange(len(ts))
    minutes = pd.DataFrame({"ts": ts, "open": close - 0.03, "high": close + 0.02, "low": close - 0.04,
                            "close": close, "volume": 1.0, "vwap": close - 0.015, "transactions": 1})
    return sessions, minutes


def load(tmp_path, rows):
    """rows: (date, strategy, time_et, direction, ref_price[, score]) -> alerts.load_alerts frame."""
    out = []
    for date, strategy, time_et, direction, ref, *score in rows:
        out.append({"date": date, "weekday": pd.Timestamp(date).strftime("%a"), "strategy": strategy,
                    "time_et": time_et, "direction": direction, "action": alerts.ACTIONS[strategy][direction],
                    "ticker": "SPY", "ref_price": ref, "score": score[0] if score else np.nan, "flag": "",
                    "alerts_posted": alerts.POSTED[direction], "notes": "", "source_line": 1})
    path = tmp_path / "alerts.csv"
    pd.DataFrame(out, columns=COLUMNS).to_csv(path, index=False)
    return alerts.load_alerts(path)


def price(minutes, text, field="close"):
    """`field` of the minute bar starting at `text` ET."""
    return float(minutes.loc[minutes["ts"] == et(text), field].iloc[0])


def study(tmp_path, rows, today, minutes=None, dividends=None):
    sessions, all_minutes = market(today)
    a = load(tmp_path, rows)
    return alerts.run_study(a, all_minutes if minutes is None else minutes(all_minutes), sessions,
                            pd.Timestamp(today), dividends, reps=200)


def test_reference_and_every_horizon_use_explicit_bar_timestamps(tmp_path):
    _, m = market("2024-12-04")
    at = lambda t: price(m, t)  # noqa: E731  close of the bar starting at t, i.e. the price at t + 1 min
    rows = [("2024-11-27", "intraday", "10:00", "BULLISH", at("2024-11-27 09:59")),
            ("2024-11-27", "overnight", "15:55", "BEARISH", at("2024-11-27 15:54"), -3),
            ("2024-11-29", "intraday", "10:00", "BEARISH", at("2024-11-29 09:59"))]
    t = study(tmp_path, rows, "2024-12-04")["table"]
    intraday, overnight, early = t.iloc[0], t.iloc[1], t.iloc[2]

    # The price known at the alert time is the close of the bar that ENDS then (starts one minute earlier).
    assert intraday["massive_price"] == at("2024-11-27 09:59")
    assert intraday["ref_matches"].split(",") == ["T"] and not intraday["ref_off_0.1pct"]
    ret = lambda end, start: (end / start - 1) * 100  # noqa: E731
    ref = intraday["massive_price"]
    assert np.isclose(intraday["ret_close_pct"], ret(at("2024-11-27 15:59"), ref))
    # Next session skips Thanksgiving, and its close is the 13:00 early close.
    assert np.isclose(intraday["ret_next_close_pct"], ret(at("2024-11-29 12:59"), ref))
    assert np.isnan(intraday["ret_next_open_pct"]) and intraday["status_next_open"] == ""  # not an intraday horizon

    ref = overnight["massive_price"]
    assert overnight["next_session"] == pd.Timestamp("2024-11-29")
    assert np.isclose(overnight["ret_next_open_pct"], ret(price(m, "2024-11-29 09:30", "open"), ref))
    assert np.isclose(overnight["ret_next_1000_pct"], ret(at("2024-11-29 09:59"), ref))
    assert np.isclose(overnight["ret_next_close_pct"], ret(at("2024-11-29 12:59"), ref))
    assert np.isclose(overnight["signed_next_open_pct"], -overnight["ret_next_open_pct"])  # BEARISH flips the sign
    assert np.isclose(early["ret_close_pct"], ret(at("2024-11-29 12:59"), early["massive_price"]))
    assert np.isclose(early["ret_next_close_pct"], ret(at("2024-12-02 15:59"), early["massive_price"]))

    # Naive-rule inputs: prior session close for intraday, today's first regular open for overnight.
    assert intraday["prior_close"] == at("2024-11-26 15:59") and intraday["momentum_side"] == 1
    assert overnight["session_open_price"] == price(m, "2024-11-27 09:30", "open")


def test_intraday_path_excludes_the_reference_bar_and_measures_touches(tmp_path):
    def spikes(m):
        m = m.copy()
        m.loc[m["ts"] == et("2024-11-29 09:59"), "low"] = 1.0  # the reference bar: before the alert, ignored
        ref = m.loc[m["ts"] == et("2024-11-29 09:59"), "close"].iloc[0]
        m.loc[m["ts"] == et("2024-11-29 11:00"), "low"] = ref * 0.994  # 0.6% against a BULLISH call
        return m

    _, m = market("2024-12-04")
    rows = [("2024-11-29", "intraday", "10:00", "BULLISH", price(m, "2024-11-29 09:59"))]
    r = study(tmp_path, rows, "2024-12-04", minutes=spikes)["table"].iloc[0]
    assert np.isclose(r["adverse_to_close_pct"], -0.6)
    assert r["touched_0.5pct"] and not r["touched_1pct"]
    assert r["signed_close_pct"] > 0 and r["closed_within_0pct"]  # the rising path still closes higher


def test_pending_and_unavailable_outcomes_are_kept_but_left_out_of_statistics(tmp_path):
    today = "2024-12-02"  # minutes exist only through 11-29; anything ending on 12-02 or later is pending
    _, m = market(today)
    drop = lambda m: m[m["ts"] != et("2024-11-27 14:00")].reset_index(drop=True)  # noqa: E731  one missing minute
    rows = [("2024-11-27", "intraday", "10:00", "BULLISH", price(m, "2024-11-27 09:59")),   # path has the gap
            ("2024-11-27", "overnight", "15:55", "BULLISH", price(m, "2024-11-27 15:54"), 2),  # after the gap
            ("2024-11-29", "intraday", "10:00", "BULLISH", price(m, "2024-11-29 09:59")),   # next close = today
            ("2024-11-29", "overnight", "12:55", "BEARISH", price(m, "2024-11-29 12:54"), -2),  # before the early
            # close; its next open is today
            ("2024-12-02", "intraday", "10:00", "BEARISH", 101.0)]                          # today itself
    result = study(tmp_path, rows, today, minutes=drop)
    t = result["table"]
    assert len(t) == len(rows)
    assert t["status_close"].tolist()[0::2] == ["unavailable", "ok", "pending"]
    assert t["status_next_close"].tolist()[0::2] == ["unavailable", "pending", "pending"]
    assert t.iloc[1][["status_next_open", "status_next_1000", "status_next_close"]].tolist() == ["ok"] * 3
    assert t.iloc[3][["status_next_open", "status_next_1000", "status_next_close"]].tolist() == ["pending"] * 3
    assert np.isnan(t.iloc[4]["massive_price"])  # today's session is never loaded
    unavailable_or_pending = t.loc[[0, 4], ["ret_close_pct", "signed_close_pct"]].to_numpy()
    assert np.isnan(unavailable_or_pending).all()

    s = result["summary"]
    rate = alerts.pick(s, "hit_rate", "intraday", "close", "as recorded", "hit_rate")
    assert rate["n"] == 1 and rate["count"] == 1  # only 11-29 counts
    assert "1 pending, 1 unavailable" in rate["note"]
    assert alerts.pick(s, "hit_rate", "intraday", "next_close", "as recorded", "hit_rate")["n"] == 0
    assert alerts.pick(s, "hit_rate", "overnight", "next_open", "as recorded", "hit_rate")["n"] == 1


def test_alert_time_checks_and_ex_dividend_windows(tmp_path):
    _, m = market("2024-12-04")
    rows = [("2024-11-27", "intraday", "10:00", "BULLISH", price(m, "2024-11-27 09:59")),
            ("2024-11-27", "overnight", "15:54", "BULLISH", price(m, "2024-11-27 15:54"), 3),  # quotes the 15:55 price
            ("2024-11-29", "intraday", "10:00", "BEARISH", price(m, "2024-11-29 09:59")),
            ("2024-11-29", "overnight", "15:55", "NEUTRAL", 100.0, 0)]  # after the 13:00 early close
    dividends = pd.DataFrame({"ex_dividend_date": ["2024-11-29"], "cash_amount": [1.5], "pay_date": ["2025-01-30"],
                              "dividend_type": ["CD"]})
    t = study(tmp_path, rows, "2024-12-04", dividends=dividends)["table"]

    assert t["issues"].iloc[0] == "" and t["issues"].iloc[2] == ""
    assert t["issues"].iloc[1] == "time is off the posting schedule"
    assert t["ref_matches"].iloc[1] == "T+1"  # equals the price one minute after the stamped time
    assert "outside the session's regular hours" in t["issues"].iloc[3]
    assert t.iloc[3][["status_next_open", "status_next_close"]].tolist() == ["unavailable"] * 2

    # Windows from 11-27 into 11-29 span the ex-date; the 11-29 intraday alert comes after that open.
    assert t["crosses_ex_div"].tolist() == [True, True, False, False]
    assert t["ex_div_cash"].iloc[1] == 1.5


def test_statistics_helpers():
    lo, hi = alerts.wilson(5, 10)
    assert np.isclose(lo, 23.66, atol=0.01) and np.isclose(hi, 76.34, atol=0.01)
    assert alerts.wilson(0, 0) == (pytest.approx(np.nan, nan_ok=True), pytest.approx(np.nan, nan_ok=True))
    assert np.isclose(alerts.binomial_tail(10, 10), 1 / 1024)
    assert np.isclose(alerts.binomial_tail(0, 7), 1) and np.isclose(alerts.binomial_head(7, 7), 1)

    # Calls that always match the move beat almost every reshuffle of the same calls; reversed, almost none.
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 1, 40)
    side = np.sign(raw)
    p_better, p_worse, pct = alerts.shuffle_test(side, raw, 2000, 0)
    assert p_better < 0.01 and p_worse > 0.99 and pct > 99
    p_better, p_worse, _ = alerts.shuffle_test(-side, raw, 2000, 0)
    assert p_better > 0.99 and p_worse < 0.01

    rho, lo, hi, p = alerts.spearman_test(np.arange(30.0), np.arange(30.0) ** 3, 2000, 0)
    assert np.isclose(rho, 1) and p < 0.01
