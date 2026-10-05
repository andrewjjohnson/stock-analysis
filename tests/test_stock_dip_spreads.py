"""stock_dip_spreads.py on SYNTHETIC data: strikes from a listed grid, weekly expiries, the earnings flags,
pricing a single-stock spread, pooling tickers one spread at a time, and the split guard."""

import numpy as np
import pandas as pd
import pytest

import dip_spreads as ds
import features
import stock_dip_spreads as sds
import synthetic

NY = "America/New_York"


def test_strikes_come_from_the_listed_grid():
    grid = [220, 222.5, 225, 227.5, 230, 232.5, 235]
    # 1% below 231.30 = 228.99 -> 227.5; 1.5% of the price lower (3.47) -> nearest listed 225.
    assert sds.pick_strikes(grid, 231.30, 1.0) == (227.5, 225.0)
    assert sds.pick_strikes(grid, 231.30, 0.0) == (230.0, 227.5)  # at the money: highest strike at or below
    assert sds.pick_strikes(grid, 219.0, 0.0) is None  # nothing listed at or below
    assert sds.pick_strikes([220], 221.0, 0.0) is None  # no lower strike for the long leg
    # A tie between two lower strikes goes to the lower one (the wider spread).
    assert sds.pick_strikes([95, 97.5, 100], 100.0, 0.0, width_pct=3.75) == (100.0, 95.0)


def test_expiry_is_the_first_listed_one_at_least_two_sessions_out():
    fridays = np.array([4, 9, 14])  # session positions of the weekly expiries
    assert sds.pick_expiry(0, fridays) == 4   # Monday -> this Friday (4 sessions)
    assert sds.pick_expiry(2, fridays) == 4   # Wednesday -> this Friday (2 sessions)
    assert sds.pick_expiry(3, fridays) == 9   # Thursday: Friday is 1 session away -> next Friday (6)
    assert sds.pick_expiry(12, fridays) == 14
    assert sds.pick_expiry(13, fridays) is None  # nothing listed 2-7 sessions out


def test_earnings_flags_the_stock_specific_volume_jump_and_gap_with_a_session_either_side():
    sessions = features.trading_sessions("2024-12-02", "2025-03-31", warmup_sessions=0)
    days = sessions.index
    stock = pd.DataFrame({"open": 100.0, "close": 100.0, "volume": 1e6}, index=days)
    spy = pd.DataFrame({"open": 500.0, "close": 500.0, "volume": 5e7}, index=days)
    stock.loc["2025-01-15", "volume"] = 9e6               # biggest jump, but outside the reporting window
    stock.loc["2025-01-27", "volume"] = 3e6               # market-wide: SPY jumps just as much
    spy.loc["2025-01-27", "volume"] = 1.5e8
    stock.loc["2025-01-30", "volume"] = 5e6               # the stock's own jump inside the window
    stock.loc["2025-02-04", "open"] = 105.0               # the stock's own 5% overnight gap
    flagged = sds.earnings_days(stock, spy)
    assert list(flagged.strftime("%m-%d")) == ["01-29", "01-30", "01-31", "02-03", "02-04", "02-05"]


def test_spreads_are_skipped_only_when_an_earnings_session_falls_inside_them():
    events = pd.DatetimeIndex(["2025-01-31"])
    assert sds.spans_earnings(pd.Timestamp("2025-01-29"), pd.Timestamp("2025-01-31"), events)
    assert not sds.spans_earnings(pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-07"), events)  # opened after it
    assert not sds.spans_earnings(pd.Timestamp("2025-01-24"), pd.Timestamp("2025-01-30"), events)  # expired before it


def leg(sessions, value):
    ts = synthetic.session_minute_starts(sessions)
    return pd.DataFrame({"ts": ts, "open": value, "high": value, "low": value, "close": value, "volume": 1.0})


def test_a_stock_spread_is_planned_from_the_chain_and_priced_like_spys():
    sessions = features.trading_sessions("2025-01-27", "2025-01-31", warmup_sessions=0)
    days = sessions.index
    raw = pd.DataFrame({"snap": [231.30, 230.0, 229.0, 228.0, 227.0],
                        "close": [231.0, 230.0, 229.0, 228.0, 226.0]}, index=days)
    chain = pd.DataFrame({"expiry": pd.Timestamp("2025-01-31"), "strike": [225, 227.5, 230, 232.5, 235.0]})
    plans = sds.plan_trades("AAPL", days, raw, chain, distances=(1.0,))
    assert [p.get("dte") for p in plans[:3]] == [4, 3, 2]
    assert plans[3]["status"].startswith("no listed expiry")  # Thursday: Friday is only 1 session away
    mon = plans[0]
    assert (mon["short_ticker"], mon["long_ticker"], mon["width"]) == (
        "O:AAPL250131P00227500", "O:AAPL250131P00225000", 2.5)
    bars = {"O:AAPL250131P00227500": leg(sessions, 1.50), "O:AAPL250131P00225000": leg(sessions, 0.70)}
    trades = sds.price_trades(plans, sessions, raw, lambda ticker, start, end: bars[ticker])
    tr = trades[days[0], 1.0]
    assert tr["status"] == "ok" and tr["credit_traded"] == pytest.approx(0.80)
    assert pd.Timestamp(tr["entry_time"], tz="UTC") == pd.Timestamp("2025-01-27 15:50", tz=NY)
    assert tr["settle"] == pytest.approx(1.5)  # closed at 226 on expiry: 227.5 - 226
    assert ds.simulate(tr, "expiry", 0.0, 0.0)[0] == pytest.approx((0.80 - 1.5) * 100)


def test_combined_view_pools_each_tickers_trades_by_exit_date():
    sig, cost = sds.PREREGISTERED[0], "base"

    def trades(days, pnl, exits):
        return pd.DataFrame({"pnl": pnl, "exit_day": pd.to_datetime(exits)}, index=pd.to_datetime(days))

    taken = {("AAPL", "skip", 1.0, "80", cost, sig): trades(["2025-01-06", "2025-01-13"], [100.0, -300.0],
                                                           ["2025-01-08", "2025-01-15"]),
             ("MSFT", "skip", 1.0, "80", cost, sig): trades(["2025-01-07"], [50.0], ["2025-01-10"]),
             ("SPY", "skip", sds.SPY["distance"], "80", cost, sig): trades(["2025-02-03"], [80.0], ["2025-02-05"])}
    months = pd.period_range("2025-01", "2025-02", freq="M")
    m, t = sds.combined(taken, ["AAPL", "MSFT", "SPY"], sig, cost, months)
    assert t["ticker"].tolist() == ["AAPL", "MSFT", "AAPL", "SPY"]  # ordered by exit date
    # Running total 100, 150, -150, -70: the worst drawdown is 150 -> -150.
    assert (m["trades"], m["total"], m["max_dd"], m["per_month"]) == (4, pytest.approx(-70.0), pytest.approx(300.0), 2.0)
    assert (m["losing_months"], m["worst_month"]) == (1, pytest.approx(-150.0))


def test_a_split_inside_the_option_window_stops_the_run(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    cols = {"execution_date": str, "split_from": float, "split_to": float}
    pd.DataFrame({"execution_date": ["2025-06-02"], "split_from": [1.0], "split_to": [4.0]}).to_parquet(
        ref / "XYZ_splits_2024-10-01_2026-09-30.parquet")
    pd.DataFrame(columns=list(cols)).astype(cols).to_parquet(ref / "ABC_splits_2024-10-01_2026-09-30.parquet")
    with pytest.raises(SystemExit, match="split inside the option window"):
        sds.check_splits("XYZ", "2024-10-01", "2026-09-30", cache_dir=tmp_path)
    sds.check_splits("ABC", "2024-10-01", "2026-09-30", cache_dir=tmp_path)  # cached, no split: no network, no error
