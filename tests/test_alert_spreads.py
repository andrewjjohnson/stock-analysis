"""alert_spreads.py on SYNTHETIC option bars: strikes and tickers, fills from both-legs minutes only,
take profit / stop / time exit, costs, pending days, the cache-only loader and the shuffled baseline.

Spreads are built as a long leg fixed at 0.10 and a short leg at spread value + 0.10, one bar per
regular minute, so each minute's spread value is set directly.
"""

import numpy as np
import pandas as pd
import pytest
from massive.exceptions import BadResponse

import alert_spreads as sp
import alerts
import features
import synthetic

NY = "America/New_York"
DAY = "2024-11-27"  # a full session


def et(text):
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def legs(day=DAY, value=0.50, closes=None, opens=None, drop_long=()):
    """(short, long) minute bars for one session: spread open/close = `value` unless overridden by
    {'HH:MM': v} in `opens` / `closes`; minutes in `drop_long` have no long-leg bar (no trade)."""
    sessions = features.trading_sessions(day, day, warmup_sessions=0)
    ts = synthetic.session_minute_starts(sessions)
    hhmm = ts.tz_convert(NY).strftime("%H:%M")
    o = np.array([(opens or {}).get(h, value) for h in hhmm])
    c = np.array([(closes or {}).get(h, value) for h in hhmm])
    long = pd.DataFrame({"ts": ts, "open": 0.10, "high": 0.10, "low": 0.10, "close": 0.10, "volume": 1.0})
    short = long.assign(open=o + 0.10, close=c + 0.10, high=np.maximum(o, c) + 0.10, low=np.minimum(o, c) + 0.10)
    return short, long[~np.isin(hhmm, list(drop_long))].reset_index(drop=True)


def trade(**kwargs):
    short, long = legs(**kwargs)
    loader = {("short", DAY): short, ("long", DAY): long}
    session = features.trading_sessions(DAY, DAY, warmup_sessions=0).loc[pd.Timestamp(DAY)]
    load = lambda ticker, day: loader["short" if ticker.endswith("761000") else "long", day]  # noqa: E731
    return sp.prepare_trade(pd.Timestamp(DAY), 1, 760.50, session, load)


GROSS, BASE = (0.0, 0.0), (0.01, 0.65)  # BASE: $0.01/share slippage, $0.65/contract commission


def test_cost_tiers_default_to_no_costs_and_always_add_wider_slippage():
    assert sp.cost_tiers() == {"base": (0.0, 0.0), "+$0.01 slippage": (0.01, 0.0), "+$0.03 slippage": (0.03, 0.0)}
    assert list(sp.cost_tiers(0.01, 0.65)) == ["gross", "base", "+$0.01 slippage", "+$0.03 slippage"]
    assert sp.cost_tiers(0.01, 0.65)["+$0.03 slippage"] == pytest.approx((0.04, 0.65))


def test_strikes_and_tickers_match_the_traders_example():
    assert sp.spread_legs(1, 760.50) == ("P", 761, 760)   # bullish: sell 761P, buy 760P
    assert sp.spread_legs(-1, 760.50) == ("C", 760, 761)  # bearish: sell 760C, buy 761C
    assert sp.spread_legs(1, 760.00) == ("P", 760, 759)   # exactly on a strike: that strike is the short leg
    assert sp.spread_legs(-1, 760.00) == ("C", 760, 761)
    assert sp.spread_legs(1, 759.999) == ("P", 760, 759)  # rounded to the cent first
    assert sp.option_ticker("2026-06-10", "P", 761) == "O:SPY260610P00761000"
    # Offsets move both legs together: + deeper in the money, - further out, for either direction.
    assert sp.spread_legs(1, 760.50, 1) == ("P", 762, 761) and sp.spread_legs(1, 760.50, -2) == ("P", 759, 758)
    assert sp.spread_legs(-1, 760.50, 1) == ("C", 759, 760) and sp.spread_legs(-1, 760.50, -2) == ("C", 762, 763)


def test_take_profit_fills_at_its_level_and_only_from_minutes_where_both_legs_traded():
    t = trade(closes={"11:00": 0.24, "12:00": 0.05}, drop_long=("12:00",))
    assert t["status"] == "ok" and t["credit_traded"] == pytest.approx(0.50)
    assert pd.Timestamp(t["entry_time"], tz="UTC") == et(f"{DAY} 10:00")
    pnl, reason, credit = sp.simulate(t, 50, None, *GROSS)
    assert (reason, pnl, credit) == ("take profit", pytest.approx(25.0), pytest.approx(0.50))  # bought back at 0.25

    # With costs, a buy limit at 0.24 needs trades at least two cents through it: 0.24 is not enough,
    # and the 0.05 print at 12:00 is unusable because the long leg didn't trade then.
    pnl, reason, credit = sp.simulate(t, 50, None, *BASE)
    assert reason == sp.EXIT and credit == pytest.approx(0.48)
    assert pnl == pytest.approx((0.48 - 0.52) * 100 - 4 * 0.65)  # 15:30 open 0.50 plus slippage


def test_stop_fills_at_the_crossing_value_capped_at_the_width_and_time_exit_waits_for_both_legs():
    t = trade(closes={"13:00": 0.80}, opens={"15:31": 0.40}, drop_long=("15:30",))
    pnl, reason, _ = sp.simulate(t, None, 50, *GROSS)  # stop level 0.50 + 50% x (1 - 0.50) = 0.75
    assert (reason, pnl) == ("stop", pytest.approx(-30.0))
    assert sp.simulate(trade(closes={"13:00": 1.20}), None, 50, *GROSS)[0] == pytest.approx(-50.0)  # max loss
    assert sp.simulate(t, 90, None, *GROSS)[:2] == (pytest.approx(10.0), sp.EXIT)  # 15:30 long missing: 15:31 open
    assert pd.Timestamp(t["exit_time"], tz="UTC") == et(f"{DAY} 15:31")


def test_entry_and_exit_need_both_legs_within_five_minutes():
    late = [f"10:0{i}" for i in range(5)]
    assert trade(drop_long=late)["status"].startswith("no entry")
    assert trade(drop_long=late[:4])["entry_time"] == et(f"{DAY} 10:04").value
    assert trade(opens={"10:00": 1.10})["status"] == "no entry: traded credit outside 0-1"
    assert trade(drop_long=[f"15:3{i}" for i in range(5)])["status"].startswith("no exit")


def test_cached_bars_load_without_a_key_and_out_of_plan_dates_are_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    short, _ = legs()
    path = tmp_path / "options" / "SPY241127P00761000_2024-11-27.parquet"
    path.parent.mkdir(parents=True)
    short.to_parquet(path, index=False)
    load = sp.option_loader(tmp_path)
    assert len(load("O:SPY241127P00761000", DAY)) == len(short) and load.state["client"] is None

    class OutOfPlan:
        def list_aggs(self, *args, **kwargs):
            raise BadResponse('{"status":"NOT_AUTHORIZED","message":"Your plan doesn\'t include this data timeframe."}')

    with pytest.raises(sp.NotInPlan):
        sp.fetch_option_minutes("O:SPY240603P00580000", "2024-06-03", OutOfPlan())


def test_run_study_pairs_both_directions_and_leaves_pending_days_out(tmp_path):
    days = ["2024-11-26", "2024-11-27", "2024-12-02"]  # the last one is "today"
    rows = pd.DataFrame({"date": days, "weekday": [pd.Timestamp(d).strftime("%a") for d in days],
                         "strategy": "intraday", "time_et": "10:00", "direction": ["BULLISH", "BEARISH", "BULLISH"],
                         "action": ["SELL PUTS", "SELL CALLS", "SELL PUTS"], "ticker": "SPY", "ref_price": 100.0,
                         "score": np.nan, "flag": "", "alerts_posted": 1, "notes": "", "source_line": 1})
    rows.to_csv(tmp_path / "log.csv", index=False)
    a = alerts.load_alerts(tmp_path / "log.csv")
    sessions = features.trading_sessions("2024-11-26", "2024-12-10", warmup_sessions=0)
    ts = synthetic.session_minute_starts(sessions[sessions.index < pd.Timestamp("2024-12-02")])
    spy = synthetic.bars_from_closes(ts, np.full(len(ts), 100.5))

    def load(ticker, day):
        # Puts: the spread decays to 0.10 by 11:00 (take profit). Calls: flat at 0.50 (15:30 exit).
        short, long = legs(day=day, closes={"11:00": 0.10} if ticker[11] == "P" else None)
        return short if ticker[12:].lstrip("0").startswith("101" if ticker[11] == "P" else "100") else long

    result = sp.run_study(a, spy, sessions, pd.Timestamp("2024-12-02"), load, reps=200)
    assert [str(d.date()) for d in result["days_used"]] == days[:2]
    assert result["trades"][pd.Timestamp(days[2]), 1]["status"] == "pending"
    u = result["user"]["base"]
    assert np.allclose(u["calls"], [u["bull"][0], u["bear"][1]])  # each alert trades its own direction
    alerts_row = sp.cell(result["grid"], "alerts", result["rule"])
    assert alerts_row["trades"] == 2 and alerts_row["total"] == pytest.approx(u["calls"].sum())
    placement = result["placement"]  # the synthetic loader only prices the offset-0 legs
    zero = placement[(placement["scope"] == "alerts") & (placement["offset"] == 0) & (placement["cost"] == "base")
                     & (placement["take_profit"] == 50)].iloc[0]
    assert result["rule"] == sp.DEFAULT_RULE and result["offset_rules"] == ((80, None), (50, None))
    assert "No completed alert trades since then" in sp.since_adopted({**result, "rule_since": "2024-12-01"})
    assert "Since adopting it: 1 trade," in sp.since_adopted({**result, "rule_since": "2024-11-27"})
    assert zero["trades"] == 2 and zero["total"] == pytest.approx(sp.cell(result["grid"], "alerts", (50, None))["total"])
    other = placement[(placement["scope"] == "alerts") & (placement["offset"] != 0)]
    assert (other["trades"] == 0).all() and (other.loc[other["cost"] == "base", "days_missing"] == 2).all()
    sims = sp.shuffled_totals(result["sides"], u["bull"], u["bear"], 50, 0)
    assert set(np.round(sims, 6)) <= {round(u["bull"][0] + u["bear"][1], 6), round(u["bull"][1] + u["bear"][0], 6)}


def test_consistency_score_and_month_by_month_reselection():
    assert sp.consistency([10, 10, 10]) != sp.consistency([10, 10, 10])  # no spread: undefined (NaN)
    steady, lumpy = np.array([5.0, 6, 4, 5, 6, 4]), np.array([30.0, -20, 25, -15, 5, 5])
    assert steady.mean() == lumpy.mean() and sp.consistency(steady) > sp.consistency(lumpy)
    assert np.isclose(sp.consistency(steady), steady.mean() / (steady.std(ddof=1) / np.sqrt(6)))

    # Rule A wins steadily in May-June and loses in July; rule B (the current one) is the reverse.
    days = pd.to_datetime(["2026-05-04", "2026-05-05", "2026-06-01", "2026-06-02", "2026-07-01", "2026-07-02"])
    cells = {(80, None): np.array([10.0, 12, 11, 9, -30, -40]), (50, None): np.array([1.0, -2, 0, -1, 20, 25])}
    m = sp.leave_one_month_out(cells, days, current=(50, None)).set_index("month")
    # Without July, A looks best and is picked for July, where only B made money: chosen without peeking.
    assert (m.loc["2026-07", "picked_take_profit"], m.loc["2026-07", "picked_pnl"]) == (80, -70)
    assert m.loc["2026-07", "current_pnl"] == 45 and m.loc["2026-07", "hindsight_take_profit"] == 50
    assert m["trades"].tolist() == [2, 2, 2]


def test_parity_pricing_values_a_spread_as_one_dollar_minus_its_twin():
    # Bull put 762/761 <-> call spread on the same strikes (long 761C, short 762C); bear call 759/760 <-> puts.
    assert sp.parity_legs("P", 762, 761) == ("C", 761, 762)
    assert sp.parity_legs("C", 759, 760) == ("P", 760, 759)
    # Twin call spread worth 0.30 all day -> the put spread is priced at 0.70.
    short, long = legs(value=0.30)  # short = the 761C here (a leg), long = the 762C (b leg)
    session = features.trading_sessions(DAY, DAY, warmup_sessions=0).loc[pd.Timestamp(DAY)]
    load = lambda ticker, day: short if ticker.endswith("C00761000") else long  # noqa: E731
    t = sp.prepare_trade(pd.Timestamp(DAY), 1, 760.50, session, load, offset=1, parity=True)
    assert (t["short_ticker"], t["priced_from"]) == ("O:SPY241127P00762000", "parity twin")
    assert t["credit_traded"] == pytest.approx(0.70) and t["exit_value"] == pytest.approx(0.70)


def test_block_resampling_keeps_runs_and_drawdown_anatomy_counts_the_drop():
    pnl = np.arange(10.0)  # distinct values reveal the resampled runs
    paths = sp.resample_paths(pnl, 12, 50, 0, block=5)
    assert paths.shape == (50, 12)
    for path in paths:  # each run of 5 is consecutive (mod 10) in the original order
        for start in (0, 5):
            run = path[start:start + 5][: len(path) - start]
            assert ((np.diff(run) == 1) | (np.diff(run) == -9)).all()

    # Up to 30, then -20, +5, -20, -10: the drawdown is 30 -> -15 = 45 over 4 trades (3 losers, 1 winner),
    # and the longest losing streak inside it is only 2 (-20, -10).
    dd, n, losers, winners, streak = sp.drawdown_anatomy([10, 20, -20, 5, -20, -10, 5])
    assert (dd, n, losers, winners, streak) == (45, 4, 3, 1, 2)
    assert sp.longest_losing_streak(np.array([[1, -1, -1, 2, -1, -1, -1]]))[0] == 3


def test_breakeven_stop_after_40pct_profit_closes_winners_that_turn_back():
    # Credit 0.50. From 11:00 the spread is worth 0.28 (a 44% profit, so the stop arms); at 12:00 it jumps to
    # 0.55: the stop exits there for a $5 loss instead of riding to the 15:30 exit at 0.90.
    low = {f"11:{m:02d}": 0.28 for m in range(60)}
    t = trade(closes={**low, "12:00": 0.55}, opens={"15:30": 0.90})
    pnl, reason, _ = sp.simulate(t, 80, None, *GROSS)
    assert reason == sp.EXIT and pnl == pytest.approx(-40.0)
    pnl, reason, _ = sp.simulate(t, 80, None, *GROSS, breakeven_at=40)
    assert reason == "breakeven stop" and pnl == pytest.approx(-5.0)
    # Only 30% profit (0.35) never arms it; a take profit reached first still wins (50% = 0.25 at 13:00).
    assert sp.simulate(trade(closes={"11:00": 0.35, "12:00": 0.60}), None, None, *GROSS, breakeven_at=40)[1] == sp.EXIT
    lower = {**low, **{f"12:{m:02d}": 0.28 for m in range(60)}, "13:00": 0.24, "14:00": 0.60}
    pnl, reason, _ = sp.simulate(trade(closes=lower), 50, None, *GROSS, breakeven_at=40)
    assert reason == "take profit" and pnl == pytest.approx(25.0)
