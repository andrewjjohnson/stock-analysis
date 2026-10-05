"""goose_spreads.py on SYNTHETIC option bars: the value at expiry, prices from a both-legs minute or each leg apart, one
spread through entry and every exit with slippage, and which spreads each rule takes."""

import numpy as np
import pandas as pd
import pytest

import alert_spreads as asp
import features
import goose_spreads as gs


def test_value_at_expiry():
    assert gs.settle("P", 601, 602.0) == 0 and gs.settle("P", 601, 600.4) == pytest.approx(0.6)
    assert gs.settle("P", 601, 598.0) == 1
    assert gs.settle("C", 600, 599.0) == 0 and gs.settle("C", 600, 600.3) == pytest.approx(0.3)
    assert gs.settle("C", 600, 603.0) == 1


def bars(rows):
    """rows: (timestamp, open, close)."""
    return pd.DataFrame({"ts": pd.to_datetime([r[0] for r in rows], utc=True).as_unit("ns"),
                         "open": [r[1] for r in rows], "high": [max(r[1:]) for r in rows],
                         "low": [min(r[1:]) for r in rows], "close": [r[2] for r in rows]})


T = pd.Timestamp("2025-03-12 13:35", tz="UTC")  # 9:35 ET


def test_prices_come_from_a_both_legs_minute_else_each_leg_apart():
    m = pd.Timedelta(minutes=1)
    short = bars([(T + m, 1.0, 1.1), (T + 2 * m, 1.2, 1.25)])
    long = bars([(T, 0.4, 0.4), (T + 2 * m, 0.5, 0.55)])
    assert gs.value_at(short, long, T.value)[::2] == (pytest.approx(0.7), "both legs")  # minute 2: 1.2 - 0.5
    long_apart = bars([(T, 0.4, 0.4), (T + 3 * m, 0.5, 0.55)])
    short_apart = bars([(T + m, 1.0, 1.1)])
    assert gs.value_at(short_apart, long_apart, T.value)[::2] == (pytest.approx(0.6), "legs apart")  # 1.0 - 0.4
    late = bars([(T + 6 * m, 1.0, 1.0)])  # outside the 5 minutes
    assert np.isnan(gs.value_at(late, long, T.value)[0])


def test_one_spread_through_entry_and_every_exit():
    sessions = features.trading_sessions("2025-03-10", "2025-03-14", warmup_sessions=0)
    day, nxt = pd.Timestamp("2025-03-11"), pd.Timestamp("2025-03-12")
    entry = sessions.loc[day, "close"] - pd.Timedelta(minutes=10)
    o2 = sessions.loc[nxt, "open"]
    at = lambda h, mi: pd.Timestamp(f"2025-03-12 {h:02d}:{mi:02d}", tz=features.NY).tz_convert("UTC")  # noqa: E731
    legs = {  # bull put spread at spot 600.4: sell the 601 put, buy the 600 put
        (asp.option_ticker(nxt, "P", 601), "2025-03-11"): bars([(entry, 1.10, 1.08), (entry + pd.Timedelta(minutes=1),
                                                                                       1.08, 1.05)]),
        (asp.option_ticker(nxt, "P", 600), "2025-03-11"): bars([(entry, 0.50, 0.49), (entry + pd.Timedelta(minutes=1),
                                                                                       0.49, 0.48)]),
        (asp.option_ticker(nxt, "P", 601), "2025-03-12"): bars([(o2 + pd.Timedelta(minutes=5), 0.80, 0.78),
                                                               (at(11, 0), 0.20, 0.10), (at(15, 30), 0.10, 0.09)]),
        (asp.option_ticker(nxt, "P", 600), "2025-03-12"): bars([(o2 + pd.Timedelta(minutes=5), 0.45, 0.44),
                                                               (at(11, 0), 0.12, 0.05), (at(15, 30), 0.05, 0.05)]),
    }
    t = gs.spread_trade(day, nxt, 1, 600.4, sessions, lambda tk, d: legs[(tk, d)], close_next=602.0)
    assert t["status"] == "ok" and (t["right"], t["short_strike"], t["long_strike"]) == ("P", 601, 600)
    assert t["credit"] == pytest.approx(0.60) and t["v935"] == pytest.approx(0.35) and t["v_late"] == pytest.approx(0.05)
    assert t["watch"].tolist() == pytest.approx([0.57, 0.34, 0.05])  # 15:51, then the next day before the 15:30 exit
    assert t["expiry_value"] == 0
    slip = 0.02  # credit after slippage 0.56
    assert gs.trade_pnl(t, "9:35", slip) == pytest.approx((0.56 - 0.35 - 0.04) * 100)
    assert gs.trade_pnl(t, gs.EXITS[1], slip) == pytest.approx((0.56 - 0.56 * 0.2) * 100)  # 0.05 <= 0.112 - 0.04
    assert gs.trade_pnl(t, "hold to expiry", slip) == pytest.approx(56.0)
    missing = {k: v for k, v in legs.items() if k[1] == "2025-03-11"}
    missing.update({k: v.iloc[:0] for k, v in legs.items() if k[1] == "2025-03-12"})
    assert gs.spread_trade(day, nxt, 1, 600.4, sessions, lambda tk, d: missing[(tk, d)], 602.0)["status"] == \
        "no exit price"


def test_each_rule_takes_its_side():
    days = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"])
    trades = pd.DataFrame({"session": np.repeat(days, 2), "side": [1, -1] * 3, "status": "ok"})
    d = pd.Series([1, -1, 0], index=days)
    goose = gs.choose(trades, d, "Green Goose")
    assert list(zip(goose["session"], goose["side"])) == [(days[0], 1), (days[1], -1)]
    against = gs.choose(trades, d, "against Green Goose")
    assert list(zip(against["session"], against["side"])) == [(days[0], -1), (days[1], 1)]
    assert (gs.choose(trades, d, "always put spreads")["side"] == 1).all()
    assert len(gs.choose(trades, d, "always call spreads")) == 3
