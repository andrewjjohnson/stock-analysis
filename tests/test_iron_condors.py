"""iron_condors.py on SYNTHETIC data: the condor's value path from its two spreads, exits on the total, and put
strikes chosen by delta through put-call parity (call_spreads.choose_shorts)."""

import numpy as np
import pandas as pd
import pytest

import call_spreads as cs
import features
import iron_condors as ic

NY = "America/New_York"


def spread(entry, ts, values, days, credit, settle, width=5.0, strike=610.0, expiry="2025-03-07"):
    return {"status": "ok", "entry_time": entry, "watch_ts": np.array(ts, dtype=np.int64),
            "watch": np.array(values, float), "watch_days": np.array(pd.to_datetime(days), dtype="datetime64[ns]"),
            "credit_traded": credit, "settle": settle, "width": width, "short_strike": strike,
            "expiry": pd.Timestamp(expiry), "expiry_close": 600.0}


def test_condor_value_sums_each_sides_latest_value_from_the_later_entry():
    days = ["2025-03-03"] * 4
    call = spread(10, [10, 20, 40], [1.0, 0.9, 0.7], days[:3], 1.0, 0.0)
    put = spread(15, [15, 30, 40], [1.2, 1.1, 1.0], days[:3], 1.2, 0.0)
    ts, value, _ = ic.condor_path(call, put)
    assert ts.tolist() == [15, 20, 30, 40]
    assert value.tolist() == pytest.approx([1.0 + 1.2, 0.9 + 1.2, 0.9 + 1.1, 0.7 + 1.0])


def condor(values, settle=0.0, expiry="2025-03-07"):
    sessions = features.trading_sessions("2025-03-03", "2025-03-07", warmup_sessions=0)
    days = sessions.index
    ts = [(sessions.loc[d, "open"] + pd.Timedelta(hours=2)).value for d in days[:len(values)]]
    call = spread(ts[0], ts, [v / 2 for v in values], days[:len(values)], 1.0, settle, expiry=expiry)
    put = spread(ts[0], ts, [v / 2 for v in values], days[:len(values)], 1.0, 0.0, strike=590.0, expiry=expiry)
    c = ic.make_condors({("d", 45, 0.3, 5, "C"): call, ("d", 45, 0.3, 5, "P"): put})[("d", 45, 0.3, 5)]
    c["day"] = days[0]
    return c, sessions


def test_exits_apply_to_the_condor_total():
    c, sessions = condor([2.0, 1.4, 0.9, 1.5])  # credit 2.0 in total
    pnl, reason, day = ic.simulate(c, {"tp": 50}, 0.0, sessions)
    assert reason == "take profit" and pnl == pytest.approx(100.0) and day == sessions.index[2]
    c, sessions = condor([2.0, 4.2, 6.5])  # 6.5 >= 3 x 2.0: a loss of 2x the credit, capped at the $5 width
    pnl, reason, _ = ic.simulate(c, {"tp": 50, "stop": 2.0}, 0.0, sessions)
    assert reason == "stop" and pnl == pytest.approx((2.0 - 5.0) * 100)
    c, sessions = condor([2.0, 1.8], settle=1.5)
    pnl, reason, _ = ic.simulate(c, {}, 0.01, sessions)
    assert reason == "expiry" and pnl == pytest.approx((2.0 - 0.04 - 1.5) * 100)  # four legs of entry slippage


def test_put_shorts_are_chosen_by_delta_through_parity():
    F, tau, vol = 600.0, 45 / 365.25, 0.18
    strikes = np.arange(500.0, 700.0, 5.0)
    expiry, day = pd.Timestamp("2025-04-17"), pd.Timestamp("2025-03-03")
    plan = pd.DataFrame({"expiry": [expiry], "day": [day], "forward": [F], "tau": [tau], "atm_vol": [vol]})

    def price(e, right, k, t):  # Black-76 at one flat volatility
        c = cs.b76_call(F, k, vol, tau)
        return c if right == "C" else c - F + k

    cs.choose_shorts(plan, {expiry: strikes}, "P", price, load=lambda *a: None, workers=1)  # nothing to download
    for d in cs.DELTAS:
        k = plan.at[0, f"put_short_{d:g}"]
        expect = strikes[np.argmin(np.abs((1 - cs.call_delta(F, strikes, vol, tau)) - d))]
        assert k == expect and k < F and plan.at[0, f"put_delta_{d:g}"] == pytest.approx(d, abs=0.03)
