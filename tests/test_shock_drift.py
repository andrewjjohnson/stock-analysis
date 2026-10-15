"""shock_drift.py on SYNTHETIC daily panels: z-scores and volume ratios use only sessions up to t (no lookahead),
events are spaced per stock, and trades enter at the next open and hedge with SPY to the close of t+h."""

import numpy as np
import pandas as pd

import shock_drift as sd


def panels(n=120, seed=0):
    days = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    cols = ["AAA", "BBB", "SPY"]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)), index=days, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, (n, 3)))
    vol = pd.DataFrame(rng.uniform(1e6, 2e6, (n, 3)), index=days, columns=cols)
    return {"open": open_, "close": close, "volume": vol, "dollar_volume": vol * close, "raw_close": close.copy()}


def test_event_measures_use_only_sessions_up_to_t():
    p = panels()
    uni = {pd.Period("2023-05", "M"): ["AAA", "BBB"]}
    t = sd.event_table(p, uni)
    cut = p["close"].index[p["close"].index.to_period("M") == pd.Period("2023-05", "M")][10]
    q = {k: v.copy() for k, v in p.items()}
    q["close"].loc[q["close"].index > cut, "AAA"] *= 5
    q["volume"].loc[q["volume"].index > cut, "AAA"] *= 50
    t2 = sd.event_table(q, uni)
    a = t[t["session"] <= cut].set_index(["session", "ticker"])
    b = t2[t2["session"] <= cut].set_index(["session", "ticker"])
    assert np.allclose(a[["z", "ar", "vr"]].to_numpy(), b[["z", "ar", "vr"]].to_numpy(), equal_nan=True)


def test_events_are_spaced_per_stock():
    days = pd.bdate_range("2023-01-02", periods=20)
    ev = pd.DataFrame({"session": days[[1, 3, 6, 7, 2]], "ticker": ["A", "A", "A", "A", "B"]})
    kept = sd.space_events(ev, days, gap=5)
    assert sorted(zip(kept["ticker"], kept["session"])) == [("A", days[1]), ("A", days[6]), ("B", days[2])]


def test_trade_enters_at_the_next_open_and_hedges_with_spy_to_the_close_of_t_plus_h():
    p = panels()
    days = p["close"].index
    ev = pd.DataFrame({"session": [days[30]], "ticker": ["AAA"], "ar": [-0.05]})
    tr = sd.trade_returns(ev, p, horizons=(5,)).iloc[0]
    stock = p["close"].at[days[35], "AAA"] / p["open"].at[days[31], "AAA"] - 1
    spy = p["close"].at[days[35], "SPY"] / p["open"].at[days[31], "SPY"] - 1
    assert tr["direction"] == -1 and np.isclose(tr["gross_5"], -(stock - spy) * 1e4)
    assert np.isclose(tr["net_5"], tr["gross_5"] - tr["cost_rt"])


def test_the_spy_trend_filter_uses_only_closes_up_to_the_event_day():
    p = panels(n=260)
    days = p["close"].index
    ev = pd.DataFrame({"session": [days[230]], "ticker": ["AAA"], "ar": [0.05]})
    a = sd.trade_returns(ev, p, horizons=(5,)).iloc[0]["spy_vs_sma"]
    q = {k: v.copy() for k, v in p.items()}
    q["close"].loc[days[231]:, "SPY"] *= 3
    b = sd.trade_returns(ev, q, horizons=(5,)).iloc[0]["spy_vs_sma"]
    spy = p["close"]["SPY"]
    assert np.isclose(a, b) and np.isclose(a, spy.iloc[230] / spy.iloc[31:231].mean())


def test_final_tests_apply_the_filter_only_to_the_filtered_test():
    days = pd.bdate_range("2025-01-02", periods=60)
    t = pd.DataFrame({"session": days, "volume": "heavy", "net_5": 10.0, "net_20": np.r_[[50.0] * 30, [-50.0] * 30],
                      "spy_vs_sma": np.r_[[1.1] * 30, [0.9] * 30], "cost_rt": 5.0})
    f = sd.final_tests(t)
    assert list(f["trades"]) == [60, 60, 30]
    assert np.isclose(f.loc[1, "mean"], 0.0) and np.isclose(f.loc[2, "mean"], 50.0)
