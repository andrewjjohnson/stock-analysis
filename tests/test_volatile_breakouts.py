"""volatile_breakouts.py and portfolio_check.py on SYNTHETIC data: the monthly universe uses only sessions before the
month (no lookahead) and applies its filters, download spans, costs, the bodies-only seconds, the fake-level
excess and one position at a time."""

import numpy as np
import pandas as pd

import portfolio_check as pc
import volatile_breakouts as vb


def panels(n_days=60, tickers=("AAA", "BBB", "CCC", "TSLA", "LOW"), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    vol = {"AAA": 0.05, "BBB": 0.03, "CCC": 0.01, "TSLA": 0.08, "LOW": 0.06}
    close = pd.DataFrame({t: 50 * np.exp(np.cumsum(rng.normal(0, vol[t], n_days))) for t in tickers}, index=days)
    high = close * (1 + pd.DataFrame({t: np.full(n_days, vol[t]) for t in tickers}, index=days))
    low = close * (1 - pd.DataFrame({t: np.full(n_days, vol[t]) for t in tickers}, index=days))
    dv = pd.DataFrame(1e9, index=days, columns=list(tickers))
    raw = close.copy()
    raw["LOW"] = 5.0                                    # trades below $10: never eligible
    return {"close": close, "high": high, "low": low, "dollar_volume": dv, "raw_close": raw}


def test_universe_uses_only_sessions_before_the_month_and_applies_the_filters():
    p = panels()
    month = pd.Period("2024-03", "M")
    u = vb.pick_universe(p, [month], lookback=21, n=3, core=2)
    assert list(u["ticker"]) == ["AAA", "BBB", "CCC"]          # by volatility; TSLA excluded, LOW under $10
    assert list(u["group"]) == ["core", "core", "related"]
    first = p["close"].index[p["close"].index.searchsorted(month.start_time)]
    changed = {k: v.copy() for k, v in p.items()}
    for k in ("close", "high", "low"):
        changed[k].loc[first:, "CCC"] *= np.linspace(1, 50, (changed[k].index >= first).sum())
    changed["dollar_volume"].loc[first:, "AAA"] = 0.0
    assert vb.pick_universe(changed, [month], lookback=21, n=3, core=2).equals(u)
    before = p["close"].index[p["close"].index < first][-5]
    gone = {k: v.copy() for k, v in p.items()}
    gone["close"].loc[before, "AAA"] = np.nan                  # a missing session inside the lookback
    assert "AAA" not in set(vb.pick_universe(gone, [month], lookback=21, n=3, core=2)["ticker"])


def test_download_spans_add_warmup_and_split_far_apart_months():
    sessions = pd.DataFrame(index=pd.bdate_range("2023-01-02", "2023-12-29"))
    u = pd.DataFrame({"month": [pd.Period(m, "M") for m in ("2023-03", "2023-04", "2023-10")],
                      "ticker": "AAA", "rank": 1, "tr_pct": 5.0, "group": "core"})
    s = vb.download_spans(u, sessions, warmup=5, gap=3)
    assert len(s) == 2
    assert s.iloc[0]["months"] == (pd.Period("2023-03", "M"), pd.Period("2023-04", "M"))
    first_march = sessions.index[sessions.index >= "2023-03-01"][0]
    assert s.iloc[0]["first"] == sessions.index[sessions.index.get_loc(first_march) - 5]
    assert s.iloc[0]["last"] == pd.Timestamp("2023-04-28")


def test_costs_add_half_a_cent_a_share_at_the_traded_price():
    assert np.allclose(vb.side_cost_bps([10.0, 100.0]), [1 + 5.0, 1 + 0.5])


def test_bodies_only_seconds_drop_the_wicks():
    arr = np.array([[0, 10.0, 10.9, 9.1, 10.2]])

    def load(ticker, start, end):
        return arr

    out = vb.bodies(load)("X", 0, 999)
    assert out.tolist() == [[0, 10.0, 10.2, 10.0, 10.2]] and arr[0, 2] == 10.9


def test_excess_over_fakes_is_zero_for_identical_trades_and_positive_when_real_is_better():
    days = pd.bdate_range("2024-01-02", periods=40).repeat(2)
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"session": days, "net_primary": rng.normal(0, 10, len(days))})
    d, lo, hi, p = vb.excess(real, real.copy())
    assert abs(d) < 1e-12 and lo <= 0 <= hi
    d, lo, hi, p = vb.excess(real.assign(net_primary=real["net_primary"] + 20), real)
    assert d > 0 and lo > 0 and p < 0.01


def test_one_at_a_time_skips_trades_that_start_while_another_is_open():
    t = pd.DataFrame({"start": [0, 5, 10, 12], "end": [8, 9, 11, 20]})
    assert list(pc.one_at_a_time(t).index) == [0, 2, 3]
