"""rebalancing.py on SYNTHETIC prices: the 60/40 drift signal uses only snapshots up to its session (no lookahead)
and resets each month, the month-end trade's timing and direction, and the reversal."""

import numpy as np
import pandas as pd

import rebalancing as rb


def prices(n=90, seed=0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    stock = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    bond = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=idx)
    return stock, bond


def test_signal_uses_only_snapshots_up_to_its_session_and_measures_drift_since_the_last_month_end():
    stock, bond = prices()
    sig = rb.calendar_signal(stock, bond)
    cut = 50
    later = stock.copy()
    later.iloc[cut:] *= 3.0
    assert np.allclose(rb.calendar_signal(later, bond).iloc[:cut], sig.iloc[:cut], equal_nan=True)
    feb = stock.index[stock.index.to_period("M") == pd.Period("2023-02", "M")]
    jan_last = stock.index[stock.index < feb[0]][-1]
    t = feb[10]
    gs, gb = stock[t] / stock[jan_last], bond[t] / bond[jan_last]
    assert np.isclose(sig[t], 0.6 * gs / (0.6 * gs + 0.4 * gb) - 0.6)
    assert sig[stock.index.to_period("M") == pd.Period("2023-01", "M")].isna().all()   # no earlier month end


def test_month_end_trade_enters_four_sessions_before_the_last_and_trades_against_the_drift():
    stock, bond = prices()
    sig = rb.calendar_signal(stock, bond)
    tr = rb.window_trades(stock, sig)
    feb = stock.index[stock.index.to_period("M") == pd.Period("2023-02", "M")]
    row = tr[tr["month"] == pd.Period("2023-02", "M")].iloc[0]
    assert row["entry"] == feb[-5] and row["exit"] == feb[-1]
    assert row["position"] == -np.sign(sig[feb[-5]])
    expect = row["position"] * (stock[feb[-1]] / stock[feb[-5]] - 1) * 1e4
    assert np.isclose(row["gross"], expect) and np.isclose(row["net"], expect - 2 * rb.COST_BPS)


def test_reversal_takes_the_opposite_position_over_the_first_two_sessions_after_the_month():
    stock, bond = prices()
    sig = rb.calendar_signal(stock, bond)
    me = rb.window_trades(stock, sig)
    rev = rb.reversal_trades(stock, me)
    r = rev[rev["month"] == pd.Period("2023-02", "M")].iloc[0]
    m = me[me["month"] == pd.Period("2023-02", "M")].iloc[0]
    i = stock.index.get_loc(m["exit"])
    assert r["entry"] == stock.index[i] and r["exit"] == stock.index[i + 2] and r["position"] == -m["position"]
