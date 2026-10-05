"""goose_indicators.py on SYNTHETIC data: the slid-copy edge test, the screen's roles, the vote rule, the conditions,
the option pick, and indicators that later prices cannot change."""

import numpy as np
import pandas as pd
import pytest

import features
import goose_indicators as gi
import meanrev as mr


def test_edge_test_finds_a_real_edge_and_not_a_random_one():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 30, 500)
    edge, p = gi.edge_test(r, r > 20)  # sessions picked by their own outcome
    assert edge > 20 and p < 0.01
    edge, p = gi.edge_test(r, rng.random(500) < 0.3)
    assert abs(edge) < 6 and p > 0.05
    assert np.isnan(gi.edge_test(r, np.zeros(500, bool))[1])


def test_slides_skip_shifts_that_line_a_pattern_up_with_itself():
    weekly = np.arange(200) % 5 == 4  # every fifth session
    s = gi.slides([weekly])
    assert len(s) and not np.any(s % 5 == 0) and s.min() >= gi.MIN_SHIFT


def design_set(seed, flip_d=False):
    rng = np.random.default_rng(seed)
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    r = rng.normal(10, 20, n)
    c = pd.DataFrame(False, index=idx, columns=["A", "B", "C", "D", "E"])
    pos = np.random.default_rng(99).permutation(n)  # the same random sessions on every ticker
    for k, name in enumerate("ABCD"):
        c.iloc[pos[k * 50:(k + 1) * 50], k] = True  # A call: +60 bps; B put; C warning; D the other way on QQQ
    c.iloc[pos[200:205], 4] = True                  # E: strong but only 5 sessions
    r[c["A"]] += 60
    r[c["B"]] -= 60
    r[c["C"]] = rng.normal(0.5, 1, c["C"].sum()) + 0.5
    r[c["D"]] += -60 if flip_d else 60
    r[c["E"]] += 200
    c["usable"] = True
    return pd.Series(r, index=idx), c


def test_screen_gives_call_put_and_warning_roles_and_needs_confirmation_and_enough_sessions():
    design = {"SPY": design_set(1), "QQQ": design_set(2, flip_d=True), "IWM": design_set(3)}
    s = gi.screen(design).set_index("condition")
    assert s.loc["A", "role"] == "call" and s.loc["A", "p"] < 0.05
    assert s.loc["B", "role"] == "put" and s.loc["B", "mean"] < 0
    assert s.loc["C", "role"] == "warning" and s.loc["C", "edge"] < 0 <= s.loc["C", "mean"]
    assert s.loc["D", "role"] == "not used" and s.loc["D", "edge_QQQ"] < 0 < s.loc["D", "edge"]
    assert s.loc["E", "role"] == "not used" and s.loc["E", "sessions"] == 5


def test_votes_decide_the_direction():
    roles = {"c1": "call", "c2": "call", "p1": "put", "w1": "warning"}
    rows = [  # c1, c2, p1, w1 -> direction, margin
        ((1, 0, 0, 0), 1, 1), ((1, 0, 0, 1), 0, 0), ((0, 0, 1, 0), -1, 1), ((1, 0, 1, 0), 0, 0),
        ((1, 1, 1, 0), 1, 1), ((1, 1, 0, 0), 1, 2), ((0, 0, 1, 1), -1, 1), ((0, 0, 0, 1), 0, 0), ((1, 1, 0, 1), 1, 1),
    ]
    cond = pd.DataFrame([r[0] for r in rows], columns=list(roles), dtype=bool)
    out = gi.rule_direction(cond, roles)
    assert out["direction"].tolist() == [r[1] for r in rows]
    assert out["margin"].tolist() == [r[2] for r in rows]


def test_conditions_from_the_inputs():
    base = {k: 1.0 for k in gi.INPUTS}
    base.update(rsi2=50, adx=20, pdi=30, mdi=25, rsi2_y=50, adx_y=20, pdi_y=30, mdi_y=25, ibs=0.5, bb_z20=0,
                streak=0, ret_1d=0.2, mfi=50, macd_hist=0.1, last_half=0.05, gap=-0.1, vixy_1d=0, vol_ratio=1,
                candle=0.3, tom_next=0, days_to_next=1)
    zone = {**base, "adx": 27, "pdi": 25, "mdi": 30}                 # ADX moves between the lines, -DI on top
    through = {**base, "rsi2": 20}                                    # from above both lines to below both
    low = {**base, "ibs": 0.1, "bb_z20": -2.5, "streak": -3, "ret_1d": -1.2, "mfi": 15, "vixy_1d": 6,
           "tom_next": 1, "days_to_next": 3}
    missing = {**base, "mfi": np.nan}
    c = gi.conditions(pd.DataFrame([base, zone, through, low, missing]))
    assert c.loc[1, "ADX(5) entered the DI zone, -DI on top"] and not c.loc[1, "ADX(5) entered the DI zone, +DI on top"]
    assert not c.loc[0, "ADX(5) entered the DI zone, -DI on top"]
    assert c.loc[2, "RSI(2) stabbed the zone from above"] and not c.loc[0, "RSI(2) stabbed the zone from above"]
    for name in ("IBS below 0.2", "Below the lower Bollinger band", "3+ down closes in a row", "Down 1%+ on the day",
                 "MFI(14) below 20", "VIXY up 5%+ on the day", "Turn of the month next session",
                 "Weekend or holiday before the next session"):
        assert c.loc[3, name] and not c.loc[0, name], name
    assert c.loc[0, "Candle up"] and c.loc[0, "Opened below yesterday's close"] and c.loc[0, "Above the 50-day average"]
    assert c.loc[0, "usable"] and not c.loc[4, "usable"] and not c.loc[4, "MFI(14) below 20"]


def test_option_pnl_takes_the_contract_matching_the_direction():
    days = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"])
    t = pd.DataFrame({"session": np.repeat(days, 2), "right": ["C", "P"] * 4,
                      "status": ["ok", "ok", "ok", "ok", "ok", "no strike in the delta band", "ok", "ok"],
                      "paid": [2.0, 1.5, 3.0, 2.5, 1.0, np.nan, 2.0, 2.0],
                      "exit_v1": [2.5, 1.0, 2.0, 3.5, 0.5, np.nan, 2.2, 1.0]})
    d = pd.Series([1, -1, -1, 0], index=days)
    pnl = gi.option_pnl(d, t, slip=0.02)
    assert pnl.index.tolist() == days[:2].tolist()  # 01-08 wanted the put, which had no contract; 01-09 no trade
    assert pnl.tolist() == pytest.approx([100 * (2.5 - 2.0 - 0.04), 100 * (3.5 - 2.5 - 0.04)])


def test_stock_check_rewards_foresight_against_slid_copies():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2024-10-01", periods=300)
    r = pd.Series(rng.normal(3, 40, 300), index=idx)
    st = gi.stock_check(pd.Series(np.sign(r).astype(int), index=idx), r)
    assert st["hit"] == 100 and st["p_up"] < 0.01 and st["slid"] < st["mean"]


def make_minutes(sessions, seed=0):
    stamps = [pd.date_range(o, c - pd.Timedelta(minutes=1), freq="min") for o, c in zip(sessions["open"],
                                                                                      sessions["close"])]
    ts = stamps[0].append(stamps[1:])
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.0008, len(ts))))
    return pd.DataFrame({"ts": ts, "open": close * 0.9999, "high": close * 1.0004, "low": close * 0.9996,
                         "close": close, "volume": 1000.0 + np.arange(len(ts)) % 7})


def test_later_prices_cannot_change_earlier_conditions():
    sessions = features.trading_sessions("2024-01-02", "2024-05-31", warmup_sessions=0)
    m, vx = make_minutes(sessions, 1), make_minutes(sessions, 2)
    no_divs = pd.DataFrame({"ex_date": pd.to_datetime([]), "cash_amount": []})
    cut = sessions.index[80]
    after = m["ts"] >= sessions.loc[cut, "close"] - pd.Timedelta(minutes=10)  # from 15:50 on the cut day

    def run(minutes):
        vx_daily, _ = mr.daily_table(vx, sessions)
        vixy = mr.ticker_features(vx_daily, full=False)["ret_1d"]
        cond, r, goose, _ = gi.ticker_tables(minutes, no_divs, sessions, vixy)
        return cond, r, goose

    c1, r1, g1 = run(m)
    m2 = m.copy()
    m2.loc[after, ["open", "high", "low", "close"]] *= np.linspace(0.8, 1.3, after.sum())[:, None]
    c2, r2, g2 = run(m2)
    early = c1.index <= cut
    assert c1["usable"][early].sum() > 20
    pd.testing.assert_frame_equal(c1[early], c2[early])
    pd.testing.assert_series_equal(g1[early], g2[early])
    assert not r1["open"].loc[:cut].iloc[:-1].ne(r2["open"].loc[:cut].iloc[:-1]).any()  # only the cut day's move changes
    assert r1["open"].loc[cut] != r2["open"].loc[cut]
