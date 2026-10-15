"""close_momentum_stocks.py on SYNTHETIC day tables: big days are picked in sd20 units within a group, trades go
with the move, and costs come off both sides."""

import numpy as np
import pandas as pd

import close_momentum_stocks as cms


def test_big_days_in_volatility_units_within_a_group():
    days = pd.bdate_range("2024-01-02", periods=5)
    d = pd.DataFrame({"move_m30": [300.0, -500.0, 100.0, 450.0, -50.0], "last30": [10.0, -20.0, 5.0, -4.0, 3.0],
                      "sd20": [250.0, 250.0, 250.0, 400.0, 250.0], "group": ["core", "core", "core", "related", "core"],
                      "ticker": "X", "p_m30": 50.0, "ratio": 1.0}, index=days)
    t = cms.pick(d, "core")
    assert list(t["gross"]) == [10.0, 20.0]                       # 1.2 and 2.0 sd; shorts gain on a further drop
    assert list(cms.pick(d, "core", lo=2.0)["gross"]) == [20.0]
    assert list(cms.pick(d, "core", lo=0.0, hi=0.5)["gross"]) == [5.0, -3.0]   # 0.4 and 0.2 sd
    assert list(cms.pick(d, "related")["gross"]) == [-4.0]
    s = cms.stats(pd.concat([t] * 2, ignore_index=True).assign(session=pd.bdate_range("2024-02-01", periods=4)))
    assert np.isclose(s["mean"], 15.0 - 2 * (1 + 0.005 / 50.0 * 1e4))   # 1 bp + half a cent at $50, each side
