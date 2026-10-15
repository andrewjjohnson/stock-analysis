"""trident_fx.py on SYNTHETIC data: forex trading days, the yearly cache and its truncation guard, gold's
close-based stop, the spread lookup, and a design run that cannot see 2024."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trident as td
import trident_fx as fx


def et(s):
    return pd.Timestamp(s, tz=fx.NY).tz_convert("UTC")


def test_minutes_belong_to_the_trading_day_ending_at_17_et():
    sessions = fx.fx_sessions("2024-07-01", "2024-07-12")
    stamps = [et("2024-07-07 17:30"),   # Sunday evening -> Monday
              et("2024-07-10 16:50"),   # Wednesday afternoon -> Wednesday
              et("2024-07-10 17:15"),   # Wednesday evening -> Thursday
              et("2024-07-12 16:50"),   # Friday -> Friday
              et("2024-07-12 17:15"),   # Friday evening -> Saturday: no trading day
              et("2024-07-13 12:00")]   # Saturday
    m = pd.DataFrame({"ts": stamps, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    got = fx.fx_minutes(m, sessions)
    assert got["session"].dt.strftime("%Y-%m-%d").tolist() == ["2024-07-08", "2024-07-10", "2024-07-11",
                                                               "2024-07-12"]
    assert (sessions.loc["2024-07-08", "open"], sessions.loc["2024-07-08", "close"]) == (
        et("2024-07-07 17:00"), et("2024-07-08 17:00"))
    roll = pd.DataFrame({"ts": [et("2024-07-10 16:54"), et("2024-07-10 16:55"), et("2024-07-10 17:09"),
                                et("2024-07-10 17:10")], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 1.0})
    assert fx.fx_minutes(roll, sessions)["ts"].tolist() == [et("2024-07-10 16:54"), et("2024-07-10 17:10")]
    winter = fx.fx_sessions("2024-01-08", "2024-01-08")
    assert winter["close"].iloc[0] == pd.Timestamp("2024-01-08 22:00", tz="UTC")  # 17:00 EST


class FakeClient:
    """list_aggs returns the bars inside the requested dates."""
    def __init__(self, bars):
        self.bars, self.calls = bars, 0

    def list_aggs(self, ticker, mult, span, start, end, **kw):
        self.calls += 1
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        return iter([b for b in self.bars if lo.value // 1_000_000 <= b.timestamp < hi.value // 1_000_000])


def agg(ts):
    ms = int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)
    return SimpleNamespace(timestamp=ms, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, vwap=1.0,
                           transactions=1)


def test_years_are_cached_and_a_short_download_is_refused(tmp_path, monkeypatch):
    full = FakeClient([agg("2015-01-02 10:00"), agg("2015-12-31 20:00")])
    df = fx.load_fx("EURUSD", "2015-01-01", "2015-12-31", tmp_path, client=full)
    assert len(df) == 2 and full.calls == 1
    monkeypatch.setattr(fx, "make_client", lambda: pytest.fail("the cache should be read without a client"))
    assert len(fx.load_fx("EURUSD", "2015-01-01", "2015-12-31", tmp_path)) == 2
    path = fx.chunk_path(tmp_path, "EURUSD", pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31"))

    class Truncated(FakeClient):  # the SDK stops paging, though Massive has July bars
        def list_aggs(self, ticker, mult, span, start, end, **kw):
            got = list(super().list_aggs(ticker, mult, span, start, end, **kw))
            return iter(got if start != "2016-01-01" else [b for b in got if b.timestamp < self.bars[2].timestamp])
    with pytest.raises(RuntimeError):
        fx.load_fx("EURUSD", "2016-01-01", "2016-12-31", tmp_path,
                   client=Truncated([agg("2016-01-04 10:00"), agg("2016-06-30 20:00"), agg("2016-07-01 09:00"),
                                     agg("2016-12-30 20:00")]))
    assert not path.exists()
    gap = FakeClient([agg("2016-01-04 10:00"), agg("2016-06-30 20:00")])  # Massive really has nothing later
    assert len(fx.load_fx("EURUSD", "2016-01-01", "2016-12-31", tmp_path, client=gap)) == 2 and path.exists()


def test_gold_stops_only_on_a_30_minute_close_beyond_the_stop():
    sessions = fx.fx_sessions("2024-03-05", "2024-03-06")
    ts = pd.date_range(et("2024-03-05 03:00"), periods=90, freq="min")
    close = np.full(90, 100.0)
    low, high = close - 0.1, close + 0.1
    low[10] = 98.5        # a wick through the stop inside the first bar
    close[29] = 99.6      # first bar closes above the stop
    close[59] = 98.8      # second bar closes below it
    rth = fx.fx_minutes(pd.DataFrame({"ts": ts, "open": close, "high": np.maximum(high, close),
                                      "low": np.minimum(low, close), "close": close, "volume": 1.0}), sessions)
    daily = pd.DataFrame({"long_now": 1.0, "short_now": 0.0}, index=sessions.index)
    bars = pd.DataFrame({"bar_end": pd.Series([et("2024-03-05 03:30"), et("2024-03-05 04:00"),
                                               et("2024-03-05 04:30")]), "close": [99.6, 98.8, 100.0]})
    e = pd.DataFrame({"side": [1], "session": [pd.Timestamp("2024-03-05")], "ordinal": [0], "price": [100.0],
                      "stop": [99.0], "atr": [1.0], "first": [0], "stop_first": [0], "entry_ns": [0], "tod": ["03:00"]})
    touch = td.outcomes(e, td.minute_arrays(rth, sessions, daily)).iloc[0]
    assert touch.stopped and touch.bracket == -1.0
    gold = td.outcomes(e, td.minute_arrays(rth, sessions, daily, bars)).iloc[0]
    assert gold.stopped and abs(gold.bracket + 1.2) < 1e-9 and abs(gold.mfe0 - 0.1) < 1e-9


def test_spreads_are_median_quotes_cached_by_entry(tmp_path):
    calls = []

    class Quotes:
        def list_quotes(self, ticker, **kw):
            calls.append((ticker, kw["timestamp_gte"], kw["timestamp_lt"]))
            return [SimpleNamespace(ask_price=1.1002, bid_price=1.1000), SimpleNamespace(ask_price=1.1001,
                    bid_price=1.1002), SimpleNamespace(ask_price=1.1004, bid_price=1.1000)]

    spread_of = fx.spread_loader(tmp_path, client=Quotes())
    got = spread_of("EURUSD", np.array([120_000_000_000, 120_000_000_000]))
    assert np.allclose(got, 0.0002) and calls == [("C:EURUSD", 60_000_000_000, 120_000_000_000)]
    spread_of.save()
    again = fx.spread_loader(tmp_path, client=SimpleNamespace())  # no list_quotes: must come from the cache
    assert np.allclose(again("EURUSD", np.array([120_000_000_000])), 0.0002)


def synthetic_fx(sessions, seed):
    ts = pd.date_range(sessions["open"].iloc[0], sessions["close"].iloc[-1], freq="min", inclusive="left")
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.0004, len(ts))))
    wig = np.abs(rng.normal(0, 0.0003, len(ts)))
    return pd.DataFrame({"ts": ts, "open": close * (1 - wig / 2), "high": close * (1 + wig),
                         "low": close * (1 - wig), "close": close, "volume": 1.0})


def test_design_run_ignores_2024():
    sessions = fx.fx_sessions("2023-10-02", "2024-02-28")
    data = {"EURUSD": synthetic_fx(sessions, 1), "XAUUSD": synthetic_fx(sessions, 2)}
    kw = dict(draws=3, prepare=fx.prepare, ends=fx.FVG_ENDS, last_entry=fx.LAST_ENTRY, groups=fx.GROUPS,
              close_stop=fx.CLOSE_STOP)
    res = td.run_study(data, sessions, **kw)
    later = {}
    for k, v in data.items():
        v = v.copy()
        late = v["ts"] >= pd.Timestamp("2023-12-29 22:00", tz="UTC")
        v.loc[late, ["open", "high", "low", "close"]] *= 1.3
        later[k] = v
    res2 = td.run_study({k: (lambda v=v: v) for k, v in later.items()}, sessions, **kw)
    t = res["trades"]
    assert len(t) and (t["session"] <= fx.DESIGN_END).all()
    assert t["tod"].between("03:00", "06:30").all()
    pd.testing.assert_frame_equal(t, res2["trades"])


def test_the_green_daily_candle_is_the_previous_days_direction():
    sessions = fx.fx_sessions("2024-03-04", "2024-03-07")
    frames = []
    for day, (a, b) in zip(sessions.index, [(1.10, 1.11), (1.11, 1.10), (1.10, 1.10), (1.10, 1.12)]):
        ts = pd.date_range(sessions.loc[day, "open"], sessions.loc[day, "close"], freq="min", inclusive="left")
        px = np.linspace(a, b, len(ts))
        frames.append(pd.DataFrame({"ts": ts, "open": px, "high": px, "low": px, "close": px, "volume": 1.0}))
    rth = fx.fx_minutes(pd.concat(frames, ignore_index=True), sessions)
    d = td.daily_states(rth, sessions)
    assert d["green_long"].tolist() == [False, True, False, False]   # Tuesday sees Monday's up candle
    assert d["green_short"].tolist() == [False, False, True, False]  # Wednesday sees Tuesday's down candle
