"""Massive download: pagination, cache hits without network, no caching of truncated data."""

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
from massive import RESTClient

import download
import features
import run
import synthetic


class FakeMassive:
    """Stands in for massive.RESTClient: yields Agg-like objects and counts requests."""

    def __init__(self, minutes):
        self.calls = []
        self.aggs = [SimpleNamespace(timestamp=r.ts.value // 10**6, open=r.open, high=r.high, low=r.low,
                                     close=r.close, volume=r.volume, vwap=r.vwap, transactions=r.transactions)
                     for r in minutes.itertuples()]

    def list_aggs(self, ticker, multiplier, timespan, from_, to, **kwargs):
        self.calls.append((ticker, multiplier, timespan, from_, to, kwargs))
        return iter(self.aggs)


def test_repeated_cli_run_reads_cache_without_massive(tmp_path, monkeypatch):
    sessions = features.trading_sessions("2024-03-05", "2024-03-06", warmup_sessions=2)
    fake = FakeMassive(synthetic.random_walk_minutes(sessions, seed=1, drop_sessions=0))
    monkeypatch.setenv("MASSIVE_API_KEY", "placeholder-not-a-real-key")
    monkeypatch.setattr(download, "RESTClient", lambda **kwargs: fake)
    argv = ["--start", "2024-03-05", "--end", "2024-03-06", "--warmup-sessions", "2", "--no-daily-filter",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "out")]

    run.main(argv)
    assert fake.calls == [("SPY", 1, "minute", "2024-03-01", "2024-03-06",
                           {"adjusted": True, "sort": "asc", "limit": 50_000})]
    first = pd.read_parquet(tmp_path / "out" / "candidates.parquet")

    monkeypatch.delenv("MASSIVE_API_KEY")
    monkeypatch.setattr(download, "RESTClient", lambda **kwargs: pytest.fail("Massive used on a cache hit"))
    run.main(argv)
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(first, pd.read_parquet(tmp_path / "out" / "candidates.parquet"))

    download.load_minute_bars("SPY", "2024-03-01", "2024-03-06", cache_dir=tmp_path / "cache",
                              refresh=True, client=fake)
    assert len(fake.calls) == 2  # --refresh downloads again


@pytest.mark.parametrize("shape", ["final session missing", "stops mid final session",
                                   "only pre-market bars on the final date"])
def test_truncated_download_raises_and_is_not_cached(tmp_path, shape):
    sessions = features.trading_sessions("2024-03-05", "2024-03-06", warmup_sessions=0)
    full = synthetic.random_walk_minutes(sessions, seed=1, drop_sessions=0)
    final_open, final_close = sessions["open"].iloc[-1], sessions["close"].iloc[-1]
    before_final = full[full["ts"] < final_open]
    premarket = synthetic.bars_from_closes(
        pd.date_range(final_open - pd.Timedelta(hours=1), periods=10, freq="1min"), [100.0] * 10)
    minutes = {
        "final session missing": before_final,
        "stops mid final session": full[full["ts"] < final_open + pd.Timedelta(minutes=90)],
        "only pre-market bars on the final date": pd.concat([before_final, premarket], ignore_index=True),
    }[shape]
    with pytest.raises(RuntimeError, match="truncated"):
        download.load_minute_bars("SPY", "2024-03-05", "2024-03-06", cache_dir=tmp_path,
                                  final_close=final_close, client=FakeMassive(minutes))
    assert not any(tmp_path.iterdir())


PAGE_2_URL = "https://api.massive.com/v2/aggs/ticker/SPY/range/1/minute/2024-03-05/2024-03-05?cursor=p2"


def minute_page(t0_ms, minutes, price, next_url=None):
    page = {"results": [{"t": t0_ms + 60_000 * i, "o": price, "h": price, "l": price, "c": price, "v": 1}
                        for i in minutes]}
    return json.dumps(page | ({"next_url": next_url} if next_url else {})).encode()


def stub_client(pages):
    """The real massive.RESTClient with its HTTP layer stubbed to return `pages` in order."""
    requests = []

    class StubHTTP:
        def request(self, method, url, fields=None, headers=None):
            requests.append((urlparse(url), fields))
            return SimpleNamespace(status=200, data=pages[len(requests) - 1])

    client = RESTClient(api_key="placeholder-not-a-real-key")
    client.client = StubHTTP()  # the SDK's urllib3 pool; nothing leaves the machine
    return client, requests


def test_page_that_fails_to_decode_mid_session_is_not_cached(tmp_path):
    """The real client stops without an error when a page fails to decode. Here the data
    ends at 11:29 ET on the final date, which a date-only check would have accepted."""
    session = features.trading_sessions("2024-03-05", "2024-03-05", warmup_sessions=0).iloc[0]
    client, requests = stub_client([minute_page(session["open"].value // 10**6, range(120), 1, PAGE_2_URL),
                                    b"<html>502 Bad Gateway</html>"])
    with pytest.raises(RuntimeError, match="truncated"):
        download.load_minute_bars("SPY", "2024-03-05", "2024-03-05", cache_dir=tmp_path,
                                  final_close=session["close"], client=client)
    assert len(requests) == 2 and not any(tmp_path.iterdir())


def test_download_follows_every_page_through_the_real_client():
    """Drive the real massive.RESTClient with its HTTP layer stubbed: all pages must be kept."""
    t0 = pd.Timestamp("2024-03-05 14:30", tz="UTC").value // 10**6
    client, requests = stub_client([minute_page(t0, range(3), 1, PAGE_2_URL), minute_page(t0, range(3, 5), 2)])
    df = download.fetch_minute_bars("SPY", "2024-03-05", "2024-03-05", client=client)

    assert len(requests) == 2 and len(df) == 5
    assert df["close"].tolist() == [1, 1, 1, 2, 2]
    first, params = requests[0]
    assert first.path == "/v2/aggs/ticker/SPY/range/1/minute/2024-03-05/2024-03-05"
    assert params == {"adjusted": "true", "sort": "asc", "limit": 50_000}
    assert parse_qs(requests[1][0].query) == {"cursor": ["p2"]}
    assert df["ts"].iloc[0] == pd.Timestamp("2024-03-05 14:30", tz="UTC")  # Massive `t` = bar start, UTC
