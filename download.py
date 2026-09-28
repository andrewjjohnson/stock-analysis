"""Download one-minute bars from Massive and cache them locally as Parquet.

The cache file name encodes ticker, date range and adjustment setting, so an
identical request is read from disk without any network access. Pass
refresh=True (CLI: --refresh) to download again.
"""

import os
from pathlib import Path

import pandas as pd
from massive import RESTClient

# Massive `adjusted=true`: prices are split-adjusted. They are NOT dividend-adjusted,
# so returns computed from them are price returns, not total returns.
ADJUSTED = True

COLUMNS = ["ts", "open", "high", "low", "close", "volume", "vwap", "transactions"]


def cache_path(cache_dir, ticker, start, end, adjusted=ADJUSTED):
    tag = "splitadj" if adjusted else "unadjusted"
    return Path(cache_dir) / f"{ticker.upper()}_1min_{start}_{end}_{tag}.parquet"


def fetch_minute_bars(ticker, start, end, adjusted=ADJUSTED, client=None):
    """Every one-minute bar Massive has for [start, end] (YYYY-MM-DD, inclusive).

    list_aggs follows the API's next_url pages until the range is exhausted;
    `limit` is only the page size. `ts` is the bar START (Massive's `t`), in UTC.
    """
    if client is None:
        # Read the key now: RESTClient's own default is captured at import time.
        key = os.environ.get("MASSIVE_API_KEY")
        if not key:
            raise SystemExit("MASSIVE_API_KEY is not set. Export it (see README) or run demo.py for the offline demo.")
        # retries=10: the SDK's exponential backoff then waits ~100 s in total, enough to
        # ride out per-minute rate limits (HTTP 429) during a multi-page download.
        client = RESTClient(api_key=key, retries=10)
    aggs = client.list_aggs(ticker, 1, "minute", start, end, adjusted=adjusted, sort="asc", limit=50_000)
    rows = [(a.timestamp, a.open, a.high, a.low, a.close, a.volume, a.vwap, a.transactions) for a in aggs]
    df = pd.DataFrame(rows, columns=["t"] + COLUMNS[1:])
    df.insert(0, "ts", pd.to_datetime(df.pop("t"), unit="ms", utc=True).dt.as_unit("ns"))
    return df


def load_minute_bars(ticker, start, end, *, cache_dir="data/cache", refresh=False,
                     adjusted=ADJUSTED, expect_through=None, client=None):
    """Return (minute_bars, source) for the request, using the Parquet cache when present.

    expect_through: last session (date) that must appear in a fresh download. If the
    data stops earlier, the download is treated as truncated: nothing is cached and
    an error is raised rather than silently studying a shorter interval.
    """
    path = cache_path(cache_dir, ticker, start, end, adjusted)
    if path.exists() and not refresh:
        return pd.read_parquet(path), f"cache {path}"

    df = fetch_minute_bars(ticker, start, end, adjusted, client)
    if df.empty:
        raise RuntimeError(f"Massive returned no minute bars for {ticker} {start}..{end}.")
    last_day = df["ts"].max().tz_convert("America/New_York").date()
    if expect_through is not None and last_day < pd.Timestamp(expect_through).date():
        raise RuntimeError(
            f"Massive data for {ticker} stops on {last_day}, but sessions through "
            f"{pd.Timestamp(expect_through).date()} were requested. Not caching a possibly "
            "truncated download; retry, or choose an earlier --end if no data exists there."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df, f"Massive API ({len(df):,} bars; cached to {path})"
