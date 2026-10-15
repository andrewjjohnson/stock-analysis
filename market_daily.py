"""Whole-market daily bars and the list of common stocks, from Massive, cached locally.

  uv run --env-file .env python market_daily.py            # fills the cache (about 10 minutes the first time)

Used to choose stock universes by rule (for example "the most volatile liquid stocks last month") without
naming the stocks by hand, which would bring in hindsight. Delisted stocks are included, so a universe chosen
from these bars has no survivorship bias.

- `load_grouped(sessions, adjusted)`: one row per ticker and session from Massive's grouped daily endpoint,
  regular and extended-hours volume included as Massive reports it. Each session is cached as its own small
  Parquet file under data/cache/grouped/<adjusted|unadjusted>/, so later sessions add files without touching
  earlier ones. Split-adjusted bars give comparable returns over time; unadjusted bars give the price that
  actually traded that day (for price filters and per-share costs).
- `stock_types()`: ticker -> Massive type ("CS" common stock, "ADRC" ADR) for active and delisted stocks,
  cached as data/cache/reference/stock_types.parquet.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from massive.exceptions import BadResponse

import features
import stock_dip_spreads as sds

COLUMNS = ["ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]
TYPES = ("CS", "ADRC")


def _folder(cache_dir, adjusted):
    return Path(cache_dir) / "grouped" / ("splitadj" if adjusted else "unadjusted")


def _fetch(client, day, adjusted):
    try:
        rows = client.get_grouped_daily_aggs(day.strftime("%Y-%m-%d"), adjusted=adjusted)
    except BadResponse as e:  # before the plan's rolling history window
        if "NOT_AUTHORIZED" not in str(e):
            raise
        rows = []
    return pd.DataFrame([(a.ticker, a.open, a.high, a.low, a.close, a.volume, a.vwap, a.transactions)
                         for a in rows], columns=COLUMNS)


def load_grouped(sessions, adjusted=True, cache_dir="data/cache", refresh=False, workers=8):
    """Daily bars of every US stock and ETF for the given sessions (a DatetimeIndex of session dates), stacked
    with a `session` column. A session Massive returns nothing for (outside the plan's history) is not cached and
    is left out."""
    folder = _folder(cache_dir, adjusted)
    days = [pd.Timestamp(d) for d in sessions]
    todo = [d for d in days if refresh or not (folder / f"{d:%Y-%m-%d}.parquet").exists()]
    if todo:
        folder.mkdir(parents=True, exist_ok=True)
        client = sds.make_client()

        def get(day):
            df = _fetch(client, day, adjusted)
            if len(df):
                df.to_parquet(folder / f"{day:%Y-%m-%d}.parquet", index=False)
            return len(df)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(get, todo))
    frames = []
    for d in days:
        path = folder / f"{d:%Y-%m-%d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path).assign(session=d))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + ["session"])


def stock_types(cache_dir="data/cache", refresh=False):
    """{ticker: type} for common stocks and ADRs, active and delisted (Massive's tickers reference). A ticker
    reused by a later company keeps the type of whichever listing Massive returns last; both are stocks."""
    path = Path(cache_dir) / "reference" / "stock_types.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        client, rows = sds.make_client(), []
        for kind in TYPES:
            for active in (True, False):
                rows += [(t.ticker, t.type) for t in client.list_tickers(type=kind, market="stocks", active=active,
                                                                         limit=1000)]
        df = pd.DataFrame(rows, columns=["ticker", "type"]).drop_duplicates("ticker", keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return dict(zip(df["ticker"], df["type"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default="2021-10-04")
    p.add_argument("--end", default="2026-09-30")
    p.add_argument("--cache-dir", default="data/cache")
    a = p.parse_args(argv)
    sessions = features.trading_sessions(a.start, a.end, warmup_sessions=0).index
    for adjusted in (True, False):
        g = load_grouped(sessions, adjusted, a.cache_dir)
        print(f"{'split-adjusted' if adjusted else 'unadjusted'}: {len(g):,} rows, "
              f"{g['session'].nunique()} of {len(sessions)} sessions")
    print(f"stock types: {len(stock_types(a.cache_dir)):,} tickers")


if __name__ == "__main__":
    main()
