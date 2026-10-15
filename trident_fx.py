"""TG Capital's "Trident" on what it was built for: the playbook's five forex pairs and gold, in the London kill zone,
on Massive currency minute bars. The same signal study as trident.py (in R, against random entries with the same
filters and stop size), with the playbook's own times and gold's close-based stop.

  uv run --env-file .env python trident_fx.py --download-only                  # cache every year, 2010 to DATA_END
  uv run --env-file .env python trident_fx.py --out output/trident_fx          # design, 2010-03 to 2023-12
  uv run --env-file .env python trident_fx.py --final-test --out output/trident_fx   # 2024-01 to DATA_END, once
  uv run --env-file .env python trident_fx.py --as-traded --out output/trident_fx    # the as-traded rules (below)

Fixed before any result was seen (trident.py's rules, already confirmed by the user, except where noted):
- Symbols (the playbook's list): EURUSD, GBPUSD, USDJPY, NZDUSD, USDCAD and XAUUSD (Massive C: tickers).
  Groups: the five forex pairs, and gold.
- Trading day: 17:00 ET to 17:00 ET, named by the date it ends (the usual forex daily candle), Monday to Friday;
  Sunday evening belongs to Monday. 30-minute bars from 17:00 ET (on the half hour), each needing 80% of its
  minutes; EMAs and ATR run continuously across days.
- Kill zone as written: the FVG forms between 2:30 and 4:00 ET (choice: candle 3 lies inside it, so it closes at
  3:00, 3:30 or 4:00 ET), and entries come by 6:30 ET (the limit variant's order is live until 6:30).
- Pattern, entry, stop, filters, outcomes, random matches, variants and diagnostics: exactly trident.py's.
- Gold (as written: "a hard stop isn't used", "a closing candle filter"): the stop is a 30-minute close beyond the
  stop level, exited at that close, in the trades and their random matches alike. The forex pairs use the normal
  touch stop.
- Rollover (found 2026-10-15 while checking the playbook's example trades, before any as-traded result): Massive's
  currency minutes around the 17:00 ET daily roll often hold off-market quotes (USDCAD's 16:59 low on 2024-10-04 is
  250 pips under the market). In 2010-23, 34-48% of NZDUSD and USDCAD minute spikes (range > 15x the local median
  and > 20 bps) fall at 16:55-17:15, against news times elsewhere. Minutes from 16:55 to 17:10 ET are dropped, so
  a day closes at 16:54. The as-written results were re-run with this rule.
- Costs: for each as-written trade, the median quoted spread (ask - bid, floored at 0) in the minute before the
  entry, charged once a round trip. Reported, not part of the test (the test compares with random entries, which
  would pay a similar spread).
- Test: as trident.py, per group (the forex pairs; gold), BH q < 0.10 across the two and an excess above 0.
- Data gaps (found while downloading, before any result): Massive has no NZDUSD minutes from 2020-10-26 to
  2020-12-31 and no gold minutes from 2010-09-27 to mid-July 2013 (apart from part of 2011). Gold is studied from
  its data's return on 2013-07-15; every symbol's trades start 60 days after its first minute.
- Pip-rounded lows (found 2026-10-15, before any as-traded result): from January 2021 Massive's currency bars (built
  from bids) take most minute lows, and some highs, from a source quoting whole pips, about 1-2 pips below the other
  bids (EURUSD 2023: 90% of lows have four decimals against 38% of closes). That shrinks long FVGs, widens short
  ones and brings long stops a little closer. Not corrected; every report splits 2010-2020 from 2021 on.
- Periods: minute data starts in 2010. Design 2010-03-01 to 2023-12-29 (January and February 2010 warm the
  indicators). Held back: 2024-01-02 to DATA_END, read only with --final-test, once. The design run never loads a
  file after 2023.

As traded (`--as-traded`; written 2026-10-15 from two of his winning trades shown on TradingView, USDJPY 2024-11-12
and USDCAD 2024-10-08, before any result of these rules was seen):
- The FVG's displacement candle (candle 2) opens between 2:30 and 4:00 ET inclusive, so candle 3 closes at 3:30,
  4:00, 4:30 or 5:00 ET. (USDJPY's displacement candle opened at 4:00, USDCAD's at 2:30.)
- The doji is the first of the 4 candles after candle 3 with a body <= 25% of its range whose wick reaches the 50%
  level and whose body stays beyond it. (USDCAD's came 3 candles after.) The setup is void if a candle closes
  beyond the gap's far side first (below candle 1's high for longs).
- The confirmation candle follows the doji, unchanged; the entry, at its close, must come by 6:30 ET.
- Everything else as above. Rows: as traded (primary), stop at candle 2, gap >= 0.1 ATR, limit at the 50% level
  (variants); no daily bias, no trend filters, every FVG (diagnostics). The doji-body variants are left out: here
  the body limit chooses which candle is the doji.
- His two example days are excluded from every run (they were picked as winners). Checked in our data: USDJPY
  2024-11-12 is found candle for candle (entry 06:00 at 154.046, stop 153.837). USDCAD 2024-10-08 is not: on
  Massive's feed candle 3's low (1.36360, from pip-rounded bids) equals candle 1's high, so there is no gap, and the
  daily EMAs were not stacked up at the previous close.

Green daily candle (written 2026-10-15, before any result of it was seen): he says he only trades on "bright green"
candles from the BullTrading Easy Scalping indicator (closed source). On his USDJPY chart its green marked every
up candle while the higher-timeframe trend was up, so the filter is: for a long, the previous trading day's candle
closed above its open (and the daily bias already requires its EMAs stacked up); shorts mirror it. It is added to
the as-written and as-traded rows as "+ green daily candle", and their random matches need it too. The test: its
excess over random per group (pairs, gold) in each version, four tests in all, BH q < 0.10 across the four.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from massive import RESTClient

import download
import trident as td
from run import timed

NY = td.NY
SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "NZDUSD", "USDCAD", "XAUUSD")
GROUPS = {"forex pairs": SYMBOLS[:5], "gold": ("XAUUSD",)}
CLOSE_STOP = ("XAUUSD",)
FVG_ENDS, LAST_ENTRY = ("03:00", "03:30", "04:00"), "06:30"
DATA_START, DATA_END = pd.Timestamp("2010-01-04"), pd.Timestamp("2026-10-09")
DESIGN_START, DESIGN_END = pd.Timestamp("2010-03-01"), pd.Timestamp("2023-12-29")
HOLDOUT_START = pd.Timestamp("2024-01-02")
ROUNDED_FROM = pd.Timestamp("2021-01-01")   # pip-rounded lows from here on
AS_TRADED_ENDS = ("03:30", "04:00", "04:30", "05:00")
AS_TRADED_WINDOW = 4
AS_TRADED_ROWS = {"as traded": {}, "stop at candle 2": {"stop": 2}, "gap >= 0.1 ATR": {"min_gap": 0.1},
                  "limit at the 50% level": {"entry": "limit"}, "no daily bias": {"bias": False},
                  "no trend filters": {"trend": False, "bias": False}, "every FVG, no trident": {"entry": "fvg"}}
EXAMPLES = (("USDJPY", "2024-11-12"), ("USDCAD", "2024-10-08"))   # his shown winners: never in a test
GREEN = {"+ green daily candle": {"green": True}}                    # the pre-declared addition (docstring)
WARMUP_DAYS = 180                       # the held-back run loads this much earlier data for the indicators
WARMUP = pd.Timedelta(days=60)          # trades start this long after a symbol's first minute
# Massive has no gold minutes from 2010-09-27 to 2013-07 (bar a few months of 2011), so gold's study starts after
# that gap rather than bridging it with stale indicators. (NZDUSD also has no bars 2020-10-26 to 2020-12-31.)
FIRST_DATA = {"XAUUSD": pd.Timestamp("2013-07-15")}
PROBE_DAYS = 3                          # weekdays checked after a download that stops early
DAY_ROLL = pd.Timedelta(hours=7)        # 17:00 ET + 7 hours = midnight: the day a minute belongs to
ROLLOVER = (16 * 60 + 55, 17 * 60 + 10)  # ET minutes [16:55, 17:10) are dropped: quotes there are often off-market
CHUNK_SLACK = pd.Timedelta(days=4)      # a year's download must reach within 4 days of its last date


# ---------------------------------------------------------------- days and minutes

def fx_sessions(start, end):
    """Forex trading days Monday-Friday from `start` to `end`: each runs from 17:00 ET the day before to 17:00 ET."""
    days = pd.bdate_range(start, end).as_unit("ns")
    close = (days + pd.Timedelta(hours=17)).tz_localize(NY).tz_convert("UTC")
    open_ = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(NY).tz_convert("UTC")
    return pd.DataFrame({"open": open_, "close": close, "in_study": True}, index=days)


def fx_minutes(minutes, sessions):
    """Sorted, de-duplicated minutes with `session` (the trading day) and its `session_open` / `session_close`, the
    shape features.regular_session_minutes gives stocks. Weekend minutes (Friday 17:00 to Sunday 17:00) and the
    rollover minutes (16:55 to 17:10 ET) drop out."""
    m = minutes.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    ts = m["ts"].dt.tz_convert("UTC").dt.as_unit("ns")
    et = ts.dt.tz_convert(NY)
    mod = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    keep = (mod < ROLLOVER[0]) | (mod >= ROLLOVER[1])
    ts, et, m = ts[keep], et[keep], m[keep]
    day = (et + DAY_ROLL).dt.tz_localize(None).dt.normalize().dt.as_unit("ns")
    m = m.assign(ts=ts, session=day)
    bounds = sessions[["open", "close"]].rename(columns={"open": "session_open", "close": "session_close"})
    m = m.join(bounds, on="session", how="inner")
    return m[(m["ts"] >= m["session_open"]) & (m["ts"] < m["session_close"])].reset_index(drop=True)


def prepare(minutes, sessions, final):
    """trident.run_study's `prepare`: the design run keeps days through DESIGN_END only; the held-back run keeps
    WARMUP_DAYS before HOLDOUT_START onwards. Trades start no sooner than 60 days after the first minute."""
    if final:
        first, used = HOLDOUT_START, sessions[sessions.index >= HOLDOUT_START - pd.Timedelta(days=WARMUP_DAYS)]
    else:
        first, used = DESIGN_START, sessions[sessions.index <= DESIGN_END]
    rth = fx_minutes(minutes, used)
    return rth, used, max(first, rth["session"].min() + WARMUP), used.index[-1]


# ---------------------------------------------------------------- data

def chunks(start, end):
    """[(first date, last date)] per calendar year."""
    out = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(f"{y}-01-01")), min(end, pd.Timestamp(f"{y}-12-31"))
        out.append((a, b))
    return out


def chunk_path(cache_dir, symbol, a, b):
    return Path(cache_dir) / "fx" / f"C_{symbol}_1min_{a:%Y-%m-%d}_{b:%Y-%m-%d}.parquet"


def make_client():
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set (see README).")
    return RESTClient(api_key=key, retries=10)


def load_fx(symbol, start, end, cache_dir="data/cache", refresh=False, client=None):
    """Minute bars for C:`symbol` over [start, end], one cached Parquet file per calendar year. A client is made only
    on a cache miss. A fresh year that stops more than 4 days before its last date is checked: if Massive has bars
    on any of the next 3 weekdays, the SDK stopped paging silently, so it raises and caches nothing; if not, it is a
    gap in Massive's data, cached as it is and reported."""
    frames = []
    for a, b in chunks(pd.Timestamp(start), pd.Timestamp(end)):
        path = chunk_path(cache_dir, symbol, a, b)
        if path.exists() and not refresh:
            frames.append(pd.read_parquet(path))
            continue
        client = client or make_client()
        df = download.fetch_minute_bars(f"C:{symbol}", f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}", client=client)
        last = a - pd.Timedelta(days=1) if df.empty else df["ts"].max().tz_convert(None).normalize()
        if last < b - CHUNK_SLACK:
            probe = pd.bdate_range(last + pd.Timedelta(days=1), b)[:PROBE_DAYS]
            if any(next(iter(client.list_aggs(f"C:{symbol}", 1, "minute", f"{d:%Y-%m-%d}", f"{d:%Y-%m-%d}",
                                              limit=10)), None) is not None for d in probe):
                raise RuntimeError(f"C:{symbol} {a:%Y-%m-%d}..{b:%Y-%m-%d} stops at {last:%Y-%m-%d} although "
                                   "Massive has later bars; not caching it.")
            print(f"C:{symbol}: no Massive minutes after {last:%Y-%m-%d} in {a:%Y}; cached as a data gap")
        df = df[["ts", "open", "high", "low", "close", "volume"]] if len(df) else pd.DataFrame(
            {"ts": pd.Series(dtype="datetime64[ns, UTC]"), **{k: pd.Series(dtype=float) for k in
                                                              ("open", "high", "low", "close", "volume")}})
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def spread_loader(cache_dir="data/cache", client=None):
    """spread_of(symbol, entry_ns array) -> the median quoted spread (ask - bid, floored at 0) in the minute before
    each entry, NaN when there are no quotes. Cached in data/cache/fx/spreads.parquet; call .save() at the end."""
    path = Path(cache_dir) / "fx" / "spreads.parquet"
    known = {}
    if path.exists():
        old = pd.read_parquet(path)
        known = dict(zip(zip(old["symbol"], old["entry_ns"]), old["spread"]))
    box = {"client": client}

    def one(symbol, ns):
        box["client"] = box["client"] or make_client()
        q = box["client"].list_quotes(f"C:{symbol}", timestamp_gte=int(ns) - 60_000_000_000, timestamp_lt=int(ns),
                                      limit=50_000)
        s = [max(0.0, x.ask_price - x.bid_price) for x in q if x.ask_price and x.bid_price]
        return float(np.median(s)) if s else np.nan

    def spread_of(symbol, entry_ns):
        todo = sorted({int(ns) for ns in entry_ns if (symbol, int(ns)) not in known})
        if todo:
            with ThreadPoolExecutor(8) as pool:
                for ns, v in zip(todo, pool.map(lambda n: one(symbol, n), todo)):
                    known[symbol, ns] = v
        return np.array([known[symbol, int(ns)] for ns in entry_ns], float)

    def save():
        if known:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([(s, n, v) for (s, n), v in known.items()],
                         columns=["symbol", "entry_ns", "spread"]).to_parquet(path, index=False)

    spread_of.save = save
    return spread_of


# ---------------------------------------------------------------- report

def green_section(res):
    """The pre-declared green-daily-candle addition against its own random matches, per group."""
    s = res["stats"]
    g = s[(s["row"] == "+ green daily candle") & (s["who"] == "pattern") & (s["sides"] == "both")
          & s["scope"].isin(["all", *GROUPS])]
    rows = []
    for r in g.itertuples():
        base = td.get(s, row=r.row, scope=r.scope, sides="both", who="random")
        p = r.p_up if pd.notna(r.p_up) else np.nan
        rows.append([r.scope, f"{int(r.trades):,}", td.pct(getattr(r, "hit1", np.nan)), td.fmt(r.bracket),
                     td.fmt(base["bracket"]) if base is not None else "n/a",
                     f"{td.fmt(r.excess)} ({td.fmt(r.ex_lo)} to {td.fmt(r.ex_hi)})",
                     "n/a" if pd.isna(p) else f"{p:.3f}"])
    return ("## Green daily candle (pre-declared addition)\n\nThe previous trading day's candle closed in the "
            "trade's direction, on top of every other rule; random matches need it too. One-sided p per group; the "
            "four tests (two groups, as written and as traded) share BH q < 0.10.\n\n" + td.alerts.md_table(
                ["Scope", "Trades", "+1R first", "Bracket, mean R", "Random, mean R", "Excess R (95%)", "p"], rows)
            + "\n")


def era_section(res):
    """The primary row before and after the pip-rounded lows began, against its random matches."""
    t = res["trades"][res["trades"]["row"] == res["primary"]]
    rnd = res["random"][res["random"]["row"] == res["primary"]]
    rows = []
    for label, keep in ((f"before {ROUNDED_FROM:%Y}", t["session"] < ROUNDED_FROM),
                        (f"{ROUNDED_FROM:%Y} on (pip-rounded lows)", t["session"] >= ROUNDED_FROM)):
        for side, name in ((None, "both"), (1, "long"), (-1, "short")):
            g = t[keep & ((t["side"] == side) if side else True)]
            r = {**td.row_stats(g, 1.0), **td.excess(g, rnd[rnd["pid"].isin(g["pid"])])}
            rows.append([label if name == "both" else "", name, len(g), td.pct(r.get("hit1")),
                         td.fmt(r.get("bracket")), f"{td.fmt(r.get('excess'))} ({td.fmt(r.get('ex_lo'))} to "
                         f"{td.fmt(r.get('ex_hi'))})"])
    return (f"## By data era ({res['primary']}, all symbols)\n\n" + td.alerts.md_table(
        ["Era", "Side", "Trades", "+1R first", "Bracket, mean R", "Excess R (95%)"], rows) +
        "\n\nFrom 2021 Massive's lows are often a pip-rounded bid 1-2 pips under the market (see trident_fx.py).\n")


INTRO = ("30-minute bars of the playbook's five forex pairs and gold, trading days from 17:00 ET. Long: a 3-candle "
         "FVG whose third candle closes at 3:00-4:00 ET, a doji (body <= 25% of its range) wicking into the gap's "
         "50% level with its body above it, then a candle closing below the doji's high; EMA 5 > 9 > 13 > 21, close "
         "above the 200 EMA, daily EMAs stacked the same way at the previous day's close. Entry at the confirmation "
         "close, stop at candle 1's low (gold: a 30-minute close beyond it); shorts mirror it. Everything is in R "
         "(entry to stop). A signal study: no position limits, overlapping trades all counted. Every open choice is "
         "listed in trident_fx.py and trident.py.")
AS_TRADED_INTRO = (" As traded: the displacement candle opens 2:30-4:00 ET (candle 3 closes by 5:00), the doji is "
                   "the first doji of the next 4 candles wicking into the 50% level (void if a candle closes beyond "
                   "the gap first), and the entry comes by 6:30. His two example days are left out.")
NOTES = ("Minute bars are Massive's currency aggregates, built from quotes, so they can differ a little from a "
         "broker's chart. A minute reaching a touch stop and a target counts as the stop; a stop gapped through "
         "(for example over a weekend) exits at that minute's open.",
         "Gold's stop is a 30-minute close beyond the stop level, exited at that close, so a gold loss can exceed "
         "1R; its random matches use the same rule.",
         "Minutes from 16:55 to 17:10 ET are dropped (off-market quotes at the daily roll).")


def main(argv=None):
    p = argparse.ArgumentParser(description="TG Capital Trident on forex and gold in the London kill zone.")
    p.add_argument("--out", default="output/trident_fx")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="read 2024-01 to DATA_END (once)")
    p.add_argument("--download-only", action="store_true", help="cache every year through DATA_END, then stop")
    p.add_argument("--as-traded", action="store_true", help="the as-traded rules (see the docstring)")
    args = p.parse_args(argv)
    if args.download_only:
        with ThreadPoolExecutor(len(args.symbols)) as pool:
            for sym, df in zip(args.symbols, pool.map(
                    lambda s: load_fx(s, DATA_START, DATA_END, args.cache_dir, args.refresh), args.symbols)):
                print(f"{sym}: {len(df):,} minutes, {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}")
        return
    timings = {}
    sessions = fx_sessions(DATA_START, DATA_END)
    start, end = ((HOLDOUT_START - pd.Timedelta(days=WARMUP_DAYS), DATA_END) if args.final_test
                  else (DATA_START, DESIGN_END))
    data = {s: (lambda s=s: load_fx(s, max(start, FIRST_DATA.get(s, start)), end, args.cache_dir, args.refresh))
            for s in args.symbols}
    spread_of = spread_loader(args.cache_dir)
    try:
        rules = (dict(ends=AS_TRADED_ENDS, doji_window=AS_TRADED_WINDOW, rows={**AS_TRADED_ROWS, **GREEN})
                 if args.as_traded else dict(ends=FVG_ENDS, rows={**td.ROWS, **GREEN}))
        res = td.run_study(data, sessions, final=args.final_test, timings=timings, prepare=prepare,
                           last_entry=LAST_ENTRY, groups=GROUPS, close_stop=CLOSE_STOP, spread_of=spread_of,
                           exclude=EXAMPLES, **rules)
    finally:
        spread_of.save()
    out = Path(args.out) / ("as_traded" if args.as_traded else "") / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    res["random"].to_parquet(out / "random.parquet", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "first": str(res["first"].date()),
        "last": str(res["last"].date()), "symbols": args.symbols, "random_draws": td.RANDOM_DRAWS,
        "seed": td.SEED}, indent=2))
    held = f"{HOLDOUT_START:%Y-%m} to {DATA_END:%Y-%m}"
    td.plot(res, out / "races.png", title="TG Capital Trident on forex and gold, London kill zone",
            subtitle="30-minute bars, FVG by 4:00 ET, entries by 6:30 ET; targets scanned up to 20 trading days.")
    title = "TG Capital Trident on forex and gold" + (", as traded" if args.as_traded else "")
    intro = INTRO + (AS_TRADED_INTRO if args.as_traded else "")
    report = td.render_report(res, title=title, intro=intro, notes=NOTES, held=held)
    head, tail = report.split("## By year", 1)
    (out / "report.md").write_text(head + green_section(res) + "\n" + era_section(res) + "\n## By year" + tail)
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
