"""What happens after a stock opens with a gap down? A price study: no options, no P&L.

  uv run --env-file .env python gap_recovery.py --out output/gap_recovery

For every session in which a ticker opens at least 1% below the previous close (on an ex-dividend morning
the payout is taken out of the gap, since it never comes back), it measures:
- whether the price trades back up to the previous close that session (the gap "fills"), and when;
- if not, how much of the gap it got back at best, and where it closed, both as a % of the gap;
- how far it fell below the open first, and whether it fell another gap's worth (the opposite of a fill);
- whether the gap filled within 1, 2, 3, 5 or 10 sessions.
Gaps are split by size and by kind: earnings (the reporting-window flags from stock_dip_spreads.py, the
flagged sessions themselves), with the market (SPY also opened 1%+ lower) or the stock alone.

Prices are Massive split-adjusted minute bars, regular hours only: the open is the first regular minute's open
and the previous close the last regular minute's close, close to (not exactly) the official auction prints.
Fills use minute highs, so a fill means the price traded at the previous close, not that an order filled.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import alerts  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "MSFT", "AAPL", "AMZN", "META")
MIN_GAP = 1.0                                     # % below the previous close
SIZES = ((0.25, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, np.inf))  # gap buckets, % (lower bound inclusive)
DAYS = (1, 2, 3, 5, 10)                           # sessions after the gap day, for later fills
BY_TIME = {"10:00": 30, "10:30": 60, "11:30": 120, "13:00": 210}  # minutes after a 9:30 open
MARKET_GAP = 1.0  # SPY opening this much lower (or --min-gap, if smaller) makes a stock's gap "with the market"
BEST_BINS = (0, 25, 50, 75, 100)                  # best recovery of unfilled gaps, % of the gap
EPS = 1e-9
WAIT = 30                                         # minutes after the open: the filter tests start at 10:00
# Renamed tickers: Massive files some sessions before a rename only under the old ticker (META was FB until
# 2022-06-09, and Massive has no META bars for 2022-01-31..2022-06-08), so missing sessions are filled from it.
FORMER = {"META": "FB"}


def _label(lo, hi):
    return f"{lo:g}%+" if np.isinf(hi) else f"{lo:g}-{hi:g}%"


def size_label(gap):
    g = abs(gap)
    for lo, hi in SIZES:
        if lo <= g < hi:
            return _label(lo, hi)
    return None


SIZE_LABELS = [_label(lo, hi) for lo, hi in SIZES]


def size_groups(min_gap):
    """For the filter tables: the first two size buckets that hold gaps of at least min_gap on their own, the rest
    together (1%: 1-2%, 2-3%, 3%+; 0.25%: 0.25-0.5%, 0.5-1%, 1%+)."""
    used = [(lo, hi) for lo, hi in SIZES if hi > min_gap]
    groups = [(_label(lo, hi), [_label(lo, hi)]) for lo, hi in used[:2]]
    if len(used) > 2:
        groups.append((f"{used[2][0]:g}%+", [_label(lo, hi) for lo, hi in used[2:]]))
    return groups


def gap_series(daily, dividends):
    """(target, gap %): the previous close net of a dividend paid that morning, and the open's gap from it.
    NaN when either session has no data."""
    paid = dividends.groupby("ex_date")["cash_amount"].sum() if len(dividends) else pd.Series(dtype=float)
    target = daily["close"].shift(1) - paid.reindex(daily.index).fillna(0.0).to_numpy()
    return target, (daily["open"] / target - 1) * 100


def first_hit(hit, ext):
    """Which came first: the fill (bar indices `hit`) or falling as far again (`ext`)."""
    return ("fill" if hit.size and (not ext.size or hit[0] < ext[0]) else
            "extension" if ext.size and (not hit.size or ext[0] < hit[0]) else
            "same minute" if hit.size else "neither")


def after_wait(hi, lo, cl, minutes, price, target):
    """The same-session measures taken from the 10:00 price instead of the open, over the bars that end after
    WAIT minutes: the fill, the best and closing recovery as a % of the gap left at 10:00, and the race against
    falling that much again below the 10:00 price. Only for gaps not yet filled by 10:00."""
    late = minutes > WAIT
    if not late.any():
        return {}
    left = target - price
    h, l = hi[late], lo[late]
    hit = np.flatnonzero(h >= target - EPS)
    ext = np.flatnonzero(l <= price - left + EPS)
    return {"filled_10": bool(hit.size), "fill_minutes_10": minutes[late][hit[0]] if hit.size else np.nan,
            "best_10": (h.max() - price) / left * 100, "close_10": (cl[-1] - price) / left * 100,
            "extended_10": bool(ext.size), "first_10": first_hit(hit, ext)}


def gap_events(ticker, rth, daily, dividends, spy_gap=None, earnings=None, min_gap=MIN_GAP, days_ahead=DAYS,
               market_gap=MARKET_GAP):
    """One row per session that opened at least `min_gap`% below the previous close. `daily` is indexed by
    every calendar session (NaN rows for sessions without data), so a missing session never stretches a gap
    across two days."""
    target, gap = gap_series(daily, dividends)
    by_session = dict(tuple(rth.groupby("session", sort=False)))
    days, high = daily.index, daily["high"].to_numpy(float)
    rows = []
    for i in np.flatnonzero((gap <= -min_gap).to_numpy()):
        t, g = days[i], by_session.get(days[i])
        if g is None or g.empty:
            continue
        o, tgt = float(daily["open"].iloc[i]), float(target.iloc[i])
        size = tgt - o
        hi, lo, cl = (g[c].to_numpy(float) for c in ("high", "low", "close"))
        minutes = (g["ts"] - g["session_open"]).dt.total_seconds().to_numpy() / 60 + 1  # bar end, minutes after open
        hit = np.flatnonzero(hi >= tgt - EPS)
        ext = np.flatnonzero(lo <= o - size + EPS)
        upto = hit[0] + 1 if hit.size else len(lo)  # lows up to the fill bar (its low may come before its high)
        first = first_hit(hit, ext)
        early = minutes <= WAIT
        price_10 = cl[early][-1] if early.any() else np.nan
        filled_by_10 = bool(hit.size) and minutes[hit[0]] <= WAIT
        if ticker == "SPY":
            kind = "SPY"
        elif earnings is not None and t in earnings:
            kind = "earnings"
        elif spy_gap is not None and spy_gap.get(t, np.nan) <= -market_gap:
            kind = "with the market"
        else:
            kind = "stock alone"
        row = {"ticker": ticker, "day": t, "prev_close": float(daily["close"].iloc[i - 1]), "target": tgt, "open": o,
               "gap": float(gap.iloc[i]), "size": size_label(gap.iloc[i]), "kind": kind,
               "spy_gap": np.nan if spy_gap is None else spy_gap.get(t, np.nan),
               "filled": bool(hit.size), "fill_minutes": minutes[hit[0]] if hit.size else np.nan,
               "best": (hi.max() - o) / size * 100, "close_rec": (cl[-1] - o) / size * 100,
               "further_drop": (lo[:upto].min() / o - 1) * 100, "further_drop_gaps": (o - lo[:upto].min()) / size,
               "extended": bool(ext.size), "first": first, "price_10": price_10, "filled_by_10": filled_by_10,
               **(after_wait(hi, lo, cl, minutes, price_10, tgt) if early.any() and not filled_by_10 else {})}
        later = np.flatnonzero(high[i:i + max(days_ahead) + 1] >= tgt - EPS)
        row["days_to_fill"] = float(later[0]) if later.size else np.nan
        for k in days_ahead:
            window = high[i:i + k + 1]
            row[f"filled_{k}d"] = (np.nan if i + k >= len(days) else True if (window >= tgt - EPS).any()
                                   else np.nan if np.isnan(window).any() else False)
        rows.append(row)
    return pd.DataFrame(rows)


# The same measures from the open, or from the 10:00 price for gaps not yet filled then (after_wait).
COLUMNS = {"open": {"filled": "filled", "fill_minutes": "fill_minutes", "best": "best", "close": "close_rec",
                    "extended": "extended", "first": "first"},
           "10:00": {"filled": "filled_10", "fill_minutes": "fill_minutes_10", "best": "best_10", "close": "close_10",
                     "extended": "extended_10", "first": "first_10"}}


def _waited(e):
    return (e["kind"] != "earnings") & e.get("filled_10", pd.Series(np.nan, index=e.index)).notna()


# Filter tests, fixed before any filtered result was seen: (where the measures start, which gaps).
FILTERS = {
    "All gaps, from the open": ("open", lambda e: pd.Series(True, index=e.index)),
    "Skip earnings, from the open": ("open", lambda e: e["kind"] != "earnings"),
    "Skip earnings, from 10:00": ("10:00", _waited),
    "Skip earnings, from 10:00, up since the open": ("10:00", lambda e: _waited(e) & (e["price_10"] > e["open"])),
    "Skip earnings, from 10:00, down since the open": ("10:00", lambda e: _waited(e) & (e["price_10"] <= e["open"])),
}


def summarize(ev, start="open"):
    """Fill rates (with 95% Wilson intervals), timing, recoveries and later fills for a set of gaps, measured from
    the open or (start="10:00") from the 10:00 price."""
    c = COLUMNS[start]
    n = len(ev)
    out = {"gaps": n}
    if not n:
        return out
    filled = ev[c["filled"]].astype(bool)
    k = int(filled.sum())
    lo, hi = alerts.wilson(k, n)
    unfilled, close = ev[~filled], ev[c["close"]]
    out.update(filled=k / n * 100, filled_lo=lo, filled_hi=hi, fill_minutes=ev.loc[filled, c["fill_minutes"]].median(),
               closed_up=(close > 0).mean() * 100, closed_filled=(close >= 100 - EPS).mean() * 100,
               close_rec=close.median(), extended=ev[c["extended"]].astype(bool).mean() * 100,
               fill_first=(ev[c["first"]] == "fill").mean() * 100,
               ext_first=(ev[c["first"]] == "extension").mean() * 100, **race(ev[c["first"]]),
               unfilled=len(unfilled), unfilled_best=unfilled[c["best"]].median() if len(unfilled) else np.nan)
    if start == "open":
        out.update(further_drop=ev["further_drop"].median(), further_drop_gaps=ev["further_drop_gaps"].median())
        for label, m in BY_TIME.items():
            out[f"by_{label}"] = (ev["fill_minutes"] <= m).mean() * 100
    for d in DAYS:
        known = ev[f"filled_{d}d"].dropna()
        out[f"within_{d}d"] = known.astype(bool).mean() * 100 if len(known) else np.nan
        out[f"within_{d}d_n"] = len(known)
    best = unfilled[c["best"]]
    for a, b in zip(BEST_BINS[:-1], BEST_BINS[1:]):
        out[f"best_{a}_{b}"] = ((best >= a) & (best < b)).mean() * 100 if len(unfilled) else np.nan
    out.update(close_below_open=(close < 0).mean() * 100, close_0_50=((close >= 0) & (close < 50)).mean() * 100,
               close_50_100=((close >= 50) & (close < 100 - EPS)).mean() * 100)
    return out


def race(first):
    """Of the gaps that either filled or fell as far again first, the share that filled first, with its 95%
    Wilson interval: about 50% if the price just wandered from the entry point."""
    fill, ext = int((first == "fill").sum()), int((first == "extension").sum())
    lo, hi = alerts.wilson(fill, fill + ext)
    return {"race_n": fill + ext, "race_fill": fill / (fill + ext) * 100 if fill + ext else np.nan,
            "race_lo": lo, "race_hi": hi}


def summary_table(events, min_gap=MIN_GAP):
    """Summaries by ticker, by size, by kind, by ticker x size, and for the filter tests."""
    rows = []
    stocks = events[events["ticker"] != "SPY"]
    groups = [("ticker", tk, "all", events[events["ticker"] == tk]) for tk in events["ticker"].unique()]
    groups.append(("ticker", "the four stocks", "all", stocks))
    for who, ev in (("SPY", events[events["ticker"] == "SPY"]), ("the four stocks", stocks)):
        groups += [("size", who, s, ev[ev["size"] == s]) for s in SIZE_LABELS]
    groups += [("kind", "the four stocks", k, stocks[stocks["kind"] == k])
               for k in ("earnings", "with the market", "stock alone")]
    groups += [("ticker x size", tk, s, events[(events["ticker"] == tk) & (events["size"] == s)])
               for tk in events["ticker"].unique() for s in SIZE_LABELS]
    for view, who, part, ev in groups:
        rows.append({"view": view, "who": who, "part": part, "start": "open", **summarize(ev)})
    for who, ev in [("the four stocks", stocks), *[(tk, events[events["ticker"] == tk])
                                                   for tk in events["ticker"].unique()]]:
        for name, (start, rule) in FILTERS.items():
            keep = rule(ev).fillna(False).to_numpy(bool) if len(ev) else np.zeros(0, bool)
            rows.append({"view": "filters", "who": who, "part": name, "start": start, **summarize(ev[keep], start)})
    for who, ev in (("the four stocks", stocks), ("SPY", events[events["ticker"] == "SPY"])):
        for label, sizes in size_groups(min_gap):
            for name in list(FILTERS)[2:]:
                start, rule = FILTERS[name]
                keep = (rule(ev).fillna(False).to_numpy(bool) & ev["size"].isin(sizes).to_numpy()) if len(ev) else \
                    np.zeros(0, bool)
                rows.append({"view": "filters x size", "who": who, "part": f"{label} | {name}", "start": start,
                             **summarize(ev[keep], start)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the study

def run_study(data, sessions, *, tickers=TICKERS, min_gap=MIN_GAP, timings=None):
    """data: {ticker: (regular-hours minutes, daily table, dividends)}; SPY must be included. No file I/O."""
    timings = {} if timings is None else timings
    with timed(timings, "gaps"):
        spy_rth, spy_daily, spy_divs = data["SPY"]
        _, spy_gap = gap_series(spy_daily, spy_divs)
        frames, flags = [], {}
        for tk in tickers:
            rth, daily, divs = data[tk]
            flags[tk] = None if tk == "SPY" else sds.earnings_days(daily, spy_daily, margin=0)
            frames.append(gap_events(tk, rth, daily, divs, spy_gap=spy_gap, earnings=flags[tk], min_gap=min_gap,
                                     market_gap=min(MARKET_GAP, min_gap)))
        events = pd.concat(frames, ignore_index=True)
        summary = summary_table(events, min_gap)
    sessions_with_data = {tk: int(data[tk][1]["close"].notna().sum()) for tk in tickers}
    return {"events": events, "summary": summary, "tickers": list(tickers), "min_gap": min_gap,
            "market_gap": min(MARKET_GAP, min_gap),
            "first": sessions.index[0], "last": sessions.index[-1], "sessions": sessions_with_data, "flags": flags}


# ---------------------------------------------------------------- report

def pct(v, digits=0):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}%"


def pick(summary, view, who, part="all"):
    m = summary[(summary["view"] == view) & (summary["who"] == who) & (summary["part"] == part)]
    return m.iloc[0] if len(m) and m.iloc[0]["gaps"] else None


def render_report(res):
    s = res["summary"]
    lines = []
    w = lines.append
    w("# Gap-down recoveries\n")
    w(f"Sessions {res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}, regular hours. A **gap down** is an open at least "
      f"{res['min_gap']:g}% below the previous close (net of any dividend paid that morning). The gap **fills** when "
      "the price trades back at the previous close. *Best recovery* is the highest price after the open, and *close* "
      "the session's last price, both as a % of the gap (100% = back to the previous close; below 0% = under the "
      "open). Ranges are 95% Wilson intervals; gaps on the same morning across tickers are not independent, so the "
      "pooled intervals are a little too narrow.\n")
    w(bottom_line(res))
    w("## 1. By ticker\n")
    w(ticker_section(res))
    w("## 2. By gap size\n")
    w(size_section(res))
    w("## 3. When the gap doesn't fill that day\n")
    w(unfilled_section(res))
    w("## 4. What time the fills happen\n")
    w(timing_section(res))
    w("## 5. Earnings, the market, or the stock alone\n")
    w(kind_section(res))
    w("## 6. Fill first, or another gap's worth down first?\n")
    w(race_section(res))
    w("## 7. Filters: skip earnings, wait until 10:00\n")
    w(filters_section(res))
    w("## 8. Data\n")
    w(data_section(res))
    return "\n".join(lines) + "\n"


def bottom_line(res):
    s = res["summary"]
    spy, st = pick(s, "ticker", "SPY"), pick(s, "ticker", "the four stocks")
    out = ["**Bottom line.** "]
    if st is not None:
        out.append(f"The four stocks opened {res['min_gap']:g}%+ lower {st['gaps']} times; the gap filled the same day "
                   f"{pct(st['filled'])} of the time ({pct(st['filled_lo'])}-{pct(st['filled_hi'])}), within 5 "
                   f"sessions {pct(st['within_5d'])}. ")
    if spy is not None:
        out.append(f"SPY gapped down {spy['gaps']} times and filled the same day {pct(spy['filled'])} of the time. ")
    if res["min_gap"] < 1:
        parts = [(label, pick(s, "size", "SPY", label)) for label in SIZE_LABELS]
        parts = [f"{label} {pct(r['filled'])} (n={r['gaps']})" for label, r in parts if r is not None]
        out.append("By gap size, SPY filled the same day: " + ", ".join(parts) + ". ")
    big = pick(s, "size", "the four stocks", "3-5%"), pick(s, "size", "the four stocks", "5%+")
    if big[0] is not None:
        out.append(f"Bigger gaps fill less: 3-5% gaps filled the same day {pct(big[0]['filled'])}")
        out.append(f", 5%+ gaps {pct(big[1]['filled'])}. " if big[1] is not None else ". ")
    if st is not None:
        out.append(f"Unfilled gaps got back a median {pct(st['unfilled_best'])} of the gap at best that day. ")
    up = pick(s, "filters", "the four stocks", "Skip earnings, from 10:00, up since the open")
    down = pick(s, "filters", "the four stocks", "Skip earnings, from 10:00, down since the open")
    if up is not None and down is not None:
        out.append(f"Skipping earnings and starting at 10:00, the four stocks reached the previous close before falling "
                   f"as far again {pct(up['race_fill'])} of the time when they were up since the open "
                   f"({pct(up['race_lo'])}-{pct(up['race_hi'])}) and {pct(down['race_fill'])} when they were down "
                   f"({pct(down['race_lo'])}-{pct(down['race_hi'])}); 50% is a coin flip.")
    return "".join(out) + "\n"


def ticker_section(res):
    s = res["summary"]
    rows = []
    for who in [*res["tickers"], "the four stocks"]:
        r = pick(s, "ticker", who)
        if r is None:
            continue
        rows.append([who, f"{r['gaps']}", f"{pct(r['filled'])} ({pct(r['filled_lo'])}-{pct(r['filled_hi'])})",
                     f"{r['fill_minutes']:.0f} min" if pd.notna(r["fill_minutes"]) else "", pct(r["within_1d"]),
                     pct(r["within_5d"]), pct(r["closed_up"]), pct(r["unfilled_best"]),
                     f"{r['further_drop']:+.1f}%"])
    return alerts.md_table(["Ticker", "Gap downs", "Filled same day", "Median time to fill", "By next day",
                            "Within 5 sessions", "Closed above the open", "Unfilled: best recovery (median)",
                            "Further drop after the open (median)"], rows) + "\n"


def size_section(res):
    s = res["summary"]
    out = []
    for who in ("the four stocks", "SPY"):
        rows = []
        for part in SIZE_LABELS:
            r = pick(s, "size", who, part)
            if r is None:
                continue
            rows.append([part, f"{r['gaps']}", f"{pct(r['filled'])} ({pct(r['filled_lo'])}-{pct(r['filled_hi'])})",
                         *[pct(r[f"within_{d}d"]) for d in DAYS], pct(r["closed_up"]), f"{r['close_rec']:.0f}%",
                         f"{r['further_drop']:+.1f}%"])
        out.append(f"**{who[0].upper() + who[1:]}:**\n")
        out.append(alerts.md_table(["Gap", "Gap downs", "Filled same day", *[f"Within {d}d" for d in DAYS],
                                    "Closed above the open", "Close, % of gap (median)", "Further drop (median)"],
                                   rows) + "\n")
    out.append("*Within N d* counts the gap day plus the next N sessions.")
    return "\n".join(out) + "\n"


def unfilled_section(res):
    s = res["summary"]
    out = ["Gaps that did not fill the same day, by the most they got back (highest price after the open, as a % of "
           "the gap), and every gap by where it closed:\n"]
    rows = []
    for who in ("the four stocks", "SPY"):
        for part in SIZE_LABELS:
            r = pick(s, "size", who, part)
            if r is None:
                continue
            rows.append([who, part, f"{int(r['unfilled'])}", *[pct(r[f"best_{a}_{b}"]) for a, b in
                                                          zip(BEST_BINS[:-1], BEST_BINS[1:])],
                         pct(r["close_below_open"]), pct(r["close_0_50"]), pct(r["close_50_100"]),
                         pct(r["closed_filled"])])
    out.append(alerts.md_table(["Who", "Gap", "Unfilled", "Best 0-25%", "25-50%", "50-75%", "75-99%",
                                "Closed below the open", "Closed 0-50% back", "50-99% back", "Closed filled"], rows))
    return "\n".join(out) + "\n"


def timing_section(res):
    s = res["summary"]
    out = ["Share of all gap downs filled by each time (Eastern), and by the close:\n"]
    rows = []
    for who in [*res["tickers"], "the four stocks"]:
        r = pick(s, "ticker", who)
        if r is not None:
            rows.append([who, *[pct(r[f"by_{t}"]) for t in BY_TIME], pct(r["filled"])])
    out.append(alerts.md_table(["Ticker", *[f"By {t}" for t in BY_TIME], "By the close"], rows))
    return "\n".join(out) + "\n"


def kind_section(res):
    s = res["summary"]
    out = ["The four stocks' gap downs by cause. *Earnings*: the session flagged as the report reaction (the biggest "
           "stock-specific volume jump or gap in the late-month reporting window). The flags are inferred, so a few "
           "are other big news in the same weeks. *With the market*: SPY also opened "
           f"{res['market_gap']:g}%+ lower. *Stock alone*: everything else (other news, downgrades, sector moves).\n"]
    rows = []
    for part in ("earnings", "with the market", "stock alone"):
        r = pick(s, "kind", "the four stocks", part)
        if r is None:
            continue
        rows.append([part, f"{r['gaps']}", f"{pct(r['filled'])} ({pct(r['filled_lo'])}-{pct(r['filled_hi'])})",
                     pct(r["within_5d"]), pct(r["closed_up"]), pct(r["unfilled_best"]), f"{r['further_drop']:+.1f}%"])
    out.append(alerts.md_table(["Kind", "Gap downs", "Filled same day", "Within 5 sessions", "Closed above the open",
                                "Unfilled: best recovery", "Further drop"], rows))
    ev = res["events"]
    big = ev[(ev["kind"] == "earnings")].sort_values("gap").head(8)
    if len(big):
        out.append("\nLargest earnings gap downs: " + "; ".join(
            f"{r.ticker} {r.day:%Y-%m-%d} {r.gap:+.1f}% ({'filled' if r.filled else f'best {r.best:.0f}%'}, "
            f"close {r.close_rec:.0f}%)" for r in big.itertuples()) + ".")
    return "\n".join(out) + "\n"


def race_section(res):
    s = res["summary"]
    out = ["If the price after a gap-down open simply wandered, rising back by the gap's size and falling another "
           "gap's worth would be about equally likely. *Another gap down* means the low reached the open minus the gap "
           "size (a 2% gap at $100: $98 open, $96 low):\n"]
    rows = []
    for who in [*res["tickers"], "the four stocks"]:
        r = pick(s, "ticker", who)
        if r is not None:
            rows.append([who, f"{r['gaps']}", pct(r["filled"]), pct(r["extended"]), pct(r["fill_first"]),
                         pct(r["ext_first"]), f"{pct(r['race_fill'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])})",
                         f"{r['further_drop_gaps']:.2f}"])
    out.append(alerts.md_table(["Ticker", "Gap downs", "Filled that day", "Fell another gap's worth", "Filled first",
                                "Fell another gap first", "Filled first, of those that did either",
                                "Further drop before any fill (median, in gaps)"], rows))
    out.append("\nA share of *filled first* whose range includes 50% is no better than a coin flip.")
    return "\n".join(out) + "\n"


SHORT = {"Skip earnings, from 10:00": "all", "Skip earnings, from 10:00, up since the open": "up since the open",
         "Skip earnings, from 10:00, down since the open": "down since the open"}


def filters_section(res):
    s = res["summary"]
    out = ["*From 10:00* skips the first 30 minutes: gaps that had already filled by 10:00 are left out (they were "
           "missed), and the rest are measured from the 10:00 price, with the gap left at 10:00 as the yardstick. "
           "*Filled first* asks whether the price reached the previous close before falling as far again below the "
           "entry price (the open, or the 10:00 price); about 50% means no edge. *Up since the open*: the 10:00 price "
           "was above the opening price. The five filters were fixed before any filtered result was seen.\n"]
    for who in ("the four stocks", "SPY"):
        rows = []
        for name in FILTERS:
            if who == "SPY" and name == "Skip earnings, from the open":
                continue
            r = pick(s, "filters", who, name)
            if r is None:
                continue
            rows.append([name.replace("Skip earnings, f", "F") if who == "SPY" else name, f"{r['gaps']}",
                         f"{pct(r['filled'])} ({pct(r['filled_lo'])}-{pct(r['filled_hi'])})",
                         f"{pct(r['race_fill'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])})", pct(r["unfilled_best"]),
                         pct(r["closed_up"]), pct(r["within_5d"])])
        out.append(f"**{who[0].upper() + who[1:]}**{' (no earnings to skip)' if who == 'SPY' else ''}:\n")
        out.append(alerts.md_table(["Filter", "Gaps", "Filled that day", "Filled first (95% range)",
                                    "Unfilled: best recovery (median)", "Closed above the entry price",
                                    "Filled within 5 sessions"], rows) + "\n")
    a = pick(s, "filters", "the four stocks", "Skip earnings, from the open")
    b = pick(s, "filters", "the four stocks", "Skip earnings, from 10:00")
    if a is not None and b is not None:
        out.append(f"Skipping earnings, {a['gaps'] - b['gaps']} of the four stocks' {a['gaps']} gaps "
                   f"({(a['gaps'] - b['gaps']) / a['gaps'] * 100:.0f}%) had already filled by 10:00 and are not in the "
                   "*from 10:00* rows.\n")
    rows = []
    for tk in [t for t in res["tickers"] if t != "SPY"]:
        for name, short in SHORT.items():
            r = pick(s, "filters", tk, name)
            if r is not None:
                rows.append([tk, short, f"{r['gaps']}", pct(r["filled"]),
                             f"{pct(r['race_fill'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])})", pct(r["closed_up"])])
    out.append("Each stock, skipping earnings and starting at 10:00:\n")
    out.append(alerts.md_table(["Ticker", "At 10:00", "Gaps", "Filled that day", "Filled first (95% range)",
                                "Closed above the 10:00 price"], rows) + "\n")
    for who in ("the four stocks", "SPY"):
        rows = []
        for label, _ in size_groups(res["min_gap"]):
            for name in list(FILTERS)[2:]:
                r = pick(s, "filters x size", who, f"{label} | {name}")
                if r is not None:
                    rows.append([label, SHORT[name], f"{r['gaps']}", pct(r["filled"]),
                                 f"{pct(r['race_fill'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])})",
                                 pct(r["closed_up"])])
        out.append(f"{'The four stocks' if who != 'SPY' else 'SPY'} by gap size, skipping earnings and starting at 10:00:\n")
        out.append(alerts.md_table(["Gap", "At 10:00", "Gaps", "Filled that day", "Filled first (95% range)",
                                    "Closed above the 10:00 price"], rows) + "\n")
    out.append("\nWith five filters, five tickers and the size splits, one range that clears 50% could be luck.")
    return "\n".join(out) + "\n"


def data_section(res):
    out = [f"- Sessions with data: " + ", ".join(f"{tk} {n:,}" for tk, n in res["sessions"].items()) + "."]
    out += [f"- {n}." for n in res.get("notes", [])]
    out.append("- Earnings flags use stock_dip_spreads.earnings_days without the extra margin (inferred from volume "
               "and gaps in the late-month reporting windows; see that module).")
    out.append("- `gaps.csv` has one row per gap down; `summary.csv` every summary, including each ticker by gap size.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- charts

def plot_outcomes(res, path):
    s = res["summary"]
    rows = [(who, part) for who in ("the four stocks", "SPY") for part in SIZE_LABELS
            if pick(s, "size", who, part) is not None]
    labels = ["Filled by 10:30", "Filled later that day", "Best 50-99% of the gap", "Best under 50%"]
    colors = [SERIES[0], "#8fb8ec", TARGET, MUTED]
    height = 1.9 + 0.36 * len(rows)
    fig = plt.figure(figsize=(8, height), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.26, 0.75 / height, 0.68, 1 - 1.55 / height])
    y = np.arange(len(rows))[::-1]
    for yy, (who, part) in zip(y, rows):
        r = pick(s, "size", who, part)
        unfilled = 100 - r["filled"]
        parts = [r["by_10:30"], r["filled"] - r["by_10:30"],
                 unfilled * (r["best_50_75"] + r["best_75_100"]) / 100, unfilled * (r["best_0_25"] + r["best_25_50"]) / 100]
        left = 0.0
        for v, c in zip(parts, colors):
            ax.barh(yy, v, left=left, color=c, height=0.62)
            left += v
        ax.annotate(f"n={r['gaps']}", (101, yy), va="center", fontsize=7, color=INK_2, annotation_clip=False)
    ax.set_yticks(y, [f"{'Stocks' if who != 'SPY' else 'SPY'}, {part} gap" for who, part in rows], fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Share of gap downs (%)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "What happened the same day after a gap-down open", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "MSFT, AAPL, AMZN and META pooled, and SPY. Best = highest price after the open, as a share "
             "of the gap.", color=INK_2, fontsize=7.5, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False, fontsize=7.5,
               labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_later(res, path):
    s = res["summary"]
    colors = [SERIES[0], SERIES[1], TARGET, INK_2, MUTED, "#8fb8ec"]
    fig = plt.figure(figsize=(8, 5.0), dpi=150, facecolor=SURFACE)
    x = [0, *DAYS]
    for k, (who, title) in enumerate((("the four stocks", "MSFT, AAPL, AMZN and META"), ("SPY", "SPY"))):
        ax = sds._axes(fig, [0.08 + k * 0.48, 0.34, 0.42, 0.48])
        for c, part in enumerate(SIZE_LABELS):
            r = pick(s, "size", who, part)
            if r is None or r["gaps"] < 5:
                continue
            ys = [r["filled"], *[r[f"within_{d}d"] for d in DAYS]]
            ax.plot(x, ys, marker="o", ms=3.5, lw=1.8, color=colors[c], label=f"{part} (n={r['gaps']})")
        ax.set_xticks(x, ["Same\nday", *[f"{d}" for d in DAYS]], fontsize=7)
        ax.set_ylim(0, 100)
        ax.axhline(50, color=BASELINE, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("Sessions after the gap day", color=INK_2, fontsize=7.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6.5,
                  labelcolor=INK_2)
    fig.text(0.02, 0.97, "How many gaps fill within a few sessions", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.915, "Share of gap downs that traded back at the previous close, by gap size.", color=INK_2,
             fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_filters(res, path):
    s = res["summary"]
    rows = [(who, name) for who in ("the four stocks", "SPY") for name in FILTERS
            if not (who == "SPY" and name == "Skip earnings, from the open") and pick(s, "filters", who, name) is not None]
    fig = plt.figure(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.42, 0.14, 0.52, 0.68])
    y = np.arange(len(rows))[::-1]
    for yy, (who, name) in zip(y, rows):
        r = pick(s, "filters", who, name)
        color = SERIES[0] if who != "SPY" else SERIES[1]
        ax.errorbar(r["race_fill"], yy, xerr=[[r["race_fill"] - r["race_lo"]], [r["race_hi"] - r["race_fill"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=3, ms=5)
        ax.annotate(f"n={int(r['race_n'])}", (r["race_lo"], yy), xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7, color=INK_2)
    ax.axvline(50, color=BASELINE, lw=1.2)
    labels = [("Stocks: " if who != "SPY" else "SPY: ") + (name.replace("Skip earnings, f", "f") if who == "SPY" else
                                                           name.replace("Skip earnings", "skip earnings")) for who, name in rows]
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(15, 100)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Reached the previous close before falling as far again (%, 95% interval)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "Do the filters beat a coin flip?", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Of gaps that did one or the other first; 50% (line) is what random wandering gives. Blue: "
             "MSFT, AAPL, AMZN and META pooled; orange: SPY.", color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def load_minutes(ticker, sessions, cache_dir="data/cache", refresh=False):
    """(minute bars, note) over the sessions' span. Sessions missing under the ticker are filled from its former
    ticker (FORMER) when it has one. Shared with scalp_meanrev.py."""
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    minutes, _ = download.load_minute_bars(ticker, first, last, cache_dir=cache_dir, refresh=refresh,
                                           final_close=sessions["close"].iloc[-1])
    note = ""
    if ticker in FORMER:
        daily, _ = mr.daily_table(minutes, sessions)
        missing = daily.index[daily["close"].isna()]
        if len(missing):
            span = sessions.loc[missing.min():missing.max()]
            old, _ = download.load_minute_bars(FORMER[ticker], str(span.index[0].date()), str(span.index[-1].date()),
                                               cache_dir=cache_dir, refresh=refresh, final_close=span["close"].iloc[-1])
            old_days = old["ts"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
            minutes = pd.concat([minutes, old[old_days.isin(missing)]], ignore_index=True)
            note = (f"{len(missing)} {ticker} sessions without {ticker} bars ({missing.min():%Y-%m-%d} to "
                    f"{missing.max():%Y-%m-%d}) filled from {FORMER[ticker]}, its ticker before the rename")
    return minutes, note


def load_ticker(ticker, sessions, cache_dir, refresh):
    """(regular-hours minutes, daily table, dividends, note)."""
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    minutes, note = load_minutes(ticker, sessions, cache_dir, refresh)
    rth, _ = features.regular_session_minutes(minutes, sessions)
    daily, _ = mr.daily_table(minutes, sessions)
    return rth, daily, mr.load_dividends(ticker, first, last, cache_dir, refresh), note


def write_outputs(out_dir, res):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["events"].to_csv(out / "gaps.csv", index=False)
    res["summary"].to_csv(out / "summary.csv", index=False)
    plot_outcomes(res, out / "outcomes.png")
    plot_later(res, out / "later.png")
    plot_filters(res, out / "filters.png")
    (out / "report.md").write_text(render_report(res))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="How often do stocks recover after a gap-down open?")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS), help="SPY is always included (for the market gap)")
    p.add_argument("--min-gap", type=float, default=MIN_GAP, help="%% below the previous close (default 1)")
    p.add_argument("--out", default="output/gap_recovery")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tickers = ["SPY", *[t.upper() for t in args.tickers if t.upper() != "SPY"]]
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        loaded = {tk: load_ticker(tk, sessions, args.cache_dir, args.refresh) for tk in tickers}
    data = {tk: v[:3] for tk, v in loaded.items()}
    res = run_study(data, sessions, tickers=tickers, min_gap=args.min_gap, timings=timings)
    res["notes"] = [v[3] for v in loaded.values() if v[3]]
    with timed(timings, "outputs"):
        out = write_outputs(args.out, res)
    s = res["summary"]
    for tk in tickers:
        r = pick(s, "ticker", tk)
        if r is not None:
            print(f"  {tk}: {r['gaps']} gap downs of {args.min_gap:g}%+, filled the same day {pct(r['filled'])}, "
                  f"within 5 sessions {pct(r['within_5d'])}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
