"""One-time check of the TSLA breakout trades on 2025-26, with shares and with options. A trade simulation on
Massive minute and one-second bars: shares are per share, gross of commissions; options are per contract (100
shares) at traded prices plus slippage, without commissions.

  uv run --env-file .env python breakout_check.py --out output/breakout_check                 # dry run on 2021-24
  uv run --env-file .env python breakout_check.py --final-test --out output/breakout_check    # 2025-26, once

A follow-up to breakouts.py: the TSLA breakout trades replayed on one-second bars, which settle what happens
inside the fill minute, then checked once on data no study had looked at.

Fixed on 2026-10-02, before any 2025-26 TSLA result was seen:
- TSLA. Levels, sides, the touch window and ATR as in levels.py and breakouts.py. Three rules: long through
  yesterday's high (the session opens below it), through yesterday's close (long from below, short from above),
  and short through yesterday's low (the session opens above it).
- Shares: a stop order at the level; stop and target 0.1 ATR from the level; otherwise out at the last close.
  The fill minute and every minute that reaches the stop or the target are replayed on one-second bars: the fill
  is the first second reaching the level (at the level, or at that second's open if already through it); after
  it, the first second reaching the stop or the target ends the trade. A second reaching both counts as stopped,
  and so does the fill second when it reaches the stop; a stop gapped through fills at that second's open.
  Costs 1 bp per side (0 and 2 shown).
- Options, one per share trade: a call for longs, a put for shorts; the first listed expiry after the trade date
  (never same-day); the listed strike nearest the level. Bought at the first option trade at or after the share
  fill (within 60 seconds, otherwise skipped and counted) and sold at the first option trade at or after the
  share exit (within 60 seconds, otherwise the last trade in the 60 seconds before it), each fill paying $0.05 a
  share of slippage ($0.02 and $0.10 shown).
- Tests: shares at 1 bp per side and options at $0.05 per fill: for each rule, mean net P&L per trade > 0
  (one-sided, standard errors clustered by session); BH q < 0.10 within each set of three.
- Context: the same share trades at 200 random fake levels (a seeded sample of levels.py's fakes), replayed the
  same way.
- Without --final-test the script runs on 2021-24 (options only from 2024-10-01, when the option data starts).
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from massive import RESTClient  # noqa: E402
from massive.exceptions import BadResponse  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alert_spreads as asp  # noqa: E402
import alerts  # noqa: E402
import breakouts as bo  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import levels as lv  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKER = "TSLA"
RULES = {"long through yesterday's high": "yesterday's high", "through yesterday's close": "yesterday's close",
         "short through yesterday's low": "yesterday's low (check)"}   # -> breakouts.RULES
WIDTH = 0.1                                 # stop and target, ATR from the level
COSTS = (0, 1, 2)                           # shares: bps per side
PRIMARY_COST = 1
SLIPS = (0.02, 0.05, 0.10)                  # options: $ per share per fill
PRIMARY_SLIP = 0.05
WINDOW_MS = 60_000                          # how far to look for an option trade
FAKE_SAMPLE = 200
OPTIONS_START = pd.Timestamp("2024-10-01")
SEED = 5
FDR = 0.10
MINUTE_MS = 60_000
COLORS = dict(zip(RULES, (SERIES[0], INK_2, SERIES[1])))


# ---------------------------------------------------------------- data

def second_loader(cache_dir, refresh=False, volume=False):
    """load(ticker, start_ms, end_ms) -> one-second bars (array of ms, open, high, low, close, plus volume when
    volume=True) for that inclusive window, cached in <cache_dir>/seconds (one pickle per stock, one per stock's
    options; the volume bars in their own pickles). The Massive client is created only on a miss. A window the data
    plan doesn't cover comes back empty, is listed in load.state["denied"] and is not cached.
    load.prefetch(requests) fills the cache in parallel; load.save() writes it."""
    folder = Path(cache_dir) / "seconds"
    state = {"client": None, "fetched": 0, "tables": {}, "dirty": set(), "denied": set()}
    width = 6 if volume else 5

    def group(ticker):
        g = f"{ticker[2:].rstrip('0123456789CP')}_options" if ticker.startswith("O:") else ticker
        return f"{g}_volume" if volume else g

    def table(ticker):
        g = group(ticker)
        if g not in state["tables"]:
            path = folder / f"{g}.pkl"
            state["tables"][g] = pickle.loads(path.read_bytes()) if path.exists() and not refresh else {}
        return state["tables"][g]

    def client():
        if state["client"] is None:
            key = os.environ.get("MASSIVE_API_KEY")
            if not key:
                raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python breakout_check.py")
            state["client"] = RESTClient(api_key=key, retries=10)
        return state["client"]

    def fetch(ticker, start, end):
        try:
            aggs = list(client().list_aggs(ticker, 1, "second", int(start), int(end), sort="asc", limit=50_000))
        except BadResponse as e:
            if "NOT_AUTHORIZED" not in str(e):
                raise
            return None
        cols = [(a.timestamp, a.open, a.high, a.low, a.close) + ((a.volume,) if volume else ()) for a in aggs]
        return np.array(cols, float).reshape(-1, width)

    def load(ticker, start, end):
        key = (ticker, start, end)
        t = table(ticker)
        if key in state["denied"]:
            return np.zeros((0, width))
        if key not in t:
            arr = fetch(ticker, start, end)
            state["fetched"] += 1
            if arr is None:
                state["denied"].add(key)
                return np.zeros((0, width))
            t[key] = arr
            state["dirty"].add(group(ticker))
        return t[key]

    def prefetch(requests, workers=8):
        todo = sorted({r for r in requests if (r[0], r[1], r[2]) not in table(r[0])})
        if not todo:
            return
        client()
        with ThreadPoolExecutor(workers) as ex:
            for r, arr in zip(todo, ex.map(lambda r: fetch(*r), todo)):
                if arr is None:
                    state["denied"].add((r[0], r[1], r[2]))
                    continue
                table(r[0])[(r[0], r[1], r[2])] = arr
                state["dirty"].add(group(r[0]))
        state["fetched"] += len(todo)

    def save():
        folder.mkdir(parents=True, exist_ok=True)
        for g in state["dirty"]:
            (folder / f"{g}.pkl").write_bytes(pickle.dumps(state["tables"][g]))
        state["dirty"].clear()

    load.prefetch, load.save, load.state = prefetch, save, state
    return load


def load_chain(ticker, first, last, cache_dir="data/cache", refresh=False):
    """{expiry: sorted strikes} of listed puts expiring in [first, last], expired or not, from Massive's contracts
    reference (one query per week and status, so each answer is a single page). Calls are listed at the same
    strikes."""
    path = Path(cache_dir) / "options_chains" / f"{ticker}_all_puts_{first}_{last}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        client, rows = sds.make_client(), []
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        for monday in pd.date_range(start - pd.Timedelta(days=start.dayofweek), end, freq="7D"):
            lo, hi = max(monday, start), min(monday + pd.Timedelta(days=6), end)
            for expired in (True, False):
                rows += [(c.expiration_date, c.strike_price) for c in client.list_options_contracts(
                    underlying_ticker=ticker, contract_type="put", expiration_date_gte=str(lo.date()),
                    expiration_date_lte=str(hi.date()), expired=expired, limit=1000)]
        df = pd.DataFrame(rows, columns=["expiry", "strike"]).drop_duplicates()
        df = df.assign(expiry=pd.to_datetime(df["expiry"]).astype("datetime64[ns]"), strike=df["strike"].astype(float))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return {pd.Timestamp(e): np.sort(g["strike"].to_numpy(float)) for e, g in df.groupby("expiry")}


# ---------------------------------------------------------------- shares, second by second

def replay(minute_ms, high, low, close, e, price, d, stop, target, seconds, close_ms):
    """One share trade from minute e of its session on (minute starts in ms and highs/lows/closes, time order).
    seconds(minute start ms) -> that minute's one-second bars. The fill minute and every minute reaching the stop
    or the target are read second by second. Returns (fill, fill ms, exit, exit ms, reason: 1 target, -1 stop,
    0 close), or None when the seconds don't show the touch the minute bar shows."""
    up = d > 0

    def hit(h, l):
        return (l <= stop, h >= target) if up else (h >= stop, l <= target)

    fill = fill_ms = None
    for m in range(e, len(minute_ms)):
        if m > e and not any(hit(high[m], low[m])):
            continue
        sec = seconds(minute_ms[m])
        first = 0
        if m == e:
            reach = sec[:, 2] >= price if up else sec[:, 3] <= price
            if not reach.any():
                return None
            first = int(np.argmax(reach))
            fill = max(price, sec[first, 1]) if up else min(price, sec[first, 1])
            fill_ms = sec[first, 0]
        for k in range(first, len(sec)):
            ms, so, sh, sl = sec[k, :4]
            s_stop, s_target = hit(sh, sl)
            if s_stop:
                gapped = (m > e or k > first) and (so <= stop if up else so >= stop)
                return fill, fill_ms, so if gapped else stop, ms, -1
            if s_target:
                at_fill = m == e and k == first
                return fill, fill_ms, (max(target, fill) if up else min(target, fill)) if at_fill else target, ms, 1
    return fill, fill_ms, close[-1], close_ms, 0


def candidates(minutes, sessions, period):
    """Real and fake breakout entries (minute level) of the three rules in one period, with the session arrays."""
    rth, _ = features.regular_session_minutes(minutes, sessions)
    table = lv.level_table(rth, minutes, sessions)
    real = bo.rule_levels(lv.real_levels(table))
    real = real[real["rule"].isin(RULES.values()) & (real["period"] == period)].reset_index(drop=True)
    fakes = lv.fake_levels(real.drop(columns="rule"), table)
    rule_of = {(level, s): name for name, (level, sides) in bo.RULES.items() for s in sides}
    fakes["rule"] = [rule_of[(a, b)] for a, b in zip(fakes["level"], fakes["side"])]
    both = pd.concat([real, fakes], ignore_index=True)
    trades = bo.run_trades(rth, both, table, exits={"check": (WIDTH, WIDTH)})
    trades = trades[trades["fills"] == "worst"].reset_index(drop=True)  # the entry minute is the same either way
    trades["atr"] = table["atr"].reindex(trades["session"]).to_numpy()
    return trades, rth


def replay_all(trades, rth, seconds_of, ticker=TICKER):
    """Replays each trade (rows of `candidates`) second by second; adds fill, exit, their times and gross bps."""
    days = rth.groupby("session").indices
    ms, hi, lo, cl = _ns(rth["ts"]) // 1_000_000, *(rth[k].to_numpy(float) for k in ("high", "low", "close"))
    open_ms = _ns(rth["session_open"]) // 1_000_000
    load = getattr(seconds_of, "prefetch", None)
    if load is not None:  # the fill minute, plus the next minute that reaches the stop or the target
        want = []
        for r in trades.itertuples():
            b = days[r.session]
            e = int(np.searchsorted(ms[b], open_ms[b[0]] + int(r.entry_minute) * MINUTE_MS))
            want.append((ticker, int(ms[b][e]), int(ms[b][e]) + MINUTE_MS - 1))
            d = -r.side
            stop, target = r.price - d * WIDTH * r.atr, r.price + d * WIDTH * r.atr
            touch = (lo[b] <= stop) | (hi[b] >= target) if d > 0 else (hi[b] >= stop) | (lo[b] <= target)
            later = np.flatnonzero(touch[e + 1:])
            if later.size:
                m = e + 1 + later[0]
                want.append((ticker, int(ms[b][m]), int(ms[b][m]) + MINUTE_MS - 1))
        load(want)
    out = []
    for r in trades.itertuples():
        b = days[r.session]
        e = int(np.searchsorted(ms[b], open_ms[b[0]] + int(r.entry_minute) * MINUTE_MS))
        d = -r.side
        stop, target = r.price - d * WIDTH * r.atr, r.price + d * WIDTH * r.atr
        res = replay(ms[b], hi[b], lo[b], cl[b], e, r.price, d, stop, target,
                     lambda start: seconds_of(ticker, start, start + MINUTE_MS - 1), ms[b][-1] + MINUTE_MS - 1)
        if res is None:
            out.append((np.nan, np.nan, np.nan, np.nan, 0, "no seconds"))
            continue
        fill, fill_ms, exit_, exit_ms, reason = res
        out.append((fill, fill_ms, exit_, exit_ms, reason, "ok"))
    o = pd.DataFrame(out, columns=["fill", "fill_ms", "exit", "exit_ms", "reason", "status"], index=trades.index)
    t = trades.drop(columns=["fill", "exit", "reason", "gross", "exit_minute"]).join(o)
    t["gross"] = -t["side"] * (t["exit"] / t["fill"] - 1) * 1e4
    return t


# ---------------------------------------------------------------- options

def option_leg(day, d, price, chain):
    """(expiry, right, strike): the first listed expiry after `day`, the listed strike nearest `price`; a call for
    longs (d > 0), a put for shorts. None if no later expiry is listed."""
    later = [e for e in chain if e > day]
    if not later:
        return None
    e = min(later)
    k = chain[e]
    return e, "C" if d > 0 else "P", float(k[np.argmin(np.abs(k - price))])


def first_trade(sec, at_ms, window=WINDOW_MS):
    """Open of the first one-second bar at or after at_ms (within the window), else None."""
    k = int(np.searchsorted(sec[:, 0], at_ms))
    return sec[k, 1] if k < len(sec) and sec[k, 0] <= at_ms + window else None


def last_trade(sec, at_ms, window=WINDOW_MS):
    """Close of the last one-second bar before at_ms (within the window), else None."""
    k = int(np.searchsorted(sec[:, 0], at_ms)) - 1
    return sec[k, 4] if k >= 0 and sec[k, 0] >= at_ms - window else None


def option_trades(shares, chain, seconds_of, underlying=TICKER):
    """One option per replayed share trade from OPTIONS_START on: its contract, the entry and exit prices (before
    slippage) and why a trade could not be priced."""
    t = shares[(shares["status"] == "ok") & (shares["session"] >= OPTIONS_START)].copy()
    legs = [option_leg(r.session, -r.side, r.price, chain) for r in t.itertuples()]
    t["contract"] = [None if g is None else asp.option_ticker(g[0], g[1], g[2], underlying) for g in legs]
    t["expiry"] = [None if g is None else g[0] for g in legs]
    t["strike"] = [np.nan if g is None else g[2] for g in legs]
    load = getattr(seconds_of, "prefetch", None)
    if load is not None:
        load([(c, int(f), int(f) + WINDOW_MS) for c, f in zip(t["contract"], t["fill_ms"]) if c] +
             [(c, int(x) - WINDOW_MS, int(x) + WINDOW_MS) for c, x in zip(t["contract"], t["exit_ms"]) if c])
    denied = getattr(seconds_of, "state", {}).get("denied", set())
    entry, exit_, status = [], [], []
    for r in t.itertuples():
        if r.contract is None:
            entry.append(np.nan), exit_.append(np.nan), status.append("no listed expiry")
            continue
        windows = [(r.contract, int(r.fill_ms), int(r.fill_ms) + WINDOW_MS),
                   (r.contract, int(r.exit_ms) - WINDOW_MS, int(r.exit_ms) + WINDOW_MS)]
        if any(w in denied for w in windows):
            entry.append(np.nan), exit_.append(np.nan), status.append("outside the data plan")
            continue
        a = first_trade(seconds_of(r.contract, int(r.fill_ms), int(r.fill_ms) + WINDOW_MS), int(r.fill_ms))
        sec = seconds_of(r.contract, int(r.exit_ms) - WINDOW_MS, int(r.exit_ms) + WINDOW_MS)
        b = first_trade(sec, int(r.exit_ms))
        b = last_trade(sec, int(r.exit_ms)) if b is None else b
        if a is None or b is None:
            entry.append(np.nan), exit_.append(np.nan)
            status.append("no option trade near the entry" if a is None else "no option trade near the exit")
            continue
        entry.append(a), exit_.append(b), status.append("ok")
    return t.assign(opt_entry=entry, opt_exit=exit_, opt_status=status)


# ---------------------------------------------------------------- statistics

def share_stats(t, cost, years):
    s = bo.trade_stats(t, cost, years)
    net = t["gross"].to_numpy(float) - 2 * cost
    return {**s, "avg_win": net[net > 0].mean() if (net > 0).any() else np.nan,
            "avg_loss": net[net <= 0].mean() if (net <= 0).any() else np.nan}


def option_stats(o, slip, years):
    """Per contract: P&L after `slip` per share on each fill, with a session-clustered 95% interval and one-sided
    p, win rate, average premium, P&L as % of the premium, dollars a year and the worst run (one contract a trade)."""
    pnl = 100 * (o["opt_exit"].to_numpy(float) - o["opt_entry"].to_numpy(float) - 2 * slip)
    n = len(pnl)
    out = {"trades": n}
    if n < 2:
        return out
    m, se = sm.cluster_mean(pnl, o["session"].to_numpy())
    z = m / se if se else np.nan
    premium = 100 * (o["opt_entry"].to_numpy(float) + slip)
    cum = np.cumsum(pnl[np.argsort(o["fill_ms"].to_numpy())])
    return {**out, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "win": (pnl > 0).mean() * 100,
            "premium": premium.mean(), "pct_premium": (pnl / premium).mean() * 100, "per_year": pnl.sum() / years,
            "worst": pnl.min(), "drawdown": float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max())}


def run_check(minutes, sessions, chain, seconds_of, *, final=False, fake_sample=FAKE_SAMPLE, seed=SEED, timings=None):
    """The whole check for one period. No file I/O apart from the injected loaders' caches."""
    timings = {} if timings is None else timings
    period = "holdout" if final else "design"
    with timed(timings, "levels and entries"):
        cand, rth = candidates(minutes, sessions, period)
        real = cand[~cand["fake"]]
        fakes = cand[cand["fake"]]
        pick = np.random.default_rng(seed).choice(len(fakes), size=min(fake_sample, len(fakes)), replace=False)
        sample = fakes.iloc[np.sort(pick)]
    with timed(timings, "share replay"):
        shares = replay_all(pd.concat([real, sample]), rth, seconds_of)
    with timed(timings, "options"):
        opts = option_trades(shares[~shares["fake"]], chain, seconds_of)
    first = sessions.index[sessions.index >= (lv.HOLDOUT_START if final else shares["session"].min())][0]
    last = sessions.index[-1] if final else lv.HOLDOUT_START - pd.Timedelta(days=1)
    years = (last - first).days / 365.25
    opt_years = (last - max(first, OPTIONS_START)).days / 365.25
    rows, orows = [], []
    for label, rule in RULES.items():
        g = shares[(shares["rule"] == rule) & (shares["status"] == "ok")]
        for control, part in (("real", g[~g["fake"]]), ("fake sample", g[g["fake"]])):
            for cost in COSTS:
                rows.append({"rule": label, "control": control, "cost": cost, **share_stats(part, cost, years)})
        o = opts[(opts["rule"] == rule) & (opts["opt_status"] == "ok")]
        for slip in SLIPS:
            orows.append({"rule": label, "slip": slip, "skipped": int(((opts["rule"] == rule)
                                                                       & (opts["opt_status"] != "ok")).sum()),
                          **option_stats(o, slip, opt_years)})
    st, ost = pd.DataFrame(rows), pd.DataFrame(orows)
    for table in (st, ost):  # rules without enough trades have no statistics
        for col in ("mean", "lo", "hi", "p_up"):
            if col not in table:
                table[col] = np.nan
    tests = []
    for name, table, key in (("shares", st, (st["control"] == "real") & (st["cost"] == PRIMARY_COST)),
                             ("options", ost, ost["slip"] == PRIMARY_SLIP)):
        t = table[key].copy()
        t["q"] = mr.bh_qvalues(t["p_up"].fillna(1).to_numpy())
        t["passes"] = (t["q"] < FDR) & (t["mean"] > 0)
        tests.append(t.assign(instrument=name))
    return {"shares": shares, "options": opts, "share_stats": st, "option_stats": ost,
            "tests": pd.concat(tests, ignore_index=True), "final": final, "period": period, "years": years,
            "option_years": opt_years, "first": first, "last": last,
            "status": shares.groupby(["fake", "status"]).size().to_dict()}


# ---------------------------------------------------------------- report

def money(v):
    return "n/a" if v is None or pd.isna(v) else f"{'-' if v < 0 else '+'}${abs(v):,.0f}"


def render_report(res):
    st, ost, tests = res["share_stats"], res["option_stats"], res["tests"]
    bps, find = bo.bps, lv.find
    lines = []
    w = lines.append
    w("# TSLA breakout trades: " + ("the 2025-26 check" if res["final"] else "dry run on 2021-24") + "\n")
    w(f"{res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}. Rules and settings were fixed before any 2025-26 result "
      "was seen (see `breakout_check.py`). Shares: a stop order at the level, a stop and target 0.1 ATR either side, "
      "replayed second by second; P&L in basis points of the fill after costs (1 bp = $1 per $10,000 traded). "
      "Options: an at-the-money call (longs) or put (shorts), the first expiry after the trade date, bought and sold "
      "at the first option trades after the share fill and exit, plus slippage; P&L per contract (100 shares)."
      + ("" if res["final"] else " Option data starts 2024-10-01, so the dry run prices options for 2024-10 to 12 "
                                 "only.") + "\n")
    ok = tests[tests["passes"]]
    w(f"**Bottom line.** {len(ok)} of {len(tests)} tests passed (mean net P&L > 0, one-sided, BH q < {FDR:g} within "
      f"shares at {PRIMARY_COST} bp per side and within options at ${PRIMARY_SLIP:.2f} per fill)" + (": " + "; ".join(
          f"{r.instrument} {r.rule}" for r in ok.itertuples()) if len(ok) else ".") + "\n")
    w("## Shares\n")
    rows = []
    for label in RULES:
        r = find(st, rule=label, control="real", cost=PRIMARY_COST)
        f = find(st, rule=label, control="fake sample", cost=PRIMARY_COST)
        r0, r2 = (find(st, rule=label, control="real", cost=c) for c in (0, 2))
        t = find(tests, instrument="shares", rule=label)
        if r is None or pd.isna(r.get("mean")):
            rows.append([label, f"{int(r['trades']) if r is not None else 0}", *["n/a"] * 7])
            continue
        rows.append([label, f"{int(r['trades']):,} ({r['per_year']:.0f})", lv.pct(r["win"]),
                     f"{bps(r['avg_win'])} / {bps(r['avg_loss'])}",
                     f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})",
                     f"{bps(bo.val(r0, 'mean'))} / {bps(bo.val(r2, 'mean'))}", f"{t['q']:.3f}",
                     bps(bo.val(f, "mean")), f"{r['pct_year']:+.1f}%"])
    w(alerts.md_table(["Rule", "Trades (a year)", f"Wins at {PRIMARY_COST} bp", "Avg win / loss, bps",
                       f"Net bps a trade at {PRIMARY_COST} bp (95%)", "At 0 / 2 bp", "q",
                       f"Fake levels at {PRIMARY_COST} bp", "% a year (whole account per trade)"], rows) + "\n")
    w("## Options\n")
    rows = []
    for label in RULES:
        r = find(ost, rule=label, slip=PRIMARY_SLIP)
        t = find(tests, instrument="options", rule=label)
        cheap, wide = (find(ost, rule=label, slip=s) for s in (SLIPS[0], SLIPS[-1]))
        if r is None or pd.isna(r.get("mean")):
            rows.append([label, f"{int(r['trades']) if r is not None else 0}", *["n/a"] * 8])
            continue
        rows.append([label, f"{int(r['trades']):,} ({int(r['skipped'])} skipped)", money(r["premium"]),
                     lv.pct(r["win"]), f"{money(r['mean'])} ({money(r['lo'])} to {money(r['hi'])})",
                     f"{money(bo.val(cheap, 'mean'))} / {money(bo.val(wide, 'mean'))}",
                     f"{r['pct_premium']:+.1f}%", "" if t is None or pd.isna(t.get("q")) else f"{t['q']:.3f}",
                     money(r["per_year"]), f"{money(r['worst'])} / {money(-r['drawdown'])}"])
    w(alerts.md_table(["Rule", "Contracts priced", "Avg premium", f"Wins at ${PRIMARY_SLIP:.2f}",
                       f"Per contract at ${PRIMARY_SLIP:.2f} per fill (95%)",
                       f"At ${SLIPS[0]:.2f} / ${SLIPS[-1]:.2f}", "% of premium", "q",
                       "A year (1 contract a trade)", "Worst trade / worst run"], rows) + "\n")
    w("## Notes\n")
    counts = res["status"]
    w(f"- Share trades replayed: {counts.get((False, 'ok'), 0)} real, {counts.get((True, 'ok'), 0)} fake; without "
      f"the touch in their seconds: {counts.get((False, 'no seconds'), 0)} real, "
      f"{counts.get((True, 'no seconds'), 0)} fake (left out).")
    w("- Option prices are trades, not quotes (the data plan has no quotes): a trade can print at the bid or the "
      "ask, which the slippage settings stand in for. Option P&L includes no commissions or exchange fees "
      "(a few cents a contract).")
    w("- Shares: per share of TSLA, gross of commissions. Options: per contract, not combined with the shares.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    """Cumulative P&L of each rule: shares (% with the whole account in each trade, 1 bp per side) and options
    ($ with one contract a trade, $0.05 per fill)."""
    sh, op = res["shares"], res["options"]
    fig = plt.figure(figsize=(10, 4.4), dpi=150, facecolor=SURFACE)
    for k, (title, ylabel) in enumerate((("Shares", "Cumulative net %, whole account per trade"),
                                         ("Options", "Cumulative $, one contract a trade"))):
        ax = sds._axes(fig, [0.08 + k * 0.48, 0.2, 0.4, 0.6])
        for label, rule in RULES.items():
            if k == 0:
                g = sh[(sh["rule"] == rule) & ~sh["fake"] & (sh["status"] == "ok")].sort_values("fill_ms")
                y = ((g["gross"] - 2 * PRIMARY_COST) / 100).cumsum()
            else:
                g = op[(op["rule"] == rule) & (op["opt_status"] == "ok")].sort_values("fill_ms")
                y = (100 * (g["opt_exit"] - g["opt_entry"] - 2 * PRIMARY_SLIP)).cumsum()
            if len(g):
                ax.plot(g["session"], y, color=COLORS[label], lw=1.2, label=label if k == 0 else None)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_ylabel(ylabel, color=INK_2, fontsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=6.5)
    fig.text(0.02, 0.97, "TSLA breakout trades: " + ("2025-26 check" if res["final"] else "dry run 2021-24"),
             color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Stop and target 0.1 ATR from the level, fills replayed second by second; shares at "
             f"{PRIMARY_COST} bp per side, options at ${PRIMARY_SLIP:.2f} a share per fill.", color=INK_2,
             fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TSLA breakout trades with shares and options: dry run or the 2025-26 check.")
    p.add_argument("--out", default="output/breakout_check")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="run the one-time 2025-26 check")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        minutes, _ = gr.load_minutes(TICKER, sessions, args.cache_dir, args.refresh)
        first = lv.HOLDOUT_START if args.final_test else OPTIONS_START
        last = sessions.index[-1] + pd.Timedelta(days=14) if args.final_test else lv.HOLDOUT_START + pd.Timedelta(days=14)
        chain = load_chain(TICKER, f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}", args.cache_dir, args.refresh)
    seconds_of = second_loader(args.cache_dir, args.refresh)
    try:
        res = run_check(minutes, sessions, chain, seconds_of, final=args.final_test, timings=timings)
    finally:
        seconds_of.save()
    out = Path(args.out) / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["shares"].to_parquet(out / "shares.parquet", index=False)
    res["options"].to_parquet(out / "options.parquet", index=False)
    res["share_stats"].to_csv(out / "share_stats.csv", index=False)
    res["option_stats"].to_csv(out / "option_stats.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": res["period"], "first": str(res["first"].date()),
        "last": str(res["last"].date()), "seconds_fetched": seconds_of.state["fetched"]}, indent=2))
    plot(res, out / "check.png")
    (out / "report.md").write_text(render_report(res))
    print(f"tests passed: {int(res['tests']['passes'].sum())} of {len(res['tests'])}; "
          f"one-second requests: {seconds_of.state['fetched']}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
