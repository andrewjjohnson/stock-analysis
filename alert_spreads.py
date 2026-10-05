"""$1 SPY credit spreads on the intraday alerts, from Massive option minute bars: a small trade simulation.

  uv run --env-file .env python alert_spreads.py --csv data-files/<log>.csv --out output/alert_spreads

Each intraday alert (10:00 ET) sells a same-day $1 vertical spread with its short strike just
in the money. BULLISH sells the put at the first strike at or above SPY and buys the put $1
below; BEARISH sells the call at the first strike at or below SPY and buys the call $1 above.
NEUTRAL is no trade. The trader's rule (--take-profit, --stop; default an 80% take profit and
no stop) closes at its levels, otherwise at 15:30 ET. A grid of take profits (% of the credit)
and stops (% of the max loss) is also run, and every rule runs on the same days with the
same exits for always-bullish, always-bearish and shuffled calls. With --context (default),
both directions also run on every session the data plan covers, with no alerts at all.

Results are dollars per one spread (100 shares per contract), before taxes. The data plan
has minute bars of trades but no bid/ask quotes, so fills are approximations:
- The spread trades in a minute only if both legs traded in it. Entry and the 15:30 exit use
  the opens of the first such minute at or after that time (within 5 minutes). Take profits
  and stops are checked on the spread's value at the close of each such minute. Prices are
  never carried forward from earlier minutes.
- A take profit fills at its level, but only once the spread traded at least the slippage
  below it; a stop fills at the value that crossed it, plus slippage. Every leg of every fill
  pays --slippage per share and --commission per contract. Both default to 0: the user's
  broker charges no commission, and replaying their actual fills at traded prices matched
  them within about $1 a trade. Wider slippage is always shown alongside.
- SPY options are American; early assignment of the short leg is ignored.
- Strike placements deeper in the money (offset > 0) are priced from their out-of-the-money
  twins by put-call parity ($1 minus the other right's spread on the same strikes): the
  in-the-money options trade too thinly, and priced from their own prints they showed
  profits in both directions without any alerts.
"""

import argparse
import math
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from massive import RESTClient  # noqa: E402
from massive.exceptions import BadResponse  # noqa: E402

import alerts  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import outcomes  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
NS_MIN = outcomes.NS_PER_MINUTE
TICKER = "SPY"
WIDTH = 1.0
SHARES = 100
ENTRY, EXIT = "10:00", "15:30"
WINDOW_MIN = 5                 # entry and time exit need a both-legs minute within this many minutes
EXIT_BEFORE_CLOSE_MIN = 30     # early-close sessions: the time exit moves to 30 minutes before the close
TAKE_PROFITS = (10, 20, 30, 40, 50, 60, 70, 80, 90, None)  # % of the credit kept; None = no take profit
STOPS = (None, 10, 20, 30, 40, 50, 60, 70, 80, 90)          # % of the max loss; None = no stop
# The user's rule (--take-profit / --stop): 80% of the credit, no stop, since 2026-10-01 (50% before).
DEFAULT_RULE = (80, None)
DEFAULT_RULE_SINCE = "2026-10-01"  # when the user adopted DEFAULT_RULE; earlier trades were used to choose it
REFERENCE_RULES = ((50, None), (80, None))  # strike placements always show these next to the user's rule
BREAKEVEN_AT = 40  # tested on request: once profit reaches 40% of the credit, a stop at breakeven is armed
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)  # strike placement: +k = both legs $k deeper in the money (see spread_legs)
SIZES = (1, 5, 10, 20)               # contracts per trade in the scaling tables (--contracts)
HORIZONS_MONTHS = (3, 12)
BLOCK = 5                            # resampled trades stay in runs of 5 so clusters of losses survive
STREAK = 8                           # report the chance of this many losses in a row
SLIPPAGE_STEPS = (0.01, 0.03)  # sensitivity: extra slippage $/share per leg per fill on top of the setting


def cost_tiers(slippage=0.0, commission=0.0):
    """Cost assumptions, each (slippage $/share per leg per fill, commission $/contract per leg per
    fill): 'base' from the settings, wider-slippage sensitivities, and 'gross' when base has costs."""
    tiers = {"base": (slippage, commission),
             **{f"+${x:.2f} slippage": (slippage + x, commission) for x in SLIPPAGE_STEPS}}
    return tiers if slippage == commission == 0 else {"gross": (0.0, 0.0), **tiers}


def cost_text(cost):
    slippage, commission = cost
    if slippage == commission == 0:
        return "fills at traded prices, no commission"
    return (f"${slippage:.2f}/share slippage and ${commission:.2f}/contract commission per leg per fill, "
            f"${4 * slippage * SHARES + 4 * commission:.2f} a round trip")
SIDE_NAMES = {1: "always bullish", -1: "always bearish"}
STATUS_OK = "ok"


class NotInPlan(Exception):
    """The data plan does not cover this date."""


# ---------------------------------------------------------------- contracts and data

def spread_legs(side, spot, offset=0):
    """(right, short strike, long strike) of the $1 spread. Offset 0 puts the short leg at or just in
    the money; each +1 moves both legs $1 deeper in the money (more credit, smaller max loss, needs
    the move), each -1 moves them $1 further out of the money."""
    spot = round(float(spot), 2)
    if side > 0:
        k = math.ceil(spot) + offset
        return "P", k, k - 1
    k = math.floor(spot) - offset
    return "C", k, k + 1


def option_ticker(day, right, strike, underlying=TICKER):
    """Massive option ticker, e.g. O:SPY260610P00761000 for the 761 put expiring 2026-06-10."""
    return f"O:{underlying}{pd.Timestamp(day):%y%m%d}{right}{round(strike * 1000):08d}"


def fetch_option_minutes(ticker, day, client):
    """Every minute bar Massive has for one contract on one day (bar starts, UTC). A day fits in a
    single page, so the SDK's silent stop between pages cannot truncate it."""
    try:
        aggs = list(client.list_aggs(ticker, 1, "minute", day, day, limit=50_000))
    except BadResponse as e:
        if "NOT_AUTHORIZED" in str(e):
            raise NotInPlan(f"{ticker} {day}") from None
        raise
    df = pd.DataFrame([(a.timestamp, a.open, a.high, a.low, a.close, a.volume) for a in aggs],
                      columns=["t", "open", "high", "low", "close", "volume"])
    df.insert(0, "ts", pd.to_datetime(df.pop("t"), unit="ms", utc=True).dt.as_unit("ns"))
    return df


def option_loader(cache_dir, refresh=False):
    """load(ticker, day) -> minute bars, from the Parquet cache when present. The Massive client
    is created only on the first cache miss. An empty result (no trades) is cached too."""
    state = {"client": None, "fetched": 0}

    def load(ticker, day):
        path = Path(cache_dir) / "options" / f"{ticker[2:]}_{day}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        if state["client"] is None:
            key = os.environ.get("MASSIVE_API_KEY")
            if not key:
                raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python alert_spreads.py ...")
            state["client"] = RESTClient(api_key=key, retries=10)
        df = fetch_option_minutes(ticker, day, state["client"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        state["fetched"] += 1
        return df

    load.state = state
    return load


# ---------------------------------------------------------------- one spread

def spread_path(short, long, session_open, session_close):
    """Minutes in which both legs traded: (minute start ns, spread open value, spread close value)."""
    legs = []
    for df in (short, long):
        t = outcomes._ns(df["ts"]) if len(df) else np.array([], dtype=np.int64)
        keep = (t >= session_open) & (t < session_close)
        legs.append((t[keep], df["open"].to_numpy(float)[keep], df["close"].to_numpy(float)[keep]))
    (ts, so, sc), (tl, lo, lc) = legs
    both, i, j = np.intersect1d(ts, tl, assume_unique=True, return_indices=True)
    return both, so[i] - lo[j], sc[i] - lc[j]


def parity_legs(right, short_k, long_k):
    """(twin right, a, b) with spread value = width - (a - b): the same strikes in the other right.
    Bull put (short high, long low) = 1 - (C low - C high); bear call = 1 - (P high - P low)."""
    lo, hi = sorted((short_k, long_k))
    return ("C", lo, hi) if right == "P" else ("P", hi, lo)


def prepare_trade(day, side, spot, session, load, offset=0, parity=False):
    """Legs, entry and time-exit minutes for one day and direction (no exit rule applied yet).
    parity=True prices the spread from its out-of-the-money twin instead of its own legs."""
    right, short_k, long_k = spread_legs(side, spot, offset)
    trade = {"day": day, "side": side, "spot": spot, "right": right, "short_strike": short_k, "long_strike": long_k,
             "short_ticker": option_ticker(day, right, short_k), "long_ticker": option_ticker(day, right, long_k),
             "priced_from": "parity twin" if parity else "own legs"}
    date = str(day.date())
    twin, a_k, b_k = parity_legs(right, short_k, long_k) if parity else (right, short_k, long_k)
    try:
        first, second = load(option_ticker(day, twin, a_k), date), load(option_ticker(day, twin, b_k), date)
    except NotInPlan:
        return {**trade, "status": "not in data plan"}
    o, c = outcomes._ns(pd.Series([session["open"]]))[0], outcomes._ns(pd.Series([session["close"]]))[0]
    both, v_open, v_close = spread_path(first, second, o, c)
    if parity:
        v_open, v_close = WIDTH - v_open, WIDTH - v_close
    short, long = first, second  # bar counts below describe the legs actually priced
    entry_ns = outcomes._ns(pd.Series([pd.Timestamp(f"{date} {ENTRY}", tz=NY)]))[0]
    exit_ns = min(outcomes._ns(pd.Series([pd.Timestamp(f"{date} {EXIT}", tz=NY)]))[0],
                  c - EXIT_BEFORE_CLOSE_MIN * NS_MIN)
    e = int(np.searchsorted(both, entry_ns))
    if e == len(both) or both[e] >= entry_ns + WINDOW_MIN * NS_MIN:
        return {**trade, "status": "no entry: legs did not both trade by 10:05"}
    credit = v_open[e]
    if not 0 < credit < WIDTH:
        return {**trade, "status": "no entry: traded credit outside 0-1", "credit_traded": credit}
    end = int(np.searchsorted(both, exit_ns))  # watch minutes e .. end-1 (they start before the time exit)
    if end == len(both) or both[end] >= min(exit_ns + WINDOW_MIN * NS_MIN, c):
        # Every rule needs the time-exit price unless a level is hit first; requiring it keeps the
        # same days in every comparison.
        return {**trade, "status": "no exit: legs did not both trade by 15:35", "credit_traded": credit}
    return {**trade, "status": STATUS_OK, "credit_traded": credit, "entry_time": both[e], "exit_time": both[end],
            "watch": v_close[e:end], "watch_times": both[e:end], "exit_value": v_open[end],
            "short_bars": len(short), "long_bars": len(long), "both_minutes": len(both)}


def simulate(trade, take_profit, stop, slippage, commission, breakeven_at=None):
    """One spread under one exit rule. Returns (P&L $ per spread, exit reason, credit received).

    breakeven_at: once a minute close shows profit >= that % of the credit, a stop at breakeven is
    armed; a later close at or above the credit exits there (stop-market: the fill is that close,
    so a jump past breakeven still loses), plus slippage."""
    credit = trade["credit_traded"] - 2 * slippage  # sold both legs, each giving up the slippage
    watch = trade["watch"]
    first = len(watch)
    reason, paid = None, np.nan
    if take_profit is not None:
        level = credit * (1 - take_profit / 100)
        hit = np.flatnonzero(watch <= level - 2 * slippage)  # a buy limit at `level` needs trades through it
        if hit.size and hit[0] < first:
            first, reason, paid = hit[0], "take profit", level
    if stop is not None:
        level = credit + stop / 100 * (WIDTH - credit)
        hit = np.flatnonzero(watch >= level)
        if hit.size and hit[0] < first:
            first, reason, paid = hit[0], "stop", min(watch[hit[0]], WIDTH) + 2 * slippage
    if breakeven_at is not None:
        armed = np.flatnonzero(watch <= credit * (1 - breakeven_at / 100))
        if armed.size:
            back = np.flatnonzero(watch[armed[0] + 1:] >= credit)
            if back.size and armed[0] + 1 + back[0] < first:
                first = armed[0] + 1 + back[0]
                reason, paid = "breakeven stop", min(watch[first], WIDTH) + 2 * slippage
    if reason is None:
        reason, paid = EXIT, min(max(trade["exit_value"], 0.0), WIDTH) + 2 * slippage
    return (credit - paid) * SHARES - 4 * commission, reason, credit


# ---------------------------------------------------------------- statistics

def max_drawdown(pnl):
    equity = np.cumsum(pnl)
    return float((np.maximum.accumulate(np.r_[0, equity])[1:] - equity).max()) if len(pnl) else np.nan


def consistency(pnl):
    """Average trade divided by its standard error: how many standard errors the average sits above
    zero. High when profits are both positive on average and steady from trade to trade."""
    pnl = np.asarray(pnl, float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else np.nan
    return pnl.mean() / (sd / math.sqrt(len(pnl))) if len(pnl) > 1 and sd > 0 else np.nan


def describe(pnl, reps, seed, days=None):
    """Trades, total, mean per trade with a bootstrap interval over dates, win rate, drawdown and
    consistency measures (with `days`, also profitable months)."""
    pnl = np.asarray(pnl, float)
    n, wins = len(pnl), int((pnl > 0).sum())
    lo, hi = alerts.bootstrap(n, lambda i: pnl[i].mean(axis=1), reps, seed)
    wlo, whi = alerts.wilson(wins, n)
    losses = -pnl[pnl < 0].sum()
    out = {"trades": n, "total": pnl.sum(), "mean": pnl.mean() if n else np.nan, "mean_lo": lo, "mean_hi": hi,
           "wins": wins, "win_rate": wins / n * 100 if n else np.nan, "win_lo": wlo, "win_hi": whi,
           "max_drawdown": max_drawdown(pnl), "std": pnl.std(ddof=1) if n > 1 else np.nan,
           "consistency": consistency(pnl), "worst_trade": pnl.min() if n else np.nan,
           "profit_factor": pnl[pnl > 0].sum() / losses if losses > 0 else np.inf}
    if days is not None and n:
        monthly = pd.Series(pnl).groupby(pd.DatetimeIndex(days).strftime("%Y-%m")).sum()
        out.update(months=len(monthly), months_positive=int((monthly > 0).sum()), worst_month=monthly.min())
    return out


def leave_one_month_out(cell_pnl, days, current=DEFAULT_RULE):
    """For each month, pick the most consistent rule from the other months only, then record what
    it and the current rule made in the held-out month. If re-optimizing does not beat the current
    rule on months it never saw, the in-sample winner's edge is mostly selection luck."""
    months = np.asarray(pd.DatetimeIndex(days).strftime("%Y-%m"))
    rows = []
    for m in sorted(set(months)):
        test = months == m
        picked = max(cell_pnl, key=lambda c: np.nan_to_num(consistency(cell_pnl[c][~test]), nan=-np.inf))
        hindsight = max(cell_pnl, key=lambda c: cell_pnl[c][test].sum())
        rows.append({"month": m, "trades": int(test.sum()), "picked_take_profit": picked[0], "picked_stop": picked[1],
                     "picked_pnl": cell_pnl[picked][test].sum(), "current_pnl": cell_pnl[current][test].sum(),
                     "hindsight_take_profit": hindsight[0], "hindsight_stop": hindsight[1],
                     "hindsight_pnl": cell_pnl[hindsight][test].sum()})
    return pd.DataFrame(rows)


def shuffled_totals(sides, bull, bear, reps, seed):
    """Total P&L of the same calls reassigned across the same days, keeping the bullish/bearish counts."""
    perm = sides[alerts.shuffled_orders(len(sides), reps, seed)]
    return np.where(perm > 0, bull, bear).sum(axis=1)


# ---------------------------------------------------------------- the study

def run_study(alert_rows, minutes, sessions, today, load, *, rule=DEFAULT_RULE, rule_since=None, costs=None,
              context_start=None, offsets=OFFSETS, sizes=SIZES, reps=10_000, seed=0, timings=None):
    """Trades for every alert day (both directions) and, with context_start, every covered session.
    costs: cost_tiers(...); the default is no slippage and no commission. No file I/O beyond the
    option loader's cache."""
    timings = {} if timings is None else timings
    costs = cost_tiers() if costs is None else costs
    today = pd.Timestamp(today)
    a = alerts.check_rows(alert_rows, sessions)
    a = a[a["strategy"] == "intraday"].reset_index(drop=True)
    rth, _ = features.regular_session_minutes(minutes, sessions)

    with timed(timings, "option data"):
        days = set(a["session"])
        if context_start is not None:
            days |= set(sessions.index[(sessions.index >= pd.Timestamp(context_start)) & (sessions.index < today)])
        days = sorted(d for d in days if d in sessions.index)
        at_entry = [pd.Timestamp(f"{d.date()} {ENTRY}", tz=NY).tz_convert("UTC") for d in days]
        spots = dict(zip(days, alerts.price_known_at(rth, at_entry)))
        trades = {}
        for d in days:
            for side in (1, -1):
                if d >= today:
                    trades[d, side] = {"day": d, "side": side, "status": "pending"}
                elif np.isnan(spots[d]):
                    trades[d, side] = {"day": d, "side": side, "status": "no SPY price at 10:00"}
                else:
                    trades[d, side] = prepare_trade(d, side, spots[d], sessions.loc[d], load)
        offset_trades, itm_prints = {}, {}  # in-the-money placements: parity-priced, and own prints for comparison
        for d in days if offsets else ():
            for side in (1, -1):
                for k in offsets:
                    t = trades[d, side]
                    if k == 0 or t["status"] in ("pending", "no SPY price at 10:00"):
                        offset_trades[d, side, k] = t
                        continue
                    offset_trades[d, side, k] = prepare_trade(d, side, spots[d], sessions.loc[d], load, k, parity=k > 0)
                    if k > 0:
                        itm_prints[d, side, k] = prepare_trade(d, side, spots[d], sessions.loc[d], load, k)

    with timed(timings, "simulation"):
        ok = lambda d, s: trades[d, s]["status"] == STATUS_OK  # noqa: E731
        directional = a[a["side"] != 0]
        usable = directional[np.array([ok(d, 1) and ok(d, -1) for d in directional["session"]], dtype=bool)]
        days_used = list(usable["session"])
        sides = usable["side"].to_numpy(int)

        def pnl(day_side, rule, cost):
            return np.array([simulate(trades[k], *rule, *costs[cost])[0] for k in day_side])

        grid, user, cell_pnl = [], {}, {}
        wider = next(k for k in costs if k not in ("base", "gross"))  # every cell also runs with wider slippage
        for tp in TAKE_PROFITS:
            for sl in STOPS:
                this = (tp, sl)
                for cost in (costs if this == rule else ("base", wider)):
                    bull = pnl([(d, 1) for d in days_used], this, cost)
                    bear = pnl([(d, -1) for d in days_used], this, cost)
                    calls = np.where(sides > 0, bull, bear)
                    sims = shuffled_totals(sides, bull, bear, reps, seed)
                    row = {"take_profit": tp, "stop": sl, "cost": cost}
                    for name, values in (("alerts", calls), ("always bullish", bull), ("always bearish", bear)):
                        grid.append({**row, "strategy": name, **describe(values, reps, seed, days_used)})
                    if cost == "base":
                        cell_pnl[this] = calls
                    gaps = {}
                    for name, base in (("bullish", bull), ("bearish", bear)):
                        diff = calls - base
                        lo, hi = alerts.bootstrap(len(diff), lambda i, d=diff: d[i].mean(axis=1), reps, seed)
                        gaps.update({f"alerts_minus_{name}_mean": diff.mean(), f"alerts_minus_{name}_lo": lo,
                                     f"alerts_minus_{name}_hi": hi})
                    grid.append({**row, "strategy": "shuffled calls", "trades": len(calls), "total": sims.mean(),
                                 "total_p5": np.percentile(sims, 5), "total_p95": np.percentile(sims, 95),
                                 "p_alerts_or_better": (1 + (sims >= calls.sum() - 1e-9).sum()) / (reps + 1), **gaps})
                    if this == rule:
                        user[cost] = {"calls": calls, "bull": bull, "bear": bear, "sims_seed": seed}
        grid = pd.DataFrame(grid)
        monthly_check = leave_one_month_out(cell_pnl, days_used, rule)

        # Flag sensitivity for the user's rule (base costs).
        variants = {}
        for name, rows, col in (("as recorded", a, "side"), ("flagged rows excluded", a[a["flag"] == ""], "side"),
                                ("alternative calls from notes", a.assign(side=a["alt_direction"].map(alerts.SIDES)),
                                 "side")):
            r = rows[(rows[col] != 0).to_numpy() & np.array([ok(d, 1) and ok(d, -1) for d in rows["session"]])]
            values = np.array([simulate(trades[d, s], *rule, *costs["base"])[0]
                               for d, s in zip(r["session"], r[col])])
            variants[name] = describe(values, reps, seed)

        context = None
        if context_start is not None:
            cdays = [d for d in days if d < today and d >= pd.Timestamp(context_start) and ok(d, 1) and ok(d, -1)]
            rows = []
            for tp in TAKE_PROFITS:
                for sl in STOPS:
                    for side in (1, -1):
                        values = pnl([(d, side) for d in cdays], (tp, sl), "base")
                        rows.append({"take_profit": tp, "stop": sl, "strategy": SIDE_NAMES[side],
                                     **describe(values, min(reps, 2000), seed, cdays)})
            context = {"grid": pd.DataFrame(rows), "days": cdays,
                       "user": {(side, cost): pnl([(d, side) for d in cdays], rule, cost)
                                for side in (1, -1) for cost in costs}}

        placement, offset_rules = None, tuple(dict.fromkeys((rule, *REFERENCE_RULES)))
        if offsets:
            cdays = [d for d in days if d < today and (context_start is None or d >= pd.Timestamp(context_start))]
            placement = strike_offsets(offset_trades, a, cdays if context_start is not None else [], offsets, costs,
                                       reps, seed, itm_prints, offset_rules)

        breakeven = breakeven_table(trades, days_used, sides, tuple(dict.fromkeys((rule, *REFERENCE_RULES))),
                                    costs, wider, context["days"] if context else [], reps, seed)
        chosen_here = rule_since is not None and (pd.DatetimeIndex(days_used) < pd.Timestamp(rule_since)).any()
        scenarios = scaling_scenarios(user["base"]["calls"], user[wider]["calls"], wider,
                                      monthly_check if chosen_here else None)
        span = max((pd.Timestamp(days_used[-1]) - pd.Timestamp(days_used[0])).days / 30.44, 1.0)  # months
        scaling = scaling_table(scenarios, len(days_used) / span, reps, seed)

    return {"alerts": a, "trades": trades, "days_used": days_used, "sides": sides, "grid": grid, "user": user,
            "placement": placement, "rule": rule, "offset_rules": offset_rules, "rule_since": rule_since,
            "breakeven": breakeven,
            "variants": variants, "context": context, "today": today, "reps": reps, "seed": seed,
            "context_start": context_start, "costs": costs, "cell_pnl": cell_pnl, "monthly_check": monthly_check,
            "wider_cost": wider, "scaling": scaling, "scenarios": scenarios, "sizes": tuple(sizes),
            "trades_per_month": len(days_used) / span}


def strike_offsets(otrades, alert_rows, context_days, offsets, costs, reps, seed, itm_prints=None,
                   rules=REFERENCE_RULES):
    """Every strike offset under each of `rules`: the alert days (each offset on the days both of its
    directions have prices, with shuffled calls and the offset-0 result on the days it loses) and,
    with context days, each direction every session without alerts. itm_prints: the in-the-money
    placements priced from their own prints, reported next to the parity prices as a check."""
    itm_prints = itm_prints or {}

    def own_prints_mean(pairs, k, rule, slip, comm):
        values = [simulate(itm_prints[d, s, k], *rule, slip, comm)[0] for d, s in pairs
                  if (d, s, k) in itm_prints and itm_prints[d, s, k]["status"] == STATUS_OK]
        return np.mean(values) if values else np.nan

    ok = lambda d, s, k: otrades[d, s, k]["status"] == STATUS_OK  # noqa: E731
    directional = alert_rows[alert_rows["side"] != 0]
    side_of = dict(zip(directional["session"], directional["side"]))
    days0 = [d for d in side_of if ok(d, 1, 0) and ok(d, -1, 0)]
    rows = []
    for k in offsets:
        days_k = [d for d in side_of if ok(d, 1, k) and ok(d, -1, k)]
        sides_k = np.array([side_of[d] for d in days_k], dtype=int)
        dropped = [d for d in days0 if d not in set(days_k)]
        for rule in rules:
            for cost, (slip, comm) in costs.items():
                pnl = lambda pairs: np.array([simulate(otrades[d, s, k], *rule, slip, comm)[0] for d, s in pairs])  # noqa: E731
                calls = pnl(zip(days_k, sides_k))
                credit = np.array([otrades[d, s, k]["credit_traded"] for d, s in zip(days_k, sides_k)]) - 2 * slip
                risk = (WIDTH - credit) * SHARES
                n = len(calls)
                row = {"scope": "alerts", "offset": k, "take_profit": rule[0], "stop": rule[1], "cost": cost,
                       **describe(calls, reps, seed, days_k), "credit": credit.mean() if n else np.nan,
                       "max_loss": risk.mean() if n else np.nan,
                       "return_on_risk": calls.sum() / risk.sum() * 100 if n else np.nan}
                if cost == "base":
                    bull, bear = pnl((d, 1) for d in days_k), pnl((d, -1) for d in days_k)
                    sims = shuffled_totals(sides_k, bull, bear, reps, seed)
                    dropped_pnl = pnl((d, side_of[d]) for d in dropped) if k == 0 else np.array(
                        [simulate(otrades[d, side_of[d], 0], *rule, slip, comm)[0] for d in dropped])
                    same = np.array([simulate(otrades[d, side_of[d], 0], *rule, slip, comm)[0] for d in days_k])
                    row["mean_own_prints"] = own_prints_mean(list(zip(days_k, sides_k)), k, rule, slip, comm)
                    row.update(p_shuffled=(1 + (sims >= calls.sum() - 1e-9).sum()) / (reps + 1),
                               always_bullish=bull.sum(), always_bearish=bear.sum(), days_missing=len(dropped),
                               offset0_pnl_on_missing_days=dropped_pnl.sum(),
                               offset0_mean_same_days=same.mean() if len(same) else np.nan,
                               # Bound: every missing day counted as a full max loss at this placement.
                               mean_if_missing_all_max_loss=(calls.sum() - len(dropped) * risk.mean())
                               / (n + len(dropped)) if n else np.nan)
                rows.append(row)
        for side in (1, -1):
            cd = [d for d in context_days if ok(d, side, k)]
            for rule in rules:
                for cost in list(costs)[:2]:
                    slip, comm = costs[cost]
                    values = np.array([simulate(otrades[d, side, k], *rule, slip, comm)[0] for d in cd])
                    risk = (WIDTH - np.array([otrades[d, side, k]["credit_traded"] for d in cd]) + 2 * slip) * SHARES
                    rows.append({"scope": f"no alerts: {SIDE_NAMES[side]}", "offset": k, "take_profit": rule[0],
                                 "stop": rule[1], "cost": cost, **describe(values, min(reps, 2000), seed, cd),
                                 "mean_own_prints": own_prints_mean([(d, side) for d in cd], k, rule, slip, comm),
                                 "max_loss": risk.mean() if len(cd) else np.nan,
                                 "return_on_risk": values.sum() / risk.sum() * 100 if len(cd) else np.nan})
    return pd.DataFrame(rows)


def scaling_scenarios(calls, wider_calls, wider, monthly_check=None):
    """Per-spread trade sequences (in date order) for the scaling tables: the rule as measured; an
    honest estimate when the rule was chosen on these trades (shifted to the month-by-month
    re-picking average); the wider-slippage fills; and no edge at all (shifted to a $0 average)."""
    out = {"as measured": calls}
    if monthly_check is not None and len(monthly_check):
        honest = monthly_check["picked_pnl"].sum() / monthly_check["trades"].sum()
        out[f"honest estimate (${honest:.2f})"] = calls - calls.mean() + honest
    out[f"with {wider}"] = wider_calls
    out["no edge ($0 average)"] = calls - calls.mean()
    return out


def resample_paths(pnl, n, reps, seed, block=BLOCK):
    """`reps` paths of `n` trades: a moving-block bootstrap of the trade sequence (runs of `block`
    consecutive trades, wrapping at the end), so streaks and clusters of losses are kept."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(pnl), (reps, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)) % len(pnl)
    return np.asarray(pnl, float)[idx.reshape(reps, -1)[:, :n]]


def drawdown_anatomy(path):
    """(max drawdown, trades from peak to trough, losers, winners, longest losing streak inside)."""
    eq = np.r_[0, np.cumsum(path)]
    peak = np.maximum.accumulate(eq)
    trough = int(np.argmax(peak - eq))
    start = int(np.flatnonzero(eq[:trough + 1] == peak[trough])[-1])
    inside = np.asarray(path)[start:trough]
    return peak[trough] - eq[trough], len(inside), int((inside < 0).sum()), int((inside > 0).sum()), \
        longest_losing_streak(inside[None, :])[0] if len(inside) else 0


def longest_losing_streak(paths):
    loss, run = np.asarray(paths) < 0, np.zeros(len(paths), int)
    longest = np.zeros(len(paths), int)
    for j in range(loss.shape[1]):
        run = np.where(loss[:, j], run + 1, 0)
        longest = np.maximum(longest, run)
    return longest


def scaling_table(scenarios, trades_per_month, reps, seed):
    """Resampled results per spread over each horizon and scenario; dollars scale with contracts."""
    rows = []
    for months in HORIZONS_MONTHS:
        n = int(round(trades_per_month * months))
        for name, pnl in scenarios.items():
            paths = resample_paths(pnl, n, reps, seed)
            total = paths.sum(axis=1)
            eq = np.cumsum(paths, axis=1)
            dd = (np.maximum.accumulate(np.c_[np.zeros(reps), eq], axis=1)[:, 1:] - eq).max(axis=1)
            dd = np.maximum(dd, 0)
            streak = longest_losing_streak(paths)
            bad = np.flatnonzero(dd >= np.percentile(dd, 95))
            anatomy = np.array([drawdown_anatomy(paths[i])[1:] for i in bad])
            rows.append({"months": months, "trades": n, "scenario": name, "mean_trade": pnl.mean(),
                         "median_total": np.median(total), "p5_total": np.percentile(total, 5),
                         "p_loss": (total < 0).mean() * 100, "median_drawdown": np.median(dd),
                         "p95_drawdown": np.percentile(dd, 95), "p95_streak": np.percentile(streak, 95),
                         f"p_streak_{STREAK}": (streak >= STREAK).mean() * 100,
                         "bad_dd_trades": np.median(anatomy[:, 0]), "bad_dd_losers": np.median(anatomy[:, 1]),
                         "bad_dd_winners": np.median(anatomy[:, 2]), "bad_dd_streak": np.median(anatomy[:, 3])})
    return pd.DataFrame(rows)


def breakeven_table(trades, days_used, sides, rules, costs, wider, context_days, reps, seed):
    """Each rule with and without the breakeven stop, trade by trade on the same spreads: on the alert
    days (alerts, always bullish, always bearish) and on every context session (both directions)."""
    scopes = [("alerts", list(zip(days_used, sides)), ("base", wider)),
              ("always bullish", [(d, 1) for d in days_used], ("base",)),
              ("always bearish", [(d, -1) for d in days_used], ("base",)),
              ("no alerts: bull put every session", [(d, 1) for d in context_days], ("base",)),
              ("no alerts: bear call every session", [(d, -1) for d in context_days], ("base",))]
    rows = []
    for scope, keys, cost_names in scopes:
        if not keys:
            continue
        for rule in rules:
            for cost in cost_names:
                plain = [simulate(trades[k], *rule, *costs[cost]) for k in keys]
                stopped = [simulate(trades[k], *rule, *costs[cost], breakeven_at=BREAKEVEN_AT) for k in keys]
                a = np.array([x[0] for x in plain])
                b = np.array([x[0] for x in stopped])
                hit = np.array([x[1] == "breakeven stop" for x in stopped])
                saved, cut = hit & (a < 0), hit & (a > 0)
                diff = b - a
                lo, hi = alerts.bootstrap(len(diff), lambda i: diff[i].mean(axis=1), reps, seed)
                rows.append({"scope": scope, "take_profit": rule[0], "stop": rule[1], "cost": cost, "trades": len(a),
                             "mean_without": a.mean(), "mean_with": b.mean(), "diff_lo": lo, "diff_hi": hi,
                             "win_without": (a > 0).mean() * 100, "win_with": (b > 0).mean() * 100,
                             "worst_without": a.min(), "worst_with": b.min(), "stopped": int(hit.sum()),
                             "saved": int(saved.sum()), "saved_without": a[saved].sum(), "saved_with": b[saved].sum(),
                             "cut": int(cut.sum()), "cut_without": a[cut].sum(), "cut_with": b[cut].sum(),
                             "losers_without": int((a < 0).sum()), "losers_with": int((b < 0).sum())})
    return pd.DataFrame(rows)


def breakeven_note(result):
    """Report rows: the user's rule (and the reference rules) with vs without the breakeven stop."""
    t = result["breakeven"]
    if t is None or t.empty:
        return ""
    rule = result["rule"]
    rows = []
    for r in t.itertuples():
        this = (None if pd.isna(r.take_profit) else int(r.take_profit), None if pd.isna(r.stop) else int(r.stop))
        if r.scope != "alerts" and this != rule:
            continue
        label = f"{r.scope}, {rule_label(*this)}" + (" (your rule)" if this == rule else "") + \
            ("" if r.cost == "base" else f", {r.cost}")
        verdict = "helped" if r.diff_lo > 0 else "hurt" if r.diff_hi < 0 else "no clear difference"
        rows.append([label, r.trades, f"{money(r.mean_without, cents=True)} → {money(r.mean_with, cents=True)}",
                     f"{money(r.mean_with - r.mean_without, cents=True)} ({money(r.diff_lo, cents=True)} to "
                     f"{money(r.diff_hi, cents=True)}): {verdict}", f"{r.win_without:.0f}% → {r.win_with:.0f}%",
                     f"{r.losers_without} → {r.losers_with}", r.stopped,
                     f"{r.saved}: {money(r.saved_without)} → {money(r.saved_with)}",
                     f"{r.cut}: {money(r.cut_without)} → {money(r.cut_with)}"])
    return (f"\n**Breakeven stop after {BREAKEVEN_AT}% profit** (tested on request): once a minute close shows the "
            f"spread worth {100 - BREAKEVEN_AT}% of the credit or less, a stop at the credit is armed; a later close at "
            "or above the credit exits there, filled at that close. Same spreads with and without it:\n\n" +
            md_table(["Trades", "Spreads", "Average P&L without → with", "Difference per spread (95% interval)",
                      "Winners", "Losers", "Stopped at breakeven", "Losers it saved (total)",
                      "Winners it cut short (total)"], rows) + "\n")


def trade_table(result):
    """trades.csv: one row per intraday alert, with both directions under the user's rule (base costs)."""
    rows = []
    for r in result["alerts"].itertuples():
        row = {"date": r.date, "direction": r.direction, "flag": r.flag, "alt_direction": r.alt_direction}
        for side, label in ((1, "bull"), (-1, "bear")):
            t = result["trades"].get((r.session, side), {"status": "no row"})
            row[f"{label}_status"] = t["status"]
            row[f"{label}_short"], row[f"{label}_long"] = t.get("short_ticker"), t.get("long_ticker")
            if t["status"] == STATUS_OK:
                pnl, reason, credit = simulate(t, *result["rule"], *result["costs"]["base"])
                row.update({f"{label}_credit": round(credit, 4), f"{label}_exit": reason, f"{label}_pnl": pnl})
        row["spot_1000"] = result["trades"].get((r.session, 1), {}).get("spot")
        row["alert_pnl"] = (np.nan if r.side == 0 else row.get("bull_pnl" if r.side > 0 else "bear_pnl", np.nan))
        row["used_in_comparisons"] = r.session in set(result["days_used"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts

DIVERGING = LinearSegmentedColormap.from_list("pnl", [SERIES[1], "#f3f2ee", SERIES[0]])  # loss orange, gain blue


def _axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7.5)
    return ax


def plot_equity(result, path):
    """Cumulative P&L per spread under the user's rule: alerts, always bullish/bearish, shuffled-call band."""
    u, days = result["user"]["base"], pd.DatetimeIndex(result["days_used"])
    sims = result["sides"][alerts.shuffled_orders(len(days), 2000, result["seed"])]
    sim_curves = np.cumsum(np.where(sims > 0, u["bull"], u["bear"]), axis=1)
    fig = plt.figure(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.1, 0.2, 0.86, 0.58])
    ax.fill_between(days, np.percentile(sim_curves, 5, axis=0), np.percentile(sim_curves, 95, axis=0), color=GRID,
                    lw=0, label="Shuffled calls, 5th–95th percentile", zorder=1)
    ax.axhline(0, color=BASELINE, lw=1, zorder=2)
    for values, color, label in ((u["calls"], SERIES[0], "Following the alerts"),
                                 (u["bull"], SERIES[1], "Bull put spread every day"),
                                 (u["bear"], TARGET, "Bear call spread every day")):
        curve = np.cumsum(values)
        ax.plot(days, curve, color=color, lw=2, label=label, zorder=3)
        ax.annotate(money(curve[-1]).replace("$", "\\$"), (days[-1], curve[-1]), xytext=(4, 0),
                    textcoords="offset points",
                    color=INK_2, fontsize=7.5, va="center")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel("Cumulative P&L per spread ($)", color=INK_2, fontsize=8)
    ax.margins(x=0.06)
    fig.text(0.02, 0.97, "Your rule on the alert days: $1 SPY credit spreads", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Sell at 10:00 ET; {rule_label(*result['rule'])}, otherwise buy back at {EXIT} ET. "
             f"{len(days)} alert days, one spread each,\ncosts: "
             f"{cost_text(result['costs']['base']).replace('$', chr(92) + '$')}.", color=INK_2, fontsize=7.5,
             va="top", linespacing=1.5)
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.text(0.98, 0.02, "Fills from minute trade prices; no quotes", color=MUTED, fontsize=7, ha="right")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _heatmap(ax, values, title, fmt, rule=None, center=0.0):
    """values: 2-D array [take profit, stop]; diverging around `center`, `rule` outlined."""
    limit = np.nanmax(np.abs(values - center)) or 1
    ax.imshow(values - center, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt(values[i, j]), ha="center", va="center", fontsize=5.6, color=INK)
    ax.set_xticks(range(len(STOPS)), ["none" if s is None else f"{s}%" for s in STOPS], fontsize=6.5)
    ax.set_yticks(range(len(TAKE_PROFITS)), ["none" if t is None else f"{t}%" for t in TAKE_PROFITS], fontsize=6.5)
    ax.set_xlabel("Stop loss (% of max loss)", color=INK_2, fontsize=7.5)
    ax.set_ylabel("Take profit (% of credit)", color=INK_2, fontsize=7.5)
    ax.set_title(title, color=INK, fontsize=8.5, loc="left")
    if rule is not None:
        i, j = TAKE_PROFITS.index(rule[0]), STOPS.index(rule[1])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))


def grid_matrix(frame, column, **filters):
    f = frame
    for k, v in filters.items():
        f = f[f[k] == v]
    key = lambda v: "none" if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)  # noqa: E731
    lookup = {(key(r.take_profit), key(r.stop)): getattr(r, column) for r in f.itertuples()}
    return np.array([[lookup.get((key(tp), key(sl)), np.nan) for sl in STOPS] for tp in TAKE_PROFITS], float)


def plot_grid(result, path):
    g = result["grid"][result["grid"]["cost"] == "base"]
    total = grid_matrix(g, "total", strategy="alerts")
    edge = total - grid_matrix(g, "total", strategy="shuffled calls")
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    for k, (values, title) in enumerate(((total, "Following the alerts: total P&L ($ per spread)"),
                                         (edge, "Alerts minus the shuffled-call average ($)"))):
        ax = _axes(fig, [0.08 + k * 0.49, 0.13, 0.4, 0.66])
        _heatmap(ax, values, title, lambda v: f"{v:,.0f}", result["rule"])
        if k:
            ax.set_ylabel("")
    n = len(result["days_used"])
    fig.text(0.02, 0.97, "Every take-profit and stop combination on the alert days", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{n} alert days, time exit {EXIT} ET, base costs. Outlined: your rule. Exploratory: "
             f"with {len(TAKE_PROFITS) * len(STOPS)} combinations, the best cells look good partly by chance.",
             color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_consistency(result, path):
    g = result["grid"][result["grid"]["cost"] == "base"]
    panels = ((grid_matrix(g, "consistency", strategy="alerts"), "Consistency score (average ÷ standard error)",
               lambda v: f"{0.0 if abs(v) < 0.05 else v:.1f}", 0.0),
              (grid_matrix(g, "win_rate", strategy="alerts"), "Winning trades (%)", lambda v: f"{v:.0f}", 50.0))
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    for k, (values, title, fmt, center) in enumerate(panels):
        ax = _axes(fig, [0.08 + k * 0.49, 0.13, 0.4, 0.66])
        _heatmap(ax, values, title, fmt, result["rule"], center=center)
        if k:
            ax.set_ylabel("")
    fig.text(0.02, 0.97, "How steady is each exit rule on the alert days?", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{len(result['days_used'])} alert days, time exit {EXIT} ET, base costs. Blue: steadier "
             "profits (left) or more winners than losers (right). Outlined: your rule.\nA high win rate alone "
             "is not consistency: small wins can be outweighed by a few large losses.", color=INK_2, fontsize=7.5,
             va="top", linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def placement_rows(result, scope, rule=None, cost="base"):
    p, rule = result["placement"], rule or result["rule"]
    p = p[(p["scope"] == scope) & (p["cost"] == cost) & (p["take_profit"] == rule[0])]
    return p[p["stop"].isna()] if rule[1] is None else p[p["stop"] == rule[1]]


def plot_offsets(result, path):
    """Return on risk by strike placement under the user's rule: alerts (base and wider slippage) and
    each direction every session without alerts."""
    wider = result["wider_cost"]
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.1, 0.2, 0.86, 0.58])
    ax.axhline(0, color=BASELINE, lw=1, zorder=1)
    series = [("alerts", "base", SERIES[0], "-", "Following the alerts"),
              ("alerts", wider, SERIES[0], (0, (4, 3)), f"Following the alerts, {wider}"),
              ("no alerts: always bullish", "base", SERIES[1], "-", "Bull put every session, no alerts"),
              ("no alerts: always bearish", "base", TARGET, "-", "Bear call every session, no alerts")]
    for scope, cost, color, style, label in series:
        r = placement_rows(result, scope, cost=cost).sort_values("offset")
        if r.empty or r["trades"].max() == 0:
            continue
        ax.plot(r["offset"], r["return_on_risk"], color=color, ls=style, lw=2, label=label, zorder=3)
        full = r["trades"] >= 0.99 * r["trades"].max()  # hollow only when more than 1% of days are missing
        ax.scatter(r["offset"][full], r["return_on_risk"][full], s=36, color=color, ec=SURFACE, lw=1.5, zorder=4)
        ax.scatter(r["offset"][~full], r["return_on_risk"][~full], s=36, color=SURFACE, ec=color, lw=1.5, zorder=4)
    ax.set_xticks(list(OFFSETS), [placement_label(k).replace(" ", "\n") for k in OFFSETS])
    ax.set_xlabel("Strike placement: $ deeper in the money (+) or further out of the money (−)", color=INK_2,
                  fontsize=8)
    ax.set_ylabel("Return on risk (total P&L ÷ total max loss, %)", color=INK_2, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    fig.text(0.02, 0.97, "Moving the $1 spread deeper in or further out of the money", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Your rule: {rule_label(*result['rule'])}, otherwise {EXIT} ET. Hollow points: over 1% of "
             "days had no usable prices at that placement,\nand those days are not random, so treat them with care.",
             color=INK_2, fontsize=7.5, va="top", linespacing=1.5)
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_context(result, path):
    ctx = result["context"]
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    for k, side in enumerate((1, -1)):
        ax = _axes(fig, [0.08 + k * 0.49, 0.13, 0.4, 0.66])
        _heatmap(ax, grid_matrix(ctx["grid"], "mean", strategy=SIDE_NAMES[side]),
                 f"{'Bull put' if side > 0 else 'Bear call'} spread every session: mean $ per trade",
                 lambda v: f"{v:.1f}", result["rule"])
        if k:
            ax.set_ylabel("")
    days = ctx["days"]
    fig.text(0.02, 0.97, "The spread structure with no alerts", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{len(days)} sessions, {days[0].date()} to {days[-1].date()}: the same spread sold at "
             f"10:00 ET every day in one direction.\nTime exit {EXIT} ET, base costs. Outlined: your rule.",
             color=INK_2, fontsize=7.5, va="top", linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- report.md

def money(v, sign=True, cents=False):
    if v is None or pd.isna(v):
        return "n/a"
    v = round(v, 2 if cents else 0) + 0.0  # no "−$0" for values that round to zero
    return f"{'+' if v >= 0 and sign else '−' if v < 0 else ''}${abs(v):,.{2 if cents else 0}f}"


def rule_label(tp, sl):
    tp_text = "no take profit" if tp is None else f"take profit {tp}%"
    return f"{tp_text}, {'no stop' if sl is None else f'stop {sl}% of max loss'}"


def cell(grid, strategy, rule, cost="base"):
    tp, sl = rule
    g = grid[(grid["strategy"] == strategy) & (grid["cost"] == cost)]
    g = g[(g["take_profit"].isna() if tp is None else g["take_profit"] == tp) &
          (g["stop"].isna() if sl is None else g["stop"] == sl)]
    return g.iloc[0]


def render_report(result, source):
    a, trades, grid = result["alerts"], result["trades"], result["grid"]
    days = result["days_used"]
    lines = []
    w = lines.append
    rule = result["rule"]
    tp, sl = rule
    w(f"# $1 SPY credit spreads on the intraday alerts: {a['date'].min()} to {a['date'].max()}\n")
    w(f"Generated by `alert_spreads.py` on {result['today'].date()}. Every intraday alert (10:00 ET) sells a "
      "same-day $1 vertical spread with the short strike just in the money: BULLISH sells the put at the first "
      "strike at or above SPY and buys the put $1 below (for SPY at 760.50: sell 761P, buy 760P); BEARISH sells "
      "the call at the first strike at or below SPY and buys the call $1 above (sell 760C, buy 761C). NEUTRAL is "
      "no trade.\n")
    target = (f"buy the spread back once it is worth {100 - tp}% of the credit "
              f"({'an' if str(tp).startswith('8') else 'a'} {tp}% take profit)"
              if tp is not None else "no take profit")
    stop = f"a stop once the loss reaches {sl}% of the max loss" if sl is not None else "no stop"
    w(f"**Your rule:** {target}, {stop}, otherwise close at {EXIT} ET (`--take-profit`, `--stop`). Dollars are per "
      "one spread (100 shares per contract), after assumed costs, before taxes.\n")
    w("## How to read this\n")
    w("- **Baselines on the same days, with the same exits:** selling the bull put spread every day (always "
      "bullish), the bear call spread every day (always bearish), and the alerts' own calls reshuffled across "
      f"the same days {result['reps']:,} times (same number of bullish and bearish calls). The alerts add value "
      "only if they beat these, not merely if the equity curve rises.")
    w("- **95% interval** on the average trade: the range of averages that fit the data. A wide interval that "
      "includes zero means the sample can't tell a winning rule from a losing one.")
    w("- **p (shuffled calls):** the share of reshuffles that made at least as much as the alerts. Small "
      "(below 0.05) means the timing of the calls is hard to explain by luck.")
    w("- **Your rule is the main result.** The grid of other take profits and stops is exploratory: the best "
      "of 100 combinations will look good partly by chance. Alerts posted from now on are the real test.\n")

    # ------------------------------------------------ data
    w("## 1. Data and fills\n")
    statuses = pd.Series([trades[d, s]["status"] for d in a.loc[a["side"] != 0, "session"]
                          for s in (1, -1) if (d, s) in trades]).value_counts()
    n_dir = int((a["side"] != 0).sum())
    w(f"- Intraday alerts: {len(a)} rows, {n_dir} directional, {int((a['side'] == 0).sum())} NEUTRAL (no trade). "
      f"**{len(days)}** directional days have usable prices for both directions and are used in every comparison.")
    w("- Spread status on the directional days (both directions counted): " +
      ", ".join(f"{k}: {v}" for k, v in statuses.items()) + ".")
    pending = [str(d.date()) for d in a.loc[a["side"] != 0, "session"] if trades[d, 1]["status"] == "pending"]
    if pending:
        w(f"- Pending (today's session, not loaded): {', '.join(pending)}.")
    ok_trades = [t for (d, s), t in trades.items() if t["status"] == STATUS_OK and d in set(days)]
    for side, name in ((1, "Bull put"), (-1, "Bear call")):
        credits = np.array([t["credit_traded"] for t in ok_trades if t["side"] == side])
        w(f"- {name} spreads: traded credit at entry (before costs) median ${np.median(credits):.2f}, range "
          f"${credits.min():.2f} to ${credits.max():.2f}, so the max loss ($1 minus the credit) is about "
          f"${(1 - np.median(credits)) * 100:.0f}" +
          (f" and {'an' if str(tp).startswith('8') else 'a'} {tp}% take profit about "
           f"${np.median(credits) * tp:.0f}" if tp is not None else "") +
          ", before costs.")
    cover = np.array([len(t["watch"]) for t in ok_trades])
    minutes_window = np.array([(t["watch_times"][-1] - t["watch_times"][0]) / NS_MIN + 1 if len(t["watch_times"])
                               else np.nan for t in ok_trades])
    w(f"- Both legs traded in a median {np.nanmedian(cover / minutes_window) * 100:.0f}% of minutes between entry and "
      f"{EXIT}; take profits and stops are checked only in those minutes.")
    w(f"- Option data: Massive minute bars (trades). The data plan has no bid/ask quotes. SPY price at 10:00: "
      f"{source}.")
    costs = result["costs"]
    w(f"- Costs (base): {cost_text(costs['base'])} (`--slippage`, `--commission`). Your rule is also shown with " +
      ", ".join(k for k in costs if k != "base") + ". Regulatory fees of a few cents per contract are not "
      "included; the wider-slippage rows more than cover them.\n")

    # ------------------------------------------------ your rule
    w(f"## 2. Your rule ({rule_label(*rule)}, exit {EXIT})\n")
    rows = []
    for name in ("alerts", "always bullish", "always bearish"):
        r = cell(grid, name, rule)
        rows.append([{"alerts": "**Following the alerts**", "always bullish": "Bull put spread every day",
                      "always bearish": "Bear call spread every day"}[name], int(r["trades"]),
                     f"**{money(r['total'])}**" if name == "alerts" else money(r["total"]),
                     f"{money(r['mean'])} ({money(r['mean_lo'])} to {money(r['mean_hi'])})",
                     f"{r['win_rate']:.0f}% ({int(r['wins'])}/{int(r['trades'])})", money(r["max_drawdown"], False)])
    sh = cell(grid, "shuffled calls", rule)
    rows.append([f"Shuffled calls (average of {result['reps']:,})", int(sh["trades"]), money(sh["total"]),
                 f"5th–95th percentile total: {money(sh['total_p5'])} to {money(sh['total_p95'])}", "", ""])
    for other in REFERENCE_RULES:
        if other != rule:
            o = cell(grid, "alerts", other)
            rows.append([f"Following the alerts, {rule_label(*other)} (for comparison)", int(o["trades"]),
                         money(o["total"]), f"{money(o['mean'])} ({money(o['mean_lo'])} to {money(o['mean_hi'])})",
                         f"{o['win_rate']:.0f}% ({int(o['wins'])}/{int(o['trades'])})",
                         money(o["max_drawdown"], False)])
    w(md_table(["", "Trades", "Total", "Average trade (95% interval)", "Winners", "Max drawdown"], rows))
    r = cell(grid, "alerts", rule)
    w("")
    w(f"- **Against shuffled calls:** reshuffling the same calls across the same days made at least as much in "
      f"{sh['p_alerts_or_better'] * 100:.1f}% of shuffles (p = {alerts.p_text(sh['p_alerts_or_better'])}). " +
      ("That is hard to explain by luck." if sh["p_alerts_or_better"] < 0.05 else
       "That is well within what luck produces." if sh["p_alerts_or_better"] > 0.2 else
       "That is suggestive at most."))
    for name, spread in (("bullish", "bull put"), ("bearish", "bear call")):
        m, lo, hi = (sh[f"alerts_minus_{name}_{k}"] for k in ("mean", "lo", "hi"))
        w(f"- **Against selling the {spread} spread every day:** the alerts made {money(abs(m), False)} per trade "
          f"{'more' if m >= 0 else 'less'} (difference {money(m)}, 95% interval {money(lo)} to {money(hi)}). " +
          ("The interval excludes zero." if lo > 0 or hi < 0 else
           "The interval includes zero, so the sample can't separate them."))
    reasons = pd.Series([simulate(trades[d, s], *rule, *result["costs"]["base"])[1]
                         for d, s in zip(days, result["sides"])]).value_counts()
    w("- **How the alert trades ended:** " + ", ".join(f"{k}: {v}" for k, v in reasons.items()) + ".")
    w(since_adopted(result))
    w(breakeven_note(result))
    w("\n![Equity curve](equity.png)\n")
    w("**Costs.** The same rule under each cost assumption, next to selling the bull put spread every day:\n")
    w(md_table(["Costs", "Alerts total", "Average trade", "Always bullish total"],
               [[c, money(cell(grid, 'alerts', rule, c)["total"]), money(cell(grid, 'alerts', rule, c)["mean"]),
                 money(cell(grid, 'always bullish', rule, c)["total"])] for c in result["costs"]]))
    w("\n**Flagged rows.** Your rule, base costs:\n")
    w(md_table(["Rows", "Trades", "Total", "Average trade (95% interval)"],
               [[name, v["trades"], money(v["total"]), f"{money(v['mean'])} ({money(v['mean_lo'])} to "
                 f"{money(v['mean_hi'])})"] for name, v in result["variants"].items()]))

    # ------------------------------------------------ grid
    w("\n## 3. Take-profit and stop grid (exploratory)\n")
    g = grid[(grid["cost"] == "base")]
    al, sh_all = g[g["strategy"] == "alerts"].copy(), g[g["strategy"] == "shuffled calls"]
    bull = g[g["strategy"] == "always bullish"]
    al["label"] = [rule_label(None if pd.isna(t) else int(t), None if pd.isna(s) else int(s))
                   for t, s in zip(al["take_profit"], al["stop"])]
    al["p"] = sh_all["p_alerts_or_better"].to_numpy()
    al["vs_bull"] = al["total"].to_numpy() - bull["total"].to_numpy()
    rank = int((al["total"] > r["total"]).sum()) + 1
    w(f"{len(al)} combinations of take profit (% of the credit kept) and stop (% of the max loss), all with the "
      f"{EXIT} time exit and base costs. Your rule ranks {rank} of {len(al)} by total P&L. The alerts made money "
      f"in {int((al['total'] > 0).sum())} combinations, beat selling the bull put spread every day in "
      f"{int((al['vs_bull'] > 0).sum())}, and beat shuffled calls with p < 0.05 in "
      f"{int((al['p'] < 0.05).sum())} (about {len(al) * 0.05:.0f} would by chance alone if the calls carried "
      "no information).\n")
    w(md_table(["Rule", "Alerts total", "Average trade (95% interval)", "Winners", "vs bull put every day",
                "p (shuffled)"],
               [[x.label, money(x.total), f"{money(x.mean)} ({money(x.mean_lo)} to {money(x.mean_hi)})",
                 f"{x.win_rate:.0f}%", money(x.vs_bull), alerts.p_text(x.p)]
                for x in pd.concat([al.nlargest(5, "total"), al.nsmallest(3, "total")]).itertuples()]))
    w("\nTop five and bottom three by total. Every combination is in `grid.csv`.\n")
    w("![Grid](grid.png)\n")
    consistency_section(w, result)
    if result["placement"] is not None:
        placement_section(w, result)
    scaling_section(w, result)

    # ------------------------------------------------ context
    ctx = result["context"]
    if ctx is not None and len(ctx["days"]):
        w("## 7. The spread with no alerts (context)\n")
        cd = ctx["days"]
        skipped = sorted({(str(d.date()), t["status"]) for (d, _), t in trades.items()
                          if d < result["today"] and d >= pd.Timestamp(result["context_start"])
                          and t["status"] != STATUS_OK})
        w(f"The same spread sold at 10:00 ET in one direction on every session from {cd[0].date()} to "
          f"{cd[-1].date()} ({len(cd)} sessions with usable prices for both directions), as far back as the data "
          "plan goes. This shows what the structure and exits do on their own." +
          (" Sessions left out: " + "; ".join(f"{d} ({st})" for d, st in skipped) + "." if skipped else "") + "\n")
        rows = []
        for side in (1, -1):
            for cost in result["costs"]:
                v = describe(ctx["user"][side, cost], 2000, result["seed"])
                rows.append([f"{'Bull put' if side > 0 else 'Bear call'} spread every session", cost, v["trades"],
                             money(v["total"]), f"{money(v['mean'])} ({money(v['mean_lo'])} to "
                             f"{money(v['mean_hi'])})", f"{v['win_rate']:.0f}%", money(v["max_drawdown"], False)])
        w(f"Your rule ({rule_label(*rule)}), under each cost assumption:\n")
        w(md_table(["", "Costs", "Trades", "Total", "Average trade (95% interval)", "Winners", "Max drawdown"],
                   rows))
        cg = ctx["grid"]
        best = cg.loc[cg["mean"].idxmax()]
        w(f"\nAcross the whole grid, the best single direction and rule was {best['strategy']} with "
          f"{rule_label(None if pd.isna(best['take_profit']) else int(best['take_profit']), None if pd.isna(best['stop']) else int(best['stop']))}"
          f": {money(best['mean'])} per trade (95% interval {money(best['mean_lo'])} to {money(best['mean_hi'])}).\n")
        w("![Context grid](context_grid.png)\n")

    # ------------------------------------------------ caveats
    w("## 8. Caveats\n")
    w("- **Fills are approximations.** No bid/ask quotes: entries and exits use traded prices, and the spread's "
      "value comes from two legs' prints in the same minute, which can be seconds apart. Real fills on a $1 "
      "spread depend on the spread's own bid/ask, so compare the wider-slippage rows.")
    w("- **Minute closes only.** A take profit or stop that was touched inside a minute and reverted is missed; "
      "a stop fills at the minute's closing value, which can be worse than the stop level.")
    w("- **Early assignment** of the in-the-money short leg is ignored (it is rare for same-day SPY options "
      "before the close, but possible).")
    w(f"- **Small sample, one period.** {len(days)} trades over five months. The grid is in-sample; the alerts "
      "posted from now on are the honest test of whichever rule you settle on.")
    w("- **One spread per trade,** no position sizing, no compounding, before taxes.")

    w("\n## 9. Bottom line\n")
    w(bottom_line(result))
    return "\n".join(lines) + "\n"


def md_table(header, rows):
    return alerts.md_table(header, rows)


def rule_of(row):
    return (None if pd.isna(row["take_profit"]) else int(row["take_profit"]),
            None if pd.isna(row["stop"]) else int(row["stop"]))


def consistency_section(w, result):
    """Most consistent take profit / stop on the alert days, with the checks against selection luck."""
    grid, ctx, rule = result["grid"], result["context"], result["rule"]
    g = grid[(grid["strategy"] == "alerts") & (grid["cost"] == "base")].copy()
    g["rule"] = [rule_of(r) for _, r in g.iterrows()]
    wider = {rule_of(r): r["consistency"] for _, r in grid[(grid["strategy"] == "alerts")
                                                           & (grid["cost"] == result["wider_cost"])].iterrows()}
    w("## 4. Most consistent take profit and stop (exploratory)\n")
    w("**Consistency score** = the average trade divided by its standard error: how many standard errors the "
      "average sits above zero. It is high only when trades are profitable on average *and* steady from trade "
      "to trade. As a rough guide, about 2 or more would be hard to get by luck for a rule chosen in advance; "
      "the best of 100 rules picked after the fact needs much more. Win rate alone misleads: a 10% take "
      "profit wins most trades and still loses money.\n")
    top = g.sort_values("consistency", ascending=False).head(6)
    if rule not in set(top["rule"]):
        top = pd.concat([top, g[g["rule"] == rule]])
    months = int(g["months"].max())
    rows = []
    for _, r in top.iterrows():
        label = rule_label(*r["rule"]) + (" (**your rule**)" if r["rule"] == rule else "")
        rows.append([label, f"{money(r['mean'])} ({money(r['mean_lo'])} to {money(r['mean_hi'])})",
                     f"**{r['consistency']:.2f}**", f"{wider[r['rule']]:.2f}", f"{r['win_rate']:.0f}%",
                     f"{r['profit_factor']:.2f}", money(r["worst_trade"]), money(r["max_drawdown"], False),
                     f"{int(r['months_positive'])}/{months} (worst {money(r['worst_month'])})"])
    w(md_table(["Rule", "Average trade (95% interval)", "Score", f"Score, {result['wider_cost']}", "Winners",
                "Profit factor", "Worst trade", "Worst drawdown", "Profitable months"], rows))
    w("\nProfit factor = total won ÷ total lost. Every rule closes at 15:30 if neither level is hit.\n")

    score = grid_matrix(grid, "consistency", strategy="alerts", cost="base")
    padded = np.pad(score, 1, constant_values=np.nan)
    hood = np.array([[np.nanmean(padded[i:i + 3, j:j + 3]) for j in range(score.shape[1])]
                     for i in range(score.shape[0])])
    bi, bj = np.unravel_index(np.nanargmax(hood), hood.shape)
    no_stop = score[:, STOPS.index(None)]
    worse = int(sum((score[i, j] < no_stop[i]) for i in range(score.shape[0]) for j in range(score.shape[1])
                    if STOPS[j] is not None))
    w(f"- **Plateau, not a peak:** averaging each rule with its neighbours, the steadiest area is around "
      f"{rule_label(TAKE_PROFITS[bi], STOPS[bj])} (neighbourhood score {hood[bi, bj]:.2f}). A rule surrounded by "
      "similar scores is less likely to be a fluke than a lone high cell.")
    cases = score.shape[0] * (score.shape[1] - 1)
    w(f"- **Stops:** adding a stop lowered the score in {worse} of {cases} cases compared with the same take "
      "profit and no stop." + (" A $1 spread already caps the loss at $1 minus the credit, and on these trades "
                               "stops mostly closed positions that would have recovered by 15:30."
                               if worse > cases / 2 else ""))

    m = result["monthly_check"]
    w("\n**Does picking the best rule work on data it hasn't seen?** For each month, the most consistent rule was "
      "chosen from the *other* months only, then traded in the held-out month:\n")
    reference = next((r for r in REFERENCE_RULES if r != rule), None)
    months = np.asarray(pd.DatetimeIndex(result["days_used"]).strftime("%Y-%m"))
    ref_pnl = {mo: result["cell_pnl"][reference][months == mo].sum() for mo in m["month"]} if reference else {}
    w(md_table(["Month", "Trades", "Rule picked without this month", "Its P&L", "Your rule's P&L",
                *([f"{rule_label(*reference)}"] if reference else []), "Best rule in hindsight"],
               [[r.month, r.trades, rule_label(*rule_of(r._asdict())), money(r.picked_pnl), money(r.current_pnl),
                 *([money(ref_pnl[r.month])] if reference else []),
                 f"{rule_label(*rule_of({'take_profit': r.hindsight_take_profit, 'stop': r.hindsight_stop}))}: "
                 f"{money(r.hindsight_pnl)}"] for r in m.assign(take_profit=m["picked_take_profit"],
                                                                 stop=m["picked_stop"]).itertuples()]))
    picked, current = m["picked_pnl"].sum(), m["current_pnl"].sum()
    chosen_here = result["rule_since"] is not None and (
        pd.DatetimeIndex(result["days_used"]) < pd.Timestamp(result["rule_since"])).any()
    if chosen_here:
        ref_total = sum(ref_pnl.values()) if reference else None
        w(f"\nOn months they never saw, the re-picked rules made **{money(picked)}**: the honest estimate of what "
          "choosing a rule this way is worth" +
          (f", against **{money(ref_total)}** for {rule_label(*reference)} ({money(picked - ref_total)})"
           if reference else "") +
          f". Your rule's **{money(current)}** on the same months is in-sample, because it was chosen using every "
          f"one of them; the gap down to {money(picked)} is roughly how much hindsight flatters it.")
    else:
        w(f"\nOn months they never saw, the re-picked rules made **{money(picked)}** in total against "
          f"**{money(current)}** for your rule ({money(picked - current)}). " +
          ("Re-optimizing helped out of sample, so part of the in-sample advantage looks real, but it is five "
           "months of data." if picked > current else
           "Re-optimizing did not beat your rule out of sample, so the in-sample winner's edge is mostly selection "
           "luck."))

    if ctx is not None and len(ctx["days"]):
        cg = ctx["grid"].copy()
        cg["rule"] = [rule_of(r) for _, r in cg.iterrows()]
        rows = []
        for this in list(dict.fromkeys(list(top["rule"].head(3)) + [rule, (50, None)])):
            c = {side: cg[(cg["rule"] == this) & (cg["strategy"] == SIDE_NAMES[side])].iloc[0] for side in (1, -1)}
            rows.append([rule_label(*this), *(f"{money(c[s]['mean'], cents=True)} (score {c[s]['consistency']:.2f}, "
                                              f"{c[s]['win_rate']:.0f}% winners)" for s in (1, -1))])
        w(f"\n**Without alerts** ({len(ctx['days'])} sessions since {ctx['days'][0].date()}, each direction every "
          "day, base costs), the same exits do this. If an exit rule is better in general, it should show up "
          "here too, on six times as many trades:\n")
        w(md_table(["Rule", "Bull put every session", "Bear call every session"], rows))
    w("\n![Consistency](consistency.png)\n")


def since_adopted(result):
    """Split the user's rule into the trades it was chosen on and the trades since it was adopted."""
    since = result["rule_since"]
    if since is None:
        return ("- **In-sample:** if you chose this rule after seeing these results, they flatter it; pass "
                "`--rule-since` to track the trades after you adopted it separately.")
    calls = result["user"]["base"]["calls"]
    after = pd.DatetimeIndex(result["days_used"]) >= pd.Timestamp(since)
    n_before = int((~after).sum())
    before = (f"- **Chosen on these trades:** you adopted this rule on {since}. The {n_before} "
              f"trade{'s' if n_before != 1 else ''} before then were used to choose it, so their results flatter it. ")
    if not after.any():
        return before + "**No completed alert trades since then yet**; rerun as they come in: they are the real test."
    v = describe(calls[after], result["reps"], result["seed"])
    return before + (f"**Since adopting it: {v['trades']} trade{'s' if v['trades'] != 1 else ''}, {money(v['total'])}** ({money(v['mean'], cents=True)} "
                     f"a trade, 95% interval {money(v['mean_lo'])} to {money(v['mean_hi'])}, {v['win_rate']:.0f}% "
                     "winners). Those are the honest numbers.")


def scaling_section(w, result):
    """What the rule's ups and downs look like at larger size, from resampled trade sequences."""
    sc, sizes, rule = result["scaling"], result["sizes"], result["rule"]
    calls = result["user"]["base"]["calls"]
    w("## 6. Scaling and drawdowns (exploratory)\n")
    w(f"What {rule_label(*rule)} could look like over the next few months, if they resemble these "
      f"{len(calls)} alert days (about {result['trades_per_month']:.0f} trades a month). Each estimate resamples "
      f"your trades in runs of {BLOCK} in a row ({result['reps']:,} times), so losing clusters stay together. "
      "**Drawdown** = the largest drop from a high point to a later low, with any winners in between. "
      "**1 in 20** = the result that only 5% of resamples did worse than. Dollars below are per spread; multiply "
      "by your contracts.\n")
    for months in HORIZONS_MONTHS:
        m = sc[sc["months"] == months]
        w(f"**Next {months} months (~{int(m['trades'].iloc[0])} trades), per spread:**\n")
        w(md_table(["If the true average trade is…", "Median result", "1-in-20 bad result", "Chance of losing money",
                    "Typical drawdown", "1-in-20 drawdown"],
                   [[f"{r.scenario}" + ("" if "($" in r.scenario else f" ({money(r.mean_trade, cents=True)})"),
                     money(r.median_total), money(r.p5_total),
                     "under 1%" if r.p_loss < 0.5 else f"{r.p_loss:.0f}%", money(-r.median_drawdown),
                     money(-r.p95_drawdown)] for r in m.itertuples()]))
        w("")
    worst10 = min(calls[i:i + 10].sum() for i in range(max(1, len(calls) - 9)))
    hist_dd = max_drawdown(calls)
    m3 = sc[sc["months"] == HORIZONS_MONTHS[0]].set_index("scenario")
    honest = next((x for x in m3.index if x.startswith("honest")), "as measured")
    cols = ["as measured", honest, "no edge ($0 average)"]
    w(f"**At size** (worst so far from your {len(calls)} trades; 1-in-20 drawdown over the next "
      f"{HORIZONS_MONTHS[0]} months):\n")
    w(md_table(["Contracts", "Worst trade so far", "Worst 10-trade stretch so far", "Worst drawdown so far",
                *(f"1-in-20 drawdown, {c}" for c in dict.fromkeys(cols)), f"1-in-20 {HORIZONS_MONTHS[0]}-month result, "
                f"{honest}"],
               [[n, money(calls.min() * n), money(worst10 * n), money(-hist_dd * n),
                 *(money(-m3.loc[c, "p95_drawdown"] * n) for c in dict.fromkeys(cols)),
                 money(m3.loc[honest, "p5_total"] * n)] for n in sizes]))
    a = m3.loc[honest]
    win = (calls > 0).mean()
    w(f"\n**What a bad-case drawdown looks like inside.** In the worst 5% of {HORIZONS_MONTHS[0]}-month resamples "
      f"({honest}), the drawdown typically runs over about **{a['bad_dd_trades']:.0f} trades, roughly "
      f"{a['bad_dd_losers']:.0f} losers and {a['bad_dd_winners']:.0f} winners**, with a longest losing streak of "
      f"about {a['bad_dd_streak']:.0f} inside it. So it isn't one long streak: it's a choppy stretch where losers "
      f"clearly outnumber winners (your average winner is {money(calls[calls > 0].mean())} and average loser "
      f"{money(calls[calls < 0].mean())} per spread). With {win:.0%} of trades "
      f"winning, {STREAK} losses in a row came up in {a[f'p_streak_{STREAK}']:.1f}% of {HORIZONS_MONTHS[0]}-month "
      f"resamples; your longest so far is {int(longest_losing_streak(calls[None, :])[0])}.")
    w("\nThese assume the future looks like these five months. A rougher market can produce bigger drawdowns than "
      "anything resampled from a calm stretch, and fills at 10–20 contracts may be worse than single-lot prints.\n")
    w("![Scaling](scaling.png)\n")


def plot_scaling(result, path):
    """Fan chart of cumulative P&L at the largest size over the longest horizon: median and 5-95% range."""
    size, months = max(result["sizes"]), max(HORIZONS_MONTHS)
    n = int(round(result["trades_per_month"] * months))
    scen = result["scenarios"]
    honest = next((k for k in scen if k.startswith("honest")), "as measured")
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.11, 0.2, 0.85, 0.58])
    x = np.arange(1, n + 1)
    for name, color in ((honest, SERIES[0]), ("no edge ($0 average)", SERIES[1])):
        eq = np.cumsum(resample_paths(scen[name], n, 5000, result["seed"]), axis=1) * size
        lo, mid, hi = np.percentile(eq, [5, 50, 95], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, mid, color=color, lw=2, zorder=3, label=f"{name}: median, with 5–95% range")
        ax.annotate(money(mid[-1]).replace("$", "\\$"), (x[-1], mid[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=7.5, va="center")
    ax.axhline(0, color=BASELINE, lw=1, zorder=2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel(f"Trades from now (about {result['trades_per_month']:.0f} a month)", color=INK_2, fontsize=8)
    ax.set_ylabel(f"Cumulative P&L at {size} contracts ($)", color=INK_2, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.margins(x=0.08)
    fig.text(0.02, 0.97, f"{size} contracts a trade for {months} months: the range of outcomes", color=INK,
             fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{rule_label(*result['rule']).capitalize()}, resampled from your {len(scen['as measured'])} alert "
             f"trades in runs of {BLOCK}. Shaded: 90% of resamples. Orange: the same trades with no edge.",
             color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=1, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def placement_label(k):
    return "0 (yours)" if k == 0 else f"{k:+d}".replace("-", "−")


def placement_section(w, result):
    """Strike placement: the same rules with the $1 spread moved in or out of the money."""
    wider = [c for c in result["costs"] if c not in ("base", "gross")]
    w("## 5. Strike placement (exploratory)\n")
    w("The same $1 spread moved in whole dollars. **+1** moves both legs $1 deeper in the money: more credit and "
      "a smaller max loss, but SPY has to move your way to keep it. **−1** moves them $1 further out of the "
      "money: less credit and a bigger max loss, but more room to be wrong. For SPY at 760.50:\n")
    w(md_table(["Placement", "Bullish (put spread)", "Bearish (call spread)"],
               [[placement_label(k), "sell {}P / buy {}P".format(*spread_legs(1, 760.50, k)[1:]),
                 "sell {}C / buy {}C".format(*spread_legs(-1, 760.50, k)[1:])] for k in OFFSETS]))
    w("\n**How deeper placements are priced.** In-the-money same-day options trade thinly, and priced from their own "
      "prints the deeper placements made money in *both* directions on every session without alerts, which a fair "
      "market doesn't hand out. So every placement deeper than yours is priced from its out-of-the-money twin "
      "instead: a $1 put spread is worth $1 minus the call spread on the same strikes (put-call parity), and those "
      "options trade far more. The own-prints result is shown next to it as a check.")
    for rule in result["offset_rules"]:
        rows = []
        r = placement_rows(result, "alerts", rule).set_index("offset")
        extra = {c: placement_rows(result, "alerts", rule, c).set_index("offset") for c in wider}
        full = int(r["trades"].max())
        for k in OFFSETS:
            x = r.loc[k]
            missing = x["trades"] < full
            rows.append([placement_label(k), f"{int(x['trades'])}" + (" ⚠" if missing else ""),
                         money(x["credit"] * 100, False), money(x["max_loss"], False), f"{x['win_rate']:.0f}%",
                         f"{money(x['mean'], cents=True)} ({money(x['mean_lo'])} to {money(x['mean_hi'])})",
                         f"**{x['return_on_risk']:+.1f}%**", f"{x['consistency']:.2f}",
                         money(x["offset0_mean_same_days"], cents=True) if missing else "",
                         money(x["mean_if_missing_all_max_loss"], cents=True) if missing else "",
                         money(x["mean_own_prints"], cents=True) if k > 0 else "",
                         *(money(extra[c].loc[k, "mean"], cents=True) for c in wider),
                         alerts.p_text(x["p_shuffled"])])
        w(f"\n**{rule_label(*rule)}**" + (" (your rule)" if rule == result["rule"] else "") + ", alert days:\n")
        w(md_table(["Placement", "Days", "Avg credit", "Avg max loss", "Winners", "Average trade (95% interval)",
                    "Return on risk", "Score", "Yours (0) on the same days", "If missing days were max losses",
                    "Avg trade priced from own prints", *(f"Avg trade, {c}" for c in wider), "p (shuffled)"], rows))
    r = placement_rows(result, "alerts").set_index("offset")
    gaps = r[r["days_missing"] > 0]
    if len(gaps):
        w("\n⚠ **Missing days are not random.** Deep in-the-money same-day options trade thinly, and their prices are "
          "most often unusable on days with big moves. " + "; ".join(
              f"at {placement_label(k)}, {int(x['days_missing'])} day{'s' if x['days_missing'] != 1 else ''} "
              f"missing, on which your current spread made {money(x['offset0_pnl_on_missing_days'])}"
              for k, x in gaps.iterrows()).replace("at", "At", 1) +
          ". Leaving out days that were mostly losers flatters those placements, so compare them with your spread "
          "on the same days, and with the bound that counts every missing day as a full max loss.")
    w("\nReturn on risk = total P&L ÷ total max loss. The fills were checked against your real trades only near the "
      "money; deeper in-the-money options have wider bid/ask spreads, so the slippage columns matter more there.\n")
    ctx = [scope for scope in ("no alerts: always bullish", "no alerts: always bearish")
           if not placement_rows(result, scope).empty and placement_rows(result, scope)["trades"].max() > 0]
    if ctx:
        rows = []
        for k in OFFSETS:
            cells = []
            for scope in ctx:
                x = placement_rows(result, scope).set_index("offset").loc[k]
                y = placement_rows(result, scope, cost=wider[0]).set_index("offset").loc[k]
                cells.append(f"{money(x['mean'], cents=True)} ({x['return_on_risk']:+.1f}%); "
                             f"{wider[0]}: {money(y['mean'], cents=True)}" +
                             (f"; own prints: {money(x['mean_own_prints'], cents=True)}" if k > 0 else ""))
            rows.append([placement_label(k), *cells])
        n = int(placement_rows(result, ctx[0])["trades"].max())
        w(f"**Without alerts** (each direction every session, about {n} sessions, your rule): average trade "
          "(return on risk). If a placement only works on the alert days, the alerts are doing the work; if it "
          "works every day, it's the placement.\n")
        w(md_table(["Placement", "Bull put every session", "Bear call every session"], rows))
    w("\n![Strike placement](offsets.png)\n")


def bottom_line(result):
    grid = result["grid"]
    rule = result["rule"]
    r, s = cell(grid, "alerts", rule), cell(grid, "shuffled calls", rule)
    b, c = cell(grid, "always bullish", rule), cell(grid, "always bearish", rule)
    did = lambda v: f"{'made' if v >= 0 else 'lost'} {money(abs(v), False)}"  # noqa: E731
    beat_both = s["alerts_minus_bullish_lo"] > 0 and s["alerts_minus_bearish_lo"] > 0
    lose_any = s["alerts_minus_bullish_hi"] < 0 or s["alerts_minus_bearish_hi"] < 0
    lucky = s["p_alerts_or_better"] < 0.05
    text = (f"Following the alerts with your rule {did(r['total'])} per spread over {int(r['trades'])} trades "
            f"({money(r['mean'])} a trade, 95% interval {money(r['mean_lo'])} to {money(r['mean_hi'])}). On the same "
            f"days, selling the bull put spread every day {did(b['total'])} and the bear call spread every day "
            f"{did(c['total'])}; reshuffled calls made at least as much as the alerts "
            f"{s['p_alerts_or_better'] * 100:.1f}% of the time. ")
    if lucky and beat_both:
        text += ("On this sample the alerts' direction calls added value beyond the spread itself. It is one "
                 "five-month period, so confirm it on new alerts before sizing up.")
    elif lose_any:
        text += "On this sample the alerts did worse than simply selling one side every day."
    else:
        text += ("So there is **no evidence yet that the alerts' calls add value** beyond the spread and exits "
                 "themselves: the result is within what the same calls on random days produce. Keep logging "
                 "alerts and rerun; the next months are the real test.")
    since = result["rule_since"]
    if since is not None and (pd.DatetimeIndex(result["days_used"]) < pd.Timestamp(since)).any():
        text += (f"\n\nYour rule was chosen on these same trades, so these numbers flatter it. Judge it on the "
                 f"alerts after {since}, which this report tracks separately in section 2.")
    return text


# ---------------------------------------------------------------- CLI

def write_outputs(out_dir, result, source):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trade_table(result).to_csv(out / "trades.csv", index=False)
    result["grid"].to_csv(out / "grid.csv", index=False)
    plot_equity(result, out / "equity.png")
    plot_grid(result, out / "grid.png")
    plot_consistency(result, out / "consistency.png")
    result["scaling"].to_csv(out / "scaling.csv", index=False)
    plot_scaling(result, out / "scaling.png")
    if result["placement"] is not None:
        result["placement"].to_csv(out / "offsets.csv", index=False)
        plot_offsets(result, out / "offsets.png")
    result["monthly_check"].to_csv(out / "monthly_check.csv", index=False)
    if result["context"] is not None and len(result["context"]["days"]):
        result["context"]["grid"].to_csv(out / "context_grid.csv", index=False)
        plot_context(result, out / "context_grid.png")
    (out / "report.md").write_text(render_report(result, source))
    return out


def rule_level(text):
    """'none' or a grid level (10..90): the user's rule must be a grid cell so it can be ranked."""
    if text.lower() == "none":
        return None
    level = int(text)
    if level not in TAKE_PROFITS:
        raise argparse.ArgumentTypeError("use 10, 20, ..., 90 or none")
    return level


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="$1 SPY credit spreads on the intraday alerts, from Massive option "
                                            "minute bars (per spread, after assumed costs, before taxes).")
    p.add_argument("--csv", required=True, help="alert log (the intraday rows are used)")
    p.add_argument("--out", default="output/alert_spreads", help="output folder (default output/alert_spreads)")
    p.add_argument("--cache-dir", default="data/cache", help="Parquet cache for SPY and option minutes")
    p.add_argument("--refresh", action="store_true", help="re-download SPY and option minutes even if cached")
    p.add_argument("--context", action=argparse.BooleanOptionalAction, default=True,
                   help="also run both directions on every session the data plan covers (default on)")
    p.add_argument("--context-start", type=lambda s: str(pd.Timestamp(s).date()),
                   help="first context session (default: the data plan's start, two years before today)")
    p.add_argument("--take-profit", type=rule_level, default=DEFAULT_RULE[0],
                   help=f"your take profit, %% of the credit kept: one of {', '.join(str(t) for t in TAKE_PROFITS if t)}"
                        f" or none (default {DEFAULT_RULE[0]})")
    p.add_argument("--stop", type=rule_level, default=DEFAULT_RULE[1],
                   help="your stop, %% of the max loss: one of 10, 20, ..., 90 or none (default none)")
    p.add_argument("--rule-since", type=lambda s: str(pd.Timestamp(s).date()), default=DEFAULT_RULE_SINCE,
                   help=f"date you adopted your rule; earlier trades chose it (default {DEFAULT_RULE_SINCE})")
    p.add_argument("--contracts", type=int, nargs="+", default=list(SIZES),
                   help=f"contract counts for the scaling tables (default {' '.join(map(str, SIZES))})")
    p.add_argument("--offsets", action=argparse.BooleanOptionalAction, default=True,
                   help="also run the strike placements -3..+3 (default on)")
    p.add_argument("--slippage", type=float, default=0.0,
                   help="$/share per leg per fill (default 0: fills at traded prices)")
    p.add_argument("--commission", type=float, default=0.0,
                   help="$/contract per leg per fill (default 0: commission-free broker)")
    p.add_argument("--reps", type=int, default=10_000, help="shuffles and bootstrap resamples (default 10,000)")
    p.add_argument("--seed", type=int, default=0, help="random seed for shuffles and bootstraps (default 0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    rows = alerts.load_alerts(args.csv)
    if (rows["ticker"] != TICKER).any():
        raise SystemExit(f"Only {TICKER} alerts are supported.")
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    plan_start = (today - pd.DateOffset(years=2)).normalize()
    context_start = (pd.Timestamp(args.context_start) if args.context_start else plan_start) if args.context else None
    first = min(rows["session"].min(), context_start) if context_start is not None else rows["session"].min()
    with timed(timings, "SPY data"):
        sessions = features.trading_sessions(first, rows["session"].max() + pd.Timedelta(days=10), warmup_sessions=0)
        done = sessions[sessions.index < today]  # never load today's session: its data may be incomplete
        minutes, source = download.load_minute_bars(TICKER, str(done.index[0].date()), str(done.index[-1].date()),
                                                    cache_dir=args.cache_dir, refresh=args.refresh,
                                                    final_close=done["close"].iloc[-1])
    load = option_loader(args.cache_dir, args.refresh)
    result = run_study(rows, minutes, sessions, today, load, rule=(args.take_profit, args.stop),
                       rule_since=args.rule_since,
                       costs=cost_tiers(args.slippage, args.commission),
                       context_start=context_start, offsets=OFFSETS if args.offsets else (), sizes=args.contracts,
                       reps=args.reps,
                       seed=args.seed, timings=timings)
    with timed(timings, "outputs"):
        out = write_outputs(args.out, result, source)

    rule = result["rule"]
    r, b = cell(result["grid"], "alerts", rule), cell(result["grid"], "always bullish", rule)
    s = cell(result["grid"], "shuffled calls", rule)
    print(f"Alert spreads: {len(result['days_used'])} alert days; option bars downloaded this run: "
          f"{load.state['fetched']}")
    print(f"  your rule ({rule_label(*rule)}, exit {EXIT}): alerts {money(r['total'])} "
          f"({money(r['mean'])}/trade), bull put every day {money(b['total'])}, shuffled p = "
          f"{alerts.p_text(s['p_alerts_or_better'])}")
    print("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
    print(f"wrote {out}/: trades.csv, grid.csv, monthly_check.csv, offsets.csv, scaling.csv, report.md, equity.png, "
          "grid.png, consistency.png, offsets.png, scaling.png" +
          (", context_grid.csv, context_grid.png" if result["context"] is not None else ""))


if __name__ == "__main__":
    main()
