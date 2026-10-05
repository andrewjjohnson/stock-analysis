"""Put credit spreads after SPY dips, priced with Massive option minute bars: a small trade simulation.

  uv run --env-file .env python dip_spreads.py --out output/dip_spreads

A follow-up to meanrev.py's dip signals: this prices them as put
credit spreads opened at 15:50 ET on signal days and compares them with the same spread opened
on every session, so it measures whether dips make put spreads better than usual.

Pre-registered before any option price was looked at:
- Primary: the study's primary dip signal ("3+ down days in a row"); short put at the $1 strike
  at or below 1% under SPY's 15:50 price, long put $5 lower, expiring 5 sessions later (the
  study's pre-set horizon), held to expiry; holdout sessions (2025-01-01 on).
- Second: the Connors entry (RSI(2) < 10 above the 200-day average), same spread.
- Everything else (expiries 1-3, strikes 0% and 2% below, $1 width, take profits, other dip
  signals, Oct-Dec 2024) is exploratory. 2-3 session expiries were suggested after seeing the
  holdout's underlying odds, so they are not confirmatory.

Prices and fills (the data plan has minute trade bars, no quotes):
- Strikes and settlement use SPY's actual (not dividend-adjusted) prices.
- A spread trades in a minute only if both legs printed in it. Entry uses the opens of the first
  such minute from 15:50 to the close. A take profit triggers on a minute close at least the
  two-leg slippage below its level and fills at the level. A spread not closed earlier is
  settled at intrinsic value from SPY's close on the expiry day (early assignment ignored).
- Costs: --slippage per share per leg per fill and --commission per contract per leg per fill,
  both 0 by default (Robinhood; traded prices matched the user's own fills near the money),
  always shown with +$0.01 and +$0.03 slippage. Fills this far out of the money are unverified.
"""

import argparse
import math
import os
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from massive import RESTClient  # noqa: E402
from massive.exceptions import BadResponse  # noqa: E402

import alert_spreads as sp  # noqa: E402
import alerts  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import outcomes  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
NS_MIN = 60 * 10**9
OPTIONS_START = "2024-10-01"   # the data plan's option minute bars start here (a rolling two years)
DTES = (1, 2, 3, 5)            # sessions to expiry
# Short strike: % below SPY's 15:50 price. ITM (added on request, exploratory) is the first strike at or above
# the price, as in the alert spreads; 0.25 and 0.5 were added with it.
ITM = -1.0
DISTANCES = (ITM, 0.0, 0.25, 0.5, 1.0, 2.0)
WIDTHS = (1, 5)                # $ between the strikes
EXITS = ("expiry", 50, 80)     # hold to expiry, or take profit at 50% / 80% of the credit
RELAXED = "expiry, relaxed entry"  # robustness: each leg's average 15:50-close print, held to expiry
# Added on request (exploratory): once profit reaches 40% of the credit, a stop at breakeven is armed.
BREAKEVEN_AT = 40
EXIT_RULES = [(ex, None) for ex in EXITS] + [(ex, BREAKEVEN_AT) for ex in EXITS]
# Added on request (exploratory): close at 15:50 on the first session after entry that SPY is up from the previous
# close (the stock dip rule's exit), or after MAX_HOLD sessions; "80+up" also takes an 80% profit if that comes first.
UP_EXITS = {"up": "expiry", "80+up": 80}
MAX_HOLD = 5


def exit_key(exit_rule, breakeven_at=None):
    """The exit column's value: 'expiry', '50', '80', or with a breakeven stop 'expiry+be40' etc."""
    return str(exit_rule) if breakeven_at is None else f"{exit_rule}+be{breakeven_at}"
PRIMARY = {"dte": 5, "distance": 1.0, "width": 5, "exit": "expiry"}
SIGNALS = {
    "3+ down days in a row": mr.DIPS["3+ down days in a row"],
    "Connors: RSI(2) < 10 above 200-day": mr.DIPS["RSI(2) < 10, above 200-day"],
    "RSI(2) < 10": mr.DIPS["RSI(2) < 10"],
    "Below lower Bollinger band": mr.DIPS["Below lower Bollinger band"],
    "3+ down days and RSI(2) < 10": mr.DIPS["3+ down days and RSI(2) < 10"],
    "1-day drop ≥ 1.5%": mr.DIPS["1-day drop ≥ 1.5%"],
}
PREREGISTERED = ("3+ down days in a row", "Connors: RSI(2) < 10 above 200-day")
EVERY_DAY = "every session (no signal)"
STRUCTURES = [(h, d, w) for h in DTES for d in DISTANCES for w in WIDTHS]


# ---------------------------------------------------------------- data

def contract_loader(cache_dir, refresh=False):
    """load(ticker, start, end) -> one contract's minute bars over several sessions, from the Parquet
    cache when present (the client is created on the first miss). A few sessions of one contract fit
    in a single page, so the SDK's silent paging stop cannot truncate them."""
    state = {"client": None, "fetched": 0}

    @lru_cache(maxsize=2048)  # a contract is reused by the entries of the five sessions before its expiry
    def cached(ticker, start, end):
        path = Path(cache_dir) / "options_multi" / f"{ticker[2:]}_{start}_{end}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        if state["client"] is None:
            key = os.environ.get("MASSIVE_API_KEY")
            if not key:
                raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python dip_spreads.py")
            state["client"] = RESTClient(api_key=key, retries=10)
        try:
            aggs = list(state["client"].list_aggs(ticker, 1, "minute", start, end, limit=50_000))
        except BadResponse as e:
            if "NOT_AUTHORIZED" in str(e):
                raise sp.NotInPlan(f"{ticker} {start}..{end}") from None
            raise
        df = pd.DataFrame([(a.timestamp, a.open, a.high, a.low, a.close, a.volume) for a in aggs],
                          columns=["t", "open", "high", "low", "close", "volume"])
        df.insert(0, "ts", pd.to_datetime(df.pop("t"), unit="ms", utc=True).dt.as_unit("ns"))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        state["fetched"] += 1
        return df

    def load(ticker, start, end):
        return cached(ticker, start, end)

    load.state = state
    return load


def put_strikes(spot, distance, width):
    """(short, long): the short put at the $1 strike at or below `distance`% under spot (ITM: the first strike
    at or above spot, just in the money), long `width` lower."""
    spot = round(float(spot), 2)
    short = math.ceil(spot) if distance == ITM else math.floor(round(spot * (1 - distance / 100), 2))
    return short, short - width


def distance_label(distance, short=False):
    if distance == ITM:
        return "just ITM" if short else "just in the money (first strike above)"
    if distance == 0:
        return "ATM" if short else "at the money"
    return f"{distance:g}% below"


def both_legs_path(short, long, window):
    """Regular-hours minutes over `window` sessions in which both legs printed:
    (minute start ns, session of each minute, spread open value, spread close value)."""
    legs = []
    for df in (short, long):
        if df.empty:
            return (np.array([], dtype=np.int64),) * 2 + (np.array([]),) * 2
        rth, _ = features.regular_session_minutes(df.assign(vwap=np.nan, transactions=0), window)
        legs.append((outcomes._ns(rth["ts"]), rth["session"].to_numpy("datetime64[ns]"),
                     rth["open"].to_numpy(float), rth["close"].to_numpy(float)))
    (ts, ss, so, sc), (tl, _, lo, lc) = legs
    both, i, j = np.intersect1d(ts, tl, assume_unique=True, return_indices=True)
    return both, ss[i], so[i] - lo[j], sc[i] - lc[j]


def spread_trade(i, days, sessions, raw, h, distance, width, load):
    """The put spread opened at day i's decision time and expiring at the close h sessions later
    (no exit rule applied yet): credit, the both-legs path to expiry, and the settlement value."""
    t, expiry = days[i], days[i + h]
    spot = raw["snap"].iloc[i]
    ks, kl = put_strikes(spot, distance, width)
    trade = {"day": t, "expiry": expiry, "dte": h, "distance": distance, "width": width, "spot": spot,
             "short_strike": ks, "long_strike": kl, "short_ticker": sp.option_ticker(expiry, "P", ks),
             "long_ticker": sp.option_ticker(expiry, "P", kl)}
    if np.isnan(spot):
        return {**trade, "status": "no SPY price at 15:50"}
    start = str(max(days[max(i + h - max(DTES), 0)], pd.Timestamp(OPTIONS_START)).date())
    return price_spread(trade, start, sessions, raw, load)


def price_spread(trade, start, sessions, raw, load):
    """Price a planned spread (day, expiry, strikes, width, leg tickers; "right": "C" for a call spread, put
    by default): both legs' minute bars from `start` to the expiry, the entry from the decision time, the
    both-legs path and the settlement value. Shared with stock_dip_spreads.py and call_spreads.py."""
    t, expiry, ks, width = trade["day"], trade["expiry"], trade["short_strike"], trade["width"]
    try:
        short = load(trade["short_ticker"], start, str(expiry.date()))
        long = load(trade["long_ticker"], start, str(expiry.date()))
    except sp.NotInPlan:
        return {**trade, "status": "not in data plan"}
    window = sessions.loc[t:expiry]
    both, sess, v_open, v_close = both_legs_path(short, long, window)
    decision = (sessions.loc[t, "close"] - pd.Timedelta(minutes=mr.DECISION_MINUTES)).value
    close_t, close_e = sessions.loc[t, "close"].value, sessions.loc[expiry, "close"].value
    settle_close = raw.loc[expiry, "close"]
    if np.isnan(settle_close):
        return {**trade, "status": "no underlying close on the expiry day"}
    # Robustness entry (not the primary rule): each leg's average print from 15:50 to the close, even if the
    # legs never printed in the same minute. Noisier, but it does not drop thinly traded days.
    intrinsic = settle_close - ks if trade.get("right", "P") == "C" else ks - settle_close  # call or put spread
    trade.update(credit_relaxed=window_mean(short, decision, close_t) - window_mean(long, decision, close_t),
                 settle=min(max(intrinsic, 0.0), float(width)), expiry_close=settle_close)
    e = int(np.searchsorted(both, decision))
    until = trade.get("entry_until") or close_t  # a working order may wait past the close (call_spreads.py)
    if e == len(both) or both[e] >= until:
        return {**trade, "status": "no entry: legs did not both trade from 15:50 to the close" if until == close_t
                else "no entry: legs did not both trade in the entry window"}
    credit = v_open[e]
    if not 0 < credit < width:
        return {**trade, "status": "no entry: traded credit outside 0 to the width", "credit_traded": credit}
    end = int(np.searchsorted(both, close_e))
    return {**trade, "status": "ok", "credit_traded": credit, "entry_time": both[e], "watch": v_close[e:end],
            "watch_open": v_open[e:end], "watch_ts": both[e:end], "watch_days": sess[e:end]}


def window_mean(df, start_ns, end_ns):
    """Average minute close of one contract in [start, end); NaN if it never printed then."""
    if df.empty:
        return np.nan
    t = outcomes._ns(df["ts"])
    m = (t >= start_ns) & (t < end_ns)
    return df["close"].to_numpy(float)[m].mean() if m.any() else np.nan


def simulate(trade, exit_rule, slippage, commission, breakeven_at=None, exit_at=None):
    """(P&L $ per spread, exit reason, exit session, credit received) under one exit rule.

    breakeven_at: once a minute close shows profit >= that % of the credit, a stop at breakeven is
    armed; a later close at or above the credit exits there (stop-market: the fill is that close,
    so a jump past breakeven, e.g. overnight, still loses), plus slippage.
    exit_at: a time (ns) to close the spread, at the open of the first both-legs minute at or after it,
    plus slippage, unless the take profit came first; past the expiry, the spread settles as usual."""
    credit = trade["credit_traded"] - 2 * slippage
    watch = trade["watch"]
    first, out = len(watch), None
    if exit_rule != "expiry":
        level = credit * (1 - exit_rule / 100)
        hit = np.flatnonzero(watch <= level - 2 * slippage)  # a buy limit needs trades through it
        if hit.size:
            first, out = hit[0], ((credit - level) * 100 - 4 * commission, "take profit")
    if breakeven_at is not None:
        armed = np.flatnonzero(watch <= credit * (1 - breakeven_at / 100))
        if armed.size:
            back = np.flatnonzero(watch[armed[0] + 1:] >= credit)
            if back.size and armed[0] + 1 + back[0] < first:
                first = armed[0] + 1 + back[0]
                paid = min(watch[first], trade["width"]) + 2 * slippage
                out = ((credit - paid) * 100 - 4 * commission, "breakeven stop")
    if exit_at is not None:
        k = int(np.searchsorted(trade["watch_ts"], exit_at))
        if k < min(first, len(watch)):
            first = k
            paid = min(trade["watch_open"][k], trade["width"]) + 2 * slippage
            late = trade["watch_ts"][k] - exit_at > 10 * NS_MIN  # no print before that day's close
            out = ((credit - paid) * 100 - 4 * commission, "first up day" + (", next print" if late else ""))
    if out is not None:
        return out[0], out[1], pd.Timestamp(trade["watch_days"][first]), credit
    return (credit - trade["settle"]) * 100 - 2 * commission, "expiry", trade["expiry"], credit


def first_up_exits(feats, days, sessions, max_hold=MAX_HOLD):
    """{entry session: exit time (ns)}: 15:50 on the first later session whose move from the previous close is up
    (feats["ret_1d"] > 0, known at that 15:50), or on the max_hold-th session after entry."""
    up = feats["ret_1d"].reindex(days).to_numpy() > 0
    out = {}
    for i, t in enumerate(days):
        for j in range(i + 1, min(i + max_hold, len(days) - 1) + 1):
            if up[j] or j - i == max_hold:
                out[t] = (sessions.loc[days[j], "close"] - pd.Timedelta(minutes=mr.DECISION_MINUTES)).value
                break
    return out


def build_trades(days, sessions, raw, load, structures=STRUCTURES, max_dte=max(DTES)):
    """Every structure opened on every session that leaves room for its expiry."""
    out = {}
    for i, t in enumerate(days):
        for h, d, w in structures:
            if i + h < len(days):
                out[t, h, d, w] = spread_trade(i, days, sessions, raw, h, d, w, load)
    return out


def pnl_table(trades, costs, exits=None):
    """One row per trade, exit rule and cost tier. exits: first_up_exits(...), for the UP_EXITS rules."""
    rows = []
    for (t, h, d, w), tr in trades.items():
        base = {"day": t, "dte": h, "distance": d, "width": w, "status": tr["status"]}
        cr = tr.get("credit_relaxed", np.nan)
        if 0 < cr < w and not np.isnan(tr.get("settle", np.nan)):
            for cost, (slip, comm) in costs.items():
                credit = cr - 2 * slip
                rows.append({**base, "status": "ok", "exit": RELAXED, "cost": cost,
                             "pnl": (credit - tr["settle"]) * 100 - 2 * comm, "reason": "expiry",
                             "exit_day": tr["expiry"], "credit": credit, "max_loss": (w - credit) * 100,
                             "settle": tr["settle"], "short_strike": tr["short_strike"], "expiry": tr["expiry"]})
        if tr["status"] != "ok":
            rows.append(base)
            continue
        rules = [(exit_key(ex, be), ex, be, None) for ex, be in EXIT_RULES]
        if exits is not None:
            rules += [(key, ex, None, exits.get(t)) for key, ex in UP_EXITS.items()]
        for key, ex, be, at in rules:
            for cost, (slip, comm) in costs.items():
                pnl, reason, exit_day, credit = simulate(tr, ex, slip, comm, be, at)
                rows.append({**base, "exit": key, "cost": cost, "pnl": pnl, "reason": reason,
                             "exit_day": exit_day,
                             "credit": credit, "max_loss": (w - credit) * 100, "settle": tr["settle"],
                             "short_strike": tr["short_strike"], "expiry": tr["expiry"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def signal_stats(mask, pnl, max_loss, idx):
    """One signal over one period's valid days: counts, mean P&L with a block-bootstrap interval,
    the excess over every day with its interval, win rate and return on risk."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    out = {"n": n, "base_mean": pnl.mean(), "base_win": (pnl > 0).mean() * 100}
    if n == 0:
        return out
    pos = np.flatnonzero(mask)
    sm, sp_ = mask[idx], pnl[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (sp_ * sm).sum(1) / sm.sum(1)
        diffs = means - sp_.mean(1)
    wl, wh = alerts.wilson(int((pnl[mask] > 0).sum()), n)
    out.update(clusters=int(1 + (np.diff(pos) > 5).sum()), mean=pnl[mask].mean(),
               mean_lo=np.nanpercentile(means, 2.5), mean_hi=np.nanpercentile(means, 97.5),
               excess=pnl[mask].mean() - pnl.mean(), excess_lo=np.nanpercentile(diffs, 2.5),
               excess_hi=np.nanpercentile(diffs, 97.5), win=(pnl[mask] > 0).mean() * 100, win_lo=wl, win_hi=wh,
               ror=pnl[mask].sum() / max_loss[mask].sum() * 100, worst=pnl[mask].min(),
               consistency=sp.consistency(pnl[mask]))
    return out


def one_at_a_time(days_sorted, mask, rows):
    """Signal days taken only when no spread from this signal is still open (closed at its exit)."""
    taken, open_until = [], pd.Timestamp.min
    for t, m in zip(days_sorted, mask):
        if m and t > open_until:
            r = rows.loc[t]
            taken.append(r)
            open_until = r["exit_day"]
    return pd.DataFrame(taken)


def analyze(table, feats, periods, reps, seed):
    """Per period, structure, exit and cost: every signal against every day, plus one-at-a-time trading."""
    ok = table[table["status"] == "ok"]
    records, single = [], []
    sig_masks = {name: rule(feats).fillna(False) for name, rule in SIGNALS.items()}
    for period, (first, last) in periods.items():
        p = ok[(ok["day"] >= first) & (ok["day"] <= last)]
        for (h, d, w, ex, cost), g in p.groupby(["dte", "distance", "width", "exit", "cost"], sort=False):
            g = g.sort_values("day").set_index("day")
            pnl, ml = g["pnl"].to_numpy(float), g["max_loss"].to_numpy(float)
            idx = mr.block_indices(len(g), reps, seed)
            key = {"period": period, "dte": h, "distance": d, "width": w, "exit": ex, "cost": cost}
            records.append({**key, "signal": EVERY_DAY, **signal_stats(np.ones(len(g), bool), pnl, ml, idx)})
            for name, m in sig_masks.items():
                mask = m.reindex(g.index).fillna(False).to_numpy(bool)
                records.append({**key, "signal": name, **signal_stats(mask, pnl, ml, idx)})
                if cost == "base":
                    t = one_at_a_time(g.index, mask, g)
                    r = t["pnl"].to_numpy(float) if len(t) else np.array([])
                    single.append({**key, "signal": name, "trades": len(r), "total": r.sum(),
                                   "mean": r.mean() if len(r) else np.nan,
                                   "win": (r > 0).mean() * 100 if len(r) else np.nan,
                                   "worst": r.min() if len(r) else np.nan,
                                   "max_drawdown": sp.max_drawdown(r) if len(r) else np.nan,
                                   "ror": r.sum() / t["max_loss"].sum() * 100 if len(r) else np.nan})
    return pd.DataFrame(records), pd.DataFrame(single)


# ---------------------------------------------------------------- the study

def run_study(daily, raw, sessions, *, costs=None, reps=2000, seed=0, load=None, timings=None):
    """Signals from meanrev features, every structure on every session with option data, and the
    comparisons. No file I/O beyond the option loader's cache."""
    timings = {} if timings is None else timings
    costs = sp.cost_tiers() if costs is None else costs
    with timed(timings, "signals"):
        feats = mr.build_features(daily, sessions)
    opt_days = raw.index[raw.index >= pd.Timestamp(OPTIONS_START)]
    opt_sessions = sessions.loc[opt_days]
    with timed(timings, "option data"):
        trades = build_trades(opt_days, opt_sessions, raw.loc[opt_days], load)
    with timed(timings, "simulation"):
        table = pnl_table(trades, costs, first_up_exits(feats.loc[opt_days], opt_days, opt_sessions))
        last = opt_days[-1]
        periods = {"holdout": (mr.HOLDOUT_START, last),
                   "Oct-Dec 2024": (pd.Timestamp(OPTIONS_START), mr.DESIGN_END),
                   "all option data": (pd.Timestamp(OPTIONS_START), last)}
        stats, single = analyze(table, feats.loc[opt_days], periods, reps, seed)
    strict = table[table["exit"].astype(str) != RELAXED] if "exit" in table else table
    statuses = strict.drop_duplicates(["day", "dte", "distance", "width"])["status"].value_counts()
    # Primary signal days in the holdout that the strict entry rule could not price (for the bounds).
    sig = SIGNALS[PREREGISTERED[0]](feats.loc[opt_days]).fillna(False)
    prim = {k: v for k, v in trades.items() if k[1:] == (PRIMARY["dte"], PRIMARY["distance"], PRIMARY["width"])}
    missing = [(t, tr["status"]) for (t, *_), tr in prim.items()
               if t >= mr.HOLDOUT_START and sig.get(t, False) and tr["status"] != "ok"]
    return {"table": table, "stats": stats, "single": single, "costs": costs, "periods": periods,
            "statuses": statuses, "feats": feats.loc[opt_days], "reps": reps, "trades": trades, "missing": missing}


# ---------------------------------------------------------------- report

def pick(stats, signal, period="holdout", cost="base", **structure):
    s = {**PRIMARY, **structure}
    m = stats[(stats["signal"] == signal) & (stats["period"] == period) & (stats["cost"] == cost)
              & (stats["dte"] == s["dte"]) & (stats["distance"] == s["distance"]) & (stats["width"] == s["width"])
              & (stats["exit"].astype(str) == str(s["exit"]))]
    return m.iloc[0] if len(m) else None


def money(v, cents=True, sign=True):
    return sp.money(v, sign=sign, cents=cents)


def structure_label(dte=PRIMARY["dte"], distance=PRIMARY["distance"], width=PRIMARY["width"],
                    exit=PRIMARY["exit"]):
    ex = "held to expiry" if exit == "expiry" else f"{exit}% take profit"
    where = distance_label(distance)
    return f"{dte}-session expiry, short put {where}, ${width} wide, {ex}"


def held(lo, hi):
    return "✓" if lo > 0 else "✗ worse" if hi < 0 else "–"


def stat_row(label, r, extra=()):
    if r is None or not r["n"]:
        return [label, 0, "", "", "", "", "", "", *extra]
    return [label, f"{int(r['n'])} ({int(r['clusters'])})", f"{r['win']:.0f}%",
            f"{money(r['mean'])} ({money(r['mean_lo'])} to {money(r['mean_hi'])})",
            f"{money(r['excess'])} ({money(r['excess_lo'])} to {money(r['excess_hi'])})",
            held(r["excess_lo"], r["excess_hi"]), f"{r['ror']:+.1f}%", money(r["worst"]), *extra]


def render_report(res):
    stats, single, costs = res["stats"], res["single"], res["costs"]
    lines = []
    w = lines.append
    h0, h1 = res["periods"]["holdout"]
    w("# Put credit spreads after SPY dips\n")
    w(f"Real SPY option prices (Massive minute bars), holdout **{h0.date()} to {h1.date()}**. Each spread is "
      "opened at 15:50 ET on a signal day and compared with **the same spread opened on every session**, so the "
      "question is whether dips make put spreads better than usual. Dollars are per one spread (100 shares), "
      f"costs: {sp.cost_text(costs['base'])}, before taxes.\n")
    w("## How to read this\n")
    w("- **Excess:** the signal's average P&L per spread minus the every-session average for the same spread. "
      "Positive means opening after a dip beat opening on a random day.")
    w("- **95% interval:** resampling whole 10-session blocks, because spreads opened on nearby days share "
      "the same market moves. Counts in brackets are separate episodes (signals more than 5 sessions apart).")
    w("- **✓ / ✗ / –:** the excess interval is above zero / below zero / includes zero.")
    w("- **Return on risk:** total P&L ÷ total max loss (width minus credit).\n")

    w("## 1. Primary result (fixed before any option price was seen)\n")
    w(f"**{structure_label()}**, on the study's primary dip signal, *3+ down days in a row*.\n")
    rows = []
    for name in (PREREGISTERED[0], EVERY_DAY):
        for cost in costs:
            r = pick(stats, name, cost=cost)
            rows.append(stat_row(f"{'**' if name != EVERY_DAY else ''}{name}{'**' if name != EVERY_DAY else ''}"
                                 f"{'' if cost == 'base' else f', {cost}'}", r))
    head = ["Opened on", "Spreads (episodes)", "Winners", "Average P&L (95% interval)", "Excess vs every session",
            "", "Return on risk", "Worst"]
    w(md_table(head, rows))
    r = pick(stats, PREREGISTERED[0])
    s = single[(single["signal"] == PREREGISTERED[0]) & (single["period"] == "holdout")
               & (single["dte"] == PRIMARY["dte"]) & (single["distance"] == PRIMARY["distance"])
               & (single["width"] == PRIMARY["width"]) & (single["exit"].astype(str) == "expiry")]
    if r is not None and r["n"]:
        w(f"\n{primary_sentence(r)}")
    if len(s):
        s = s.iloc[0]
        w(f"\n**One spread at a time** (skip signals while a spread is open): {int(s['trades'])} trades, "
          f"{money(s['total'])} in total, {money(s['mean'])} a trade, {s['win']:.0f}% winners, worst "
          f"{money(s['worst'])}, worst drawdown {money(-s['max_drawdown'])}.")
    w(robustness_section(res))
    w("\n![Signals](signals.png)\n")

    w("## 2. Second pre-registered test: the Connors RSI(2) entry\n")
    rows = [stat_row(f"{n}{'' if c == 'base' else f', {c}'}", pick(stats, n, cost=c))
            for n in (PREREGISTERED[1], EVERY_DAY) for c in costs]
    w(md_table(head, rows) + "\n")

    w("## 3. Every dip signal, same spread (exploratory)\n")
    rows = [stat_row(n, pick(stats, n)) for n in [*SIGNALS, EVERY_DAY]]
    w(md_table(head, rows))
    w("\nOct–Dec 2024 (option data before the holdout; the signals were chosen on data through 2024):\n")
    rows = [stat_row(n, pick(stats, n, period="Oct-Dec 2024")) for n in (PREREGISTERED[0], PREREGISTERED[1],
                                                                          EVERY_DAY)]
    w(md_table(head, rows) + "\n")

    w("## 4. Expiry, strike and width (exploratory)\n")
    w("Excess P&L per spread of *3+ down days in a row* over every session, held to expiry, holdout, by "
      "expiry and short-strike distance. 2–3 session expiries were suggested by the holdout's underlying odds, "
      "so treat them as leads, not confirmation.\n")
    for width in WIDTHS:
        rows = []
        for d in DISTANCES:
            row = [f"short put {distance_label(d)}"]
            for hh in DTES:
                x = pick(stats, PREREGISTERED[0], dte=hh, distance=d, width=width, exit="expiry")
                b = pick(stats, EVERY_DAY, dte=hh, distance=d, width=width, exit="expiry")
                row.append("n/a" if x is None or not x["n"] else
                           f"{money(x['excess'])} {held(x['excess_lo'], x['excess_hi'])} "
                           f"(signal {money(x['mean'])}, every day {money(b['mean'])})")
            rows.append(row)
        w(f"**${width} wide:**\n")
        w(md_table(["", *(f"{hh}-session expiry" for hh in DTES)], rows) + "\n")
    w("![Grid](grid.png)\n")
    w(closer_strikes_section(res))

    w("## 5. Take profits instead of holding to expiry (exploratory)\n")
    rows = []
    for ex in EXITS:
        for name in (PREREGISTERED[0], EVERY_DAY):
            rows.append(stat_row(f"{name}, {'held to expiry' if ex == 'expiry' else f'{ex}% take profit'}",
                                 pick(stats, name, exit=ex)))
    w(md_table(head, rows))
    w(rare_loss_note(res) + "\n")
    w(exits_across_structures(stats))
    w(breakeven_section(res))
    w(first_up_section(res))

    w("## 6. Selling put spreads every session, no signal (context)\n")
    w("What each spread earned on average when opened at 15:50 every session in the holdout, held to expiry: the "
      "structure's own expectancy before any dip signal.\n")
    rows = []
    for width in WIDTHS:
        for d in DISTANCES:
            row = [f"${width} wide, short {distance_label(d)}"]
            for hh in DTES:
                b = pick(stats, EVERY_DAY, dte=hh, distance=d, width=width, exit="expiry")
                row.append("n/a" if b is None else f"{money(b['mean'])} ({b['win']:.0f}% win)")
            rows.append(row)
    w(md_table(["", *(f"{hh}-session expiry" for hh in DTES)], rows) + "\n")

    st = res["statuses"]
    w("## 7. Data and caveats\n")
    w("- **Spreads priced:** " + ", ".join(f"{k}: {v:,}" for k, v in st.items()) + " (every structure on every "
      "session from 2024-10-01 that leaves room for its expiry).")
    w("- **Fills:** traded minute prices only (no bid/ask quotes); both legs must print in the same minute. "
      "Your real fills matched traded prices near the money, but these puts are 1–2% out of the money and up to "
      "5 sessions out, where spreads are wider: check the +$0.01 and +$0.03 slippage rows.")
    w("- **Settlement** at intrinsic value from SPY's close; early assignment and the exercise process are "
      "ignored. In practice you would close an in-the-money spread before expiry at about that value.")
    w("- **Small samples, one period.** Dip signals fire a few dozen times in 21 months, in a few episodes; "
      "the spring-2025 selloff and rebound weigh heavily. A put spread's payoff is lopsided: many small wins, "
      "occasional near-max losses, so averages swing with a handful of trades.")
    w("- **The holdout was already opened** for the underlying study; this test was fixed beforehand, but the "
      "exploratory cells are in-sample for this period.")
    w("\n## 8. Bottom line\n")
    w(bottom_line(res))
    return "\n".join(lines) + "\n"


def md_table(header, rows):
    return alerts.md_table(header, rows)


UP_LABELS = {"expiry": "held to expiry", "80": "80% take profit", "up": "first up day",
             "80+up": "first up day or 80%"}
UP_STRUCTURES = [(5, 1.0, 5), (3, 0.0, 5), (3, 0.5, 5), (2, 0.5, 5), (5, ITM, 1)]


def first_up_section(res):
    """Closing on the first up day (the stock dip rule's exit) against holding and the 80% take profit."""
    stats, single, table = res["stats"], res["single"], res["table"]
    wide = [c for c in res["costs"] if c != "base"][-1]
    out = [f"\n**Close on the first up day** (exploratory, added on request): the spread is closed at 15:50 on the "
           f"first session after entry that SPY is up from the previous close, or after {MAX_HOLD} sessions, at the "
           "open of the first minute both legs trade from 15:50 (plus slippage); if expiry comes first it settles "
           "as usual. *First up day or 80%* also takes an 80% profit if that comes sooner. After *3+ down days*, "
           f"holdout; averages per spread at traded prices and with {wide}; *one at a time* at traded prices.\n"]
    rows = []
    for h, d, w in UP_STRUCTURES:
        for ex in UP_LABELS:
            r = pick(stats, PREREGISTERED[0], dte=h, distance=d, width=w, exit=ex)
            r2 = pick(stats, PREREGISTERED[0], cost=wide, dte=h, distance=d, width=w, exit=ex)
            b = pick(stats, EVERY_DAY, dte=h, distance=d, width=w, exit=ex)
            o = single[(single["signal"] == PREREGISTERED[0]) & (single["period"] == "holdout")
                       & (single["dte"] == h) & (single["distance"] == d) & (single["width"] == w)
                       & (single["exit"].astype(str) == ex)]
            if r is None or not r["n"]:
                continue
            o = o.iloc[0] if len(o) else None
            rows.append([f"${w}, {distance_label(d, short=True)}, {h}-session" if ex == "expiry" else "",
                         UP_LABELS[ex], f"{int(r['n'])}", f"{r['win']:.0f}%", money(r["mean"]),
                         money(r2["mean"]) if r2 is not None else "", money(b["mean"]) if b is not None else "",
                         f"{r['ror']:+.1f}%", money(r["worst"]),
                         "" if o is None else f"{int(o['trades'])}, {money(o['total'], cents=False)}, "
                                              f"drawdown {money(-o['max_drawdown'], cents=False)}"])
    out.append(md_table(["Spread", "Exit", "Spreads", "Winners", "Average", f"Average, {wide}", "Every session",
                         "Return on risk", "Worst", "One at a time: trades, total, worst drawdown"], rows))
    ok = table[(table["status"] == "ok") & (table["cost"] == "base") & (table["exit"] == "up")
               & (table["day"] >= res["periods"]["holdout"][0])]
    reasons = ok["reason"].value_counts(normalize=True) * 100
    out.append("\nHow the first-up-day spreads ended (every structure and session, holdout): "
               + ", ".join(f"{k} {v:.0f}%" for k, v in reasons.items()) + ".")
    better, rows2 = {}, []
    five = [(h, d) for h in DTES for d in DISTANCES]
    for vs in ("expiry", "80"):
        diffs = []
        for h, d in five:
            a = pick(stats, PREREGISTERED[0], cost=wide, dte=h, distance=d, width=5, exit="up")
            b = pick(stats, PREREGISTERED[0], cost=wide, dte=h, distance=d, width=5, exit=vs)
            if a is not None and b is not None and a["n"] and b["n"]:
                diffs.append((a["mean"] - b["mean"], a["worst"] - b["worst"]))
        if diffs:
            dm, dw = np.array(diffs).T
            rows2.append([UP_LABELS[vs], f"{int((dm > 0).sum())} of {len(dm)}", money(np.median(dm)),
                          f"{int((dw > 0).sum())} of {len(dw)}", money(np.median(dw))])
    out.append(f"\nAcross all 24 $5-wide versions (expiry x strike), first up day against the other exits, after "
               f"3+ down days with {wide}:\n")
    out.append(md_table(["Compared with", "Higher average", "Median change in average", "Smaller worst loss",
                         "Median change in worst"], rows2))
    return "\n".join(out) + "\n"


def breakeven_section(res):
    """With vs without the breakeven stop: the primary version, every version, and what it changed."""
    stats, t = res["stats"], res["table"]
    out = [f"\n**Breakeven stop after {BREAKEVEN_AT}% profit** (exploratory, added on request): once a minute close "
           f"shows the spread worth {100 - BREAKEVEN_AT}% of the credit or less, a stop at the credit is armed; a later "
           "close at or above the credit exits there. The fill is that close, so a jump past breakeven (overnight, "
           "or a fast minute) still loses a little or more.\n"]
    rows = []
    for name in (PREREGISTERED[0], EVERY_DAY):
        for ex in EXITS:
            a = pick(stats, name, exit=exit_key(ex))
            b = pick(stats, name, exit=exit_key(ex, BREAKEVEN_AT))
            label = "hold to expiry" if ex == "expiry" else f"{ex}% take profit"
            rows.append([f"{name}, {label}", f"{money(a['mean'])} ({a['win']:.0f}%, worst {money(a['worst'])})",
                         f"{money(b['mean'])} ({b['win']:.0f}%, worst {money(b['worst'])})"])
    out.append(f"{structure_label(exit='expiry').rsplit(',', 1)[0]}, holdout: average P&L (winners, worst trade):\n")
    out.append(md_table(["Opened on, exit", "Without the stop", "With the breakeven stop"], rows))
    # Every version, and what the stop changed trade by trade (base costs, holdout).
    h0 = res["periods"]["holdout"][0]
    ok = t[(t["status"] == "ok") & (t["cost"] == "base") & (t["day"] >= h0)].copy()
    ok["exit"] = ok["exit"].astype(str)
    sig = SIGNALS[PREREGISTERED[0]](res["feats"]).fillna(False)
    rows = []
    for name in (PREREGISTERED[0], EVERY_DAY):
        g = ok if name == EVERY_DAY else ok[sig.reindex(ok["day"]).fillna(False).to_numpy()]
        for ex in EXITS:
            key = ["day", "dte", "distance", "width"]
            a = g[g["exit"] == exit_key(ex)].set_index(key)["pnl"]
            b = g[g["exit"] == exit_key(ex, BREAKEVEN_AT)].set_index(key)
            b = b.reindex(a.index)
            stopped = b["reason"] == "breakeven stop"
            saved = stopped & (a < 0)        # would have finished as a loss without the stop
            cut = stopped & (a > 0)          # would have finished as a win without the stop
            label = "hold to expiry" if ex == "expiry" else f"{ex}% take profit"
            versions = g[g["exit"].isin([exit_key(ex), exit_key(ex, BREAKEVEN_AT)])].pivot_table(
                index=["dte", "distance", "width"], columns="exit", values="pnl", aggfunc="mean")
            better = int((versions[exit_key(ex, BREAKEVEN_AT)] > versions[exit_key(ex)]).sum())
            rows.append([f"{name}, {label}", f"{money(a.mean())} → {money(b['pnl'].mean())}",
                         f"{better} of {len(versions)}", f"{int(stopped.sum()):,} of {len(a):,}",
                         f"{int(saved.sum()):,}: {money(a[saved].sum(), cents=False)} → "
                         f"{money(b.loc[saved, 'pnl'].sum(), cents=False)}",
                         f"{int(cut.sum()):,}: {money(a[cut].sum(), cents=False)} → "
                         f"{money(b.loc[cut, 'pnl'].sum(), cents=False)}"])
    out.append(f"\nAcross all {len(STRUCTURES)} expiry/strike/width versions (holdout, every spread pooled):\n")
    out.append(md_table(["Opened on, exit", "Average P&L without → with the stop", "Versions where the stop helped",
                         "Spreads stopped at breakeven", "Losers it saved (total without → with)",
                         "Winners it cut short (total without → with)"], rows))
    return "\n".join(out) + "\n"


def closer_strikes_section(res, dte=PRIMARY["dte"]):
    """Short strikes from just in the money to 2% below, $1 vs $5 wide: per spread and per dollar at risk,
    with the widest slippage tier, because a $1 spread's small credit makes fills matter much more."""
    stats, t = res["stats"], res["table"]
    h0 = res["periods"]["holdout"][0]
    wide = list(res["costs"])[-1]
    sig = SIGNALS[PREREGISTERED[0]](res["feats"]).fillna(False)
    ok = t[(t["status"] == "ok") & (t["cost"] == "base") & (t["day"] >= h0) & (t["dte"] == dte)
           & (t["exit"].astype(str) == "expiry")]
    ok = ok[sig.reindex(ok["day"]).fillna(False).to_numpy()]
    get = lambda name, d, w, ex, cost="base": pick(stats, name, dte=dte, distance=d, width=w, exit=ex,  # noqa: E731
                                                   cost=cost)
    ror = lambda r: "n/a" if r is None or not r["n"] else f"{r['ror']:+.1f}%"  # noqa: E731
    rows, flags = [], []
    for width in WIDTHS:
        for d in DISTANCES:
            g = ok[(ok["width"] == width) & (ok["distance"] == d)]
            hold, tp, tp_wide = (get(PREREGISTERED[0], d, width, ex, c) for ex, c in
                                 (("expiry", "base"), ("80", "base"), ("80", wide)))
            e_hold, e_tp, e_wide = (get(EVERY_DAY, d, width, ex, c) for ex, c in
                                    (("expiry", "base"), ("80", "base"), ("80", wide)))
            loss = g["max_loss"].mean()
            rows.append([f"${width} wide, short {distance_label(d)}",
                         f"{money(g['credit'].mean() * 100, cents=False, sign=False)} / "
                         f"{money(loss, cents=False, sign=False)}",
                         f"{money(hold['mean'])} ({hold['win']:.0f}%), {ror(hold)}" if hold is not None and hold["n"]
                         else "n/a",
                         f"{money(tp['mean'])} ({tp['win']:.0f}%), {ror(tp)}" if tp is not None and tp["n"] else "n/a",
                         f"**{ror(tp_wide)}**", f"{ror(e_hold)} / {ror(e_tp)} / {ror(e_wide)}",
                         f"{1000 / loss:.0f}" if loss else "n/a"])
            if e_hold is not None and e_tp is not None and e_tp["ror"] - e_hold["ror"] > 10:
                flags.append((width, d, e_hold["ror"], e_tp["ror"], e_wide["ror"]))
    out = [f"\n**Closer strikes and $1 vs $5 width** (exploratory, {dte}-session expiry, after *{PREREGISTERED[0]}*, "
           "holdout). Cells: average P&L per spread (winners), return on risk (P&L ÷ max loss). A $1 spread risks "
           "far less than a $5 one, so compare returns on risk, or how many spreads $1,000 of risk buys.\n",
           md_table(["Spread", "Credit / max loss per spread", "Hold to expiry", "80% take profit",
                     f"80% take profit, {wide}", "Every session: hold / 80% / 80% with " + wide.split()[0],
                     "Spreads per $1,000 at risk"], rows)]
    if flags:
        w, d, h, tp80, tpw = max(flags, key=lambda x: x[3] - x[2])
        out.append(f"\n**Caution on $1 spreads.** On ordinary sessions, ${w} spreads {distance_label(d)} return "
                   f"{h:+.1f}% held to expiry but {tp80:+.1f}% with the 80% take profit. If option prices are roughly "
                   "fair, an exit rule cannot add that much, so the take-profit fills are probably too optimistic "
                   "here: a $1 spread's value is the small difference between two nearly identical option prices, "
                   "and trade prints seconds apart can make it look cheap for a moment. With "
                   f"{wide} the same trade returns {tpw:+.1f}%. Treat the {wide} column as the realistic one for $1 "
                   "spreads, and compare signals with ordinary sessions at the same cost.")
    out.append("\n![Closer strikes](closer.png)\n")
    return "\n".join(out) + "\n"


def plot_closer(res, path, exit_rule="80"):
    """Return on risk after the primary dip signal by short strike and expiry, $1 vs $5 wide, at the widest
    slippage tier (the realistic one for $1 spreads, whose small credits make fills matter most)."""
    stats = res["stats"]
    cost = list(res["costs"])[-1]
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    for k, width in enumerate(WIDTHS):
        ax = _axes(fig, [0.12 + k * 0.46, 0.14, 0.36, 0.6])
        m = np.array([[np.nan if (x := pick(stats, PREREGISTERED[0], dte=h, distance=d, width=width,
                                              exit=exit_rule, cost=cost)) is None or not x["n"] else x["ror"]
                       for h in DTES] for d in DISTANCES], float)
        lim = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap=sp.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:+.1f}%", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(DTES)), [f"{h}d" for h in DTES])
        ax.set_yticks(range(len(DISTANCES)), [distance_label(d, short=True) for d in DISTANCES])
        ax.set_xlabel("Sessions to expiry", color=INK_2, fontsize=7.5)
        ax.set_title(f"${width} wide", color=INK, fontsize=9, loc="left")
    fig.text(0.02, 0.97, "Return on risk after 3+ down days, by short strike", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.91, f"Put spreads with an {exit_rule}% take profit, {cost}, holdout; P&L ÷ max loss per trade. "
             "Exploratory.", color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def exits_across_structures(stats):
    """Hold vs 50% vs 80% take profit over every expiry/strike/width, so one structure can't speak for all."""
    g = stats[(stats["period"] == "holdout") & (stats["cost"] == "base")].copy()
    g["exit"] = g["exit"].astype(str)
    g = g[g["exit"].isin(["expiry", "50", "80"])]
    rows = []
    for name in (*PREREGISTERED, EVERY_DAY):
        piv = g[g["signal"] == name].pivot_table(index=["dte", "distance", "width"], columns="exit", values="mean")
        best = piv.idxmax(axis=1).value_counts()
        rows.append([name, money(piv["expiry"].mean()), money(piv["50"].mean()), money(piv["80"].mean()),
                     f"{int((piv['50'] > piv['80']).sum())} of {len(piv)}",
                     ", ".join(f"{'hold' if k == 'expiry' else k + '%'} {v}" for k, v in best.items())])
    out = [f"\n**Across all {len(STRUCTURES)} expiry/strike/width versions** (holdout, average P&L per spread), so a "
           "single version cannot decide the exit rule:\n",
           md_table(["Opened on", "Hold to expiry", "50% take profit", "80% take profit", "50% beat 80% in",
                     "Best exit (count of versions)"], rows)]
    rows = []
    for width in WIDTHS:
        for hh in DTES:
            cells = []
            for ex in ("expiry", 50, 80):
                x = pick(stats, PREREGISTERED[0], dte=hh, distance=1.0, width=width, exit=ex)
                cells.append("n/a" if x is None or not x["n"] else f"{money(x['mean'])} ({x['win']:.0f}%)")
            rows.append([f"${width} wide, {hh}-session expiry", *cells])
    out += [f"\n*{PREREGISTERED[0]}*, short put 1% below: average P&L (winners) by exit:\n",
            md_table(["Version", "Hold to expiry", "50% take profit", "80% take profit"], rows)]
    return "\n".join(out) + "\n"


def rare_loss_note(res):
    """A high win rate with a lopsided payoff: how likely is the signal's loser count if its days were ordinary?"""
    t = res["table"]
    h0 = res["periods"]["holdout"][0]
    p = t[(t["status"] == "ok") & (t["dte"] == PRIMARY["dte"]) & (t["distance"] == PRIMARY["distance"])
          & (t["width"] == PRIMARY["width"]) & (t["exit"].astype(str) == "50") & (t["cost"] == "base")
          & (t["day"] >= h0)].set_index("day")
    mask = SIGNALS[PREREGISTERED[0]](res["feats"]).reindex(p.index).fillna(False).to_numpy(bool)
    if not mask.any():
        return ""
    n, losers = int(mask.sum()), int((p["pnl"].to_numpy()[mask] <= 0).sum())
    rate = (p["pnl"] <= 0).mean()
    chance = sum(math.comb(n, k) * rate ** k * (1 - rate) ** (n - k) for k in range(losers + 1))
    avg_loss = -p.loc[p["pnl"] <= 0, "pnl"].mean()
    return (f"\n**Read the 50% take profit with care.** {n - losers} of {n} signal spreads reached the target within "
            f"{PRIMARY['dte']} sessions ({losers} losers). On an ordinary session {rate:.0%} of these spreads never do, "
            f"and those lose about {money(avg_loss, sign=False)} each. Seeing {losers} or fewer losers in {n} trades by "
            f"luck alone has a probability of about {chance:.0%} (more, since signals cluster), and the resampled "
            "interval cannot include losers that simply did not happen. One max loss would cut the average by about "
            f"{money(avg_loss / n, sign=False)}.")


def robustness_section(res):
    """The strict entry rule drops thinly traded signal days: bound their effect and re-price every day
    with a looser entry rule."""
    stats, missing = res["stats"], res["missing"]
    r, b = pick(stats, PREREGISTERED[0]), pick(stats, EVERY_DAY)
    out = []
    if missing and r is not None and r["n"]:
        n, k = int(r["n"]), len(missing)
        avg_max_loss = (r["mean"] / (r["ror"] / 100)) if r["ror"] else np.nan  # mean max loss of priced spreads
        worst = (r["mean"] * n - k * avg_max_loss) / (n + k)
        neutral = (r["mean"] * n + k * b["mean"]) / (n + k)
        out.append(f"\n**Missing days.** On {k} of the {n + k} signal days the two legs never printed in the same "
                   "minute between 15:50 and the close, so the strict rule could not price them (" +
                   ", ".join(str(t.date()) for t, _ in missing) + "). Counting each as the every-session average "
                   f"gives {money(neutral)} per spread; counting each as a full max loss (about "
                   f"{money(avg_max_loss, sign=False)}) gives {money(worst)}.")
    rows = []
    for name in (PREREGISTERED[0], PREREGISTERED[1], EVERY_DAY):
        for cost in list(res["costs"])[:2]:
            x = pick(stats, name, cost=cost, exit=RELAXED)
            rows.append(stat_row(f"{name}{'' if cost == 'base' else f', {cost}'}", x))
    out.append("\n**Robustness: a looser entry rule on every day.** Each leg priced at its average print from 15:50 "
               "to the close, even when the legs never printed in the same minute (noisier on fast days, but it "
               "keeps the thinly traded days), held to expiry:\n")
    out.append(md_table(["Opened on", "Spreads (episodes)", "Winners", "Average P&L (95% interval)",
                         "Excess vs every session", "", "Return on risk", "Worst"], rows))
    return "\n".join(out) + "\n"


def primary_sentence(r):
    verdict = ("beat the same spread opened on a random session, beyond what chance explains"
               if r["excess_lo"] > 0 else "did worse than opening on a random session" if r["excess_hi"] < 0
               else "was not distinguishable from opening on a random session")
    return (f"Opening this spread after 3+ down days earned {money(r['mean'])} per spread on average "
            f"({r['win']:.0f}% winners) against {money(r['base_mean'])} on every session: an excess of "
            f"{money(r['excess'])} (95% interval {money(r['excess_lo'])} to {money(r['excess_hi'])}). It {verdict}.")


def bottom_line(res):
    stats = res["stats"]
    r, b = pick(stats, PREREGISTERED[0]), pick(stats, EVERY_DAY)
    c = pick(stats, PREREGISTERED[1])
    wide = pick(stats, PREREGISTERED[0], cost=list(res["costs"])[-1])
    parts = []
    if r is not None and r["n"]:
        parts.append(f"Primary test: {primary_sentence(r)}")
    if c is not None and c["n"]:
        parts.append(f"Connors entry: {money(c['mean'])} per spread over {int(c['n'])} spreads, excess "
                     f"{money(c['excess'])} ({held(c['excess_lo'], c['excess_hi'])}).")
    if wide is not None and wide["n"]:
        parts.append(f"With {list(res['costs'])[-1]}, the primary spread averages {money(wide['mean'])}.")
    return " ".join(parts) + (" The next months of signals are the only clean test left for any change to this "
                              "setup.")


# ---------------------------------------------------------------- charts

def _axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7.5)
    return ax


def plot_signals(res, path):
    stats = res["stats"]
    names = [*SIGNALS, EVERY_DAY]
    rows = [(n, pick(stats, n)) for n in names]
    rows = [(n, r) for n, r in rows if r is not None and r["n"]]
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.34, 0.16, 0.6, 0.66])
    y = np.arange(len(rows))[::-1]
    for yy, (n, r) in zip(y, rows):
        color = INK_2 if n == EVERY_DAY else SERIES[0] if n in PREREGISTERED else SERIES[1]
        ax.errorbar(r["mean"], yy, xerr=[[r["mean"] - r["mean_lo"]], [r["mean_hi"] - r["mean"]]], fmt="o",
                    color=color, ecolor=color, elinewidth=1.5, capsize=3, ms=6)
        ax.annotate(f"{money(r['mean'])} (n={int(r['n'])})", (r["mean_hi"], yy), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=BASELINE, lw=1)
    base = pick(stats, EVERY_DAY)
    ax.axvline(base["mean"], color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y, [n for n, _ in rows], fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Average P&L per spread, $ (95% interval)", color=INK_2, fontsize=8)
    ax.margins(x=0.25)
    fig.text(0.02, 0.97, "Put spreads opened after dips vs on every session", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{structure_label()}; holdout. Blue: pre-registered signals; dashed: every-session average.",
             color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_grid(res, path):
    stats = res["stats"]
    fig = plt.figure(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    for k, (title, signal, value) in enumerate((("Every session: average P&L per spread", EVERY_DAY, "mean"),
                                                ("3+ down days: excess over every session", PREREGISTERED[0],
                                                 "excess"))):
        ax = _axes(fig, [0.1 + k * 0.47, 0.14, 0.38, 0.58])
        m = np.array([[pick(stats, signal, dte=h, distance=d, width=5, exit="expiry")[value]
                       for h in DTES] for d in DISTANCES], float)
        lim = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap=sp.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, money(m[i, j]), ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(DTES)), [f"{h}d" for h in DTES])
        ax.set_yticks(range(len(DISTANCES)), [distance_label(d, short=True) for d in DISTANCES])
        ax.set_xlabel("Sessions to expiry", color=INK_2, fontsize=7.5)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
    fig.text(0.02, 0.97, "$5-wide put spreads held to expiry, holdout", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.91, "Left: what the spread earned on an average session. Right: how much more it earned "
             "after 3+ down days.\nExploratory except the pre-registered cell (5 sessions, 1% below).", color=INK_2,
             fontsize=7.5, va="top", linespacing=1.5)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_equity(res, path):
    t = res["table"]
    p = t[(t["status"] == "ok") & (t["dte"] == PRIMARY["dte"]) & (t["distance"] == PRIMARY["distance"])
          & (t["width"] == PRIMARY["width"]) & (t["exit"].astype(str) == "expiry") & (t["cost"] == "base")
          & (t["day"] >= res["periods"]["holdout"][0])].sort_values("day").set_index("day")
    mask = SIGNALS[PREREGISTERED[0]](res["feats"]).reindex(p.index).fillna(False).to_numpy(bool)
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.1, 0.2, 0.86, 0.58])
    sig = p[mask]
    ax.step(sig.index, sig["pnl"].cumsum(), where="post", color=SERIES[0], lw=2,
            label=f"After 3+ down days ({mask.sum()} spreads)")
    expected = np.cumsum(np.where(mask, p["pnl"].mean(), 0.0))[mask]
    ax.step(sig.index, expected, where="post", color=MUTED, lw=1.5, ls=(0, (4, 3)),
            label="Same number of spreads at the every-session average")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("Cumulative P&L per spread ($)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "The primary put spread, trade by trade", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.915, f"{structure_label()}; one spread per signal day, holdout.", color=INK_2, fontsize=7.5,
             va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def write_outputs(out_dir, res):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "spreads.csv", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["single"].to_csv(out / "one_at_a_time.csv", index=False)
    plot_signals(res, out / "signals.png")
    plot_grid(res, out / "grid.png")
    plot_closer(res, out / "closer.png")
    plot_equity(res, out / "equity.png")
    (out / "report.md").write_text(render_report(res))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Put credit spreads after SPY dips, priced with real option prices.")
    p.add_argument("--out", default="output/dip_spreads")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--slippage", type=float, default=0.0, help="$/share per leg per fill (default 0)")
    p.add_argument("--commission", type=float, default=0.0, help="$/contract per leg per fill (default 0)")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        daily, _ = mr.load_daily(sessions, args.cache_dir, args.refresh, (mr.TICKER, *mr.CONTEXT))
        first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
        minutes, _ = download.load_minute_bars(mr.TICKER, first, last, cache_dir=args.cache_dir,
                                               final_close=sessions["close"].iloc[-1])
        raw, _ = mr.daily_table(minutes, sessions)  # actual prices: strikes and settlement are not adjusted
    load = contract_loader(args.cache_dir, args.refresh)
    res = run_study(daily, raw, sessions, costs=sp.cost_tiers(args.slippage, args.commission), reps=args.reps,
                    seed=args.seed, load=load, timings=timings)
    with timed(timings, "outputs"):
        out = write_outputs(args.out, res)
    r = pick(res["stats"], PREREGISTERED[0])
    print(f"Dip put spreads: option contracts downloaded this run: {load.state['fetched']:,}")
    if r is not None and r["n"]:
        print(f"  primary ({structure_label()}): {int(r['n'])} spreads, {money(r['mean'])} each vs "
              f"{money(r['base_mean'])} every session; excess {money(r['excess'])} "
              f"[{money(r['excess_lo'])}, {money(r['excess_hi'])}]")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
