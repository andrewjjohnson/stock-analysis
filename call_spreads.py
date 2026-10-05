"""Call credit spreads on SPY, 30-45 days out with the short call near 0.30 delta: a trade simulation on Massive
option minute bars. Results are per spread (100 shares), before taxes; gross of commissions (Robinhood).

  uv run --env-file .env python call_spreads.py --out output/call_spreads

Question: does selling SPY call credit spreads 30-45 days out, short call near 0.30 delta (about a 70%
risk-neutral chance of expiring worthless), make money, and which version works best?

Fixed before any option price was looked at:
- An entry decision at 15:50 ET every session from 2024-10-01 whose expiry fits in the data. Expiry: the weekly
  expiry (Friday, or Thursday in a holiday week) closest to 30 or 45 calendar days out, and more than 21 days out.
- Short call: the $5-multiple strike (where 30-45-day SPY options trade) whose delta is closest to 0.20, 0.30 or
  0.40 (within 0.06), each strike's delta from its own implied volatility: Black-76 on the forward implied by the
  at-the-money call and put (put-call parity; rates and dividends ignored otherwise), using each contract's last
  trade from 14:50 to 15:50. Long call $5 or $10 higher.
- Exits: held to expiry; 50% take profit; closed at 21 days to expiry; 50% or 21 days, whichever first; 50% take
  profit with a stop when the loss reaches 2x the credit.
- Primary: 45 days, 0.30 delta, $5 wide, 50% or 21 days, every session. Everything else is exploratory.
- Entry filters (exploratory, known at 15:50): after 3+ up days; RSI(2) above 90; SPY below its 50-day average;
  volatile (20-day volatility in the top third of 2021-2024).
- Fills as in dip_spreads.py: both legs must trade in the same minute; entry at the first such minute from 15:50,
  as a working order through 10:30 the next morning (revised before any P&L was looked at: in the last 10 minutes
  alone the legs of these longer-dated spreads seldom print together); take profits fill at their level once a minute close is two slippages through it; the stop fills
  at the minute close that reaches it; the 21-day exit at the open of the first both-legs minute from 15:50 on the
  first session 21 or fewer days from expiry; otherwise settlement at intrinsic value from SPY's close at expiry
  (early assignment ignored). Costs: traded prices, +$0.03 and +$0.05 per share per leg per fill.
- A new spread every session, so trades overlap: averages are per spread with 95% intervals from 20-session
  blocks of entry days; one-at-a-time results (a new spread only after the last one closed) are shown too.
"""

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from massive import RESTClient  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alert_spreads as sp  # noqa: E402
import alerts  # noqa: E402
import dip_spreads as ds  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import outcomes  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
NS = outcomes.NS_PER_MINUTE
TICKER = "SPY"
DTE_TARGETS = (30, 45)                 # calendar days to expiry
DELTAS = (0.20, 0.30, 0.40)            # short call delta targets
DELTA_TOLERANCE = 0.06
CANDIDATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)  # where to look for the short strike
WIDTHS = (5, 10)
MIN_DAYS = 21                          # the expiry must be more than this many days out (the 21-day exit)
EXITS = {"held to expiry": {}, "50% take profit": {"tp": 50}, "21 days": {"days": 21},
         "50% or 21 days": {"tp": 50, "days": 21}, "50% with a 2x stop": {"tp": 50, "stop": 2.0}}
SLIPPAGES = {"traded prices": 0.0, "+$0.03": 0.03, "+$0.05": 0.05}
REALISTIC = "+$0.03"
PRIMARY = {"dte": 45, "delta": 0.30, "width": 5, "exit": "50% or 21 days", "filter": "every session"}
FILTERS = {"every session": lambda f: pd.Series(True, index=f.index),
           "after 3+ up days": lambda f: f["streak"] >= 3,
           "RSI(2) above 90": lambda f: f["rsi_2"] > 90,
           "below the 50-day average": lambda f: f["dist_sma50"] < 0,
           "volatile": lambda f: f["volatile"]}
BLOCK = 20
LEG_HISTORY_DAYS = 56                  # each contract is fetched from 8 weeks before its expiry


# ---------------------------------------------------------------- pricing helpers

def b76_call(F, K, sigma, tau):
    """Black-76 call value per share (no discounting)."""
    s = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return F * norm.cdf(d1) - K * norm.cdf(d1 - s)


def implied_vol(price, F, K, tau, lo=0.005, hi=3.0, steps=60):
    """Implied volatility of a call by bisection; NaN when the price is outside (intrinsic, F)."""
    price, F, K, tau = (np.asarray(x, float) for x in np.broadcast_arrays(price, F, K, tau))
    ok = (price > np.maximum(F - K, 0) + 1e-6) & (price < F) & (tau > 0)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(steps):
        m = 0.5 * (a + b)
        high = b76_call(F, K, m, tau) > price
        b, a = np.where(high, m, b), np.where(high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def call_delta(F, K, sigma, tau):
    s = sigma * np.sqrt(tau)
    return norm.cdf((np.log(F / K) + 0.5 * s * s) / s)


def strike_for_delta(F, sigma, tau, delta):
    """The strike whose Black-76 call delta is `delta` at volatility `sigma`."""
    s = sigma * np.sqrt(tau)
    return F * np.exp(0.5 * s * s - norm.ppf(delta) * s)


def last_print(df, start_ns, end_ns):
    """The last minute close of one contract in [start, end); NaN if it did not trade then."""
    if df is None or df.empty:
        return np.nan
    t = outcomes._ns(df["ts"])
    i = np.searchsorted(t, end_ns) - 1
    return float(df["close"].iloc[i]) if i >= 0 and t[i] >= start_ns else np.nan


# ---------------------------------------------------------------- data

def make_client():
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python call_spreads.py")
    return RESTClient(api_key=key, retries=10)


def weekly_chain(first, last, cache_dir="data/cache", refresh=False):
    """{weekly expiry: sorted call strikes}: each week's Thursday-Friday listings, keeping the later expiry."""
    path = Path(cache_dir) / "options_chains" / f"{TICKER}_weekly_calls_{first}_{last}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        client, rows = make_client(), []
        start = pd.Timestamp(first)
        for monday in pd.date_range(start - pd.Timedelta(days=start.dayofweek), last, freq="7D"):
            got = [(c.expiration_date, c.strike_price) for c in client.list_options_contracts(
                underlying_ticker=TICKER, contract_type="call", expiration_date_gte=str((monday + pd.Timedelta(days=3)).date()),
                expiration_date_lte=str((monday + pd.Timedelta(days=4)).date()), expired=True, limit=1000)]
            if got:
                latest = max(e for e, _ in got)
                rows += [(e, k) for e, k in got if e == latest]
        df = pd.DataFrame(rows, columns=["expiry", "strike"]).drop_duplicates()
        df = df.assign(expiry=pd.to_datetime(df["expiry"]).astype("datetime64[ns]"), strike=df["strike"].astype(float))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return {pd.Timestamp(e): np.sort(g["strike"].to_numpy(float)) for e, g in df.groupby("expiry")}


def leg_ticker(expiry, right, strike):
    return sp.option_ticker(expiry, right, strike, TICKER)


def leg_start(expiry):
    return str(max(pd.Timestamp(expiry) - pd.Timedelta(days=LEG_HISTORY_DAYS), pd.Timestamp(ds.OPTIONS_START)).date())


def prefetch(requests, load, workers=8):
    """Load (ticker, start, end) requests in parallel; each contract is cached on its first load."""
    def one(r):
        try:
            load(*r)
        except sp.NotInPlan:
            pass
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, sorted(set(requests))))


def pick_expiry(day, expiries, last_day, target):
    """The weekly expiry closest to `target` calendar days after `day`, more than MIN_DAYS out and in the data."""
    ok = [e for e in expiries if (e - day).days > MIN_DAYS and e <= last_day]
    return min(ok, key=lambda e: (abs((e - day).days - target), e)) if ok else None


def nearest(strikes, k):
    return float(strikes[np.argmin(np.abs(strikes - k))])


# ---------------------------------------------------------------- choosing the strikes

def plan_entries(days, sessions, raw, chain, load, workers=8, rights=("C",)):
    """For every session and target expiry: the forward and at-the-money volatility from the at-the-money call and
    put, the deltas of candidate strikes, and the short strike for each delta target on each side in `rights`
    ("C" calls, "P" puts; see choose_shorts). Returns a DataFrame."""
    last_day = days[-1]
    chain = {e: k[np.isclose(k % 5, 0)] for e, k in chain.items()}  # $5 strikes: where these expiries trade
    base = []
    for t in days:
        spot = raw.loc[t, "snap"]
        if np.isnan(spot):
            continue
        for target in DTE_TARGETS:
            e = pick_expiry(t, sorted(chain), last_day, target)
            if e is None or not len(chain[e]):
                continue
            k0 = nearest(chain[e], spot)
            base.append({"day": t, "target": target, "expiry": e, "spot": spot, "k0": k0})
    plan = pd.DataFrame(base)
    reqs = [(leg_ticker(r.expiry, right, r.k0), leg_start(r.expiry), str(r.expiry.date()))
            for r in plan.itertuples() for right in ("C", "P")]
    prefetch(reqs, load, workers)

    def window(t):
        dec = (sessions.loc[t, "close"] - pd.Timedelta(minutes=mr.DECISION_MINUTES)).value
        return dec - 60 * NS, dec

    def price(e, right, k, t):
        try:
            df = load(leg_ticker(e, right, k), leg_start(e), str(e.date()))
        except sp.NotInPlan:
            return np.nan
        return last_print(df, *window(t))

    c0 = np.array([price(r.expiry, "C", r.k0, r.day) for r in plan.itertuples()])
    p0 = np.array([price(r.expiry, "P", r.k0, r.day) for r in plan.itertuples()])
    plan["forward"] = plan["k0"] + c0 - p0
    plan["tau"] = [((sessions.loc[r.expiry, "close"] - (sessions.loc[r.day, "close"] - pd.Timedelta(minutes=10)))
                    / pd.Timedelta(days=365.25)) for r in plan.itertuples()]
    plan["atm_vol"] = implied_vol(c0, plan["forward"], plan["k0"], plan["tau"])
    for right in rights:
        choose_shorts(plan, chain, right, price, load, workers)
    return plan


def choose_shorts(plan, chain, right, price, load, workers=8):
    """Add the short strike for each delta target on one side (columns short_/delta_ for calls, put_short_/
    put_delta_ for puts, delta as a positive number): candidates at the at-the-money volatility, then each
    candidate's own implied volatility (a put through parity: C = P + F - K) and delta."""
    prefix = "" if right == "C" else "put_"
    cands = {}
    for r in plan.itertuples():
        if np.isnan(r.atm_vol):
            continue
        ks = {nearest(chain[r.expiry], strike_for_delta(r.forward, r.atm_vol, r.tau, d if right == "C" else 1 - d))
              for d in CANDIDATES}
        cands[r.Index] = sorted(ks)
    prefetch([(leg_ticker(plan.at[i, "expiry"], right, k), leg_start(plan.at[i, "expiry"]),
               str(plan.at[i, "expiry"].date())) for i, ks in cands.items() for k in ks], load, workers)
    for d in DELTAS:
        plan[f"{prefix}short_{d:g}"], plan[f"{prefix}delta_{d:g}"] = np.nan, np.nan
    for i, ks in cands.items():
        r = plan.loc[i]
        k = np.array(ks)
        px = np.array([price(r["expiry"], right, x, r["day"]) for x in ks])
        call_px = px if right == "C" else px + r["forward"] - k
        dl = call_delta(r["forward"], k, implied_vol(call_px, r["forward"], k, r["tau"]), r["tau"])
        dl = dl if right == "C" else 1 - dl
        for d in DELTAS:
            if np.isnan(dl).all():
                continue
            j = int(np.nanargmin(np.abs(dl - d)))
            if abs(dl[j] - d) <= DELTA_TOLERANCE:
                plan.at[i, f"{prefix}short_{d:g}"], plan.at[i, f"{prefix}delta_{d:g}"] = ks[j], dl[j]


def build_trades(plan, sessions, raw, chain, load, workers=8):
    """Price every planned spread (short strike per delta target, long $5 / $10 higher): dict keyed by
    (day, target, delta, width) of dip_spreads.price_spread results for calls."""
    planned = []
    later = {d: (sessions.loc[n, "open"] + pd.Timedelta(minutes=60)).value
             for d, n in zip(sessions.index[:-1], sessions.index[1:])}  # a working order lasts to 10:30 next day
    for r in plan.itertuples():
        for d in DELTAS:
            ks = plan.at[r.Index, f"short_{d:g}"]
            if np.isnan(ks):
                continue
            for w in WIDTHS:
                above = chain[r.expiry][chain[r.expiry] >= ks + w - 1e-9]
                if not len(above):
                    continue
                kl = float(above[0])
                planned.append({"day": r.day, "target": r.target, "delta": d, "delta_actual": plan.at[r.Index,
                                f"delta_{d:g}"], "width": kl - ks, "nominal_width": w, "expiry": r.expiry,
                                "entry_until": later.get(r.day),
                                "spot": r.spot, "short_strike": ks, "long_strike": kl, "right": "C",
                                "short_ticker": leg_ticker(r.expiry, "C", ks), "long_ticker": leg_ticker(r.expiry, "C", kl)})
    prefetch([(p[k], leg_start(p["expiry"]), str(p["expiry"].date())) for p in planned
              for k in ("short_ticker", "long_ticker")], load, workers)
    trades = {}
    for p in planned:
        tr = ds.price_spread(dict(p), leg_start(p["expiry"]), sessions, raw, load)
        trades[p["day"], p["target"], p["delta"], p["nominal_width"]] = tr
    return trades


# ---------------------------------------------------------------- exits

def simulate(trade, rule, slippage, sessions):
    """(P&L $ per spread, exit reason, exit session) for one exit rule (an EXITS entry)."""
    credit = trade["credit_traded"] - 2 * slippage
    watch, width = trade["watch"], trade["width"]
    first, out = len(watch), None
    if "tp" in rule:
        level = credit * (1 - rule["tp"] / 100)
        hit = np.flatnonzero(watch <= level - 2 * slippage)
        if hit.size:
            first, out = hit[0], ((credit - level) * 100, "take profit")
    if "stop" in rule:
        hit = np.flatnonzero(watch >= credit * (1 + rule["stop"]))
        if hit.size and hit[0] < first:
            first = hit[0]
            out = ((credit - min(watch[first], width) - 2 * slippage) * 100, "stop")
    if "days" in rule:
        day = next((d for d in sessions.index[(sessions.index > trade["day"]) & (sessions.index <= trade["expiry"])]
                    if (trade["expiry"] - d).days <= rule["days"]), None)
        if day is not None:
            at = (sessions.loc[day, "close"] - pd.Timedelta(minutes=mr.DECISION_MINUTES)).value
            k = int(np.searchsorted(trade["watch_ts"], at))
            if k < first:
                first = k
                out = ((credit - min(trade["watch_open"][k], width) - 2 * slippage) * 100, f"{rule['days']} days")
    if out is not None:
        return out[0], out[1], pd.Timestamp(trade["watch_days"][first])
    return (credit - trade["settle"]) * 100, "expiry", trade["expiry"]


def pnl_table(trades, sessions):
    rows = []
    for (day, target, d, w), tr in trades.items():
        base = {"day": day, "dte": target, "delta": d, "width": w, "status": tr["status"],
                "expiry": tr.get("expiry"), "delta_actual": tr.get("delta_actual")}
        if tr["status"] != "ok":
            rows.append(base)
            continue
        for name, rule in EXITS.items():
            for cost, slip in SLIPPAGES.items():
                pnl, reason, exit_day = simulate(tr, rule, slip, sessions)
                credit = tr["credit_traded"] - 2 * slip
                rows.append({**base, "exit": name, "cost": cost, "pnl": pnl, "reason": reason, "exit_day": exit_day,
                             "credit": credit, "max_loss": (tr["width"] - credit) * 100,
                             "days_held": (exit_day - day).days, "settle": tr["settle"],
                             "above_short": tr["expiry_close"] > tr["short_strike"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def stats_for(g, mask, idx):
    """A filter's spreads among one structure/exit/cost's entries: count, average with a block-bootstrap interval,
    excess over every session, win rate, return on risk, worst, days held."""
    pnl, ml = g["pnl"].to_numpy(float), g["max_loss"].to_numpy(float)
    n = int(mask.sum())
    out = {"n": n, "base_mean": pnl.mean()}
    if n == 0:
        return out
    sm, sp_ = mask[idx], pnl[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (sp_ * sm).sum(1) / sm.sum(1)
        diffs = means - sp_.mean(1)
    lo, hi = alerts.wilson(int((pnl[mask] > 0).sum()), n)
    out.update(mean=pnl[mask].mean(), mean_lo=np.nanpercentile(means, 2.5), mean_hi=np.nanpercentile(means, 97.5),
               excess=pnl[mask].mean() - pnl.mean(), excess_lo=np.nanpercentile(diffs, 2.5),
               excess_hi=np.nanpercentile(diffs, 97.5), win=(pnl[mask] > 0).mean() * 100, win_lo=lo, win_hi=hi,
               ror=pnl[mask].sum() / ml[mask].sum() * 100, worst=pnl[mask].min(),
               days=g["days_held"].to_numpy()[mask].mean(), credit=g["credit"].to_numpy()[mask].mean())
    return out


def one_at_a_time(g):
    """Entries taken only after the previous spread closed (by exit session)."""
    taken, until = [], pd.Timestamp.min
    for r in g.sort_values("day").itertuples():
        if r.day > until:
            taken.append(r)
            until = r.exit_day
    t = pd.DataFrame(taken)
    pnl = t["pnl"].to_numpy(float) if len(t) else np.array([])
    return {"trades": len(pnl), "total": pnl.sum(), "avg": pnl.mean() if len(pnl) else np.nan,
            "win": (pnl > 0).mean() * 100 if len(pnl) else np.nan,
            "max_dd": sp.max_drawdown(pnl) if len(pnl) else np.nan, "worst": pnl.min() if len(pnl) else np.nan}


def analyze(table, feats, reps, seed):
    ok = table[table["status"] == "ok"]
    masks = {name: rule(feats).fillna(False) for name, rule in FILTERS.items()}
    rows = []
    for (dte, d, w, ex, cost), g in ok.groupby(["dte", "delta", "width", "exit", "cost"], sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        idx = mr.block_indices(len(g), reps, seed, block=BLOCK)
        for name, m in masks.items():
            mask = m.reindex(g["day"]).fillna(False).to_numpy(bool)
            row = {"dte": dte, "delta": d, "width": w, "exit": ex, "cost": cost, "filter": name,
                   **stats_for(g, mask, idx)}
            if name == "every session":
                row.update({f"single_{k}": v for k, v in one_at_a_time(g).items()})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the study

def run_study(raw, feats, sessions, chain, load, *, reps=2000, seed=0, workers=8, timings=None):
    """raw: SPY daily table (actual prices); feats: meanrev features plus a boolean `volatile`; chain: weekly_chain.
    No file I/O beyond the option loader's cache."""
    timings = {} if timings is None else timings
    days = raw.index[raw.index >= pd.Timestamp(ds.OPTIONS_START)]
    opt_sessions = sessions.loc[days]
    with timed(timings, "strikes"):
        plan = plan_entries(days, opt_sessions, raw.loc[days], chain, load, workers)
    with timed(timings, "option data"):
        trades = build_trades(plan, opt_sessions, raw.loc[days], chain, load, workers)
    with timed(timings, "simulation"):
        table = pnl_table(trades, opt_sessions)
        stats = analyze(table, feats.loc[days], reps, seed)
    return {"plan": plan, "table": table, "stats": stats, "first": days[0], "last": days[-1],
            "spy": raw.loc[days, "close"]}


# ---------------------------------------------------------------- report

def money(v, cents=False):
    return sp.money(v, sign=True, cents=cents)


def pick(stats, cost=REALISTIC, **kw):
    s = {**PRIMARY, **kw}
    m = stats[(stats["dte"] == s["dte"]) & np.isclose(stats["delta"], s["delta"]) & (stats["width"] == s["width"])
              & (stats["exit"] == s["exit"]) & (stats["cost"] == cost) & (stats["filter"] == s["filter"])]
    return m.iloc[0] if len(m) else None


def label(dte=None, delta=None, width=None, exit=None):
    s = {**PRIMARY, **{k: v for k, v in dict(dte=dte, delta=delta, width=width, exit=exit).items() if v is not None}}
    return f"{s['dte']} days, {s['delta']:.2f} delta, ${s['width']} wide, {s['exit']}"


def render_report(res):
    st, table = res["stats"], res["table"]
    lines = []
    w = lines.append
    w("# Call credit spreads on SPY, 30-45 days out\n")
    w(f"Real SPY option prices (Massive minute bars), entries from {res['first']:%Y-%m-%d} to the last that expire by "
      f"{res['last']:%Y-%m-%d}. A spread is sold at 15:50 every session (so trades overlap); short call by delta, "
      "long call $5 or $10 higher. Dollars per spread (100 shares). *Return on risk* = total P&L / total max loss. "
      "Ranges are 95% intervals from 20-session blocks of entry days. Costs per share per leg per fill; the "
      f"main tables use **{REALISTIC}**.\n")
    w(bottom_line(res))
    w("## 1. The primary version\n")
    w(primary_section(res))
    w("## 2. Every expiry, delta and width (50% or 21 days)\n")
    w(grid_section(res))
    w("## 3. Exits\n")
    w(exit_section(res))
    w("## 4. Entry filters\n")
    w(filter_section(res))
    w("## 5. Did the deltas match what happened?\n")
    w(delta_section(res))
    w("## 6. Data\n")
    w(data_section(res))
    return "\n".join(lines) + "\n"


def bottom_line(res):
    st = res["stats"]
    r = pick(st)
    if r is None or not r["n"]:
        return "**Bottom line.** No primary spreads could be priced.\n"
    spy = res["spy"]
    move = (spy.iloc[-1] / spy.iloc[0] - 1) * 100
    best = st[(st["cost"] == REALISTIC) & (st["filter"] == "every session") & (st["n"] >= 100)].sort_values(
        "mean", ascending=False).iloc[0]
    return (f"**Bottom line.** The primary version ({label()}) averaged {money(r['mean'])} a spread over {int(r['n'])} "
            f"entries ({money(r['mean_lo'])} to {money(r['mean_hi'])}), {r['win']:.0f}% winners, return on risk "
            f"{r['ror']:+.1f}%, worst {money(r['worst'])}, with {REALISTIC} slippage. SPY rose {move:+.0f}% over the "
            f"period. The best of every version: {label(best['dte'], best['delta'], best['width'], best['exit'])}, "
            f"{money(best['mean'])} a spread ({best['win']:.0f}% winners).\n")


def primary_section(res):
    st = res["stats"]
    rows = []
    for cost in SLIPPAGES:
        r = pick(st, cost=cost)
        if r is None or not r["n"]:
            continue
        rows.append([cost, f"{int(r['n'])}", money(r["credit"] * 100), f"{r['win']:.0f}%",
                     f"{money(r['mean'])} ({money(r['mean_lo'])} to {money(r['mean_hi'])})", f"{r['ror']:+.1f}%",
                     money(r["worst"]), f"{r['days']:.0f}",
                     f"{int(r['single_trades'])}, {money(r['single_total'])}, drawdown {money(-r['single_max_dd'])}"])
    return (f"**{label()}**, entered every session:\n\n"
            + alerts.md_table(["Costs", "Spreads", "Avg credit", "Winners", "Average (95%)", "Return on risk", "Worst",
                               "Avg days held", "One at a time: trades, total, worst drawdown"], rows) + "\n")


def grid_section(res):
    st = res["stats"]
    rows = []
    for dte in DTE_TARGETS:
        for d in DELTAS:
            row = [f"{dte} days", f"{d:.2f}"]
            for wd in WIDTHS:
                r = pick(st, dte=dte, delta=d, width=wd, exit="50% or 21 days")
                row.append("" if r is None or not r["n"] else
                           f"{money(r['mean'])}, {r['win']:.0f}% win, {r['ror']:+.1f}% on risk (n={int(r['n'])})")
            rows.append(row)
    return (f"Average per spread with {REALISTIC}:\n\n"
            + alerts.md_table(["Expiry", "Short delta", "$5 wide", "$10 wide"], rows) + "\n\n![Grid](grid.png)\n")


def exit_section(res):
    st = res["stats"]
    rows = []
    for ex in EXITS:
        r = pick(st, exit=ex)
        if r is None or not r["n"]:
            continue
        rows.append([ex, f"{r['win']:.0f}%", money(r["mean"]), f"{r['ror']:+.1f}%", money(r["worst"]), f"{r['days']:.0f}",
                     f"{int(r['single_trades'])}, {money(r['single_total'])}, drawdown {money(-r['single_max_dd'])}"])
    out = [f"{label(exit='*')}".replace("*", "each exit") + f", {REALISTIC}:\n",
           alerts.md_table(["Exit", "Winners", "Average", "Return on risk", "Worst", "Avg days held",
                            "One at a time: trades, total, worst drawdown"], rows)]
    g = st[(st["cost"] == REALISTIC) & (st["filter"] == "every session")]
    piv = g.pivot_table(index=["dte", "delta", "width"], columns="exit", values="mean")
    best = piv.idxmax(axis=1).value_counts()
    out.append("\nBest exit by average across all 12 versions: " + ", ".join(f"{k} {v}" for k, v in best.items()) + ".")
    return "\n".join(out) + "\n"


def filter_section(res):
    st = res["stats"]
    rows = []
    for name in FILTERS:
        r = pick(st, filter=name)
        if r is None or not r["n"]:
            continue
        rows.append([name, f"{int(r['n'])}", f"{r['win']:.0f}%",
                     f"{money(r['mean'])} ({money(r['mean_lo'])} to {money(r['mean_hi'])})",
                     "" if name == "every session" else f"{money(r['excess'])} ({money(r['excess_lo'])} to "
                                                        f"{money(r['excess_hi'])})", f"{r['ror']:+.1f}%", money(r["worst"])])
    return (f"{label()}, {REALISTIC}, by entry condition (known at 15:50):\n\n"
            + alerts.md_table(["Entered when", "Spreads", "Winners", "Average (95%)", "Excess vs every session",
                               "Return on risk", "Worst"], rows) + "\n")


def delta_section(res):
    t = res["table"]
    t = t[(t["status"] == "ok") & (t["exit"] == "held to expiry") & (t["cost"] == "traded prices") & (t["width"] == 5)]
    rows = []
    for (dte, d), g in t.groupby(["dte", "delta"]):
        above = g["above_short"].astype(float)  # an object column of bools averages wrongly
        rows.append([f"{dte} days", f"{d:.2f}", f"{g['delta_actual'].astype(float).mean():.2f}",
                     f"{above.mean() * 100:.0f}%", f"{len(g)}"])
    return ("A call's delta is roughly the market's odds of it finishing in the money. Share of entries where SPY "
            "closed above the short strike at expiry, against the average delta at entry:\n\n"
            + alerts.md_table(["Expiry", "Target delta", "Average delta at entry", "Finished above the short strike",
                               "Entries"], rows) + "\n")


def data_section(res):
    t = res["table"]
    s = t.drop_duplicates(["day", "dte", "delta", "width"])["status"].value_counts()
    plan = res["plan"]
    found = {d: int(plan[f"short_{d:g}"].notna().sum()) for d in DELTAS}
    return ("- Planned entries (session x expiry target): " + f"{len(plan):,}; short strike found within "
            f"{DELTA_TOLERANCE} of the target delta: " + ", ".join(f"{d:.2f}: {n:,}" for d, n in found.items()) + ".\n"
            "- Spreads priced: " + ", ".join(f"{k}: {v:,}" for k, v in s.items()) + ".\n"
            "- Deltas come from each strike's implied volatility (Black-76 on the parity forward), with rates and "
            "dividends otherwise ignored; trade prints, not quotes.\n"
            "- The period was a strong rise for SPY with one sharp fall (spring 2025): a bearish structure like this "
            "is judged on a market that mostly went against it.\n")


def plot_grid(res, path):
    st = res["stats"]
    fig = plt.figure(figsize=(8, 3.8), dpi=150, facecolor=SURFACE)
    for k, wd in enumerate(WIDTHS):
        ax = sds._axes(fig, [0.1 + k * 0.47, 0.16, 0.36, 0.56])
        m = np.array([[pick(st, dte=dte, delta=d, width=wd, exit="50% or 21 days")["mean"] for d in DELTAS]
                      for dte in DTE_TARGETS], float)
        lim = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap=sp.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, money(m[i, j]).replace("$", r"\$"), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(DELTAS)), [f"{d:.2f}" for d in DELTAS])
        ax.set_yticks(range(len(DTE_TARGETS)), [f"{t} days" for t in DTE_TARGETS])
        ax.set_xlabel("Short call delta", color=INK_2, fontsize=7.5)
        ax.set_title(f"\\${wd} wide", color=INK, fontsize=8.5, loc="left")
    fig.text(0.02, 0.97, "SPY call credit spreads: average P&L per spread", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.9, f"Entered every session, 50% take profit or 21 days to expiry, {REALISTIC} per leg.".replace(
        "$", r"\$"), color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_equity(res, path):
    t = res["table"]
    g = t[(t["status"] == "ok") & (t["dte"] == PRIMARY["dte"]) & np.isclose(t["delta"], PRIMARY["delta"])
          & (t["width"] == PRIMARY["width"]) & (t["exit"] == PRIMARY["exit"]) & (t["cost"] == REALISTIC)]
    taken, until = [], pd.Timestamp.min
    for r in g.sort_values("day").itertuples():
        if r.day > until:
            taken.append(r)
            until = r.exit_day
    s = pd.DataFrame(taken).sort_values("exit_day")
    fig = plt.figure(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.1, 0.18, 0.78, 0.6])
    ax.step(s["exit_day"], s["pnl"].cumsum(), where="post", color=SERIES[0], lw=2, label="One spread at a time")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel(r"Cumulative P&L, \$ per spread", color=INK_2, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(res["spy"].index, res["spy"], color=MUTED, lw=1, label="SPY (right axis)")
    ax2.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7)
    for sp_ in ax2.spines.values():
        sp_.set_visible(False)
    fig.text(0.02, 0.97, "The primary call spread, one at a time", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"{label()}, {REALISTIC} per leg; by exit date.".replace("$", r"\$"), color=INK_2,
             fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPY call credit spreads 30-45 days out by delta, with real option prices.")
    p.add_argument("--out", default="output/call_spreads")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    with timed(timings, "data"):
        minutes, _ = download.load_minute_bars(TICKER, first, last, cache_dir=args.cache_dir,
                                               final_close=sessions["close"].iloc[-1])
        raw, _ = mr.daily_table(minutes, sessions)
        adj, _ = mr.adjust_dividends(raw, mr.load_dividends(TICKER, first, last, args.cache_dir, args.refresh))
        feats = mr.ticker_features(adj).reindex(sessions.index)
        feats["volatile"] = feats["rv20"] >= feats.loc[feats.index <= mr.DESIGN_END, "rv20"].quantile(2 / 3)
        chain = weekly_chain(ds.OPTIONS_START, last, args.cache_dir, args.refresh)
    load = ds.contract_loader(args.cache_dir, args.refresh)
    res = run_study(raw, feats, sessions, chain, load, reps=args.reps, seed=args.seed, workers=args.workers,
                    timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "spreads.csv", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["plan"].to_csv(out / "entries.csv", index=False)
    plot_grid(res, out / "grid.png")
    plot_equity(res, out / "equity.png")
    (out / "report.md").write_text(render_report(res))
    print(bottom_line(res))
    print(f"option contracts downloaded this run: {load.state['fetched']:,}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
