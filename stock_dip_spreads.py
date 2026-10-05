"""Put credit spreads after dips in large single stocks, priced with Massive option minute bars.

  uv run --env-file .env python stock_dip_spreads.py --out output/stock_dip_spreads

A follow-up to dip_spreads.py's SPY put spreads after 3+ down days: this tests whether the signal
carries over to single stocks (MSFT, AAPL, AMZN and META by
default), whether adding RSI(2) < 10 as a second trigger adds good trades, and how many trades the tickers
give together, with SPY's spread alongside.

Declared before any single-stock option price was looked at:
- Primary, per ticker: "3+ down days in a row" at 15:50 ET. Short put at the listed strike at or below 1%
  under the 15:50 price; long put at the listed strike nearest 1.5% of the price lower (about SPY's $5
  width once the stocks' higher volatility is allowed for); the first listed expiry at least 2 sessions out
  (weeklies expire on Fridays, so 2-6 sessions); 80% take profit, else held to expiry. Spreads that would be
  open over an earnings report are skipped. Judged against the same spread opened on every session (also
  skipping earnings), at traded prices and with +$0.05 slippage per leg.
- Second: the same with "3+ down days or RSI(2) < 10".
- Exploratory: at the money and 2% below, holding to expiry, RSI(2) < 10 alone, keeping the earnings
  trades, other slippage, and the combined view with SPY.

Data notes:
- Earnings dates are not in the data plan. In each reporting window (the 22nd of Jan/Apr/Jul/Oct to the 8th
  of the next month, which fits these four companies) two sessions are flagged: the biggest stock-specific
  volume jump (volume over its prior 20-session median, divided by the same ratio for SPY) and the biggest
  stock-specific overnight gap (the stock's open-to-previous-close move minus SPY's). Spreads open over a
  flagged session or the session either side are skipped. Either signal alone is fooled by big news days;
  together, with the margin, they err on the side of skipping a few extra trades. This uses hindsight, but
  only to stand in for the published earnings calendar, which is known weeks ahead. --earnings-csv replaces
  the flags with exact dates (columns ticker,date: the first session after each report). The report lists
  the sessions flagged.
- Strikes and expiries come from Massive's options contracts reference (the puts listed for each week).
- A stock split inside the option window would make split-adjusted prices disagree with the strikes, so it
  stops the run.
- Entry, take profit and settlement follow dip_spreads.py: both legs must trade in the same minute from
  15:50, early assignment is ignored, and results are per spread (100 shares), before commissions.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from massive import RESTClient  # noqa: E402

import alert_spreads as sp  # noqa: E402
import alerts  # noqa: E402
import dip_spreads as ds  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("MSFT", "AAPL", "AMZN", "META")
DISTANCES = (0.0, 1.0, 2.0)    # short strike: % below the 15:50 price (0 = at the money)
WIDTH_PCT = 1.5                # long strike: the listed strike nearest this % of the price lower
MIN_DTE, MAX_DTE = 2, 7        # the first listed expiry at least MIN_DTE sessions out, at most MAX_DTE
EXITS = ("expiry", 80)
SLIPPAGE_STEPS = (0.03, 0.05, 0.10)
REALISTIC = "+$0.05 slippage"  # single-stock options trade wider than SPY's
SIGNALS = {
    "3+ down days in a row": lambda f: f["streak"] <= -3,
    "3+ down days or RSI(2) < 10": lambda f: (f["streak"] <= -3) | (f["rsi_2"] < 10),
    "RSI(2) < 10": lambda f: f["rsi_2"] < 10,
}
PREREGISTERED = ("3+ down days in a row", "3+ down days or RSI(2) < 10")
EVERY_DAY = ds.EVERY_DAY
PRIMARY = {"distance": 1.0, "exit": "80", "earnings": "skip"}
# SPY's spread for the side-by-side and combined views: dip_spreads.py's best-supported version (exploratory).
SPY = {"dte": 3, "distance": 0.5, "width": 5}
REPORT_WINDOWS = ((1, 22, 2, 8), (4, 22, 5, 8), (7, 22, 8, 8), (10, 22, 11, 8))  # (month, day) to (month, day)


def make_client():
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python stock_dip_spreads.py")
    return RESTClient(api_key=key, retries=10)


# ---------------------------------------------------------------- reference data

def check_splits(ticker, first, last, cache_dir="data/cache", refresh=False):
    """Stop if the ticker split in [first, last]: split-adjusted prices would not match the option strikes."""
    path = Path(cache_dir) / "reference" / f"{ticker}_splits_{first}_{last}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        rows = [(s.execution_date, s.split_from, s.split_to) for s in make_client().list_splits(
            ticker=ticker, execution_date_gte=first, execution_date_lte=last, limit=1000)]
        df = pd.DataFrame(rows, columns=["execution_date", "split_from", "split_to"]).astype(
            {"execution_date": str, "split_from": float, "split_to": float})
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    if len(df):
        raise SystemExit(f"{ticker} split inside the option window ({', '.join(df['execution_date'])}): "
                         "split-adjusted prices would not match the strikes. Choose a later start.")


def load_chain(ticker, first, last, cache_dir="data/cache", refresh=False):
    """Listed put strikes per expiry for expiries in [first, last] (expired contracts), from Massive's
    contracts reference: one query per week, so each answer is a single page."""
    path = Path(cache_dir) / "options_chains" / f"{ticker}_puts_{first}_{last}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    client, rows = make_client(), []
    start, end = pd.Timestamp(first), pd.Timestamp(last)
    for monday in pd.date_range(start - pd.Timedelta(days=start.dayofweek), end, freq="7D"):
        lo, hi = max(monday, start), min(monday + pd.Timedelta(days=6), end)
        rows += [(c.expiration_date, c.strike_price) for c in client.list_options_contracts(
            underlying_ticker=ticker, contract_type="put", expiration_date_gte=str(lo.date()),
            expiration_date_lte=str(hi.date()), expired=True, limit=1000)]
    df = pd.DataFrame(rows, columns=["expiry", "strike"]).drop_duplicates()
    df = df.assign(expiry=pd.to_datetime(df["expiry"]).astype("datetime64[ns]"), strike=df["strike"].astype(float))
    if df.empty:
        raise RuntimeError(f"Massive listed no {ticker} puts expiring {first}..{last}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def earnings_days(daily, spy_daily, margin=1):
    """Sessions treated as earnings events: in each reporting window the data fully covers, the biggest
    stock-specific volume jump and the biggest stock-specific overnight gap, each with `margin` sessions
    either side."""
    vol, spy = daily["volume"].astype(float), spy_daily["volume"].reindex(daily.index).astype(float)
    ratio = vol / vol.shift(1).rolling(20, min_periods=10).median()
    spy_ratio = spy / spy.shift(1).rolling(20, min_periods=10).median()
    jump = (ratio / spy_ratio).dropna()
    gap = (daily["open"] / daily["close"].shift(1)
           - (spy_daily["open"] / spy_daily["close"].shift(1)).reindex(daily.index)).abs().reindex(jump.index)
    if jump.empty:
        return pd.DatetimeIndex([])
    days, flagged = jump.index, set()
    for year in sorted(set(days.year)):
        for m1, d1, m2, d2 in REPORT_WINDOWS:
            lo, hi = pd.Timestamp(year, m1, d1), pd.Timestamp(year, m2, d2)
            if lo < days[0] or hi > days[-1]:
                continue
            inside = (days >= lo) & (days <= hi)
            for score in (jump, gap):
                s = score[inside].dropna()
                if len(s):
                    k = days.get_loc(s.idxmax())
                    flagged.update(days[max(k - margin, 0):k + margin + 1])
    return pd.DatetimeIndex(sorted(flagged))


def load_earnings_csv(path):
    """{ticker: sessions} from a CSV with columns ticker,date (the first session after each report)."""
    df = pd.read_csv(path)
    return {tk.upper(): pd.DatetimeIndex(sorted(pd.to_datetime(g["date"]))) for tk, g in df.groupby("ticker")}


def spans_earnings(day, expiry, reactions):
    """True if an earnings session falls after the entry day, up to and including the expiry."""
    i = reactions.searchsorted(day, side="right")
    return bool(i < len(reactions) and reactions[i] <= expiry)


# ---------------------------------------------------------------- trades

def pick_expiry(i, expiry_pos):
    """Position of the first listed expiry at least MIN_DTE sessions after session i (None past MAX_DTE)."""
    k = int(np.searchsorted(expiry_pos, i + MIN_DTE))
    return int(expiry_pos[k]) if k < len(expiry_pos) and expiry_pos[k] - i <= MAX_DTE else None


def pick_strikes(strikes, spot, distance, width_pct=WIDTH_PCT):
    """(short, long) from the listed strikes: the short at or below `distance`% under spot, the long the listed
    strike nearest `width_pct`% of spot lower (at least one strike lower; a tie goes to the lower strike)."""
    strikes = np.sort(np.asarray(strikes, float))
    target = round(float(spot) * (1 - distance / 100), 2)
    below = strikes[strikes <= target + 1e-9]
    if not len(below):
        return None
    short = below[-1]
    lower = strikes[strikes < short - 1e-9]
    if not len(lower):
        return None
    return float(short), float(lower[np.argmin(np.abs(lower - (short - float(spot) * width_pct / 100)))])


def plan_trades(ticker, days, raw, chain, distances=DISTANCES):
    """One planned spread per session and distance (strikes, expiry, leg tickers), before any price is loaded."""
    strikes = {pd.Timestamp(e): g["strike"].to_numpy(float) for e, g in chain.groupby("expiry")}
    pos = np.array(sorted(days.get_loc(e) for e in strikes if e in days), dtype=int)
    plans = []
    for i, t in enumerate(days):
        spot, j = raw.loc[t, "snap"], pick_expiry(i, pos)
        for d in distances:
            plan = {"ticker": ticker, "day": t, "distance": d, "spot": spot}
            if j is None:
                plans.append({**plan, "status": f"no listed expiry {MIN_DTE}-{MAX_DTE} sessions out in the study"})
                continue
            expiry = days[j]
            plan.update(expiry=expiry, dte=j - i)
            legs = None if np.isnan(spot) else pick_strikes(strikes[expiry], spot, d)
            if legs is None:
                plans.append({**plan, "status": "no price at 15:50" if np.isnan(spot) else "no listed strike"})
                continue
            short, long = legs
            plans.append({**plan, "short_strike": short, "long_strike": long, "width": round(short - long, 2),
                          "short_ticker": sp.option_ticker(expiry, "P", short, ticker),
                          "long_ticker": sp.option_ticker(expiry, "P", long, ticker),
                          "start": str(max(days[max(j - MAX_DTE, 0)], pd.Timestamp(ds.OPTIONS_START)).date())})
    return plans


def prefetch(plans, load, workers=8):
    """Download every leg the plans need, several at a time (each contract is cached on its first load)."""
    reqs = sorted({(p[k], p["start"], str(p["expiry"].date())) for p in plans if "short_ticker" in p
                   for k in ("short_ticker", "long_ticker")})

    def one(req):
        try:
            load(*req)
        except sp.NotInPlan:
            pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, reqs))
    return len(reqs)


def price_trades(plans, sessions, raw, load):
    trades = {}
    for p in plans:
        p = dict(p)
        start = p.pop("start", None)
        trades[p["day"], p["distance"]] = ds.price_spread(p, start, sessions, raw, load) if start else p
    return trades


def spy_trades(days, sessions, raw, load):
    """SPY's spread (dip_spreads.py) on every session, from the cached SPY contracts."""
    out = {}
    for i, t in enumerate(days):
        if i + SPY["dte"] < len(days):
            tr = ds.spread_trade(i, days, sessions, raw, SPY["dte"], SPY["distance"], SPY["width"], load)
            out[t, SPY["distance"]] = {**tr, "ticker": "SPY"}
    return out


def cost_tiers(slippage=0.0, commission=0.0):
    return {"base": (slippage, commission),
            **{f"+${x:.2f} slippage": (slippage + x, commission) for x in SLIPPAGE_STEPS}}


def pnl_table(trades, costs, reactions=None):
    """One row per trade, exit rule and cost tier (one row with the status for trades that could not be priced)."""
    rows = []
    for (t, d), tr in trades.items():
        base = {"ticker": tr["ticker"], "day": t, "distance": d, "dte": tr.get("dte"), "width": tr.get("width"),
                "status": tr["status"],
                "spans_earnings": bool(reactions is not None and "expiry" in tr
                                       and spans_earnings(t, tr["expiry"], reactions))}
        if tr["status"] != "ok":
            rows.append(base)
            continue
        for ex in EXITS:
            for cost, (slip, comm) in costs.items():
                pnl, reason, exit_day, credit = ds.simulate(tr, ex, slip, comm)
                rows.append({**base, "exit": str(ex), "cost": cost, "pnl": pnl, "reason": reason,
                             "exit_day": exit_day, "credit": credit, "max_loss": (tr["width"] - credit) * 100,
                             "expiry": tr["expiry"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def analyze(table, feats, reps, seed):
    """Per ticker, earnings rule, distance, exit and cost: every signal against every session, plus the
    signal traded one spread at a time (kept for the combined view)."""
    ok = table[table["status"] == "ok"]
    records, taken = [], {}
    for ticker, g0 in ok.groupby("ticker", sort=False):
        masks = {name: rule(feats[ticker]).fillna(False) for name, rule in SIGNALS.items()}
        for earn in ("skip", "keep"):
            g1 = g0[~g0["spans_earnings"].astype(bool)] if earn == "skip" else g0
            for (d, ex, cost), g in g1.groupby(["distance", "exit", "cost"], sort=False):
                g = g.sort_values("day").set_index("day")
                pnl, ml = g["pnl"].to_numpy(float), g["max_loss"].to_numpy(float)
                idx = mr.block_indices(len(g), reps, seed)
                key = {"ticker": ticker, "earnings": earn, "distance": d, "exit": ex, "cost": cost}
                records.append({**key, "signal": EVERY_DAY, **ds.signal_stats(np.ones(len(g), bool), pnl, ml, idx)})
                for name, m in masks.items():
                    mask = m.reindex(g.index).fillna(False).to_numpy(bool)
                    t = ds.one_at_a_time(g.index, mask, g)
                    r = t["pnl"].to_numpy(float) if len(t) else np.array([])
                    records.append({**key, "signal": name, **ds.signal_stats(mask, pnl, ml, idx),
                                    "single_trades": len(r), "single_total": r.sum(),
                                    "single_dd": sp.max_drawdown(r) if len(r) else np.nan})
                    taken[ticker, earn, d, ex, cost, name] = t
    return pd.DataFrame(records), taken


def combined(taken, tickers, signal, cost, months):
    """Each ticker traded one spread at a time (primary structure; SPY its own spread), pooled by exit date."""
    parts = [taken[tk, PRIMARY["earnings"], PRIMARY["distance"], PRIMARY["exit"], cost, signal].assign(ticker=tk)
             for tk in tickers if tk != "SPY"]
    if "SPY" in tickers:
        parts.append(taken["SPY", "skip", SPY["distance"], PRIMARY["exit"], cost, signal].assign(ticker="SPY"))
    t = pd.concat([p for p in parts if len(p)]).rename_axis("day").reset_index().sort_values(["exit_day", "day"])
    pnl = t["pnl"].to_numpy(float)
    by_month = t.groupby(t["exit_day"].dt.to_period("M"))["pnl"].sum().reindex(months, fill_value=0.0)
    return {"trades": len(t), "per_month": len(t) / len(months), "total": pnl.sum(), "avg": pnl.mean(),
            "win": (pnl > 0).mean() * 100, "max_dd": sp.max_drawdown(pnl), "losing_months": int((by_month < 0).sum()),
            "months": len(months), "worst_month": by_month.min()}, t


def relatedness(feats, raw, days, tickers):
    """How often the tickers' dip signals coincide with SPY's and each other's, and daily return correlations."""
    sig = pd.DataFrame({tk: SIGNALS[PREREGISTERED[0]](feats[tk]).reindex(days).fillna(False).astype(bool)
                        for tk in [*tickers, "SPY"]})
    rets = pd.DataFrame({tk: raw[tk]["close"].reindex(days).pct_change() for tk in [*tickers, "SPY"]})
    rows = [{"ticker": tk, "signals": int(sig[tk].sum()),
             "with_spy": (sig[tk] & sig["SPY"]).sum() / max(sig[tk].sum(), 1) * 100,
             "corr_spy": rets[tk].corr(rets["SPY"])} for tk in tickers]
    together = sig[tickers].sum(axis=1)
    return pd.DataFrame(rows), together[together > 0].value_counts().sort_index()


# ---------------------------------------------------------------- the study

def run_study(stocks, spy_raw, spy_feats, sessions, *, tickers=TICKERS, costs=None, reps=2000, seed=0, load=None,
              workers=8, earnings=None, timings=None):
    """stocks: {ticker: (raw daily table, features, chain)}; earnings: optional {ticker: sessions after each
    report} replacing the inferred dates. No file I/O beyond the option loader's cache."""
    timings = {} if timings is None else timings
    costs = cost_tiers() if costs is None else costs
    days = spy_raw.index[spy_raw.index >= pd.Timestamp(ds.OPTIONS_START)]
    opt_sessions = sessions.loc[days]
    raw, feats, reactions, plans = {"SPY": spy_raw}, {"SPY": spy_feats}, {}, {}
    for tk in tickers:
        raw[tk], feats[tk], chain = stocks[tk]
        reactions[tk] = (earnings or {}).get(tk)
        if reactions[tk] is None:
            reactions[tk] = earnings_days(raw[tk], spy_raw)
        plans[tk] = plan_trades(tk, days, raw[tk].loc[days], chain)
    with timed(timings, "option data"):
        fetched = prefetch([p for tk in tickers for p in plans[tk]], load, workers)
        trades = {tk: price_trades(plans[tk], opt_sessions, raw[tk].loc[days], load) for tk in tickers}
        trades["SPY"] = spy_trades(days, opt_sessions, spy_raw.loc[days], load)
    with timed(timings, "simulation"):
        table = pd.concat([pnl_table(trades[tk], costs, reactions.get(tk)) for tk in [*tickers, "SPY"]],
                          ignore_index=True)
        stats, taken = analyze(table, feats, reps, seed)
        months = pd.period_range(days[0], days[-1], freq="M")
        views = {}
        for signal in SIGNALS:
            for cost in ("base", REALISTIC):
                for name, group in (("stocks", [t for t in tickers]), ("SPY", ["SPY"]),
                                    ("all", [*tickers, "SPY"])):
                    views[signal, cost, name] = combined(taken, group, signal, cost, months)
        related = relatedness(feats, raw, days, tickers)
    return {"table": table, "stats": stats, "taken": taken, "views": views, "related": related, "reactions": reactions,
            "tickers": list(tickers), "days": days, "costs": costs, "reps": reps, "requests": fetched,
            "earnings_source": {tk: "csv" if tk in (earnings or {}) else "inferred" for tk in tickers}}


# ---------------------------------------------------------------- report

def money(v, cents=False, sign=True):
    return ds.money(v, cents=cents, sign=sign)


def pick(stats, ticker, signal, cost="base", **structure):
    s = {**PRIMARY, **({"distance": SPY["distance"]} if ticker == "SPY" else {}), **structure}
    m = stats[(stats["ticker"] == ticker) & (stats["signal"] == signal) & (stats["cost"] == cost)
              & (stats["earnings"] == s["earnings"]) & (stats["distance"] == s["distance"])
              & (stats["exit"].astype(str) == str(s["exit"]))]
    return m.iloc[0] if len(m) else None


def interval(r, lo, hi):
    return f"{money(r[lo])} to {money(r[hi])}"


def verdict(r):
    if r["n"] < 5:
        return "too few"
    return "better" if r["excess_lo"] > 0 else "worse" if r["excess_hi"] < 0 else "unclear"


def label(ticker, distance=None):
    if ticker == "SPY":
        return f"SPY ($5, {ds.distance_label(SPY['distance'])}, {SPY['dte']} sessions)"
    return ticker


def render_report(res):
    stats, tickers = res["stats"], res["tickers"]
    lines = []
    w = lines.append
    first, last = res["days"][0].date(), res["days"][-1].date()
    w("# Put spreads after dips in single stocks\n")
    w(f"Option data {first} to {last} (Massive minute bars). Short put at the listed strike at or below 1% under "
      "the 15:50 price, long put about 1.5% of the price lower, first listed expiry 2+ sessions out (usually the "
      "coming Friday), 80% take profit, spreads open over earnings skipped. SPY's row uses its own spread from "
      f"dip_spreads.py ({label('SPY')}). Dollars are per spread (100 shares); *every session* is the same spread "
      "opened on every session, the baseline a signal has to beat. Ranges are 95% block-bootstrap intervals over "
      "sessions.\n")
    w(bottom_line(res))
    w("## 1. Each ticker, the pre-registered tests\n")
    w(primary_table(res))
    w("\n*Excess* is the signal's average minus the every-session average; *better* means its 95% range is above "
      "zero. Each ticker is a separate test, so with five tickers and two signals one *better* could be luck.\n")
    w("## 2. Adding RSI(2) < 10\n")
    w(rsi_section(res))
    w("## 3. One spread at a time, all tickers together\n")
    w(combined_section(res))
    w("## 4. Strike distance and exit (exploratory)\n")
    w(grid_section(res))
    w("## 5. Earnings\n")
    w(earnings_section(res))
    w("## 6. How related the tickers are\n")
    w(related_section(res))
    w("## 7. Data\n")
    w(data_section(res))
    return "\n".join(lines) + "\n"


def bottom_line(res):
    stats, tickers = res["stats"], res["tickers"]
    better = [tk for tk in tickers for s in PREREGISTERED
              if (r := pick(stats, tk, s, REALISTIC)) is not None and r["n"] >= 5 and r["excess_lo"] > 0]
    avg = [(tk, pick(stats, tk, PREREGISTERED[0], REALISTIC)) for tk in tickers]
    avg = [(tk, r) for tk, r in avg if r is not None and r["n"]]
    out = ["**Bottom line.** "]
    if avg:
        means = ", ".join(f"{tk} {money(r['mean'])}" for tk, r in avg)
        out.append(f"After 3+ down days, with +$0.05 slippage, the average spread made {means}. ")
    if better:
        out.append(f"The signal beat the every-session baseline clearly (95% range above zero) for: "
                   f"{', '.join(sorted(set(better)))}. ")
    else:
        out.append("No ticker's signal beat its every-session baseline clearly at +$0.05 slippage. ")
    v = res["views"]
    a, b = v[PREREGISTERED[0], REALISTIC, "all"][0], v[PREREGISTERED[1], REALISTIC, "all"][0]
    out.append(f"Trading every ticker one spread at a time, 3+ down days gave {a['trades']} trades "
               f"({a['per_month']:.1f} a month, {money(a['total'])}); adding RSI(2) < 10 gave {b['trades']} "
               f"({b['per_month']:.1f} a month, {money(b['total'])}).\n")
    return "".join(out)


def primary_table(res):
    stats = res["stats"]
    rows = []
    for tk in [*res["tickers"], "SPY"]:
        for s in PREREGISTERED:
            r, x = pick(stats, tk, s), pick(stats, tk, s, REALISTIC)
            if r is None or not r["n"]:
                rows.append([label(tk), s, "0", "", "", "", "", "", ""])
                continue
            rows.append([label(tk), s, f"{int(r['n'])}", f"{money(r['mean'])} ({interval(r, 'mean_lo', 'mean_hi')})",
                         money(r["base_mean"]), f"{money(r['excess'])} ({interval(r, 'excess_lo', 'excess_hi')})",
                         f"{r['win']:.0f}%", f"{r['ror']:+.1f}%",
                         f"{money(x['mean'])}; excess {money(x['excess'])}, {verdict(x)}"])
    return ds.md_table(["Ticker", "Signal", "Spreads", "Avg per spread", "Every session", "Excess", "Win",
                        "Return on risk", "With +$0.05 slippage"], rows)


def rsi_section(res):
    stats = res["stats"]
    out = ["RSI(2) < 10 fires after sharp drops that need not be 3 days in a row. Same spread, 80% take profit, "
           "+$0.05 slippage; *one at a time* is the signal traded one spread at a time per ticker:\n"]
    rows = []
    for tk in [*res["tickers"], "SPY"]:
        row = [label(tk)]
        for s in SIGNALS:
            r = pick(stats, tk, s, REALISTIC)
            row.append("" if r is None or not r["n"] else
                       f"{int(r['n'])} spreads, {money(r['mean'])} avg; one at a time {int(r['single_trades'])} "
                       f"for {money(r['single_total'])}")
        rows.append(row)
    out.append(ds.md_table(["Ticker", *SIGNALS], rows))
    return "\n".join(out) + "\n"


def combined_section(res):
    v = res["views"]
    out = ["Each ticker trades one spread at a time (a new signal is skipped while that ticker's spread is open); "
           "different tickers can be open together. Primary spread, 80% take profit, earnings skipped:\n"]
    rows = []
    for signal in SIGNALS:
        for cost in ("base", REALISTIC):
            for name in ("stocks", "SPY", "all"):
                m = v[signal, cost, name][0]
                rows.append([signal, cost, {"stocks": "the stocks", "SPY": "SPY alone", "all": "stocks + SPY"}[name],
                             f"{m['trades']}", f"{m['per_month']:.1f}", money(m["total"]), money(m["avg"]),
                             f"{m['win']:.0f}%", money(m["max_dd"], sign=False),
                             f"{m['losing_months']} of {m['months']}", money(m["worst_month"])])
    out.append(ds.md_table(["Signal", "Costs", "Tickers", "Trades", "Per month", "Total", "Avg", "Win",
                            "Worst drawdown", "Losing months", "Worst month"], rows))
    out.append("\nDrawdowns and months use exit dates, one spread per trade. Several tickers often dip on the same "
               "days, so their losses can land together (section 6).")
    return "\n".join(out) + "\n"


def grid_section(res):
    stats = res["stats"]
    out = ["After 3+ down days: average per spread, and the excess over every session in brackets, by short-strike "
           "distance and exit. Earnings skipped.\n"]
    rows = []
    for tk in res["tickers"]:
        for d in DISTANCES:
            row = [tk, ds.distance_label(d)]
            for ex in EXITS:
                for cost in ("base", REALISTIC):
                    r = pick(stats, tk, PREREGISTERED[0], cost, distance=d, exit=str(ex))
                    row.append("" if r is None or not r["n"] else
                               f"{money(r['mean'])} [{money(r['excess'])}] n={int(r['n'])}")
            rows.append(row)
    out.append(ds.md_table(["Ticker", "Short put", "Held, traded prices", "Held, +$0.05", "80% TP, traded prices",
                            "80% TP, +$0.05"], rows))
    return "\n".join(out) + "\n"


def earnings_section(res):
    stats, table = res["stats"], res["table"]
    out = ["Sessions treated as earnings events (inferred: each reporting window's biggest stock-specific volume "
           "jump and overnight gap, with a session either side; exact dates if given with --earnings-csv). Spreads "
           "open over any of them are skipped:\n"]
    for tk in res["tickers"]:
        dates = [d for d in res["reactions"][tk] if d >= res["days"][0]]
        runs, cur = [], []
        for d in dates:  # group consecutive flagged sessions into runs for reading
            if cur and (d - cur[-1]).days > 4:
                runs.append(cur)
                cur = []
            cur.append(d)
        if cur:
            runs.append(cur)
        text = ", ".join(f"{r[0]:%Y-%m-%d}" + (f" to {r[-1]:%m-%d}" if len(r) > 1 else "") for r in runs)
        out.append(f"- **{tk}** ({res['earnings_source'][tk]}): {text}")
    rows = []
    for tk in res["tickers"]:
        r_skip, r_keep = pick(stats, tk, PREREGISTERED[0]), pick(stats, tk, PREREGISTERED[0], earnings="keep")
        p = table[(table["ticker"] == tk) & (table["status"] == "ok") & (table["distance"] == PRIMARY["distance"])
                  & (table["exit"] == PRIMARY["exit"]) & (table["cost"] == "base") & table["spans_earnings"].astype(bool)]
        rows.append([tk, f"{int(r_keep['n'] - r_skip['n'])}" if r_keep is not None and r_skip is not None else "",
                     f"{len(p)} sessions, {money(p['pnl'].mean())} avg, worst {money(p['pnl'].min())}" if len(p) else "",
                     f"{money(r_keep['mean'])} vs {money(r_skip['mean'])}" if r_keep is not None and r_keep["n"] else ""])
    out.append("\n" + ds.md_table(["Ticker", "Dip spreads skipped", "Every-session spreads skipped (traded prices)",
                                   "Dip average, earnings kept vs skipped"], rows))
    return "\n".join(out) + "\n"


def related_section(res):
    rel, together = res["related"]
    out = ["The four stocks are large parts of SPY, so their dips often come together:\n"]
    out.append(ds.md_table(["Ticker", "3+ down-day signals", "Same day as a SPY signal", "Daily return correlation "
                            "with SPY"], [[r.ticker, f"{r.signals}", f"{r.with_spy:.0f}%", f"{r.corr_spy:.2f}"]
                                          for r in rel.itertuples()]))
    out.append("\nSessions by how many of the four stocks signalled: " +
               ", ".join(f"{k}: {v}" for k, v in together.items()) + ".")
    return "\n".join(out) + "\n"


def data_section(res):
    t = res["table"]
    strict = t.drop_duplicates(["ticker", "day", "distance"])
    out = [f"- Leg requests this run: {res['requests']:,} contracts (cached in data/cache/options_multi/)."]
    for tk in res["tickers"]:
        s = strict[strict["ticker"] == tk]
        st = s["status"].value_counts()
        prim = s[(s["distance"] == PRIMARY["distance"]) & (s["status"] == "ok")]
        out.append(f"- **{tk}:** {st.get('ok', 0)} of {len(s)} planned spreads priced "
                   f"({'; '.join(f'{k}: {v}' for k, v in st.items() if k != 'ok') or 'none missing'}). "
                   f"Primary spread: expiry {prim['dte'].min():.0f}-{prim['dte'].max():.0f} sessions out "
                   f"(average {prim['dte'].mean():.1f}), width ${prim['width'].min():g}-${prim['width'].max():g} "
                   f"(median ${prim['width'].median():g}).")
    out.append("- Earnings dates are inferred from volume (see the module notes); results skip spreads open over "
               "them unless marked *earnings kept*.")
    out.append("- Fills are minute trade prints, not quotes; single-stock options trade wider than SPY's, so read "
               "the +$0.05 slippage columns as the realistic case.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- charts

def _axes(fig, rect):
    return ds._axes(fig, rect)


def plot_tickers(res, path):
    stats = res["stats"]
    names = [*res["tickers"], "SPY"]
    fig = plt.figure(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.24, 0.14, 0.7, 0.68])
    y = np.arange(len(names))[::-1]
    for k, s in enumerate(PREREGISTERED):
        for yy, tk in zip(y, names):
            r = pick(stats, tk, s, REALISTIC)
            if r is None or not r["n"]:
                continue
            off = 0.15 - 0.3 * k
            ax.errorbar(r["mean"], yy + off, xerr=[[r["mean"] - r["mean_lo"]], [r["mean_hi"] - r["mean"]]], fmt="o",
                        color=SERIES[k], ecolor=SERIES[k], elinewidth=1.5, capsize=3, ms=5,
                        label=s if yy == y[0] else None)
            ax.plot([r["base_mean"]], [yy + off], marker="|", color=INK_2, ms=10, mew=1.5,
                    label="Every session" if yy == y[0] and k == 0 else None)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y, [label(tk).replace("$", r"\$") for tk in names], fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel(r"Average P&L per spread, \$ (95% interval), +\$0.05 slippage per leg", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "Put spreads after dips, by ticker", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Short put 1% below, ~1.5% wide, next expiry 2+ sessions out, 80% take profit, earnings "
             "skipped. Tick: every-session average.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_combined(res, path):
    v = res["views"]
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.1, 0.2, 0.86, 0.58])
    for k, (signal, name, color, style) in enumerate(((PREREGISTERED[0], "all", SERIES[0], "-"),
                                                      (PREREGISTERED[1], "all", SERIES[1], "-"),
                                                      (PREREGISTERED[0], "SPY", INK_2, (0, (4, 3))))):
        m, t = v[signal, REALISTIC, name]
        what = "stocks + SPY" if name == "all" else "SPY alone"
        ax.step(t["exit_day"], t["pnl"].cumsum(), where="post", color=color, lw=2 if name == "all" else 1.5,
                ls=style, label=f"{signal}, {what} ({m['trades']} trades)")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel(r"Cumulative P&L, one spread per trade (\$)", color=INK_2, fontsize=8)
    fig.text(0.02, 0.97, "All tickers together, one spread at a time each", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, r"Primary spread, 80% take profit, earnings skipped, +\$0.05 slippage per leg; by exit date.",
             color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def load_ticker(ticker, sessions, cache_dir, refresh):
    """(actual daily table, features from dividend-adjusted prices) for one ticker."""
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    minutes, _ = download.load_minute_bars(ticker, first, last, cache_dir=cache_dir, refresh=refresh,
                                           final_close=sessions["close"].iloc[-1])
    raw, _ = mr.daily_table(minutes, sessions)
    adj, _ = mr.adjust_dividends(raw, mr.load_dividends(ticker, first, last, cache_dir, refresh))
    return raw, mr.ticker_features(adj).reindex(sessions.index)


def write_outputs(out_dir, res):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "spreads.csv", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    plot_tickers(res, out / "tickers.png")
    plot_combined(res, out / "combined.png")
    (out / "report.md").write_text(render_report(res))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Put credit spreads after dips in single stocks, with real option prices.")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--out", default="output/stock_dip_spreads")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--slippage", type=float, default=0.0, help="$/share per leg per fill (default 0)")
    p.add_argument("--commission", type=float, default=0.0, help="$/contract per leg per fill (default 0)")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="parallel option downloads")
    p.add_argument("--earnings-csv", help="exact earnings sessions (columns ticker,date: the first session after "
                   "each report) instead of the inferred ones")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tickers = [t.upper() for t in args.tickers]
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    opt_first, last = ds.OPTIONS_START, str(sessions.index[-1].date())
    with timed(timings, "data"):
        spy_raw, spy_feats = load_ticker(mr.TICKER, sessions, args.cache_dir, args.refresh)
        stocks = {}
        for tk in tickers:
            check_splits(tk, opt_first, last, args.cache_dir, args.refresh)
            raw, feats = load_ticker(tk, sessions, args.cache_dir, args.refresh)
            stocks[tk] = (raw, feats, load_chain(tk, opt_first, last, args.cache_dir, args.refresh))
    load = ds.contract_loader(args.cache_dir, args.refresh)
    res = run_study(stocks, spy_raw, spy_feats, sessions, tickers=tickers,
                    costs=cost_tiers(args.slippage, args.commission), reps=args.reps, seed=args.seed, load=load,
                    workers=args.workers, timings=timings,
                    earnings=load_earnings_csv(args.earnings_csv) if args.earnings_csv else None)
    with timed(timings, "outputs"):
        out = write_outputs(args.out, res)
    print(f"Stock dip spreads: option contracts downloaded this run: {load.state['fetched']:,}")
    for tk in tickers:
        r = pick(res["stats"], tk, PREREGISTERED[0], REALISTIC)
        if r is not None and r["n"]:
            print(f"  {tk}: {int(r['n'])} spreads after 3+ down days, {money(r['mean'], cents=True)} each with +$0.05 "
                  f"slippage vs {money(r['base_mean'], cents=True)} every session")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
