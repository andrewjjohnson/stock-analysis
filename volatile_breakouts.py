"""Breakouts through yesterday's high and low in the most volatile liquid stocks: do the TSLA keeper rules carry
over to other volatile names? A trade simulation on Massive minute and one-second bars. Shares are per share of
the underlying, after costs; options (the 2025-26 check only) are per contract at traded prices plus slippage.

  uv run --env-file .env python volatile_breakouts.py --out output/volatile_breakouts                # design screen
  uv run --env-file .env python volatile_breakouts.py --final-test --out output/volatile_breakouts   # 2025-26, once

Mechanism. In volatile, heavily traded single stocks, yesterday's high and low are the most obvious reference
levels. Breakout buyers' stop orders and short sellers' protective stops sit just above yesterday's high (sell
stops just below the low). When price reaches the level those orders fire together and push it through, and
momentum and attention-driven buyers extend the move. The other side is liquidity providers and traders who fade
the level. The edge should last as long as stops keep being placed at obvious levels and these names stay
attention-driven. breakout_check.py tests these rules on TSLA; this asks whether they apply to other volatile
stocks.

Declared on 2026-10-04, before any trade result for these stocks was computed:
- Universe, re-picked every calendar month by rule (no names chosen by hand), from Massive's grouped daily bars
  (market_daily.py), delisted stocks included. Eligible: common stocks and ADRs that traded on each of the 21
  sessions before the month's first session, with a median daily dollar volume (VWAP x volume) of at least
  $500M over them and an unadjusted close of at least $10 on the last of them. Ranked by mean daily true range as
  % of the previous close over those 21 sessions (split-adjusted bars). Core = ranks 1-10; related = ranks 11-20.
  TSLA is left out: its design results are known and its 2025-26 data is spent on these rules.
- Rules, unchanged from the TSLA keeper (levels.py, breakouts.py): long through yesterday's regular-hours high
  when the session opens below it; short through yesterday's low when the session opens above it; levels within
  0.05 ATR of the open are skipped. A stop order at the level, first touch from 9:30 and at least 30 minutes
  before the close. Stop and target 0.1 ATR either side of the level (ATR = daily ATR(14) through yesterday,
  regular-hours bars), otherwise out at the last regular-hours price. One trade per rule, stock and session.
- Fills: the fill minute and every minute reaching the stop or the target are replayed on one-second bars
  (breakout_check.replay: a second reaching both counts as stopped, as does the fill second reaching the stop; a
  stop gapped through fills at that second's open). Trades whose seconds never show the minute bar's touch are
  left out and counted.
- Costs per side: 1 bp plus half a cent a share (half of a one-cent spread), at the actually traded (unadjusted)
  price. 0 and a flat 1 bp a side are shown too.
- Baseline: the same trades at levels.py's random fake levels (2 per real level: the same stock, session and side,
  at distances drawn from that level's distances on other sessions), replayed the same way.
- Tests (10, one-sided, standard errors clustered by session, Benjamini-Hochberg q < 0.10 across all 10), on the
  design period (2021-12 to 2024-12):
    1-2. core, long through yesterday's high / short through yesterday's low, 0.1 ATR: mean net P&L > 0
    3-4. the same two: real minus fake-level P&L > 0 (session bootstrap)
    5-6. the same two with the stop and target at 0.25 ATR: mean net > 0
    7-8. related (ranks 11-20), the same two rules at 0.1 ATR: mean net > 0
    9-10. core, long through last week's high / short through last week's low, 0.1 ATR: mean net > 0
- A rule passes the screen if: its test (1 or 2) has q < 0.10; its mean net P&L is at least its average
  round-trip cost (so the edge before costs is at least twice the costs); it beats the fake levels (point
  estimate); it is positive in at least 2 of 2022, 2023 and 2024; and the related stocks' same rule is positive.
- Also reported, not tested: each year, each stock, the trades with one-second wicks ignored (only one-second
  bodies may fill, stop or target: odd one-off prints can't trip anything), and the account view (one position at
  a time) with the daily-P&L correlation to SPY dip buying and the TSLA breakouts.

The held-back check (--final-test), declared after the screen and before 2025-26 is read, is in FINAL below.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import breakout_check as bc  # noqa: E402
import breakouts as bo  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import levels as lv  # noqa: E402
import market_daily as md  # noqa: E402
import meanrev as mr  # noqa: E402
import portfolio_check as pc  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
DATA_START = pd.Timestamp("2021-10-05")      # first session the data plan still returned on 2026-10-04
DESIGN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
FINAL_END = pd.Timestamp("2026-09-30")
FULL_RANGE = ("2021-10-04", "2026-09-30")     # existing whole-range minute caches are reused
LOOKBACK = 21                                # sessions behind each month's pick
MIN_DOLLAR_VOLUME = 5e8
MIN_PRICE = 10.0
CORE, RELATED = 10, 20                       # ranks 1-10 and 11-20
EXCLUDE = ("TSLA",)
RULES = {"long through yesterday's high": "yesterday's high",          # -> breakouts.RULES
         "short through yesterday's low": "yesterday's low (check)",
         "long through last week's high": "last week's high",
         "short through last week's low": "last week's low (check)"}
NAME = {v: k for k, v in RULES.items()}
PRIMARY_RULES = ("long through yesterday's high", "short through yesterday's low")
WIDTHS = (0.1, 0.25)                         # stop and target, ATR from the level
PRIMARY_WIDTH = 0.1
COST_BPS = 1.0                               # per side, plus HALF_TICK
HALF_TICK = 0.005                            # $ a share per side
COSTS = ("none", "1 bp", "primary")
FAKES = 2
WARMUP = 30                                  # sessions of minute bars before a ticker's first universe month
GAP_MONTHS = 3                               # universe months further apart than this get separate downloads
SEED = 11
FDR = 0.10
REPS = 2000
YEARS = (2022, 2023, 2024)
TESTS = [("core", "long through yesterday's high", 0.1, "net"),
         ("core", "short through yesterday's low", 0.1, "net"),
         ("core", "long through yesterday's high", 0.1, "vs fakes"),
         ("core", "short through yesterday's low", 0.1, "vs fakes"),
         ("core", "long through yesterday's high", 0.25, "net"),
         ("core", "short through yesterday's low", 0.25, "net"),
         ("related", "long through yesterday's high", 0.1, "net"),
         ("related", "short through yesterday's low", 0.1, "net"),
         ("core", "long through last week's high", 0.1, "net"),
         ("core", "short through last week's low", 0.1, "net")]
FINAL = None                                 # set only for rules that pass the screen, before --final-test
OPTION_SLIPS = (0.03, 0.05, 0.10)
PRIMARY_SLIP = 0.05


# ---------------------------------------------------------------- universe

def daily_panels(adj, raw, types, min_peak=1e8):
    """Session x ticker panels for stocks (Massive types CS/ADRC): split-adjusted high, low, close and dollar volume,
    and the unadjusted close. Tickers that never trade `min_peak` dollars in a day are dropped first."""
    a = adj[adj["ticker"].map(types).isin(md.TYPES)]
    dv = a["vwap"].fillna(a["close"]) * a["volume"]
    big = set(a.loc[dv >= min_peak, "ticker"])
    a = a[a["ticker"].isin(big)].assign(dollar_volume=dv)
    r = raw[raw["ticker"].isin(big)]

    def wide(df, col):
        return df.pivot_table(index="session", columns="ticker", values=col, aggfunc="last").sort_index()

    out = {k: wide(a, k) for k in ("high", "low", "close", "dollar_volume")}
    out["raw_close"] = wide(r, "close").reindex(index=out["close"].index, columns=out["close"].columns)
    return out


def pick_universe(panels, months, lookback=LOOKBACK, min_dollar_volume=MIN_DOLLAR_VOLUME, min_price=MIN_PRICE,
                  n=RELATED, core=CORE, exclude=EXCLUDE):
    """One row per (month, ticker) for the top `n` stocks by mean true range % over the `lookback` sessions before
    the month's first session in the panels; only data from before that session is used. months: monthly
    Periods."""
    close, high, low = panels["close"], panels["high"], panels["low"]
    days = close.index
    prev = close.shift(1).to_numpy()
    h, lo_ = high.to_numpy(), low.to_numpy()
    tr = np.fmax(h - lo_, np.fmax(np.abs(h - prev), np.abs(lo_ - prev))) / prev * 100
    tr[np.isnan(h) | np.isnan(prev)] = np.nan
    tr = pd.DataFrame(tr, index=days, columns=close.columns)
    rows = []
    for m in months:
        i = int(days.searchsorted(m.start_time))
        if i >= len(days) or days[i].to_period("M") != m or i < lookback + 1:
            continue
        win = slice(i - lookback, i)
        t, d, c = tr.iloc[win], panels["dollar_volume"].iloc[win], close.iloc[win]
        ok = (t.notna().all() & d.notna().all() & c.notna().all() & (d.median() >= min_dollar_volume)
              & (panels["raw_close"].iloc[i - 1] >= min_price))
        score = t.mean()[ok].drop(list(exclude), errors="ignore").sort_values(ascending=False, kind="stable")
        for rank, (tk, v) in enumerate(score.iloc[:n].items(), 1):
            rows.append({"month": m, "ticker": tk, "rank": rank, "tr_pct": v,
                         "group": "core" if rank <= core else "related"})
    return pd.DataFrame(rows, columns=["month", "ticker", "rank", "tr_pct", "group"])


def download_spans(universe, sessions, warmup=WARMUP, gap=GAP_MONTHS):
    """Per ticker, the minute-bar ranges to load: each run of its universe months no more than `gap` months apart,
    from `warmup` sessions before the run's first session (not before the first of `sessions`) to the run's last
    session. Returns rows of ticker, first, last and the months covered."""
    idx = sessions.index
    rows = []
    for tk, g in universe.groupby("ticker", sort=True):
        months = sorted(set(g["month"]))
        runs = [[months[0]]]
        for m in months[1:]:
            if (m - runs[-1][-1]).n > gap:
                runs.append([m])
            else:
                runs[-1].append(m)
        for run in runs:
            in_run = idx[(idx >= run[0].start_time) & (idx <= run[-1].end_time)]
            if not len(in_run):
                continue
            k = max(int(idx.get_loc(in_run[0])) - warmup, 0)
            rows.append({"ticker": tk, "first": idx[k], "last": in_run[-1], "months": tuple(run)})
    return pd.DataFrame(rows, columns=["ticker", "first", "last", "months"])


def load_span(ticker, first, last, sessions, cache_dir="data/cache", refresh=False):
    """(minute bars, source note) for one span; a whole-range cache of the ticker is reused when present. On a
    failed or short download returns (None, the error)."""
    full = download.cache_path(cache_dir, ticker, *FULL_RANGE)
    if full.exists() and not refresh:
        return pd.read_parquet(full), f"cache {full.name}"
    try:
        return download.load_minute_bars(ticker, f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}", cache_dir=cache_dir,
                                         refresh=refresh, final_close=sessions.loc[last, "close"])
    except RuntimeError as e:
        return None, str(e)


# ---------------------------------------------------------------- trades

def span_entries(minutes, span_sessions, groups, period, seed=SEED):
    """Minute-level entries at the real levels of RULES and at their fakes, for one ticker over one span, on the
    sessions of its universe months only (`groups`: month -> core/related). Returns (entries, rth)."""
    rth, _ = features.regular_session_minutes(minutes, span_sessions)
    if rth.empty:
        return None, rth
    table = lv.level_table(rth, minutes, span_sessions)
    real = bo.rule_levels(lv.real_levels(table))
    month = real["session"].dt.to_period("M")
    real = real[real["rule"].isin(RULES.values()) & month.isin(list(groups)) & (real["period"] == period)]
    real = real.reset_index(drop=True)
    if real.empty:
        return None, rth
    fakes = lv.fake_levels(real.drop(columns="rule"), table, k=FAKES, seed=seed)
    rule_of = {(level, s): name for name, (level, sides) in bo.RULES.items() for s in sides}
    fakes["rule"] = [rule_of[(a, b)] for a, b in zip(fakes["level"], fakes["side"])]
    both = pd.concat([real, fakes], ignore_index=True)
    e = bo.run_trades(rth, both, table, exits={"entry": (PRIMARY_WIDTH, PRIMARY_WIDTH)})
    e = e[e["fills"] == "worst"].reset_index(drop=True)  # the entry minute is the same either way
    e["atr"] = table["atr"].reindex(e["session"]).to_numpy()
    e["group"] = e["session"].dt.to_period("M").map(groups)
    return e, rth


def bodies(seconds_of):
    """A seconds loader whose bars span only each second's open and close: wicks (where odd one-off prints show up)
    can't fill, stop or target anything."""
    def load(ticker, start, end):
        s = seconds_of(ticker, start, end).copy()
        if len(s):
            s[:, 2] = np.maximum(s[:, 1], s[:, 4])
            s[:, 3] = np.minimum(s[:, 1], s[:, 4])
        return s
    load.prefetch = getattr(seconds_of, "prefetch", None)
    return load


def side_cost_bps(actual_price, flat=COST_BPS, half_tick=HALF_TICK):
    """Primary cost per side, in bps of the traded price: `flat` bps plus `half_tick` dollars a share."""
    return flat + half_tick / np.asarray(actual_price, float) * 1e4


def add_net(t, ratio):
    """Net P&L columns for each cost setting. ratio: session x ticker unadjusted / split-adjusted close, read from
    the session before the trade (the traded price = fill x ratio)."""
    prev = ratio.shift(1)
    r = np.array([prev.at[d, tk] if tk in prev.columns and d in prev.index else np.nan
                  for d, tk in zip(t["session"], t["ticker"])], float)
    r = np.where(np.isfinite(r), r, 1.0)
    t = t.assign(actual_price=t["fill"].to_numpy(float) * r)
    t["cost_side"] = side_cost_bps(t["actual_price"])
    t["net_none"], t["net_1 bp"] = t["gross"], t["gross"] - 2 * COST_BPS
    t["net_primary"] = t["gross"] - 2 * t["cost_side"]
    return t


# ---------------------------------------------------------------- statistics

def trade_stats(t, years, cost="primary"):
    net = t[f"net_{cost}"].to_numpy(float)
    n = len(net)
    out = {"trades": n, "per_year": n / years if years else np.nan}
    if n < 2:
        return out
    m, se = sm.cluster_mean(net, t["session"].to_numpy())
    z = m / se if se else np.nan
    return {**out, "win": (net > 0).mean() * 100, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "cost_rt": 2 * t["cost_side"].mean(),
            "gross": t["gross"].mean(), "stops": (t["reason"] == -1).mean() * 100,
            "targets": (t["reason"] == 1).mean() * 100}


def excess(real, fake, reps=REPS, seed=SEED, col="net_primary"):
    """Real minus fake mean P&L per trade with a session bootstrap: (difference, lo, hi, one-sided p)."""
    if len(real) < 2 or len(fake) < 2:
        return np.nan, np.nan, np.nan, np.nan
    days = np.union1d(real["session"].unique(), fake["session"].unique())
    r = real.groupby("session")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    f = fake.groupby("session")[col].agg(["sum", "count"]).reindex(days, fill_value=0)
    w = lv.boot_weights(len(days), reps, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = w @ r["sum"].to_numpy() / (w @ r["count"].to_numpy()) - w @ f["sum"].to_numpy() / (w @ f["count"].to_numpy())
    d = real[col].mean() - fake[col].mean()
    se = np.nanstd(boot)
    return d, d - 1.96 * se, d + 1.96 * se, norm.sf(d / se) if se > 0 else np.nan


def statistics(trades, years):
    """Stats per group, rule, width, fill mode, control and cost; the excess over fakes; years; stocks."""
    rows = []
    keys = ["group", "rule", "width", "mode", "fake"]
    for (g, rule, w, mode, fake), part in trades.groupby(keys, sort=False):
        for cost in COSTS:
            rows.append({"group": g, "rule": rule, "width": w, "mode": mode, "control": "fake" if fake else "real",
                         "cost": cost, **trade_stats(part, years, cost)})
    stats = pd.DataFrame(rows)
    ex = []
    for (g, rule, w, mode), part in trades.groupby(["group", "rule", "width", "mode"], sort=False):
        d, lo, hi, p = excess(part[~part["fake"]], part[part["fake"]])
        ex.append({"group": g, "rule": rule, "width": w, "mode": mode, "excess": d, "lo": lo, "hi": hi, "p_up": p})
    real = trades[~trades["fake"] & (trades["mode"] == "prints")]
    by_year = []
    for (g, rule, w, y), part in real.groupby(["group", "rule", "width", real["session"].dt.year], sort=True):
        by_year.append({"group": g, "rule": rule, "width": w, "year": y, **trade_stats(part, 1.0)})
    prim = real[(real["width"] == PRIMARY_WIDTH)]
    by_stock = [{"group": g, "rule": rule, "ticker": tk, **trade_stats(part, 1.0)}
                for (g, rule, tk), part in prim.groupby(["group", "rule", "ticker"], sort=True)]
    return stats, pd.DataFrame(ex), pd.DataFrame(by_year), pd.DataFrame(by_stock)


def find(df, **match):
    m = np.ones(len(df), bool)
    for k, v in match.items():
        m &= (df[k] == v).to_numpy()
    return None if not m.any() else df[m].iloc[0]


def test_table(stats, ex):
    rows = []
    for g, rule, w, kind in TESTS:
        if kind == "net":
            r = find(stats, group=g, rule=rule, width=w, mode="prints", control="real", cost="primary")
            mean, lo, hi, p = (np.nan,) * 4 if r is None else (r.get("mean"), r.get("lo"), r.get("hi"), r.get("p_up"))
        else:
            r = find(ex, group=g, rule=rule, width=w, mode="prints")
            mean, lo, hi, p = (np.nan,) * 4 if r is None else (r["excess"], r["lo"], r["hi"], r["p_up"])
        rows.append({"group": g, "rule": rule, "width": w, "test": kind, "mean": mean, "lo": lo, "hi": hi,
                     "p_up": p})
    t = pd.DataFrame(rows)
    t["q"] = mr.bh_qvalues(t["p_up"].fillna(1).to_numpy())
    t["qualifies"] = (t["q"] < FDR) & (t["mean"] > 0)
    return t


def screen(tests, stats, ex, by_year):
    """The screen's verdict for each primary rule, with each criterion."""
    rows = []
    for rule in PRIMARY_RULES:
        t = find(tests, group="core", rule=rule, width=PRIMARY_WIDTH, test="net")
        r = find(stats, group="core", rule=rule, width=PRIMARY_WIDTH, mode="prints", control="real", cost="primary")
        x = find(ex, group="core", rule=rule, width=PRIMARY_WIDTH, mode="prints")
        rel = find(stats, group="related", rule=rule, width=PRIMARY_WIDTH, mode="prints", control="real",
                   cost="primary")
        yrs = by_year[(by_year["group"] == "core") & (by_year["rule"] == rule) & (by_year["width"] == PRIMARY_WIDTH)
                      & by_year["year"].isin(YEARS)]
        c = {"rule": rule,
             "significant": bool(t is not None and t["q"] < FDR and t["mean"] > 0),
             "edge_2x_costs": bool(r is not None and r.get("mean", np.nan) >= r.get("cost_rt", np.inf)),
             "beats_fakes": bool(x is not None and x["excess"] > 0),
             "years_positive": int((yrs["mean"] > 0).sum()),
             "related_positive": bool(rel is not None and rel.get("mean", np.nan) > 0)}
        c["passes"] = c["significant"] and c["edge_2x_costs"] and c["beats_fakes"] and c["years_positive"] >= 2 \
            and c["related_positive"]
        rows.append(c)
    return pd.DataFrame(rows)


def daily_account(trades, index):
    """One position at a time (in order of the fill; a trade is skipped while an earlier one is open), each trade
    with the whole account at the primary cost: daily returns over `index`, plus the trades taken."""
    t = trades.assign(start=trades["fill_ms"], end=trades["exit_ms"])
    taken = pc.one_at_a_time(t)
    r = (taken["net_primary"] / 1e4).groupby(taken["session"]).apply(lambda x: (1 + x).prod() - 1)
    return r.reindex(index).fillna(0), taken


def basket_returns(panels, universe, index):
    """Daily return of holding each month's core stocks in equal weights (split-adjusted closes, no dividends)."""
    ret = panels["close"].pct_change()
    out = pd.Series(np.nan, index=index)
    for m, g in universe[universe["group"] == "core"].groupby("month"):
        days = index[index.to_period("M") == m]
        cols = [c for c in g["ticker"] if c in ret.columns]
        out.loc[days] = ret.reindex(index=days, columns=cols).mean(axis=1)
    return out.fillna(0)


# ---------------------------------------------------------------- study

def run_study(universe, spans, load, sessions, ratio, seconds_of, *, final=False, timings=None):
    """universe/spans from pick_universe/download_spans; load(ticker, first, last) -> (minutes, note). No file I/O
    besides the injected loaders' caches."""
    timings = {} if timings is None else timings
    period = "holdout" if final else "design"
    frames, notes = [], []
    for sp in spans.itertuples():
        with timed(timings, "minute bars"):
            minutes, note = load(sp.ticker, sp.first, sp.last)
        print(f"  {sp.Index + 1}/{len(spans)} {sp.ticker} {sp.first:%Y-%m-%d}..{sp.last:%Y-%m-%d}", flush=True)
        if minutes is None:
            notes.append({"ticker": sp.ticker, "first": sp.first, "last": sp.last, "note": note, "failed": True,
                          "entries": 0})
            continue
        span_sessions = sessions.loc[sp.first:sp.last]
        g = universe[(universe["ticker"] == sp.ticker) & universe["month"].isin(sp.months)]
        groups = dict(zip(g["month"], g["group"]))
        with timed(timings, "levels and entries"):
            e, rth = span_entries(minutes, span_sessions, groups, period)
        notes.append({"ticker": sp.ticker, "first": sp.first, "last": sp.last, "note": note, "failed": False,
                      "entries": 0 if e is None else int((~e["fake"]).sum())})
        if e is None or e.empty:
            continue
        with timed(timings, "one-second replay"):
            for w in WIDTHS:
                frames.append(bc.replay_all(e, rth, seconds_of, ticker=sp.ticker, width=w)
                              .assign(ticker=sp.ticker, width=w, mode="prints"))
            frames.append(bc.replay_all(e, rth, bodies(seconds_of), ticker=sp.ticker, width=PRIMARY_WIDTH)
                          .assign(ticker=sp.ticker, width=PRIMARY_WIDTH, mode="bodies"))
        seconds_of.save()
    trades = pd.concat(frames, ignore_index=True)
    trades["rule"] = trades["rule"].map(NAME)
    status = trades.groupby(["mode", "fake", "status"]).size().rename("trades").reset_index()
    trades = add_net(trades[trades["status"] == "ok"].reset_index(drop=True), ratio)
    first = sessions.index[sessions.index >= universe["month"].min().start_time][0]
    last = DESIGN_END if not final else sessions.index[-1]
    years = (last - first).days / 365.25
    with timed(timings, "statistics"):
        stats, ex, by_year, by_stock = statistics(trades, years)
        tests = test_table(stats, ex)
        verdict = screen(tests, stats, ex, by_year)
    return {"trades": trades, "stats": stats, "excess": ex, "by_year": by_year, "by_stock": by_stock,
            "tests": tests, "screen": verdict, "status": status, "spans": pd.DataFrame(notes), "years": years,
            "first": first, "last": last, "final": final, "universe": universe}


def portfolio_view(res, panels, sessions, cache_dir):
    """Account view and correlations for the primary rules on the core stocks (the rules that pass, if any)."""
    t = res["trades"]
    passing = list(res["screen"].loc[res["screen"]["passes"], "rule"]) or list(PRIMARY_RULES)
    pick = t[~t["fake"] & (t["mode"] == "prints") & (t["width"] == PRIMARY_WIDTH) & (t["group"] == "core")
             & t["rule"].isin(passing)]
    index = sessions.index[(sessions.index >= res["first"]) & (sessions.index <= res["last"])]
    daily, taken = daily_account(pick, index)
    spy = panels["spy_close"].reindex(index).pct_change().fillna(0)
    basket = basket_returns(panels, res["universe"], index)
    dip = pc.spy_dip_daily(index, cache_dir)
    tsla = pc.tsla_breakout_daily(index)
    acct = pd.DataFrame([{"what": f"{' + '.join(passing)}, one at a time", "trades": len(taken),
                          **pc.account(daily)},
                         {"what": "holding SPY", **pc.account(spy)},
                         {"what": "holding each month's core stocks, equal weight", **pc.account(basket)},
                         {"what": "SPY dip buying (shares)", **pc.account(dip)}])
    corr = pc.correlations(daily, {"SPY dip buying": dip, "TSLA breakouts": tsla, "SPY": spy})
    return {"rules": passing, "account": acct, "corr": corr, "daily": daily, "taken": taken}


# ---------------------------------------------------------------- report

def bps(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1f}"


def ci(r, k="mean"):
    if r is None or pd.isna(r.get(k)):
        return "n/a"
    return f"{bps(r[k])} ({bps(r['lo'])} to {bps(r['hi'])})"


def render_report(res, port):
    st, ex, tests, by_year, by_stock = res["stats"], res["excess"], res["tests"], res["by_year"], res["by_stock"]
    lines = []
    w = lines.append
    title = "the 2025-26 check" if res["final"] else "design screen"
    w(f"# Breakouts in the most volatile stocks: {title}\n")
    w(f"{res['first']:%Y-%m-%d} to {res['last']:%Y-%m-%d}. The TSLA keeper rules (a stop order at yesterday's high "
      "or low, stop and target 0.1 ATR from the level) on the 10 most volatile liquid stocks of each month (TSLA "
      "left out), replayed on one-second bars. P&L is in basis points of the fill per trade, after costs of 1 bp "
      "plus half a cent a share per side unless stated (1 bp = $1 per $10,000 traded). Rules and tests were fixed "
      "before any result was computed (see `volatile_breakouts.py`). A signal study's P&L, not an account's, except "
      "in the account view.\n")
    sc = res["screen"]
    w("## Verdict\n")
    w(alerts.md_table(["Rule", "Significant (q < 0.10)", "Net >= costs", "Beats fake levels", "Years positive (of 3)",
                       "Related stocks positive", "Passes"],
                      [[r.rule, *("yes" if r[k] else "no" for k in ("significant", "edge_2x_costs", "beats_fakes")),
                        str(r.years_positive), "yes" if r.related_positive else "no",
                        "**yes**" if r.passes else "no"] for _, r in sc.iterrows()]) + "\n")
    w("## The declared tests\n")
    w(alerts.md_table(["#", "Stocks", "Rule", "Stop/target", "Test", "bps a trade (95%)", "p", "q", "Qualifies"],
                      [[str(i + 1), r.group, r.rule, f"{r.width:g} ATR", r.test,
                        f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})", f"{r.p_up:.3f}",
                        f"{r.q:.3f}", "yes" if r.qualifies else "no"] for i, r in tests.iterrows()]) + "\n")
    w("## Every rule\n")
    rows = []
    for g in ("core", "related"):
        for rule in RULES:
            for wd in WIDTHS:
                r = find(st, group=g, rule=rule, width=wd, mode="prints", control="real", cost="primary")
                if r is None:
                    continue
                f = find(st, group=g, rule=rule, width=wd, mode="prints", control="fake", cost="primary")
                r0, r1 = (find(st, group=g, rule=rule, width=wd, mode="prints", control="real", cost=c)
                          for c in ("none", "1 bp"))
                x = find(ex, group=g, rule=rule, width=wd, mode="prints")
                rows.append([g, rule, f"{wd:g}", f"{int(r['trades']):,} ({r['per_year']:.0f})",
                             lv.pct(r.get("win")), ci(r), f"{bps(bo.val(r0, 'mean'))} / {bps(bo.val(r1, 'mean'))}",
                             bps(r.get("cost_rt")), bps(bo.val(f, "mean")),
                             "n/a" if x is None else f"{bps(x['excess'])} ({bps(x['lo'])} to {bps(x['hi'])})"])
    w(alerts.md_table(["Stocks", "Rule", "ATR", "Trades (a year)", "Wins", "Net bps (95%)", "Before costs / 1 bp",
                       "Avg cost a round trip", "Fake levels", "Real minus fake (95%)"], rows) + "\n")
    w("## By year (core, 0.1 ATR)\n")
    rows = []
    for rule in RULES:
        for y in sorted(by_year["year"].unique()):
            r = find(by_year, group="core", rule=rule, width=PRIMARY_WIDTH, year=y)
            if r is not None:
                rows.append([rule, str(y), f"{int(r['trades'])}", lv.pct(r.get("win")), ci(r)])
    w(alerts.md_table(["Rule", "Year", "Trades", "Wins", "Net bps (95%)"], rows) + "\n")
    w("## Odd prints: one-second wicks ignored (core, 0.1 ATR)\n")
    w("Only one-second bodies (open to close) may fill, stop or target, so a lone print far from the rest of its "
      "second can't trip anything. Trades whose bodies never reach the level are dropped.\n")
    rows = []
    for rule in RULES:
        a = find(st, group="core", rule=rule, width=PRIMARY_WIDTH, mode="prints", control="real", cost="primary")
        b = find(st, group="core", rule=rule, width=PRIMARY_WIDTH, mode="bodies", control="real", cost="primary")
        if a is not None and b is not None:
            rows.append([rule, f"{int(a['trades'])}", ci(a), f"{int(b['trades'])}", ci(b)])
    w(alerts.md_table(["Rule", "Trades (all prints)", "Net bps", "Trades (bodies only)", "Net bps"], rows) + "\n")
    w("## By stock (core, primary rules, 0.1 ATR)\n")
    rows = []
    for rule in PRIMARY_RULES:
        part = by_stock[(by_stock["group"] == "core") & (by_stock["rule"] == rule)].sort_values("trades",
                                                                                             ascending=False)
        pos = int((part["mean"] > 0).sum())
        w(f"- {rule}: {len(part)} stocks, {pos} with a positive average.")
        rows += [[rule, r["ticker"], f"{int(r['trades'])}", bps(r.get("mean"))] for _, r in part.head(12).iterrows()]
    w("")
    w(alerts.md_table(["Rule", "Stock", "Trades", "Net bps"], rows) + "\n")
    if port is not None:
        w("## Account view\n")
        w(f"Rules: {', '.join(port['rules'])}; core stocks, one position at a time (the first fill wins; nothing new "
          "while a trade is open), the whole account in each trade, primary costs. No leverage, no interest on idle "
          "cash.\n")
        rows = []
        for _, r in port["account"].iterrows():
            yrs = " / ".join(f"{r[c] * 100:+.1f}%" for c in r.index if str(c).startswith("y") and pd.notna(r[c]))
            rows.append([r["what"], "" if pd.isna(r.get("trades")) else f"{int(r['trades'])}",
                         f"{r['total'] * 100:+.1f}%", f"{r['per_year'] * 100:+.1f}%", f"{r['max_dd'] * 100:.1f}%",
                         yrs])
        w(alerts.md_table(["", "Trades", "Total", "A year", "Worst drop", "Each year"], rows) + "\n")
        w(alerts.md_table(["Daily P&L correlation with", "Correlation", "Sessions both had a position"],
                          [[r["with"], "n/a" if pd.isna(r["corr"]) else f"{r['corr']:+.2f}", f"{r['both_active']}"]
                           for _, r in port["corr"].iterrows()]) + "\n")
    w("## Universe and data\n")
    u = res["universe"]
    core = u[u["group"] == "core"]
    w(f"- {u['month'].nunique()} months; {core['ticker'].nunique()} different core stocks and "
      f"{u.loc[u['group'] == 'related', 'ticker'].nunique()} related ones. Most months in the core: "
      + ", ".join(f"{k} {v}" for k, v in core["ticker"].value_counts().head(12).items()) + ".")
    s = res["spans"]
    bad = s[s["failed"]]
    w(f"- Minute-bar spans: {len(s)}; failed or short downloads (left out): {len(bad)}"
      + (" (" + ", ".join(f"{r.ticker} {r.first:%Y-%m}" for r in bad.itertuples()) + ")" if len(bad) else "") + ".")
    stt = res["status"]
    miss = stt[stt["status"] != "ok"]
    w("- Trades without the touch in their one-second bars (left out): "
      + (", ".join(f"{r.mode} {'fake' if r.fake else 'real'} {int(r.trades)}" for r in miss.itertuples())
         if len(miss) else "none") + ".")
    w("- Costs: 1 bp a side plus half a cent a share, at the unadjusted price (Massive's grouped daily bars give the "
      "split ratio). Short trades assume shares can be borrowed; on Robinhood a short needs a put instead (the "
      "options version is priced only in the 2025-26 check, where option data exists).")
    return "\n".join(lines) + "\n"


def plot(res, path):
    t = res["trades"]
    fig = plt.figure(figsize=(10, 4.4), dpi=150, facecolor=SURFACE)
    for k, rule in enumerate(PRIMARY_RULES):
        ax = sds._axes(fig, [0.08 + k * 0.48, 0.2, 0.4, 0.6])
        for (label, fake, group), color in zip((("core", False, "core"), ("core, fake levels", True, "core"),
                                                ("related", False, "related")), (SERIES[0], MUTED, SERIES[1])):
            g = t[(t["rule"] == rule) & (t["fake"] == fake) & (t["group"] == group) & (t["mode"] == "prints")
                  & (t["width"] == PRIMARY_WIDTH)].sort_values("fill_ms")
            if len(g):
                ax.plot(g["session"], g["net_primary"].cumsum() / max(len(g), 1), color=color, lw=1.2,
                        label=label if k == 0 else None)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(rule, color=INK, fontsize=9, loc="left")
        ax.set_ylabel("Running sum of net bps / trades", color=INK_2, fontsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=6.5)
    fig.text(0.02, 0.97, "Volatile-stock breakouts: " + ("2025-26 check" if res["final"] else "design screen"),
             color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Net of 1 bp + half a cent a share per side; one-second fills; ends at the average net bps "
             "a trade.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TSLA breakout rules on the most volatile liquid stocks.")
    p.add_argument("--out", default="output/volatile_breakouts")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="run the one-time 2025-26 check (rules in FINAL)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.final_test and FINAL is None:
        raise SystemExit("FINAL is not set: write the held-back rules down before running --final-test.")
    timings = {}
    sessions = features.trading_sessions(DATA_START, FINAL_END, warmup_sessions=0)
    with timed(timings, "grouped daily"):
        adj = md.load_grouped(sessions.index, True, args.cache_dir)
        raw = md.load_grouped(sessions.index, False, args.cache_dir)
        panels = daily_panels(adj, raw, md.stock_types(args.cache_dir))
        panels["spy_close"] = adj[adj["ticker"] == "SPY"].set_index("session")["close"]
        ratio = panels["raw_close"] / panels["close"]
    lo, hi = (HOLDOUT_START, FINAL_END) if args.final_test else (DATA_START, DESIGN_END)
    months = pd.period_range(lo, hi, freq="M")
    universe = pick_universe(panels, months)
    spans = download_spans(universe, sessions)
    if not args.final_test:  # the design run never loads a 2025+ session
        sessions = sessions[sessions.index <= DESIGN_END]
        spans["last"] = spans["last"].clip(upper=DESIGN_END)
    seconds_of = bc.second_loader(args.cache_dir, args.refresh)

    def load(tk, first, last):
        return load_span(tk, first, last, sessions, args.cache_dir, args.refresh)

    try:
        res = run_study(universe, spans, load, sessions, ratio, seconds_of, final=args.final_test, timings=timings)
    finally:
        seconds_of.save()
    with timed(timings, "account view"):
        port = portfolio_view(res, panels, sessions, args.cache_dir)
    out = Path(args.out) / ("holdout" if args.final_test else "design")
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    universe.assign(month=universe["month"].astype(str)).to_csv(out / "universe.csv", index=False)
    for k in ("stats", "excess", "by_year", "by_stock", "tests", "screen", "spans"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    port["account"].to_csv(out / "account.csv", index=False)
    port["corr"].to_csv(out / "correlations.csv", index=False)
    (out / "settings.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now(tz=NY).isoformat(), "final": args.final_test, "first": str(res["first"].date()),
        "last": str(res["last"].date()), "seconds_fetched": seconds_of.state["fetched"],
        "timings": {k: round(v, 1) for k, v in timings.items()}}, indent=2))
    plot(res, out / "breakouts.png")
    (out / "report.md").write_text(render_report(res, port))
    print(res["screen"].to_string(index=False))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
