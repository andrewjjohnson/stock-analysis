"""Drift after news-sized moves in liquid stocks: when a large stock jumps or drops far more than usual on very heavy
volume (earnings, guidance, an analyst call), does it keep going over the next days, relative to SPY? A signal
study on Massive's grouped daily bars: P&L per trade in bps, stock minus SPY over the same window, after costs.

  uv run --env-file .env python shock_drift.py --out output/shock_drift                 # design screen, 2021-24
  uv run --env-file .env python shock_drift.py --final-test --out output/shock_drift    # 2025-26, once

Mechanism. Heavy volume on a big stock-specific move usually means news. Investors underreact to news that takes
work to interpret, and large holders adjust their positions over days rather than at once, so prices drift in the
direction of the news (post-earnings drift, Bernard and Thomas 1989; drift after news but reversal after big
moves without news, Chan 2003). The other side is investors anchored to the old price and traders fading big
moves. The trade is hedged with SPY, so it is market-neutral and takes both directions. The worry: drift after
earnings has largely gone in large US stocks (Martineau 2022), so this may well fail.

Declared on 2026-10-04, before any result was computed:
- Universe, re-picked each month by rule: the 300 common stocks or ADRs with the largest median daily dollar volume
  (VWAP x volume) over the 21 sessions before the month's first session, trading every one of them, with an
  unadjusted close of at least $10 on the last.
- Event on session t (prices split-adjusted, not dividend-adjusted): the stock's close-to-close return minus SPY's
  (AR) is at least 3 standard deviations of its AR over the previous 60 sessions (at least 50 present), up or down;
  "heavy volume" = the session's volume at least 3x its median over the previous 20 sessions; "light volume" = under
  2x. A stock can have at most one event in any 5 sessions (later ones are skipped).
- Trade: at the next session's open, buy the stock and sell SPY in equal dollars after an up move (the reverse
  after a down move); close both at the close of session t+5 (primary) or t+20. Costs per side: 1 bp plus half a
  cent a share on the stock (at its unadjusted price), 1 bp on SPY.
- Tests (8, one-sided; standard errors clustered by the entry week, or the entry month at 20 sessions; BH q < 0.10
  across all 8), design 2021-11 to 2024-12:
    1. heavy-volume up moves, 5 sessions: mean net drift > 0
    2. heavy-volume down moves, 5 sessions: the same (short the stock)
    3. heavy volume, both directions, 5 sessions (primary)
    4. heavy volume, both directions, 20 sessions
    5. heavy volume, both directions, 2 sessions (next open to the close of t+2)
    6. light-volume moves, both directions, 5 sessions: reversal (trading against the move) > 0
    7. heavy minus light volume: the drift difference at 5 sessions > 0
    8. the 100 largest stocks only: heavy volume, both directions, 5 sessions
- Passes the screen if test 3 or 4 has q < 0.10 with a mean net P&L at least its round-trip cost (so the edge
  before costs is at least twice the costs), is positive in at least 2 of 2022-2024, and test 8 has the same sign.
- Reported, not tested: each year, each move size, prices, the portfolio view (every open event position equally
  weighted, at most 10% of the account each) and its daily-P&L correlation with SPY dip buying, the TSLA breakouts
  and SPY.
- Held-back check (--final-test): 2025-01 to 2026-09 with the same universe rule, leaving out the stocks whose
  2025-26 daily prices earlier studies already looked at (MSFT, AAPL, AMZN, META, TSLA, NVDA).

Held-back check, declared on 2026-10-04 after the design screen (which this idea did not pass) and before any
2025-26 result was computed; run at the user's request, once:
- Events, trades, costs and universe exactly as above. 20-session trades that would end after 2026-09-30 are left out.
- Three tests (FINAL), one-sided, clustered as above, Benjamini-Hochberg q < 0.10 across the three:
    1. heavy volume, both directions, 5 sessions (the design primary)
    2. heavy volume, both directions, 20 sessions
    3. heavy volume, both directions, 20 sessions, only when SPY closed above its average of the last 200 sessions
       on the event day (a rule known in advance for "a bullish market")
- The check passes if at least one test has q < 0.10 and a mean net P&L at least its average round-trip cost; the
  passing versions are the ones to keep. Everything else in the report (years, directions, light-volume moves, the
  100 largest, unhedged) is description, not a test.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import market_daily as md  # noqa: E402
import meanrev as mr  # noqa: E402
import portfolio_check as pc  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
import volatile_breakouts as vb  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
DATA_START, FINAL_END = vb.DATA_START, vb.FINAL_END
DESIGN_END, HOLDOUT_START = vb.DESIGN_END, vb.HOLDOUT_START
TOP = 300
TOP_SUBSET = 100
LOOKBACK = 21
MIN_PRICE = 10.0
SIGMA_WINDOW, SIGMA_MIN = 60, 50
Z = 3.0
VOL_WINDOW = 20
HEAVY, LIGHT = 3.0, 2.0
GAP = 5                         # sessions between events of one stock
HORIZONS = (2, 5, 20)
PRIMARY_H = 5
COST_BPS, HALF_TICK, HEDGE_BPS = 1.0, 0.005, 1.0
YEARS = (2022, 2023, 2024)
FDR = 0.10
SEEN = ("MSFT", "AAPL", "AMZN", "META", "TSLA", "NVDA")   # 2025-26 daily prices already looked at
TESTS = [("heavy", "up", 5, "drift"), ("heavy", "down", 5, "drift"), ("heavy", "both", 5, "drift"),
         ("heavy", "both", 20, "drift"), ("heavy", "both", 2, "drift"), ("light", "both", 5, "reversal"),
         ("heavy - light", "both", 5, "difference"), ("heavy, top 100", "both", 5, "drift")]
FINAL = [("heavy, both, 5 sessions", PRIMARY_H, False), ("heavy, both, 20 sessions", 20, False),
         ("heavy, both, 20 sessions, SPY above its 200-session average", 20, True)]   # (name, sessions, filter)
SMA = 200


# ---------------------------------------------------------------- data

def panels_from(adj, raw, types):
    """Session x ticker panels for stocks (CS/ADRC) plus SPY: split-adjusted open, close, volume, dollar volume and
    the unadjusted close."""
    a = adj[adj["ticker"].map(types).isin(md.TYPES) | (adj["ticker"] == "SPY")]
    a = a.assign(dollar_volume=a["vwap"].fillna(a["close"]) * a["volume"])
    big = set(a.loc[a["dollar_volume"] >= 5e7, "ticker"]) | {"SPY"}
    a = a[a["ticker"].isin(big)]

    def wide(df, col):
        return df.pivot_table(index="session", columns="ticker", values=col, aggfunc="last").sort_index()

    out = {k: wide(a, k) for k in ("open", "close", "volume", "dollar_volume")}
    r = raw[raw["ticker"].isin(big)]
    out["raw_close"] = wide(r, "close").reindex(index=out["close"].index, columns=out["close"].columns)
    return out


def monthly_universe(p, months, top=TOP, lookback=LOOKBACK, min_price=MIN_PRICE):
    """{month: ordered tickers}: the `top` stocks by median dollar volume over the `lookback` sessions before the
    month's first session (trading on all of them), unadjusted close >= min_price on the last. SPY excluded."""
    days = p["close"].index
    out = {}
    for m in months:
        i = int(days.searchsorted(m.start_time))
        if i >= len(days) or days[i].to_period("M") != m or i < lookback:
            continue
        win = slice(i - lookback, i)
        d, c = p["dollar_volume"].iloc[win], p["close"].iloc[win]
        ok = d.notna().all() & c.notna().all() & (p["raw_close"].iloc[i - 1] >= min_price)
        med = d.median()[ok].drop("SPY", errors="ignore").sort_values(ascending=False, kind="stable")
        out[m] = list(med.index[:top])
    return out


# ---------------------------------------------------------------- events and trades

def event_table(p, universe):
    """One row per stock-session in the universe with AR, its z-score from the previous SIGMA_WINDOW sessions and
    the volume ratio to the previous VOL_WINDOW sessions' median. Everything uses sessions up to t only."""
    close, vol = p["close"], p["volume"]
    ret = close.pct_change(fill_method=None)
    ar = ret.sub(ret["SPY"], axis=0)
    sd = ar.shift(1).rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).std()
    vr = vol / vol.shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).median()
    rows = []
    days = close.index
    for m, tickers in universe.items():
        d = days[days.to_period("M") == m]
        cols = [t for t in tickers if t in close.columns]
        z = (ar.loc[d, cols] / sd.loc[d, cols]).stack()
        part = pd.DataFrame({"z": z, "ar": ar.loc[d, cols].stack().reindex(z.index),
                             "vr": vr.loc[d, cols].stack().reindex(z.index)})
        rank = {t: k + 1 for k, t in enumerate(tickers)}
        part["rank"] = part.index.get_level_values(1).map(rank).to_numpy()
        rows.append(part)
    t = pd.concat(rows)
    t.index.names = ["session", "ticker"]
    return t.reset_index()


def space_events(events, days, gap=GAP):
    """At most one event per stock in any `gap` sessions, by session position (the first wins)."""
    pos = pd.Series(np.arange(len(days)), index=days)
    e = events.assign(pos=events["session"].map(pos).to_numpy()).sort_values(["ticker", "pos"])
    keep, last = [], {}
    for i, tk, k in zip(e.index, e["ticker"], e["pos"]):
        if tk in last and k - last[tk] < gap:
            continue
        keep.append(i)
        last[tk] = k
    return e.loc[keep].drop(columns="pos")


def trade_returns(events, p, horizons=HORIZONS):
    """For each event on t: the stock and SPY from the open of t+1 to the close of t+h; signed (direction of the
    move) stock-minus-SPY gross and net bps. NaN when a price is missing or t+h is past the data."""
    o, c, raw = p["open"], p["close"], p["raw_close"]
    days = c.index
    pos = pd.Series(np.arange(len(days)), index=days)
    i = events["session"].map(pos).to_numpy()
    cols = c.columns.get_indexer(events["ticker"])
    spy = c.columns.get_loc("SPY")
    O, C, R = o.to_numpy(float), c.to_numpy(float), raw.to_numpy(float)
    out = events.copy()
    d = np.sign(events["ar"].to_numpy())
    out["direction"] = d
    ok = i + 1 < len(days)
    entry = np.where(ok, O[np.minimum(i + 1, len(days) - 1), cols], np.nan)
    spy_entry = np.where(ok, O[np.minimum(i + 1, len(days) - 1), spy], np.nan)
    price = np.where(ok, entry * (R[i, cols] / C[i, cols]), np.nan)
    out["price"] = price
    spy_close = c["SPY"]
    out["spy_vs_sma"] = (spy_close / spy_close.rolling(SMA, min_periods=SMA).mean()).to_numpy()[i]  # known at t
    cost = 2 * (COST_BPS + HALF_TICK / price * 1e4) + 2 * HEDGE_BPS
    out["cost_rt"] = cost
    for h in horizons:
        j = i + h
        okh = j < len(days)
        jj = np.minimum(j, len(days) - 1)
        stock = np.where(okh, C[jj, cols] / entry - 1, np.nan)
        mkt = np.where(okh, C[jj, spy] / spy_entry - 1, np.nan)
        out[f"exit_{h}"] = np.where(okh, days[jj], pd.NaT)
        out[f"gross_{h}"] = d * (stock - mkt) * 1e4
        out[f"net_{h}"] = out[f"gross_{h}"] - cost
        out[f"raw_{h}"] = d * stock * 1e4 - (cost - 2 * HEDGE_BPS)     # unhedged, stock costs only
    return out


# ---------------------------------------------------------------- statistics

def clustered(x, groups):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(groups)[ok]
    n = len(x)
    if n < 3:
        return {"trades": n}
    m, se = sm.cluster_mean(x, g)
    return {"trades": n, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(m / se) if se and np.isfinite(se) else np.nan, "win": (x > 0).mean() * 100}


def cluster_key(sessions, h):
    s = pd.to_datetime(pd.Series(sessions))
    return (s.dt.to_period("M") if h >= 20 else s.dt.to_period("W")).astype(str).to_numpy()


def subset(t, volume, side):
    m = t["volume"] == volume if volume in ("heavy", "light") else t["volume"] == "heavy"
    if volume == "heavy, top 100":
        m &= t["rank"] <= TOP_SUBSET
    if side == "up":
        m &= t["direction"] > 0
    elif side == "down":
        m &= t["direction"] < 0
    return t[m]


def test_table(t):
    rows = []
    for volume, side, h, kind in TESTS:
        col = f"net_{h}"
        if kind == "difference":
            a, b = subset(t, "heavy", side), subset(t, "light", side)
            # light-volume drift measured the same way (with the move); difference via stacking with signs
            x = np.r_[a[col].to_numpy(), -b[col].to_numpy()]
            n_a = len(a)
            g = np.r_[cluster_key(a["session"], h), cluster_key(b["session"], h)]
            ma, mb = a[col].mean(), b[col].mean()
            # clustered SE of a difference of two means: stack centred values, weights 1/n per group
            ca = (a[col] - ma).to_numpy() / max(n_a, 1)
            cb = -(b[col] - mb).to_numpy() / max(len(b), 1)
            sums = pd.Series(np.r_[ca, cb]).groupby(g).sum().to_numpy()
            k = len(sums)
            se = np.sqrt((sums ** 2).sum() * k / (k - 1)) if k > 1 else np.nan
            dmean = ma - mb
            r = {"trades": len(x), "mean": dmean, "lo": dmean - 1.96 * se, "hi": dmean + 1.96 * se,
                 "p_up": norm.sf(dmean / se) if se else np.nan}
        else:
            s = subset(t, volume, side)
            gross, cost = s[f"gross_{h}"].to_numpy(float), s["cost_rt"].to_numpy(float)
            x = gross - cost if kind == "drift" else -gross - cost      # reversal: trade against the move
            r = clustered(x, cluster_key(s["session"], h))
            r["cost_rt"] = s["cost_rt"].mean()
        rows.append({"volume": volume, "side": side, "sessions": h, "test": kind, **r})
    tests = pd.DataFrame(rows)
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["qualifies"] = (tests["q"] < FDR) & (tests["mean"] > 0)
    return tests


def verdict(tests, by_year):
    t = tests.reset_index(drop=True)
    prim = [t.iloc[2], t.iloc[3]]
    ok = [bool(r["q"] < FDR and r["mean"] >= r.get("cost_rt", np.inf)) for r in prim]
    yrs = by_year[(by_year["sessions"] == PRIMARY_H) & by_year["year"].isin(YEARS)]
    c = {"significant_and_2x_costs": any(ok), "years_positive": int((yrs["mean"] > 0).sum()),
         "top100_same_sign": bool(np.sign(t.iloc[7]["mean"]) == np.sign(t.iloc[2]["mean"]))}
    c["passes"] = c["significant_and_2x_costs"] and c["years_positive"] >= 2 and c["top100_same_sign"]
    return c


def final_tests(t):
    """The held-back tests in FINAL, BH across them, and the verdict."""
    heavy = t[t["volume"] == "heavy"]
    rows = []
    for name, h, bull in FINAL:
        g = heavy[heavy["spy_vs_sma"] > 1] if bull else heavy
        r = clustered(g[f"net_{h}"], cluster_key(g["session"], h))
        rows.append({"test": name, "sessions": h, **r, "cost_rt": g.loc[g[f"net_{h}"].notna(), "cost_rt"].mean()})
    out = pd.DataFrame(rows)
    out["q"] = mr.bh_qvalues(out["p_up"].fillna(1).to_numpy())
    out["passes"] = (out["q"] < FDR) & (out["mean"] >= out["cost_rt"])
    return out


def portfolio_daily(t, p, index, h=PRIMARY_H, cap=0.10):
    """Every heavy-volume event position open on a session, equally weighted (at most `cap` of the account each):
    daily return of stock minus SPY in the move's direction, costs charged on entry."""
    c, o = p["close"], p["open"]
    days = c.index
    pos = pd.Series(np.arange(len(days)), index=days)
    rc = c.pct_change(fill_method=None)
    first = c / o - 1                       # entry day: open to close
    contrib = {}
    for r in t.itertuples():
        i = pos[r.session]
        if i + h >= len(days) or not np.isfinite(getattr(r, f"net_{h}")):
            continue
        for k in range(i + 1, i + h + 1):
            day = days[k]
            if k == i + 1:
                x = first.iat[k, c.columns.get_loc(r.ticker)] - first.iat[k, c.columns.get_loc("SPY")]
                x -= r.cost_rt / 1e4
            else:
                x = rc.iat[k, c.columns.get_loc(r.ticker)] - rc.iat[k, c.columns.get_loc("SPY")]
            contrib.setdefault(day, []).append(r.direction * x)
    out = pd.Series(0.0, index=index)
    for day, xs in contrib.items():
        if day in out.index:
            w = min(cap, 1 / len(xs))
            out[day] = w * np.nansum(xs)
    return out


# ---------------------------------------------------------------- study

def run_study(p, universe, *, period="design", timings=None):
    timings = {} if timings is None else timings
    days = p["close"].index
    with timed(timings, "events"):
        table = event_table(p, universe)
        ev = table[table["z"].abs() >= Z]
        ev = ev.assign(volume=np.where(ev["vr"] >= HEAVY, "heavy", np.where(ev["vr"] < LIGHT, "light", "middle")))
        ev = space_events(ev, days)
        trades = trade_returns(ev, p)
    keep_last = DESIGN_END if period == "design" else FINAL_END
    lo = HOLDOUT_START if period == "holdout" else pd.Timestamp("1900-01-01")
    for h in HORIZONS:  # nothing may end past the period
        late = pd.to_datetime(trades[f"exit_{h}"]) > keep_last
        trades.loc[late, [f"gross_{h}", f"net_{h}", f"raw_{h}"]] = np.nan
    trades = trades[(trades["session"] >= lo) & (trades["session"] <= keep_last)]
    if period == "holdout":
        trades = trades[~trades["ticker"].isin(SEEN)]
    tests = test_table(trades)
    by_year = []
    heavy = trades[trades["volume"] == "heavy"]
    for h in HORIZONS:
        for y, g in heavy.groupby(heavy["session"].dt.year):
            by_year.append({"sessions": h, "year": y, **clustered(g[f"net_{h}"], cluster_key(g["session"], h))})
    by_year = pd.DataFrame(by_year)
    size = pd.cut(heavy["z"].abs(), [3, 4, 6, np.inf], labels=["3-4 sd", "4-6 sd", "6+ sd"], right=False)
    by_size = pd.DataFrame([{"size": k, "side": s, **clustered(g[f"net_{PRIMARY_H}"],
                                                              cluster_key(g["session"], PRIMARY_H))}
                            for (k, s), g in heavy.groupby([size, np.where(heavy["direction"] > 0, "up", "down")],
                                                           observed=True)])
    raw = pd.DataFrame([{"side": s, "sessions": h, **clustered(subset(trades, "heavy", s)[f"raw_{h}"],
                                                               cluster_key(subset(trades, "heavy", s)["session"], h))}
                        for s in ("up", "down") for h in HORIZONS])
    final = final_tests(trades) if period == "holdout" else None
    return {"trades": trades, "tests": tests, "by_year": by_year, "by_size": by_size, "raw": raw, "final": final,
            "verdict": verdict(tests, by_year) if period == "design" else
            {"passes": bool(final["passes"].any()), "passing": list(final.loc[final["passes"], "test"])},
            "period": period, "counts": trades["volume"].value_counts().to_dict()}


# ---------------------------------------------------------------- report

def bps(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.0f}"


def ci(r):
    if r is None or pd.isna(r.get("mean")):
        return "n/a"
    return f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})"


def render_report(res, port):
    t, v = res["tests"], res["verdict"]
    lines = []
    w = lines.append
    w("# Drift after news-sized moves: " + ("design screen, 2021-24" if res["period"] == "design"
                                            else "held-back check, 2025-26") + "\n")
    w("A move of 3+ standard deviations against SPY on 3x normal volume, in the 300 most traded stocks; buy (or sell) "
      "the stock and hedge with SPY at the next open, for 5 sessions. P&L in bps per trade after costs (1 bp + half "
      "a cent a share on the stock, 1 bp on SPY, each side). Split-adjusted daily bars, no dividends. Rules fixed in "
      "advance (see `shock_drift.py`). A signal study, not account P&L except in the portfolio view.\n")
    if res["final"] is not None:
        f = res["final"]
        w("## The held-back tests (declared before 2025-26 was read)\n")
        w(f"**Verdict: {'passed' if v['passes'] else 'failed'}**"
          + (f" ({', '.join(v['passing'])})" if v["passing"] else "") + ". A test passes with BH q < 0.10 across the "
          "three and a mean net P&L at least its round-trip cost. MSFT, AAPL, AMZN, META, TSLA and NVDA left out.\n")
        w(alerts.md_table(["#", "Test", "Trades", "Wins", "Net bps (95%)", "Cost a round trip", "p", "q", "Passes"],
                          [[str(i + 1), r["test"], f"{int(r['trades'])}", f"{r['win']:.0f}%", ci(r),
                            f"{r['cost_rt']:.1f}", f"{r['p_up']:.3f}", f"{r['q']:.3f}",
                            "yes" if r["passes"] else "no"] for i, r in f.iterrows()]) + "\n")
        w("## The design screen's tests on 2025-26 (description only, not tests)\n")
    else:
        w("## Verdict\n")
        w(alerts.md_table(["Significant, net >= costs", "Years positive (of 3)", "Top 100 same sign", "Passes"],
                          [["yes" if v["significant_and_2x_costs"] else "no", str(v["years_positive"]),
                            "yes" if v["top100_same_sign"] else "no", "**yes**" if v["passes"] else "no"]]) + "\n")
        w("## The declared tests\n")
    w(alerts.md_table(["#", "Volume", "Direction", "Sessions", "Test", "Trades", "Net bps (95%)", "p", "q",
                       "Qualifies"],
                      [[str(i + 1), r["volume"], r["side"], str(r["sessions"]), r["test"], f"{int(r['trades'])}",
                        ci(r), f"{r['p_up']:.3f}", f"{r['q']:.3f}", "yes" if r["qualifies"] else "no"]
                       for i, r in t.iterrows()]) + "\n")
    w(f"Events by volume: {res['counts']}. Average round-trip cost of the heavy-volume trades: "
      f"{t.iloc[2].get('cost_rt', np.nan):.1f} bps.\n")
    w("## By year (heavy volume, both directions)\n")
    w(alerts.md_table(["Sessions", "Year", "Trades", "Net bps (95%)"],
                      [[str(int(r["sessions"])), str(int(r["year"])), f"{int(r['trades'])}", ci(r)]
                       for _, r in res["by_year"].iterrows()]) + "\n")
    w("## By move size (heavy volume, 5 sessions)\n")
    w(alerts.md_table(["Size", "Direction", "Trades", "Net bps (95%)"],
                      [[str(r["size"]), r["side"], f"{int(r['trades'])}", ci(r)]
                       for _, r in res["by_size"].iterrows()]) + "\n")
    w("## Without the SPY hedge (heavy volume)\n")
    w(alerts.md_table(["Direction", "Sessions", "Trades", "Net bps (95%)"],
                      [[r["side"], str(r["sessions"]), f"{int(r['trades'])}", ci(r)]
                       for _, r in res["raw"].iterrows()]) + "\n")
    if port is not None:
        w("## Portfolio view (heavy volume, 5 sessions, every open position equally weighted, at most 10% each)\n")
        rows = []
        for _, r in port["account"].iterrows():
            yrs = " / ".join(f"{r[c] * 100:+.1f}%" for c in r.index if str(c).startswith("y") and pd.notna(r[c]))
            rows.append([r["what"], f"{r['total'] * 100:+.1f}%", f"{r['per_year'] * 100:+.1f}%",
                         f"{r['max_dd'] * 100:.1f}%", yrs])
        w(alerts.md_table(["", "Total", "A year", "Worst drop", "Each year"], rows) + "\n")
        w(alerts.md_table(["Daily P&L correlation with", "Correlation", "Sessions both had a position"],
                          [[r["with"], "n/a" if pd.isna(r["corr"]) else f"{r['corr']:+.2f}", f"{r['both_active']}"]
                           for _, r in port["corr"].iterrows()]) + "\n")
    w("## Notes\n")
    w("- Earnings dates are not in the data plan; heavy volume on a big stock-specific move stands in for news.")
    w("- Prices are split-adjusted but not dividend-adjusted; over 5 sessions this matters only on ex-dates, for "
      "both the stock and SPY.")
    w("- On Robinhood the short side would be a put and the SPY hedge an inverse ETF or SPY shares.")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Drift after news-sized moves in liquid stocks.")
    p.add_argument("--out", default="output/shock_drift")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--final-test", action="store_true")
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
        p = panels_from(adj, raw, md.stock_types(args.cache_dir))
    period = "holdout" if args.final_test else "design"
    if period == "design":   # the design run never sees a 2025+ price
        p = {k: v[v.index <= DESIGN_END] for k, v in p.items()}
    lo, hi = (HOLDOUT_START, FINAL_END) if period == "holdout" else (DATA_START, DESIGN_END)
    universe = monthly_universe(p, pd.period_range(lo, hi, freq="M"))
    res = run_study(p, universe, period=period, timings=timings)
    days = p["close"].index
    index = days[(days >= res["trades"]["session"].min()) & (days <= (DESIGN_END if period == "design" else FINAL_END))]
    with timed(timings, "portfolio view"):
        heavy = res["trades"][res["trades"]["volume"] == "heavy"]
        daily = portfolio_daily(heavy, p, index)
        spy = p["close"]["SPY"].pct_change().reindex(index).fillna(0)
        dip = pc.spy_dip_daily(index, args.cache_dir)
        port = {"account": pd.DataFrame([{"what": "heavy-volume drift, hedged", **pc.account(daily)},
                                         {"what": "holding SPY", **pc.account(spy)},
                                         {"what": "SPY dip buying (shares)", **pc.account(dip)}]),
                "corr": pc.correlations(daily, {"SPY dip buying": dip, "TSLA breakouts": pc.tsla_breakout_daily(index),
                                                "SPY": spy})}
    out = Path(args.out) / period
    out.mkdir(parents=True, exist_ok=True)
    res["trades"].to_parquet(out / "trades.parquet", index=False)
    for k in ("tests", "by_year", "by_size", "raw"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    if res["final"] is not None:
        res["final"].to_csv(out / "final_tests.csv", index=False)
        print(res["final"].to_string(index=False))
    port["account"].to_csv(out / "account.csv", index=False)
    port["corr"].to_csv(out / "correlations.csv", index=False)
    (out / "settings.json").write_text(json.dumps({"run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": period,
                                                   "verdict": res["verdict"], "counts": res["counts"],
                                                   "timings": {k: round(x, 1) for k, x in timings.items()}},
                                                  indent=2, default=str))
    (out / "report.md").write_text(render_report(res, port))
    print(res["tests"].to_string(index=False))
    print(res["verdict"])
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
