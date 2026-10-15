"""Into-the-close momentum on big days: when SPY, QQQ or IWM has moved a lot by 15:30, does the last half hour keep
going the same way? A trade simulation on Massive minute bars: P&L per trade in bps of the entry price, per share,
after costs.

  uv run --env-file .env python close_momentum.py --out output/close_momentum                # design screen
  uv run --env-file .env python close_momentum.py --final-test --out output/close_momentum   # 2025-26, once

Mechanism. Two kinds of forced trading land in the last minutes of a big day, both in the direction of the day's
move. Leveraged and inverse ETFs (TQQQ, SQQQ, UPRO, TNA and the like) must buy after up days and sell after down
days to reset their leverage, in proportion to the day's return, mostly at the close. Options dealers who are short
gamma hedge the same way. Neither cares about price, so the close is pushed further in the day's direction
(Baltussen, Da, Lammers and Martens, "Hedging demand and market intraday momentum", JFE 2021). The other side is
liquidity providers and closing-auction traders. It should last as long as leveraged ETFs reset daily. It trades
both ways (shorts on big down days) and is flat overnight.

What earlier studies here already read on 2021-24: intraday_momentum.py tested the last half hour after the morning
move (previous close to 10:00), on all days and on volatile mornings, and after the 15:00-15:30 move. The signal
here is the whole day's move to 15:30, on big days only, which those tests did not look at.

Declared on 2026-10-04, before any result was computed:
- Prices: the close of the minute bar ending at a time (intraday_momentum.price_at). The day's move = the 15:30
  price over the previous session's last-minute close net of any dividend that morning.
- Trade on a big day (|day's move| >= 1%): at 15:30 buy (after an up move) or sell short (after a down move) at the
  15:30 price, exit at the last regular minute's close. Costs 1 bp a side. On early-close days the same, 30 minutes
  before the close.
- Baselines: small days (|move| < 0.5%) traded the same way; the same rule's direction randomly flipped (zero);
  the earlier half hour (15:00 to 15:30) on the same big days, traded in the direction of the move to 15:00.
- Tests (10, one-sided, BH q < 0.10 across all 10; the pooled tests cluster standard errors by session), design
  2021-10 to 2024-12:
    1-3. QQQ, SPY, IWM: big days, mean net P&L > 0
    4. the three pooled (primary)
    5. pooled, |move| >= 2%
    6. pooled, big down days only (short trades)
    7. pooled, big up days only
    8. pooled, entering at 15:45 instead
    9. pooled: big days minus small days > 0
    10. pooled, big measured in volatility units: |move| >= 1.5x the standard deviation of the previous 20 daily
        returns
- Passes the screen if test 4 has q < 0.10 with a mean net P&L at least the 2 bp round-trip cost (so the edge
  before costs is at least twice the costs), tests 1-3 are all positive, at least 2 of 2022-2024 are positive and
  test 9 is positive.
- Reported, not tested: each year, DIA (2021-24 only) and TSLA/NVDA (single stocks with large leveraged ETFs) as
  related tickers, the official closing price as the exit instead of the last minute's close, and the account
  view (one trade a day) with the daily-P&L correlation to SPY dip buying and the TSLA breakouts.
- Held-back check (--final-test): QQQ, IWM and SPY intraday 2025-01 to 2026-09. Rules in FINAL.
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
import download  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import intraday_momentum as im  # noqa: E402
import market_daily as md  # noqa: E402
import meanrev as mr  # noqa: E402
import portfolio_check as pc  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("QQQ", "SPY", "IWM")
RELATED = ("DIA", "TSLA", "NVDA")
DESIGN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
BIG, HUGE, SMALL = 100.0, 200.0, 50.0       # bps
VOL_UNITS, VOL_WINDOW = 1.5, 20
COST_BPS = 1.0
YEARS = (2022, 2023, 2024)
FDR = 0.10
TESTS = ["QQQ", "SPY", "IWM", "pooled", "pooled, 2%+", "pooled, down days", "pooled, up days", "pooled, 15:45 entry",
         "pooled, big minus small", "pooled, volatility units"]
FINAL = None                    # set only if the screen passes, before --final-test


# ---------------------------------------------------------------- days

def day_table(rth, sessions, dividends, official_close=None):
    """One row per session: prices at 15:00, 15:30 and 15:45 (30 and 15 minutes before the close on early-close
    days), the last minute's close, the previous close net of a dividend that morning, the day's move to each time
    and the size of the previous 20 daily moves. Everything a trade decides on is known at its entry time."""
    d = pd.DataFrame(index=sessions.index)
    d["p_m60"] = im.price_at(rth, sessions["close"] - pd.Timedelta(minutes=60))
    d["p_m30"] = im.price_at(rth, sessions["close"] - pd.Timedelta(minutes=30))
    d["p_m15"] = im.price_at(rth, sessions["close"] - pd.Timedelta(minutes=15))
    d["p_close"] = im.price_at(rth, sessions["close"])
    paid = dividends.groupby("ex_date")["cash_amount"].sum() if len(dividends) else pd.Series(dtype=float)
    d["prev"] = d["p_close"].shift(1) - paid.reindex(d.index).fillna(0.0).to_numpy()
    for k in ("m60", "m30", "m15"):
        d[f"move_{k}"] = (d[f"p_{k}"] / d["prev"] - 1) * 1e4
    daily = (d["p_close"] / d["prev"] - 1) * 1e4
    d["sd20"] = daily.shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    d["last30"] = (d["p_close"] / d["p_m30"] - 1) * 1e4
    d["last15"] = (d["p_close"] / d["p_m15"] - 1) * 1e4
    d["prior30"] = (d["p_m30"] / d["p_m60"] - 1) * 1e4
    if official_close is not None:
        d["last30_official"] = (official_close.reindex(d.index) / d["p_m30"] - 1) * 1e4
    return d


def trades(d, ticker, move="move_m30", outcome="last30", lo=BIG, hi=np.inf, units=None):
    """Signed outcome (bps, before costs) on days whose |move| is in [lo, hi) (or >= `units` x sd20)."""
    m = d[move]
    size = m.abs() / d["sd20"] if units else m.abs()
    keep = (size >= (units or lo)) & (size < (np.inf if units else hi)) & d[outcome].notna()
    s = np.sign(m[keep])
    return pd.DataFrame({"session": d.index[keep], "ticker": ticker, "move": m[keep].to_numpy(),
                         "gross": (s * d.loc[keep, outcome]).to_numpy()})


def stats(t, cost=COST_BPS):
    net = t["gross"].to_numpy(float) - 2 * cost
    n = len(net)
    out = {"trades": n}
    if n < 3:
        return out
    m, se = sm.cluster_mean(net, t["session"].to_numpy())
    return {**out, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(m / se) if se and np.isfinite(se) else np.nan, "win": (net > 0).mean() * 100,
            "gross": t["gross"].mean()}


def difference(a, b, cost=COST_BPS):
    """Mean of a minus mean of b (both net), clustered by session, as one stats row."""
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return {"trades": na + nb}
    ma, mb = a["gross"].mean(), b["gross"].mean()
    ca = (a["gross"] - ma).to_numpy() / na
    cb = -(b["gross"] - mb).to_numpy() / nb
    g = np.r_[a["session"].astype(str), b["session"].astype(str)]
    sums = pd.Series(np.r_[ca, cb]).groupby(g).sum().to_numpy()
    k = len(sums)
    se = np.sqrt((sums**2).sum() * k / (k - 1))
    d = ma - mb
    return {"trades": na + nb, "mean": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "p_up": norm.sf(d / se)}


def run_study(tables, *, period="design", timings=None):
    """tables: {ticker: day_table}. No file I/O."""
    timings = {} if timings is None else timings

    def cut(d):
        keep = (d.index <= DESIGN_END) if period == "design" else (d.index >= HOLDOUT_START)
        return d[keep & (d.index > d.index[0])]

    t = {tk: cut(d) for tk, d in tables.items()}
    main = [k for k in TICKERS if k in t]
    pool = lambda **kw: pd.concat([trades(t[k], k, **kw) for k in main], ignore_index=True)  # noqa: E731
    with timed(timings, "trades"):
        big = pool()
        sets = {"QQQ": big[big["ticker"] == "QQQ"], "SPY": big[big["ticker"] == "SPY"],
                "IWM": big[big["ticker"] == "IWM"], "pooled": big, "pooled, 2%+": pool(lo=HUGE),
                "pooled, down days": big[big["move"] < 0], "pooled, up days": big[big["move"] > 0],
                "pooled, 15:45 entry": pool(move="move_m15", outcome="last15"),
                "pooled, volatility units": pool(units=VOL_UNITS)}
        small = pool(lo=0.0, hi=SMALL)
    rows = []
    for name in TESTS:
        r = difference(big, small) if name == "pooled, big minus small" else stats(sets[name])
        rows.append({"test": name, **r})
    tests = pd.DataFrame(rows)
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["qualifies"] = (tests["q"] < FDR) & (tests["mean"] > 0)
    by_year = pd.DataFrame([{"ticker": tk, "year": y, **stats(g)}
                            for (tk, y), g in big.groupby(["ticker", big["session"].dt.year])] +
                           [{"ticker": "pooled", "year": y, **stats(g)} for y, g in big.groupby(big["session"].dt.year)])
    context = []
    for tk in [k for k in RELATED if k in t]:
        context.append({"what": f"{tk}, big days", **stats(trades(t[tk], tk))})
    context.append({"what": "pooled, small days (|move| < 0.5%)", **stats(small)})
    context.append({"what": "pooled, 15:00-15:30 on big days (direction of the move to 15:00)",
                    **stats(pool(move="move_m60", outcome="prior30"))})
    if all("last30_official" in t[k] for k in main):
        context.append({"what": "pooled, exit at the official close",
                        **stats(pool(outcome="last30_official"))})
    return {"tests": tests, "by_year": by_year, "context": pd.DataFrame(context), "big": big, "period": period,
            "verdict": verdict(tests, by_year)}


def verdict(tests, by_year):
    t = tests.set_index("test")
    yrs = by_year[(by_year["ticker"] == "pooled") & by_year["year"].isin(YEARS)]
    c = {"significant_and_2x_costs": bool(t.loc["pooled", "q"] < FDR and t.loc["pooled", "mean"] >= 2 * COST_BPS),
         "each_etf_positive": bool((t.loc[["QQQ", "SPY", "IWM"], "mean"] > 0).all()),
         "years_positive": int((yrs["mean"] > 0).sum()),
         "big_beats_small": bool(t.loc["pooled, big minus small", "mean"] > 0)}
    c["passes"] = c["significant_and_2x_costs"] and c["each_etf_positive"] and c["years_positive"] >= 2 \
        and c["big_beats_small"]
    return c


def daily_account(big, index, cost=COST_BPS):
    """One trade a day: on days with several big-move ETFs, the first in TICKERS order; whole account, net."""
    t = big.copy()
    t["order"] = t["ticker"].map({k: i for i, k in enumerate(TICKERS)})
    t = t.sort_values(["session", "order"]).drop_duplicates("session")
    r = ((t["gross"] - 2 * cost) / 1e4).groupby(t["session"]).sum()
    return r.reindex(index).fillna(0)


# ---------------------------------------------------------------- report

def bps(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1f}"


def ci(r):
    if r is None or pd.isna(r.get("mean")):
        return "n/a"
    return f"{bps(r['mean'])} ({bps(r['lo'])} to {bps(r['hi'])})"


def render_report(res, port):
    t, v = res["tests"], res["verdict"]
    lines = []
    w = lines.append
    w("# Into-the-close momentum on big days: " + ("design screen, 2021-24" if res["period"] == "design"
                                                   else "held-back check, 2025-26") + "\n")
    w("When an ETF is up or down at least 1% at 15:30, trade in that direction until the last minute's close. P&L "
      "in bps per trade after 1 bp a side. Rules fixed in advance (see `close_momentum.py`). A trade simulation on "
      "minute bars, per share; not account P&L except in the account view.\n")
    w("## Verdict\n")
    w(alerts.md_table(["Pooled significant, net >= 2 bps", "QQQ, SPY, IWM all positive", "Years positive (of 3)",
                       "Big beats small days", "Passes"],
                      [["yes" if v["significant_and_2x_costs"] else "no", "yes" if v["each_etf_positive"] else "no",
                        str(v["years_positive"]), "yes" if v["big_beats_small"] else "no",
                        "**yes**" if v["passes"] else "no"]]) + "\n")
    w("## The declared tests\n")
    w(alerts.md_table(["#", "Test", "Trades", "Wins", "Net bps (95%)", "p", "q", "Qualifies"],
                      [[str(i + 1), r["test"], f"{int(r['trades'])}",
                        "" if pd.isna(r.get("win")) else f"{r['win']:.0f}%", ci(r), f"{r['p_up']:.3f}",
                        f"{r['q']:.3f}", "yes" if r["qualifies"] else "no"] for i, r in t.iterrows()]) + "\n")
    w("## By year (big days)\n")
    w(alerts.md_table(["Ticker", "Year", "Trades", "Net bps (95%)"],
                      [[r["ticker"], str(int(r["year"])), f"{int(r['trades'])}", ci(r)]
                       for _, r in res["by_year"].iterrows()]) + "\n")
    w("## Baselines and related tickers\n")
    w(alerts.md_table(["", "Trades", "Net bps (95%)"],
                      [[r["what"], f"{int(r['trades'])}", ci(r)] for _, r in res["context"].iterrows()]) + "\n")
    if port is not None:
        w("## Account view (one trade a day, whole account)\n")
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
    w("- Shorts need an inverse ETF (SH, PSQ, RWM) or a put on Robinhood. The exit is the last minute's close; a "
      "market order in the final seconds gets close to it, and the official closing price is shown as a check.")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Into-the-close momentum on big days for SPY, QQQ and IWM.")
    p.add_argument("--out", default="output/close_momentum")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true")
    return p.parse_args(argv)


def load_table(tk, sessions, cache_dir, refresh, official, upto=None):
    """Day table for one ticker; minutes come from the whole-range cache, but only sessions up to `upto` are
    built (the design run never builds a 2025+ row)."""
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    full = download.cache_path(cache_dir, tk, first, last)
    if tk == "DIA" and not full.exists():   # DIA is cached for 2021-24 only
        path = download.cache_path(cache_dir, tk, "2021-10-11", "2024-12-31")
        if not path.exists():
            return None
        minutes = pd.read_parquet(path)
    else:
        minutes, _ = gr.load_minutes(tk, sessions, cache_dir, refresh)
    if upto is not None:
        sessions = sessions[sessions.index <= upto]
    rth, _ = features.regular_session_minutes(minutes, sessions)
    try:
        divs = mr.load_dividends(tk, first, last, cache_dir, refresh)
    except SystemExit:
        divs = pd.DataFrame(columns=["ex_date", "cash_amount"])
    return day_table(rth, sessions, divs, official.get(tk))


def main(argv=None):
    args = parse_args(argv)
    if args.final_test and FINAL is None:
        raise SystemExit("FINAL is not set: write the held-back rules down before running --final-test.")
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    period = "holdout" if args.final_test else "design"
    with timed(timings, "data"):
        adj = md.load_grouped(sessions.index[sessions.index >= pd.Timestamp("2021-10-05")], True, args.cache_dir)
        official = {tk: g.set_index("session")["close"] for tk, g in adj[adj["ticker"].isin(TICKERS + RELATED)]
                    .groupby("ticker")}
        tables = {}
        # the held-back check reads only the three ETFs: TSLA's and NVDA's 2025-26 are kept out of it
        for tk in TICKERS + (() if args.final_test else RELATED):
            d = load_table(tk, sessions, args.cache_dir, args.refresh, official,
                           upto=None if args.final_test else DESIGN_END)
            if d is not None:
                tables[tk] = d
    res = run_study(tables, period=period, timings=timings)
    days = sessions.index
    index = days[(days <= DESIGN_END) if period == "design" else (days >= HOLDOUT_START)]
    index = index[index >= res["big"]["session"].min()]
    with timed(timings, "account view"):
        acct_daily = daily_account(res["big"], index)
        spy = official["SPY"].pct_change().reindex(index).fillna(0)
        dip = pc.spy_dip_daily(index, args.cache_dir)
        port = {"account": pd.DataFrame([{"what": "close momentum, one trade a day", **pc.account(acct_daily)},
                                         {"what": "holding SPY", **pc.account(spy)},
                                         {"what": "SPY dip buying (shares)", **pc.account(dip)}]),
                "corr": pc.correlations(acct_daily, {"SPY dip buying": dip,
                                                     "TSLA breakouts": pc.tsla_breakout_daily(index), "SPY": spy})}
    out = Path(args.out) / period
    out.mkdir(parents=True, exist_ok=True)
    res["big"].to_csv(out / "trades.csv", index=False)
    for k in ("tests", "by_year", "context"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    port["account"].to_csv(out / "account.csv", index=False)
    port["corr"].to_csv(out / "correlations.csv", index=False)
    (out / "settings.json").write_text(json.dumps({"run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": period,
                                                   "verdict": res["verdict"]}, indent=2, default=str))
    (out / "report.md").write_text(render_report(res, port))
    print(res["tests"].to_string(index=False))
    print(res["verdict"])
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
