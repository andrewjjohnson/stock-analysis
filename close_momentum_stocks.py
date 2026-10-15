"""Into-the-close momentum on big days in the most volatile liquid stocks: when one of them has moved far more than
usual by 15:30, does the last half hour keep going? A trade simulation on Massive minute bars: P&L per trade in bps
of the entry price, per share, after costs.

  uv run --env-file .env python close_momentum_stocks.py --out output/close_momentum_stocks                # design
  uv run --env-file .env python close_momentum_stocks.py --final-test --out output/close_momentum_stocks   # once

Mechanism. The same forced trading as close_momentum.py, but where it is far larger next to the stock's own
trading. Single-stock leveraged ETFs (TSLL, NVDL, MSTU, CONL and others, mostly launched 2022-24) must buy after up
days and sell after down days at the close, in proportion to the move. Retail investors buy calls in these names,
so dealers are often short gamma and hedge in the direction of the move too. Neither trades on price, so the close
is pushed further in the day's direction. The other side is closing-auction liquidity providers. Both directions;
flat overnight.

Where the idea came from: TSLA was a context ticker in close_momentum.py's design run (2021-24), so its big-day
closes have already been looked at for this rule. TSLA is therefore left out here; the screen uses the other
volatile stocks, whose minute bars were not used for any close-momentum result before.

Declared on 2026-10-04, before any result for these stocks was computed:
- Universe: volatile_breakouts.py's monthly rule (the 10 stocks with the highest mean daily true range among those
  trading $500M+ a day at $10+; ranks 11-20 as the related set; TSLA left out).
- Day's move = the 15:30 price (the close of the minute bar ending then) over the previous session's last-minute
  close; size in units of the standard deviation of the stock's previous 20 daily moves (sd20).
- Trade on a big day (|move| >= 1 sd20): at 15:30 buy after an up move, sell short after a down move, at the 15:30
  price; exit at the last regular minute's close. Costs per side: 1 bp plus half a cent a share at the unadjusted
  price.
- Tests (8, one-sided, standard errors clustered by session, BH q < 0.10 across all 8), design 2021-12 to 2024-12:
    1. core, big days: mean net P&L > 0 (primary)
    2. core, big down days (shorts)
    3. core, big up days
    4. core, |move| >= 2 sd20
    5. core, entering at 15:45 instead (move measured to 15:45)
    6. core, big days minus small days (|move| < 0.5 sd20) > 0
    7. related stocks (ranks 11-20), big days
    8. core, |move| >= 3% (a fixed size instead of volatility units)
- Passes the screen if test 1 has q < 0.10 with a mean net P&L at least its average round-trip cost (so the edge
  before costs is at least twice the costs), at least 2 of 2022-2024 are positive, test 6 is positive and test 7 is
  positive.
- Reported, not tested: each year, each stock, and the account view (one trade a day: the core stock with the
  largest move in sd20 units at 15:30) with the daily-P&L correlation to SPY dip buying and the TSLA breakouts.
- Held-back check (--final-test): the same rule's 2025-01 to 2026-09 universe. Rules in FINAL.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import alerts
import close_momentum as cm
import features
import market_daily as md
import meanrev as mr
import portfolio_check as pc
import scalp_meanrev as sm
import volatile_breakouts as vb
from run import timed

NY = features.NY
DESIGN_END, HOLDOUT_START, FINAL_END = vb.DESIGN_END, vb.HOLDOUT_START, vb.FINAL_END
BIG, HUGE, SMALL = 1.0, 2.0, 0.5             # sd20 units
FIXED = 300.0                                # bps
YEARS = (2022, 2023, 2024)
FDR = 0.10
TESTS = ["core, big days", "core, big down days", "core, big up days", "core, 2 sd+", "core, 15:45 entry",
         "core, big minus small", "related, big days", "core, 3%+"]
NO_DIVIDENDS = pd.DataFrame(columns=["ex_date", "cash_amount"])
FINAL = None                    # set only if the screen passes, before --final-test


def span_days(minutes, span_sessions, groups, period):
    """close_momentum day tables for one ticker over one span, kept on its universe months in the period."""
    rth, _ = features.regular_session_minutes(minutes, span_sessions)
    if rth.empty:
        return None
    d = cm.day_table(rth, span_sessions, NO_DIVIDENDS)
    month = d.index.to_period("M")
    keep = month.isin(list(groups)) & ((d.index <= DESIGN_END) if period == "design" else (d.index >= HOLDOUT_START))
    d = d[keep].copy()
    d["group"] = d.index.to_period("M").map(groups)
    return d


def pick(days, group, move="move_m30", outcome="last30", lo=BIG, hi=np.inf, units=True):
    """Trades (gross bps, signed with the move) on days of a group whose |move| (in sd20 units, or bps) is in
    [lo, hi)."""
    d = days[(days["group"] == group) & days[outcome].notna() & days[move].notna()]
    size = d[move].abs() / d["sd20"] if units else d[move].abs()
    d = d[(size >= lo) & (size < hi)]
    s = np.sign(d[move])
    entry = d["p_" + move.split("_")[1]].to_numpy(float) * d["ratio"].to_numpy(float)   # the unadjusted entry price
    return pd.DataFrame({"session": d.index, "ticker": d["ticker"].to_numpy(), "move": d[move].to_numpy(),
                         "z": (d[move] / d["sd20"]).to_numpy(), "gross": (s * d[outcome]).to_numpy(),
                         "cost_side": vb.side_cost_bps(entry)})


def stats(t):
    net = t["gross"].to_numpy(float) - 2 * t["cost_side"].to_numpy(float)
    n = len(net)
    out = {"trades": n}
    if n < 3:
        return out
    m, se = sm.cluster_mean(net, t["session"].to_numpy())
    return {**out, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(m / se) if se and np.isfinite(se) else np.nan, "win": (net > 0).mean() * 100,
            "gross": t["gross"].mean(), "cost_rt": 2 * t["cost_side"].mean()}


def difference(a, b):
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return {"trades": na + nb}
    xa = a["gross"] - 2 * a["cost_side"]
    xb = b["gross"] - 2 * b["cost_side"]
    ma, mb = xa.mean(), xb.mean()
    g = np.r_[a["session"].astype(str), b["session"].astype(str)]
    sums = pd.Series(np.r_[(xa - ma).to_numpy() / na, -(xb - mb).to_numpy() / nb]).groupby(g).sum().to_numpy()
    k = len(sums)
    se = np.sqrt((sums**2).sum() * k / (k - 1))
    d = ma - mb
    return {"trades": na + nb, "mean": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "p_up": norm.sf(d / se)}


def run_study(days, *, period="design"):
    big = pick(days, "core")
    sets = {"core, big days": big, "core, big down days": big[big["move"] < 0], "core, big up days": big[big["move"] > 0],
            "core, 2 sd+": pick(days, "core", lo=HUGE),
            "core, 15:45 entry": pick(days, "core", move="move_m15", outcome="last15"),
            "related, big days": pick(days, "related"),
            "core, 3%+": pick(days, "core", lo=FIXED, units=False)}
    small = pick(days, "core", lo=0.0, hi=SMALL)
    rows = []
    for name in TESTS:
        r = difference(big, small) if name == "core, big minus small" else stats(sets[name])
        rows.append({"test": name, **r})
    tests = pd.DataFrame(rows)
    tests["q"] = mr.bh_qvalues(tests["p_up"].fillna(1).to_numpy())
    tests["qualifies"] = (tests["q"] < FDR) & (tests["mean"] > 0)
    by_year = pd.DataFrame([{"year": y, **stats(g)} for y, g in big.groupby(big["session"].dt.year)])
    by_stock = pd.DataFrame([{"ticker": tk, **stats(g)} for tk, g in big.groupby("ticker")])
    t = tests.set_index("test")
    yrs = by_year[by_year["year"].isin(YEARS)]
    v = {"significant_and_2x_costs": bool(t.loc["core, big days", "q"] < FDR
                                          and t.loc["core, big days", "mean"] >= t.loc["core, big days", "cost_rt"]),
         "years_positive": int((yrs["mean"] > 0).sum()),
         "big_beats_small": bool(t.loc["core, big minus small", "mean"] > 0),
         "related_positive": bool(t.loc["related, big days", "mean"] > 0)}
    v["passes"] = v["significant_and_2x_costs"] and v["years_positive"] >= 2 and v["big_beats_small"] \
        and v["related_positive"]
    return {"tests": tests, "by_year": by_year, "by_stock": by_stock, "big": big, "small": small, "verdict": v,
            "period": period}


def daily_account(big, index):
    """One trade a day: the core stock with the largest |move| in sd20 units; whole account, net."""
    t = big.assign(absz=big["z"].abs()).sort_values(["session", "absz"], ascending=[True, False])
    t = t.drop_duplicates("session")
    r = ((t["gross"] - 2 * t["cost_side"]) / 1e4).groupby(t["session"]).sum()
    return r.reindex(index).fillna(0)


def render_report(res, port):
    t, v = res["tests"], res["verdict"]
    lines = []
    w = lines.append
    w("# Into-the-close momentum in volatile stocks: " + ("design screen, 2021-24" if res["period"] == "design"
                                                         else "held-back check, 2025-26") + "\n")
    w("When one of the month's 10 most volatile liquid stocks (TSLA left out) has moved at least one usual daily move "
      "(sd of its last 20) by 15:30, trade that direction until the last minute's close. P&L in bps per trade after "
      "1 bp plus half a cent a share each side. Rules fixed in advance (see `close_momentum_stocks.py`). Per share, "
      "not account P&L except in the account view.\n")
    w("## Verdict\n")
    w(alerts.md_table(["Significant, net >= costs", "Years positive (of 3)", "Big beats small days",
                       "Related stocks positive", "Passes"],
                      [["yes" if v["significant_and_2x_costs"] else "no", str(v["years_positive"]),
                        "yes" if v["big_beats_small"] else "no", "yes" if v["related_positive"] else "no",
                        "**yes**" if v["passes"] else "no"]]) + "\n")
    w("## The declared tests\n")
    w(alerts.md_table(["#", "Test", "Trades", "Wins", "Net bps (95%)", "Before costs", "Cost a round trip", "p", "q",
                       "Qualifies"],
                      [[str(i + 1), r["test"], f"{int(r['trades'])}",
                        "" if pd.isna(r.get("win")) else f"{r['win']:.0f}%", cm.ci(r), cm.bps(r.get("gross")),
                        cm.bps(r.get("cost_rt")), f"{r['p_up']:.3f}", f"{r['q']:.3f}",
                        "yes" if r["qualifies"] else "no"] for i, r in t.iterrows()]) + "\n")
    w("## By year (core, big days)\n")
    w(alerts.md_table(["Year", "Trades", "Net bps (95%)"],
                      [[str(int(r["year"])), f"{int(r['trades'])}", cm.ci(r)] for _, r in res["by_year"].iterrows()])
      + "\n")
    bs = res["by_stock"].sort_values("trades", ascending=False)
    w(f"## By stock (core, big days): {int((bs['mean'] > 0).sum())} of {len(bs)} stocks positive\n")
    w(alerts.md_table(["Stock", "Trades", "Net bps"],
                      [[r["ticker"], f"{int(r['trades'])}", cm.bps(r.get("mean"))] for _, r in bs.head(15).iterrows()])
      + "\n")
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
    w("- Prices at a time are minute-bar closes; a stock without a trade in the minute ending at 15:30 or at the "
      "close is skipped that day. Shorts would be puts on Robinhood. No dividends (these stocks rarely pay).")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Into-the-close momentum on big days in volatile stocks.")
    p.add_argument("--out", default="output/close_momentum_stocks")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.final_test and FINAL is None:
        raise SystemExit("FINAL is not set: write the held-back rules down before running --final-test.")
    timings = {}
    period = "holdout" if args.final_test else "design"
    sessions = features.trading_sessions(vb.DATA_START, FINAL_END, warmup_sessions=0)
    with timed(timings, "grouped daily"):
        adj = md.load_grouped(sessions.index, True, args.cache_dir)
        raw = md.load_grouped(sessions.index, False, args.cache_dir)
        panels = vb.daily_panels(adj, raw, md.stock_types(args.cache_dir))
        ratio = (panels["raw_close"] / panels["close"]).shift(1)
        spy_close = adj[adj["ticker"] == "SPY"].set_index("session")["close"]
    lo, hi = (HOLDOUT_START, FINAL_END) if args.final_test else (vb.DATA_START, DESIGN_END)
    universe = vb.pick_universe(panels, pd.period_range(lo, hi, freq="M"))
    spans = vb.download_spans(universe, sessions)
    if not args.final_test:
        sessions = sessions[sessions.index <= DESIGN_END]
        spans["last"] = spans["last"].clip(upper=DESIGN_END)
    frames = []
    with timed(timings, "minute bars and days"):
        for sp in spans.itertuples():
            minutes, note = vb.load_span(sp.ticker, sp.first, sp.last, sessions, args.cache_dir, args.refresh)
            if minutes is None:
                continue
            g = universe[(universe["ticker"] == sp.ticker) & universe["month"].isin(sp.months)]
            d = span_days(minutes, sessions.loc[sp.first:sp.last], dict(zip(g["month"], g["group"])), period)
            if d is not None and len(d):
                frames.append(d.assign(ticker=sp.ticker))
    days = pd.concat(frames)
    r = np.array([ratio.at[s, tk] if tk in ratio.columns and s in ratio.index else np.nan
                  for s, tk in zip(days.index, days["ticker"])], float)
    days["ratio"] = np.where(np.isfinite(r), r, 1.0)
    res = run_study(days, period=period)
    idx = sessions.index
    index = idx[(idx >= res["big"]["session"].min()) & (idx <= (DESIGN_END if period == "design" else FINAL_END))]
    with timed(timings, "account view"):
        acct = daily_account(res["big"], index)
        spy = spy_close.pct_change().reindex(index).fillna(0)
        dip = pc.spy_dip_daily(index, args.cache_dir)
        port = {"account": pd.DataFrame([{"what": "volatile-stock close momentum, one trade a day", **pc.account(acct)},
                                         {"what": "holding SPY", **pc.account(spy)},
                                         {"what": "SPY dip buying (shares)", **pc.account(dip)}]),
                "corr": pc.correlations(acct, {"SPY dip buying": dip, "TSLA breakouts": pc.tsla_breakout_daily(index),
                                               "SPY": spy})}
    out = Path(args.out) / period
    out.mkdir(parents=True, exist_ok=True)
    res["big"].to_csv(out / "trades.csv", index=False)
    for k in ("tests", "by_year", "by_stock"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    port["account"].to_csv(out / "account.csv", index=False)
    port["corr"].to_csv(out / "correlations.csv", index=False)
    (out / "settings.json").write_text(json.dumps({"run_at": pd.Timestamp.now(tz=NY).isoformat(), "period": period,
                                                   "verdict": res["verdict"],
                                                   "timings": {k: round(x, 1) for k, x in timings.items()}},
                                                  indent=2, default=str))
    (out / "report.md").write_text(render_report(res, port))
    print(res["tests"].to_string(index=False))
    print(res["verdict"])
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
