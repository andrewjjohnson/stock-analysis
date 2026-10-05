"""Iron condors on SPY, 30-45 days out: a call credit spread above and a put credit spread below, same expiry, both
short strikes at the same delta. A trade simulation on Massive option minute bars; results per condor (100 shares),
before taxes, gross of commissions (Robinhood).

  uv run --env-file .env python iron_condors.py --out output/iron_condors

The direction-neutral version of call_spreads.py: it earns when SPY stays between the short strikes.

Fixed before any condor result was seen (the call side's own results were known from call_spreads.py):
- Entries, expiries, strikes and fills as in call_spreads.py: a decision at 15:50 every session; the weekly expiry
  closest to 30 or 45 days out; short call and short put at the $5 strike whose delta (from its own implied
  volatility) is closest to 0.20, 0.30 or 0.40; wings $5 or $10 wide; each spread entered as a working order (the
  first minute both its legs trade, through 10:30 the next morning). The condor's credit is the two credits; its
  value at a minute is the sum of each spread's latest traded value.
- Exits on the condor's total: held to expiry; 50% take profit; 21 days; 50% or 21 days; 50% with a stop at a loss
  of 2x the credit. Primary: 45 days, 0.30 delta, $5 wide, 50% or 21 days, every session.
- Entry filters (known at 15:50): every session; volatile and calm (20-day volatility in its 2021-2024 top or
  bottom third); SPY below its 50-day average.
- Costs: traded prices, +$0.03 and +$0.05 per share per leg per fill (four legs each way).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import alert_spreads as sp  # noqa: E402
import alerts  # noqa: E402
import call_spreads as cs  # noqa: E402
import dip_spreads as ds  # noqa: E402
import download  # noqa: E402
import features  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
PRIMARY = {"dte": 45, "delta": 0.30, "width": 5, "exit": "50% or 21 days", "filter": "every session"}
FILTERS = {"every session": lambda f: pd.Series(True, index=f.index),
           "volatile": lambda f: f["volatile"],
           "calm": lambda f: f["calm"],
           "below the 50-day average": lambda f: f["dist_sma50"] < 0}
REALISTIC = cs.REALISTIC


# ---------------------------------------------------------------- trades

def build_sides(plan, sessions, raw, chain, load, workers=8):
    """Both credit spreads for every planned condor: {(day, target, delta, width, right): priced spread}."""
    later = {d: (sessions.loc[n, "open"] + pd.Timedelta(minutes=60)).value
             for d, n in zip(sessions.index[:-1], sessions.index[1:])}
    planned = []
    for r in plan.itertuples():
        for d in cs.DELTAS:
            for right, prefix in (("C", ""), ("P", "put_")):
                ks = plan.at[r.Index, f"{prefix}short_{d:g}"]
                if np.isnan(ks):
                    continue
                for w in cs.WIDTHS:
                    k = chain[r.expiry]
                    other = k[k >= ks + w - 1e-9] if right == "C" else k[k <= ks - w + 1e-9]
                    if not len(other):
                        continue
                    kl = float(other[0] if right == "C" else other[-1])
                    planned.append({"day": r.day, "target": r.target, "delta": d, "nominal_width": w, "right": right,
                                    "expiry": r.expiry, "spot": r.spot, "short_strike": ks, "long_strike": kl,
                                    "width": abs(kl - ks), "entry_until": later.get(r.day),
                                    "delta_actual": plan.at[r.Index, f"{prefix}delta_{d:g}"],
                                    "short_ticker": cs.leg_ticker(r.expiry, right, ks),
                                    "long_ticker": cs.leg_ticker(r.expiry, right, kl)})
    cs.prefetch([(p[k], cs.leg_start(p["expiry"]), str(p["expiry"].date())) for p in planned
                 for k in ("short_ticker", "long_ticker")], load, workers)
    return {(p["day"], p["target"], p["delta"], p["nominal_width"], p["right"]):
            ds.price_spread(dict(p), cs.leg_start(p["expiry"]), sessions, raw, load) for p in planned}


def condor_path(call, put):
    """(minute times, condor values, sessions) from the later of the two entries: each spread's latest traded value
    summed at every minute either spread traded."""
    start = max(call["entry_time"], put["entry_time"])
    ts = np.union1d(call["watch_ts"], put["watch_ts"])
    ts = ts[ts >= start]

    def latest(tr):
        return tr["watch"][np.searchsorted(tr["watch_ts"], ts, side="right") - 1]

    day_of = dict(zip(np.r_[call["watch_ts"], put["watch_ts"]], np.r_[call["watch_days"], put["watch_days"]]))
    return ts, latest(call) + latest(put), np.array([day_of[t] for t in ts])


def make_condors(sides):
    """{(day, target, delta, width): condor dict, or a status string when a side could not be priced}."""
    out = {}
    keys = {k[:4] for k in sides}
    for key in keys:
        call, put = sides.get((*key, "C")), sides.get((*key, "P"))
        if call is None or put is None:
            out[key] = "a side had no short strike near the delta"
            continue
        if call["status"] != "ok" or put["status"] != "ok":
            out[key] = "a side could not be entered"
            continue
        ts, value, days = condor_path(call, put)
        if not len(ts):
            out[key] = "no minute after both entries"
            continue
        out[key] = {"day": key[0], "expiry": call["expiry"], "credit": call["credit_traded"] + put["credit_traded"],
                    "call_credit": call["credit_traded"], "put_credit": put["credit_traded"],
                    "width": max(call["width"], put["width"]), "ts": ts, "value": value, "days": days,
                    "settle": call["settle"] + put["settle"], "call_settle": call["settle"],
                    "put_settle": put["settle"], "expiry_close": call["expiry_close"],
                    "call_strike": call["short_strike"], "put_strike": put["short_strike"]}
    return out


def simulate(c, rule, slippage, sessions):
    """(P&L $ per condor, exit reason, exit session) for one exit rule on the condor's total value."""
    credit = c["credit"] - 4 * slippage
    value, width = c["value"], c["width"]
    first, out = len(value), None
    if "tp" in rule:
        level = credit * (1 - rule["tp"] / 100)
        hit = np.flatnonzero(value <= level - 4 * slippage)
        if hit.size:
            first, out = hit[0], ((credit - level) * 100, "take profit")
    if "stop" in rule:
        hit = np.flatnonzero(value >= credit * (1 + rule["stop"]))
        if hit.size and hit[0] < first:
            first = hit[0]
            out = ((credit - min(value[first], width) - 4 * slippage) * 100, "stop")
    if "days" in rule:
        day = next((d for d in sessions.index[(sessions.index > c["day"]) & (sessions.index <= c["expiry"])]
                    if (c["expiry"] - d).days <= rule["days"]), None)
        if day is not None:
            at = (sessions.loc[day, "close"] - pd.Timedelta(minutes=mr.DECISION_MINUTES)).value
            k = int(np.searchsorted(c["ts"], at))
            if k < first:
                first = k
                out = ((credit - min(value[k], width) - 4 * slippage) * 100, f"{rule['days']} days")
    if out is not None:
        return out[0], out[1], pd.Timestamp(c["days"][first])
    return (credit - c["settle"]) * 100, "expiry", c["expiry"]


def pnl_table(condors, sessions):
    rows = []
    for (day, target, d, w), c in condors.items():
        base = {"day": day, "dte": target, "delta": d, "width": w}
        if isinstance(c, str):
            rows.append({**base, "status": c})
            continue
        for name, rule in cs.EXITS.items():
            for cost, slip in cs.SLIPPAGES.items():
                pnl, reason, exit_day = simulate(c, rule, slip, sessions)
                credit = c["credit"] - 4 * slip
                rows.append({**base, "status": "ok", "exit": name, "cost": cost, "pnl": pnl, "reason": reason,
                             "exit_day": exit_day, "credit": credit, "max_loss": (c["width"] - credit) * 100,
                             "days_held": (exit_day - day).days, "settle": c["settle"],
                             "call_side": (c["call_credit"] - c["call_settle"]) * 100,
                             "put_side": (c["put_credit"] - c["put_settle"]) * 100,
                             "inside": float(c["put_strike"] <= c["expiry_close"] <= c["call_strike"])})
    return pd.DataFrame(rows)


def analyze(table, feats, reps, seed):
    ok = table[table["status"] == "ok"]
    masks = {name: rule(feats).fillna(False) for name, rule in FILTERS.items()}
    rows = []
    for (dte, d, w, ex, cost), g in ok.groupby(["dte", "delta", "width", "exit", "cost"], sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        idx = mr.block_indices(len(g), reps, seed, block=cs.BLOCK)
        for name, m in masks.items():
            mask = m.reindex(g["day"]).fillna(False).to_numpy(bool)
            row = {"dte": dte, "delta": d, "width": w, "exit": ex, "cost": cost, "filter": name,
                   **cs.stats_for(g, mask, idx)}
            if name == "every session":
                row.update({f"single_{k}": v for k, v in cs.one_at_a_time(g).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def run_study(raw, feats, sessions, chain, load, *, reps=2000, seed=0, workers=8, timings=None):
    """No file I/O beyond the option loader's cache. feats: meanrev features plus boolean `volatile` and `calm`."""
    timings = {} if timings is None else timings
    days = raw.index[raw.index >= pd.Timestamp(ds.OPTIONS_START)]
    opt_sessions = sessions.loc[days]
    with timed(timings, "strikes"):
        plan = cs.plan_entries(days, opt_sessions, raw.loc[days], chain, load, workers, rights=("C", "P"))
    with timed(timings, "option data"):
        sides = build_sides(plan, opt_sessions, raw.loc[days], chain, load, workers)
        condors = make_condors(sides)
    with timed(timings, "simulation"):
        table = pnl_table(condors, opt_sessions)
        stats = analyze(table, feats.loc[days], reps, seed)
    return {"plan": plan, "table": table, "stats": stats, "first": days[0], "last": days[-1],
            "spy": raw.loc[days, "close"]}


# ---------------------------------------------------------------- report

def money(v):
    return sp.money(v, sign=True, cents=False)


def pick(stats, cost=REALISTIC, **kw):
    s = {**PRIMARY, **kw}
    m = stats[(stats["dte"] == s["dte"]) & np.isclose(stats["delta"], s["delta"]) & (stats["width"] == s["width"])
              & (stats["exit"] == s["exit"]) & (stats["cost"] == cost) & (stats["filter"] == s["filter"])]
    return m.iloc[0] if len(m) else None


def label(**kw):
    s = {**PRIMARY, **kw}
    return f"{s['dte']} days, {s['delta']:.2f} delta each side, ${s['width']} wings, {s['exit']}"


def render_report(res):
    st = res["stats"]
    lines = []
    w = lines.append
    w("# Iron condors on SPY, 30-45 days out\n")
    w(f"Real SPY option prices (Massive minute bars), entries from {res['first']:%Y-%m-%d} to the last that expire by "
      f"{res['last']:%Y-%m-%d}. A condor (call credit spread above, put credit spread below, same expiry and delta) "
      "is sold at 15:50 every session, so trades overlap. Dollars per condor (100 shares). *Return on risk* = total "
      "P&L / total max loss (wing width minus credit). Ranges are 95% intervals from 20-session blocks of entry days. "
      f"Main tables use **{REALISTIC}** per share per leg per fill.\n")
    r = pick(st)
    spy = res["spy"]
    if r is not None and r["n"]:
        best = st[(st["cost"] == REALISTIC) & (st["filter"] == "every session") & (st["n"] >= 100)].sort_values(
            "mean", ascending=False).iloc[0]
        w(f"**Bottom line.** The primary condor ({label()}) averaged {money(r['mean'])} over {int(r['n'])} entries "
          f"({money(r['mean_lo'])} to {money(r['mean_hi'])}), {r['win']:.0f}% winners, return on risk {r['ror']:+.1f}%, "
          f"worst {money(r['worst'])}. SPY rose {(spy.iloc[-1] / spy.iloc[0] - 1) * 100:+.0f}% over the period. The best "
          f"version: {label(dte=best['dte'], delta=best['delta'], width=best['width'], exit=best['exit'])}, "
          f"{money(best['mean'])} a condor ({best['win']:.0f}% winners).\n")
    w("## 1. The primary condor\n")
    rows = []
    for cost in cs.SLIPPAGES:
        x = pick(st, cost=cost)
        if x is None or not x["n"]:
            continue
        rows.append([cost, f"{int(x['n'])}", money(x["credit"] * 100), f"{x['win']:.0f}%",
                     f"{money(x['mean'])} ({money(x['mean_lo'])} to {money(x['mean_hi'])})", f"{x['ror']:+.1f}%",
                     money(x["worst"]), f"{x['days']:.0f}",
                     f"{int(x['single_trades'])}, {money(x['single_total'])}, drawdown {money(-x['single_max_dd'])}"])
    w(alerts.md_table(["Costs", "Condors", "Avg credit", "Winners", "Average (95%)", "Return on risk", "Worst",
                       "Avg days held", "One at a time: trades, total, worst drawdown"], rows) + "\n")
    w(sides_section(res))
    w("## 2. Every expiry, delta and wing width (50% or 21 days)\n")
    rows = []
    for dte in cs.DTE_TARGETS:
        for d in cs.DELTAS:
            row = [f"{dte} days", f"{d:.2f}"]
            for wd in cs.WIDTHS:
                x = pick(st, dte=dte, delta=d, width=wd)
                row.append("" if x is None or not x["n"] else
                           f"{money(x['mean'])}, {x['win']:.0f}% win, {x['ror']:+.1f}% on risk (n={int(x['n'])})")
            rows.append(row)
    w(f"Average per condor with {REALISTIC}:\n")
    w(alerts.md_table(["Expiry", "Short delta (each side)", "$5 wings", "$10 wings"], rows) + "\n\n![Grid](grid.png)\n")
    w("## 3. Exits\n")
    rows = []
    for ex in cs.EXITS:
        x = pick(st, exit=ex)
        if x is None or not x["n"]:
            continue
        rows.append([ex, f"{x['win']:.0f}%", money(x["mean"]), f"{x['ror']:+.1f}%", money(x["worst"]), f"{x['days']:.0f}",
                     f"{int(x['single_trades'])}, {money(x['single_total'])}, drawdown {money(-x['single_max_dd'])}"])
    w(f"{label(exit='each exit')}, {REALISTIC}:\n")
    w(alerts.md_table(["Exit", "Winners", "Average", "Return on risk", "Worst", "Avg days held",
                       "One at a time: trades, total, worst drawdown"], rows))
    g = st[(st["cost"] == REALISTIC) & (st["filter"] == "every session")]
    best = g.pivot_table(index=["dte", "delta", "width"], columns="exit", values="mean").idxmax(axis=1).value_counts()
    w("\nBest exit by average across all 12 versions: " + ", ".join(f"{k} {v}" for k, v in best.items()) + ".\n")
    w("## 4. Entry filters\n")
    rows = []
    for name in FILTERS:
        x = pick(st, filter=name)
        if x is None or not x["n"]:
            continue
        rows.append([name, f"{int(x['n'])}", f"{x['win']:.0f}%",
                     f"{money(x['mean'])} ({money(x['mean_lo'])} to {money(x['mean_hi'])})",
                     "" if name == "every session" else
                     f"{money(x['excess'])} ({money(x['excess_lo'])} to {money(x['excess_hi'])})",
                     f"{x['ror']:+.1f}%", money(x["worst"])])
    w(f"{label()}, {REALISTIC}, by entry condition (known at 15:50):\n")
    w(alerts.md_table(["Entered when", "Condors", "Winners", "Average (95%)", "Excess vs every session",
                       "Return on risk", "Worst"], rows) + "\n")
    w("## 5. Data\n")
    t = res["table"]
    s = t.drop_duplicates(["day", "dte", "delta", "width"])["status"].value_counts()
    w("- Condors: " + ", ".join(f"{k}: {v:,}" for k, v in s.items()) + ".")
    w("- Deltas and fills as in call_spreads.py (implied volatility from trade prints via the parity forward; both legs "
      "of a spread in the same minute; working orders through 10:30 the next morning). The condor's value between "
      "trades uses each spread's latest traded value, so a take profit or stop can be seen a little late.")
    w("- One market regime: SPY mostly rose, with one sharp fall (spring 2025).")
    return "\n".join(lines) + "\n"


def sides_section(res):
    t = res["table"]
    t = t[(t["status"] == "ok") & (t["exit"] == "held to expiry") & (t["cost"] == "traded prices")]
    rows = []
    for (dte, d, wd), g in t.groupby(["dte", "delta", "width"]):
        rows.append([f"{dte} days", f"{d:.2f}", f"${wd}", money(g["call_side"].mean()), money(g["put_side"].mean()),
                     f"{g['inside'].mean() * 100:.0f}%", f"{len(g)}"])
    return ("Held to expiry at traded prices, what each side earned on average, and how often SPY finished between "
            "the short strikes (the condor's full profit zone):\n\n"
            + alerts.md_table(["Expiry", "Delta", "Wings", "Call side", "Put side", "Finished between the strikes",
                               "Condors"], rows) + "\n")


def plot_grid(res, path):
    st = res["stats"]
    fig = plt.figure(figsize=(8, 3.8), dpi=150, facecolor=SURFACE)
    for k, wd in enumerate(cs.WIDTHS):
        ax = sds._axes(fig, [0.1 + k * 0.47, 0.16, 0.36, 0.56])
        m = np.array([[pick(st, dte=dte, delta=d, width=wd)["mean"] for d in cs.DELTAS] for dte in cs.DTE_TARGETS],
                     float)
        lim = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap=sp.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, money(m[i, j]).replace("$", r"\$"), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(cs.DELTAS)), [f"{d:.2f}" for d in cs.DELTAS])
        ax.set_yticks(range(len(cs.DTE_TARGETS)), [f"{t} days" for t in cs.DTE_TARGETS])
        ax.set_xlabel("Short delta, each side", color=INK_2, fontsize=7.5)
        ax.set_title(f"\\${wd} wings", color=INK, fontsize=8.5, loc="left")
    fig.text(0.02, 0.97, "SPY iron condors: average P&L per condor", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.9, rf"Entered every session, 50% take profit or 21 days to expiry, +\$0.03 per leg.", color=INK_2,
             fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_equity(res, path):
    t = res["table"]
    fig = plt.figure(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.1, 0.18, 0.78, 0.6])
    for d, color in ((0.20, SERIES[1]), (0.30, SERIES[0])):
        g = t[(t["status"] == "ok") & (t["dte"] == PRIMARY["dte"]) & np.isclose(t["delta"], d)
              & (t["width"] == PRIMARY["width"]) & (t["exit"] == PRIMARY["exit"]) & (t["cost"] == REALISTIC)]
        taken, until = [], pd.Timestamp.min
        for r in g.sort_values("day").itertuples():
            if r.day > until:
                taken.append(r)
                until = r.exit_day
        s = pd.DataFrame(taken).sort_values("exit_day")
        ax.step(s["exit_day"], s["pnl"].cumsum(), where="post", color=color, lw=2,
                label=f"{d:.2f} delta, one condor at a time ({len(s)} trades)")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel(r"Cumulative P&L, \$ per condor", color=INK_2, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(res["spy"].index, res["spy"], color=MUTED, lw=1, label="SPY (right axis)")
    ax2.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.text(0.02, 0.97, "Iron condors one at a time", color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, rf"45 days, \$5 wings, 50% or 21 days, +\$0.03 per leg; by exit date.", color=INK_2,
             fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPY iron condors 30-45 days out by delta, with real option prices.")
    p.add_argument("--out", default="output/iron_condors")
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
        minutes, _ = download.load_minute_bars(cs.TICKER, first, last, cache_dir=args.cache_dir,
                                               final_close=sessions["close"].iloc[-1])
        raw, _ = mr.daily_table(minutes, sessions)
        adj, _ = mr.adjust_dividends(raw, mr.load_dividends(cs.TICKER, first, last, args.cache_dir, args.refresh))
        feats = mr.ticker_features(adj).reindex(sessions.index)
        rv = feats.loc[feats.index <= mr.DESIGN_END, "rv20"]
        feats["volatile"] = feats["rv20"] >= rv.quantile(2 / 3)
        feats["calm"] = feats["rv20"] <= rv.quantile(1 / 3)
        chain = cs.weekly_chain(ds.OPTIONS_START, last, args.cache_dir, args.refresh)
    load = ds.contract_loader(args.cache_dir, args.refresh)
    res = run_study(raw, feats, sessions, chain, load, reps=args.reps, seed=args.seed, workers=args.workers,
                    timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "condors.csv", index=False)
    res["stats"].to_csv(out / "stats.csv", index=False)
    plot_grid(res, out / "grid.png")
    plot_equity(res, out / "equity.png")
    (out / "report.md").write_text(render_report(res))
    r = pick(res["stats"])
    print(f"primary condor: {int(r['n'])} entries, {money(r['mean'])} each, {r['win']:.0f}% winners")
    print(f"option contracts downloaded this run: {load.state['fetched']:,}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
