"""Three red bars in a row on different timeframes: does price bounce after 3+ red 5-minute, 15-minute, 1-hour,
4-hour or daily bars? A signal study on Massive minute bars (no options, no P&L).

  uv run --env-file .env python red_bars.py --out output/red_bars                # design period
  uv run --env-file .env python red_bars.py --final-test --out output/red_bars   # adds the holdout, once

A follow-up to meanrev.py's 3+ down days on daily bars: this asks whether the pattern shows on shorter bars.

Fixed before any result was seen:
- Tickers SPY (primary), QQQ, IWM. Bars from regular-hours minutes, anchored at each session's open: 5 and 15
  minutes, 1 hour (the last one 30 minutes) and 4 hours (9:30-13:30 and 13:30-16:00), plus daily bars.
- Two definitions of a red bar: a red candle (close below its own open) and a down close (close below the previous
  bar's close; the daily study's definition). The signal fires on every bar while 3 or more in a row are red.
  Runs of 5- and 15-minute bars restart each session and their outcomes stay inside it; hourly, 4-hour and daily
  runs and outcomes may cross the night.
- Outcome: the move from the signal bar's close to the close 1, 3 and 5 bars later, in basis points; excess = that
  minus the average after any bar of the same size and period. Also a race: did a later bar's high reach +1 ATR
  (14 bars of that size) before a bar's low reached -1 ATR, within 5 bars (a bar doing both is left out)?
- Design: to 2024-12-31. A test qualifies (ticker x timeframe x definition, 3-bar excess) with BH q < 0.10 across
  the 30 design tests and the same sign at 1 and 5 bars. The holdout (2025-01-01 on) is read only with
  --final-test, for the qualifiers.
- Uncertainty is clustered by session date (events on the same day, or overlapping, are not independent).
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import scalp_meanrev as sm  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM")
TIMEFRAMES = {"5 min": 5, "15 min": 15, "1 hour": 60, "4 hours": 240, "1 day": None}
WITHIN_SESSION = {"5 min", "15 min"}          # runs restart each session, outcomes stay inside it
DEFINITIONS = ("red candles", "down closes")
RUN = 3
HORIZONS = (1, 3, 5)                         # bars
PRIMARY_H = 3
HOLDOUT_START = pd.Timestamp("2025-01-01")
FDR = 0.10


def daily_bars(rth):
    """Regular-session daily bars from minutes, in the same shape as resampled bars."""
    g = rth.groupby("session")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()}).reset_index()
    return d


def make_bars(rth, minutes):
    if minutes is None:
        return daily_bars(rth)
    bars, _ = features.resample_bars(rth, minutes, min_coverage=0.8)
    return bars[["session", "bar_start", "open", "high", "low", "close"]].reset_index(drop=True)


def red_runs(bars, definition, restart_each_session):
    """Length of the current run of red bars at each bar (0 when the bar is not red)."""
    c, o = bars["close"].to_numpy(float), bars["open"].to_numpy(float)
    sess = bars["session"].to_numpy()
    new = np.r_[True, sess[1:] != sess[:-1]]
    if definition == "red candles":
        red = c < o
    else:
        red = np.r_[False, c[1:] < c[:-1]]
        if restart_each_session:
            red &= ~new  # the first bar's change would include the overnight gap
    out, run = np.zeros(len(c), dtype=int), 0
    for i in range(len(c)):
        if restart_each_session and new[i]:
            run = 0
        run = run + 1 if red[i] else 0
        out[i] = run
    return out


def forward(bars, restart_each_session, horizons=HORIZONS, race_bars=5):
    """Moves (bps) from each bar's close to the close h bars later, and the +/-1 ATR race over the next race_bars
    bars; NaN where those bars do not exist (or leave the session, when restart_each_session)."""
    c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
    sess = bars["session"].to_numpy()
    n = len(c)
    out = {}
    for k in horizons:
        r = np.full(n, np.nan)
        j = np.arange(n) + k
        ok = j < n
        if restart_each_session:
            ok[ok] &= sess[j[ok]] == sess[np.flatnonzero(ok)]
        r[ok] = (c[j[ok]] / c[ok] - 1) * 1e4
        out[f"ret_{k}"] = r
    atr = talib.ATR(h, l, c, 14)
    race = np.full(n, np.nan)
    for i in range(n - race_bars):
        if np.isnan(atr[i]) or (restart_each_session and sess[i + race_bars] != sess[i]):
            continue
        up = np.flatnonzero(h[i + 1:i + 1 + race_bars] >= c[i] + atr[i])
        dn = np.flatnonzero(l[i + 1:i + 1 + race_bars] <= c[i] - atr[i])
        u, d = (up[0] if up.size else race_bars), (dn[0] if dn.size else race_bars)
        race[i] = 1.0 if u < d else -1.0 if d < u else (0.0 if u == race_bars else np.nan)
    out["race"] = race
    return pd.DataFrame(out, index=bars.index)


def ticker_table(ticker, rth):
    """Per timeframe and definition: every bar's run length, period and forward outcomes."""
    frames = []
    for tf, minutes in TIMEFRAMES.items():
        bars = make_bars(rth, minutes)
        within = tf in WITHIN_SESSION
        fwd = forward(bars, within)
        period = np.where(pd.to_datetime(bars["session"]) >= HOLDOUT_START, "holdout", "design")
        for definition in DEFINITIONS:
            frames.append(pd.DataFrame({"ticker": ticker, "timeframe": tf, "definition": definition,
                                        "session": bars["session"].to_numpy(), "period": period,
                                        "run": red_runs(bars, definition, within), **fwd}))
    return pd.concat(frames, ignore_index=True)


def summarize(g, sig=None, side=1.0):
    """Signal (run >= 3 by default, or a boolean mask) against every bar, per horizon, for one ticker, timeframe,
    definition and period. side = -1 measures moves in the short direction (a fade). Shared with bar_patterns.py."""
    sig = (g["run"] >= RUN) if sig is None else pd.Series(np.asarray(sig, bool), index=g.index)
    out = {"bars": len(g), "signals": int(sig.sum()), "share": sig.mean() * 100}
    for k in HORIZONS:
        x = side * g.loc[sig, f"ret_{k}"]
        keep = x.notna()
        base = side * g[f"ret_{k}"].mean()
        m, se = sm.cluster_mean((x[keep] - base).to_numpy(), g.loc[sig, "session"][keep].to_numpy())
        z = m / se if se else np.nan
        out.update({f"avg_{k}": x.mean(), f"base_{k}": base, f"excess_{k}": m, f"lo_{k}": m - 1.96 * se,
                    f"hi_{k}": m + 1.96 * se, f"p_{k}": 2 * norm.sf(abs(z)) if not np.isnan(z) else np.nan,
                    f"p_up_{k}": norm.sf(z) if not np.isnan(z) else np.nan, f"hit_{k}": (x[keep] > 0).mean() * 100})
    r = side * g.loc[sig, "race"]
    wins, losses = int((r == 1).sum()), int((r == -1).sum())
    lo, hi = alerts.wilson(wins, wins + losses)
    base_r = side * g["race"]
    out.update(race=wins / (wins + losses) * 100 if wins + losses else np.nan, race_lo=lo, race_hi=hi,
               race_base=(base_r == 1).sum() / base_r.isin([1, -1]).sum() * 100)
    return out


def run_study(minutes_by_ticker, sessions, *, final=False, timings=None):
    """minutes_by_ticker: {ticker: regular-hours minutes}. No file I/O."""
    timings = {} if timings is None else timings
    with timed(timings, "bars"):
        table = pd.concat([ticker_table(tk, rth) for tk, rth in minutes_by_ticker.items()], ignore_index=True)
    with timed(timings, "statistics"):
        rows = []
        for (tk, tf, d, period), g in table.groupby(["ticker", "timeframe", "definition", "period"], sort=False):
            if period == "holdout" and not final:
                continue
            rows.append({"ticker": tk, "timeframe": tf, "definition": d, "period": period, **summarize(g)})
        stats = pd.DataFrame(rows)
        design = stats[stats["period"] == "design"].copy()
        design["q"] = mr.bh_qvalues(design[f"p_{PRIMARY_H}"].fillna(1).to_numpy())
        same = np.sign(design["excess_1"]) == np.sign(design[f"excess_{PRIMARY_H}"])
        same &= np.sign(design["excess_5"]) == np.sign(design[f"excess_{PRIMARY_H}"])
        design["qualifies"] = (design["q"] < FDR) & same
        held = None
        if final:
            keys = ["ticker", "timeframe", "definition"]
            held = stats[stats["period"] == "holdout"].merge(design.loc[design["qualifies"], keys + [
                f"excess_{PRIMARY_H}"]].rename(columns={f"excess_{PRIMARY_H}": "design_excess"}), on=keys)
            if len(held):
                side = np.sign(held["design_excess"])
                p = np.where(side > 0, held[f"p_up_{PRIMARY_H}"], 1 - held[f"p_up_{PRIMARY_H}"])
                held["holdout_q"] = mr.bh_qvalues(p)
                held["holds_up"] = held["holdout_q"] < FDR
    return {"stats": stats, "design": design, "holdout": held, "final": final, "tickers": list(minutes_by_ticker),
            "first": sessions.index[0], "last": sessions.index[-1]}


# ---------------------------------------------------------------- report

def bps(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}"


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def find(table, tk, tf, d):
    m = table[(table["ticker"] == tk) & (table["timeframe"] == tf) & (table["definition"] == d)]
    return m.iloc[0] if len(m) else None


def render_report(res):
    d = res["design"]
    lines = []
    w = lines.append
    w("# Three red bars in a row, by timeframe\n")
    w("After 3 or more red bars in a row (and on every further red bar), the move from that bar's close to the close "
      "1, 3 and 5 bars later, in basis points (1 bp = 0.01%). *Excess* = that minus the average after any bar of the "
      "same size in the same period (positive = a bounce). *Race*: price reached +1 ATR before -1 ATR within 5 bars "
      "(50% is a coin flip; *any bar* shows the same for every bar). Ranges are 95% intervals clustered by day. "
      "Design period to 2024-12-31" + (", holdout 2025-01-01 on." if res["final"] else " (holdout not read).") + "\n")
    q = d[d["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(d)} design tests qualified (a clear 3-bar excess, q < {FDR:g}, with the same "
      "sign at 1 and 5 bars)" + (": " + "; ".join(f"{r.ticker} {r.timeframe} {r.definition} "
                                                   f"({bps(r[f'excess_{PRIMARY_H}'])} bps)" for _, r in q.iterrows())
                                 if len(q) else ".") + "\n")
    for tk in res["tickers"]:
        w(f"## {tk}, design period\n")
        rows = []
        for tf in TIMEFRAMES:
            for df in DEFINITIONS:
                r = find(d, tk, tf, df)
                if r is None:
                    continue
                rows.append([tf, df, f"{int(r['signals']):,} ({r['share']:.0f}% of bars)",
                             " / ".join(bps(r[f"avg_{k}"]) for k in HORIZONS),
                             " / ".join(bps(r[f"base_{k}"]) for k in HORIZONS),
                             f"{bps(r[f'excess_{PRIMARY_H}'])} ({bps(r[f'lo_{PRIMARY_H}'])} to {bps(r[f'hi_{PRIMARY_H}'])})",
                             f"{r['q']:.2f}", pct(r[f"hit_{PRIMARY_H}"]),
                             f"{pct(r['race'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])}); any bar {pct(r['race_base'])}"])
        w(alerts.md_table(["Bars", "Red =", "Signals", "After 1 / 3 / 5 bars (bps)", "Any bar 1 / 3 / 5",
                           "Excess after 3 (95%)", "q", "Up after 3", "Race (95%)"], rows) + "\n")
    if res["final"]:
        w("## Holdout: did the qualifiers hold up?\n")
        h = res["holdout"]
        if h is None or h.empty:
            w("No qualifiers to test.\n")
        else:
            rows = [[r.ticker, r.timeframe, r.definition, f"{int(r.signals):,}", bps(r.design_excess),
                     f"{bps(getattr(r, f'excess_{PRIMARY_H}'))} ({bps(getattr(r, f'lo_{PRIMARY_H}'))} to "
                     f"{bps(getattr(r, f'hi_{PRIMARY_H}'))})", f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"]
                    for r in h.itertuples()]
            w(alerts.md_table(["Ticker", "Bars", "Red =", "Signals", "Design excess", "Holdout excess (95%)",
                               "Holdout q", "Holds up"], rows) + "\n")
    w("## Notes\n")
    w("- 5- and 15-minute runs restart each session and their outcomes stay inside it; hourly and longer may cross "
      "the night. The last hourly bar is 30 minutes and the second 4-hour bar 2.5 hours.")
    w("- scalp_meanrev.py already tested 4-6 same-direction 5-minute closes between 10:30 and 15:00 (no edge).")
    return "\n".join(lines) + "\n"


def plot(res, path):
    d = res["design"]
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    for k, df in enumerate(DEFINITIONS):
        ax = sds._axes(fig, [0.12 + k * 0.45, 0.2, 0.38, 0.6])
        x = np.arange(len(TIMEFRAMES))
        for j, (tk, color) in enumerate(zip(res["tickers"], (SERIES[0], SERIES[1], INK_2))):
            vals = [find(d, tk, tf, df) for tf in TIMEFRAMES]
            m = np.array([v[f"excess_{PRIMARY_H}"] for v in vals])
            lo = np.array([v[f"lo_{PRIMARY_H}"] for v in vals])
            hi = np.array([v[f"hi_{PRIMARY_H}"] for v in vals])
            ax.errorbar(x + (j - 1) * 0.18, m, yerr=[m - lo, hi - m], fmt="o", ms=4, color=color, ecolor=color,
                        elinewidth=1.2, capsize=2, label=tk if k == 0 else None)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_xticks(x, list(TIMEFRAMES), fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(df[0].upper() + df[1:], color=INK, fontsize=8.5, loc="left")
        if k == 0:
            ax.set_ylabel("Excess move after 3 bars, bps (95%)", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, "After 3+ red bars in a row: bounce or not?", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.915, "Design period 2021-2024. Above zero = price rose more than after an average bar of the same "
             "size.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Three red bars in a row on different timeframes (signal study).")
    p.add_argument("--out", default="output/red_bars")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true", help="also read the 2025+ holdout (run once)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        data = {}
        for tk in TICKERS:
            minutes, _ = gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)
            data[tk], _ = features.regular_session_minutes(minutes, sessions)
    res = run_study(data, sessions, final=args.final_test, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["design"].to_csv(out / "design.csv", index=False)
    if res["final"] and res["holdout"] is not None:
        res["holdout"].to_csv(out / "holdout.csv", index=False)
    plot(res, out / "red_bars.png")
    (out / "report.md").write_text(render_report(res))
    q = res["design"][res["design"]["qualifies"]]
    print(f"design tests: {len(res['design'])}, qualified: {len(q)}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
