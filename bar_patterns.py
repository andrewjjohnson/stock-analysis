"""Candle patterns on short timeframes: 3 red, then green, then red (a bottoming try) and its mirror for tops. A
signal study on Massive minute bars (no options, no P&L).

  uv run --env-file .env python bar_patterns.py --out output/bar_patterns                # design period
  uv run --env-file .env python bar_patterns.py --final-test --out output/bar_patterns   # adds the holdout, once

A follow-up to red_bars.py's runs of red bars. The idea here: a first green bar after the run,
then a red retest, means sellers are tiring and price starts to turn.

Fixed before any result was seen:
- Tickers SPY, QQQ, IWM; 5-minute, 15-minute and 1-hour bars (red_bars.make_bars). Candle colours: green = close
  above open, red = close below open (a doji breaks a pattern).
- Patterns, the signal at the fifth bar's close: red-red-red-green-red, bought; the same where the last bar's low
  stays above the lowest low of the three red bars (a higher low); green-green-green-red-green, faded (measured
  short); the same with a lower high. 5- and 15-minute patterns must sit inside one session and their outcomes stay
  inside it; hourly ones may cross the night (as in red_bars.py).
- Outcomes and statistics as in red_bars.py: the move 1, 3 and 5 bars later in the signal's direction, in bps,
  against every bar of the same size; a +/-1 ATR race within 5 bars; standard errors clustered by session.
- Design: to 2024-12-31. A test qualifies with BH q < 0.10 on the 3-bar excess across the 36 design tests and the
  same sign at 1 and 5 bars. The holdout (2025-01-01 on) is read only with --final-test, for the qualifiers.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import alerts  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import red_bars as rb  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM")
TIMEFRAMES = {"5 min": 5, "15 min": 15, "1 hour": 60}
# name: (colours, side: +1 buy / -1 fade, require a higher low / lower high)
PATTERNS = {"red x3, green, red": ("RRRGR", 1, False), "red x3, green, red, higher low": ("RRRGR", 1, True),
            "green x3, red, green": ("GGGRG", -1, False), "green x3, red, green, lower high": ("GGGRG", -1, True)}
PRIMARY_H = rb.PRIMARY_H
FDR = 0.10


def colours(bars):
    c, o = bars["close"].to_numpy(float), bars["open"].to_numpy(float)
    return np.where(c > o, "G", np.where(c < o, "R", "D"))


def pattern_mask(bars, pattern, side, extreme, within_session):
    """True at the last bar of each match. extreme: for a buy, the last bar's low stays above the lowest low of the
    first three bars; for a fade, its high stays below their highest high."""
    col, n = colours(bars), len(pattern)
    lo, hi = bars["low"].to_numpy(float), bars["high"].to_numpy(float)
    sess = bars["session"].to_numpy()
    mask = np.zeros(len(col), bool)
    for i in range(n - 1, len(col)):
        if "".join(col[i - n + 1:i + 1]) != pattern:
            continue
        if within_session and sess[i - n + 1] != sess[i]:
            continue
        if extreme:
            first = slice(i - n + 1, i - n + 4)
            if (side > 0 and lo[i] <= lo[first].min()) or (side < 0 and hi[i] >= hi[first].max()):
                continue
        mask[i] = True
    return mask


def run_study(minutes_by_ticker, *, final=False, timings=None):
    """minutes_by_ticker: {ticker: regular-hours minutes}. No file I/O."""
    timings = {} if timings is None else timings
    rows = []
    with timed(timings, "patterns"):
        for tk, rth in minutes_by_ticker.items():
            for tf, minutes in TIMEFRAMES.items():
                bars = rb.make_bars(rth, minutes)
                within = tf in rb.WITHIN_SESSION
                g = pd.concat([bars[["session"]], rb.forward(bars, within)], axis=1)
                g["period"] = np.where(pd.to_datetime(g["session"]) >= rb.HOLDOUT_START, "holdout", "design")
                days = g["session"].nunique()
                for name, (pat, side, extreme) in PATTERNS.items():
                    mask = pattern_mask(bars, pat, side, extreme, within)
                    for period in ("design", "holdout") if final else ("design",):
                        keep = (g["period"] == period).to_numpy()
                        part = g[keep]
                        rows.append({"ticker": tk, "timeframe": tf, "pattern": name, "side": side, "period": period,
                                     "per_day": mask[keep].sum() / part["session"].nunique(),
                                     **rb.summarize(part, mask[keep], side)})
    stats = pd.DataFrame(rows)
    design = stats[stats["period"] == "design"].copy()
    design["q"] = mr.bh_qvalues(design[f"p_{PRIMARY_H}"].fillna(1).to_numpy())
    same = (np.sign(design["excess_1"]) == np.sign(design[f"excess_{PRIMARY_H}"])) & (
        np.sign(design["excess_5"]) == np.sign(design[f"excess_{PRIMARY_H}"]))
    design["qualifies"] = (design["q"] < FDR) & same
    held = None
    if final:
        keys = ["ticker", "timeframe", "pattern"]
        held = stats[stats["period"] == "holdout"].merge(
            design.loc[design["qualifies"], keys + [f"excess_{PRIMARY_H}"]].rename(
                columns={f"excess_{PRIMARY_H}": "design_excess"}), on=keys)
        if len(held):
            up = np.sign(held["design_excess"]) > 0
            p = np.where(up, held[f"p_up_{PRIMARY_H}"], 1 - held[f"p_up_{PRIMARY_H}"])
            held["holdout_q"] = mr.bh_qvalues(p)
            held["holds_up"] = held["holdout_q"] < FDR
    return {"stats": stats, "design": design, "holdout": held, "final": final, "tickers": list(minutes_by_ticker)}


# ---------------------------------------------------------------- report

def bps(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1f}"


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0f}%"


def find(t, tk, tf, pat):
    m = t[(t["ticker"] == tk) & (t["timeframe"] == tf) & (t["pattern"] == pat)]
    return m.iloc[0] if len(m) else None


def render_report(res):
    d = res["design"]
    lines = []
    w = lines.append
    w("# Bottoming and topping candle patterns on short timeframes\n")
    w("At the close of the pattern's fifth bar, the move 1, 3 and 5 bars later **in the signal's direction** (up for "
      "the red-bar bottoms, down for the green-bar tops), in basis points; *excess* subtracts the same move after any "
      "bar of the same size. *Race*: price went 1 ATR the signal's way before 1 ATR against it within 5 bars (*any bar* "
      "is the same for every bar). Ranges are 95% intervals clustered by day. Design period to 2024-12-31"
      + (", holdout 2025-01-01 on." if res["final"] else " (holdout not read).") + "\n")
    q = d[d["qualifies"]]
    w(f"**Bottom line.** {len(q)} of {len(d)} design tests qualified" + (": " + "; ".join(
        f"{r.ticker} {r.timeframe} {r.pattern} ({bps(r[f'excess_{PRIMARY_H}'])} bps)" for _, r in q.iterrows())
        if len(q) else ".") + "\n")
    for tk in res["tickers"]:
        w(f"## {tk}, design period\n")
        rows = []
        for tf in TIMEFRAMES:
            for pat in PATTERNS:
                r = find(d, tk, tf, pat)
                if r is None:
                    continue
                rows.append([tf, pat, f"{int(r['signals']):,} ({r['per_day']:.1f} a day)",
                             " / ".join(bps(r[f"avg_{k}"]) for k in rb.HORIZONS),
                             f"{bps(r[f'excess_{PRIMARY_H}'])} ({bps(r[f'lo_{PRIMARY_H}'])} to {bps(r[f'hi_{PRIMARY_H}'])})",
                             f"{r['q']:.2f}", pct(r[f"hit_{PRIMARY_H}"]),
                             f"{pct(r['race'])} ({pct(r['race_lo'])}-{pct(r['race_hi'])}); any bar {pct(r['race_base'])}"])
        w(alerts.md_table(["Bars", "Pattern", "Signals", "Move after 1 / 3 / 5 bars (bps)", "Excess after 3 (95%)", "q",
                           "Right way after 3", "Race (95%)"], rows) + "\n")
    if res["final"]:
        w("## Holdout\n")
        h = res["holdout"]
        if h is None or h.empty:
            w("No qualifiers to test.\n")
        else:
            w(alerts.md_table(["Ticker", "Bars", "Pattern", "Signals", "Design excess", "Holdout excess (95%)", "q",
                               "Holds up"], [[r.ticker, r.timeframe, r.pattern, f"{int(r.signals):,}",
                                              bps(r.design_excess), f"{bps(getattr(r, f'excess_{PRIMARY_H}'))}",
                                              f"{r.holdout_q:.3f}", "yes" if r.holds_up else "no"]
                                             for r in h.itertuples()]) + "\n")
    return "\n".join(lines) + "\n"


def plot(res, path):
    d = res["design"]
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    groups = (("Bottoms (bought)", list(PATTERNS)[:2]), ("Tops (faded)", list(PATTERNS)[2:]))
    for k, (title, pats) in enumerate(groups):
        ax = sds._axes(fig, [0.1 + k * 0.47, 0.24, 0.38, 0.56])
        x = np.arange(len(TIMEFRAMES))
        for j, (tk, color) in enumerate(zip(res["tickers"], (SERIES[0], SERIES[1], INK_2))):
            for m_, pat in enumerate(pats):
                vals = [find(d, tk, tf, pat) for tf in TIMEFRAMES]
                m = np.array([v[f"excess_{PRIMARY_H}"] for v in vals])
                lo = np.array([v[f"lo_{PRIMARY_H}"] for v in vals])
                hi = np.array([v[f"hi_{PRIMARY_H}"] for v in vals])
                ax.errorbar(x + (j - 1) * 0.2 + m_ * 0.07, m, yerr=[m - lo, hi - m], fmt="o" if m_ == 0 else "s", ms=3.5,
                            color=color, ecolor=color, elinewidth=1, capsize=1.5,
                            label=f"{tk}{' (strict)' if m_ else ''}" if k == 0 else None, alpha=1 if m_ == 0 else 0.6)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_xticks(x, list(TIMEFRAMES), fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
        if k == 0:
            ax.set_ylabel("Excess move after 3 bars, signal's way, bps (95%)", color=INK_2, fontsize=7)
    fig.text(0.02, 0.97, "Red x3, green, red (and its mirror): does price turn?", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Design period 2021-2024. Above zero = the move went the pattern's way more than after an "
             "average bar. Squares: with a higher low / lower high.", color=INK_2, fontsize=7.2, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=6, frameon=False, fontsize=7, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bottoming / topping candle patterns on short timeframes.")
    p.add_argument("--out", default="output/bar_patterns")
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
    res = run_study(data, final=args.final_test, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["stats"].to_csv(out / "stats.csv", index=False)
    res["design"].to_csv(out / "design.csv", index=False)
    plot(res, out / "patterns.png")
    (out / "report.md").write_text(render_report(res))
    q = res["design"][res["design"]["qualifies"]]
    print(f"design tests: {len(res['design'])}, qualified: {len(q)}")
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
