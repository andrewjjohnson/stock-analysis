"""Comparison tables, terminal summary, output files and one static chart."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402

from outcomes import COST_BPS, EXCURSION_MINUTES, HORIZONS  # noqa: E402

# Chart colors: categorical slots 1-2 (validated pair) plus neutral ink and chrome;
# slot 3 (validated all-pairs with 1-2) only marks the target level on candidate charts.
SERIES = ["#2a78d6", "#eb6834"]
TARGET = "#1baf7a"
INK, INK_2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIGNAL_TINT, OPENING_TINT = "#cde2fb", "#f0efec"
EXIT_REASONS = ("target", "stop", "stop_gap", "close", "unresolved")
NY = "America/New_York"

CAVEAT = ("Signal study, not P&L: the reference price is the completed trigger bar's close, not a fill. "
          "Overlapping candidate windows are not independent observations.")
BARRIER_CAVEAT = ("Idealized barrier comparison, not P&L: next-minute-open entry, fixed stop/target touches, no "
                  "compounding. 1 bp/side is an illustrative friction sensitivity, not calibrated costs; "
                  "no borrow costs, and short results do not establish borrow availability.")


def summarize(candidates):
    """Counts and return/excursion statistics for one configuration and segment.

    Statistics with no available values are NaN, never 0.
    """
    n = len(candidates)
    out = {"n_candidates": n}
    for h in HORIZONS:
        v = candidates[f"fwd_ret_{h}m_pct"].dropna()
        out[f"n_{h}m"] = len(v)
        out[f"unavailable_{h}m"] = n - len(v)
        out[f"mean_{h}m_pct"] = v.mean() if len(v) else np.nan
        out[f"median_{h}m_pct"] = v.median() if len(v) else np.nan
        out[f"frac_pos_{h}m"] = (v > 0).mean() if len(v) else np.nan
    e = EXCURSION_MINUTES
    mfe = candidates[f"mfe_{e}m_pct"].dropna()
    mae = candidates[f"mae_{e}m_pct"].dropna()
    out[f"n_excursion_{e}m"] = len(mfe)
    out[f"unavailable_excursion_{e}m"] = n - len(mfe)
    for name, v in (("mfe", mfe), ("mae", mae)):
        out[f"mean_{name}_{e}m_pct"] = v.mean() if len(v) else np.nan
        out[f"median_{name}_{e}m_pct"] = v.median() if len(v) else np.nan
    return out


def summarize_barriers(candidates):
    """Entry validity, exit reasons and R for the fixed-barrier comparison.

    R and returns cover resolved exits only; statistics with no values are NaN, never 0.
    """
    status, reason = candidates["entry_status"], candidates["exit_reason"]
    out = {"n_entry_ok": int((status == "ok").sum()), "n_entry_unavailable": int((status == "unavailable").sum()),
           "n_entry_invalid": int((status == "invalid").sum()),
           **{f"n_exit_{r}": int((reason == r).sum()) for r in EXIT_REASONS},
           "n_ambiguous": int(candidates["ambiguous"].astype(bool).sum())}
    for b in COST_BPS:
        r, ret = candidates[f"barrier_r_{b}bp"].dropna(), candidates[f"barrier_ret_{b}bp_pct"].dropna()
        out[f"n_resolved_{b}bp"] = len(r)
        out[f"mean_r_{b}bp"] = r.mean() if len(r) else np.nan
        out[f"median_r_{b}bp"] = r.median() if len(r) else np.nan
        out[f"frac_pos_r_{b}bp"] = (r > 0).mean() if len(r) else np.nan
        out[f"mean_ret_{b}bp_pct"] = ret.mean() if len(ret) else np.nan
    held = candidates["holding_minutes"].dropna()
    out["median_holding_min"] = held.median() if len(held) else np.nan
    return out


def summarize_sides(candidates, counts, patterns, barriers=False):
    """One row per side (pattern "all", carrying the session counts) and one per pattern group.

    Pattern groups are mutually exclusive, so they add up to their side row; session
    counts appear only on side rows so that nothing is double counted.
    """
    rows = []
    for side, groups in patterns.items():
        s = candidates[candidates["side"] == side]
        rows.append({"side": side, "pattern": "all", **counts[side], **summarize(s),
                     **(summarize_barriers(s) if barriers else {})})
        for p in groups:
            g = s[s["pattern"] == p]
            rows.append({"side": side, "pattern": p, **summarize(g), **(summarize_barriers(g) if barriers else {})})
    return rows


def _fmt(value, n=None, sign=True):
    text = "—" if np.isnan(value) else (f"{value:+.3f}" if sign else f"{value:.2f}")
    return text if n is None else f"{text} ({n})"


def _group(r):
    return r["side"] if r["pattern"] == "all" else f"  {r['pattern']}"


def format_table(summary):
    """Compact comparison table: means with their available counts in parentheses.

    With side/pattern rows, empty pattern rows are left out of the printout (not the CSV).
    """
    e = EXCURSION_MINUTES
    sides = "side" in summary
    header = f"{'config':<33}{'segment':<11}" + (f"{'side / pattern':<26}" if sides else "") + f"{'cands':>6}"
    header += "".join(f"{f'{h}m % (n)':>16}" for h in HORIZONS) + f"{'>0 @30m':>9}{f'MFE{e} %':>9}{f'MAE{e} %':>9}"
    lines = [header]
    for r in summary.to_dict("records"):
        if sides and r["pattern"] != "all" and r["n_candidates"] == 0:
            continue
        line = f"{r['config']:<33}{r['segment']:<11}" + (f"{_group(r):<26}" if sides else "") + f"{r['n_candidates']:>6}"
        line += "".join(f"{_fmt(r[f'mean_{h}m_pct'], r[f'n_{h}m']):>16}" for h in HORIZONS)
        line += f"{_fmt(r['frac_pos_30m'], sign=False):>9}"
        line += f"{_fmt(r[f'mean_mfe_{e}m_pct']):>9}{_fmt(r[f'mean_mae_{e}m_pct']):>9}"
        lines.append(line)
    return "\n".join(lines)


def format_counts(summary):
    """Session funnel per configuration and segment, from the side rows present.

    After a split, the later segment holds only each side's selected threshold.
    """
    lines = []
    sides = summary[summary["pattern"] == "all"]
    for (config, segment), g in sides.groupby(["config", "segment"], sort=False):
        first = g.iloc[0]
        per_side = lambda col: " / ".join(f"{side} {int(n)}" for side, n in zip(g["side"], g[col]))  # noqa: E731
        lines.append(
            f"{config:<33}{segment:<11}{int(first['sessions'])} sessions -> {int(first['eligible_sessions'])} eligible "
            f"(opening incomplete {int(first['or_incomplete'])}, prior ATR unavailable {int(first['atr_unavailable'])})"
            f" -> qualifying openings {per_side('qualifying_openings')} (flat skipped "
            f"{int(first['flat_openings_skipped'])}) -> signals {per_side('n_candidates')}")
    return "\n".join(lines)


def format_barrier_table(summary):
    """Barrier comparison per configuration, segment and side (pattern rows as in format_table)."""
    b0, b1 = COST_BPS
    header = (f"{'config':<33}{'segment':<11}{'side / pattern':<26}{'cands':>6}{'entry ok':>9}{'no entry':>9}"
              f"{'invalid':>8}{'target':>7}{'stop':>6}{'gap':>5}{'close':>6}{'unres.':>7}{'ambig.':>7}"
              f"{f'R@{b0}bp mean (n)':>18}{f'median':>8}{f'R>0':>6}{f'R@{b1}bp mean':>13}{f'% @{b1}bp':>9}")
    lines = [header]
    for r in summary.to_dict("records"):
        if r["pattern"] != "all" and r["n_candidates"] == 0:
            continue
        line = f"{r['config']:<33}{r['segment']:<11}{_group(r):<26}{r['n_candidates']:>6}"
        line += f"{r['n_entry_ok']:>9}{r['n_entry_unavailable']:>9}{r['n_entry_invalid']:>8}"
        line += "".join(f"{r[f'n_exit_{x}']:>{w}}" for x, w in zip(EXIT_REASONS, (7, 6, 5, 6, 7)))
        line += f"{r['n_ambiguous']:>7}{_fmt(r[f'mean_r_{b0}bp'], r[f'n_resolved_{b0}bp']):>18}"
        line += f"{_fmt(r[f'median_r_{b0}bp']):>8}{_fmt(r[f'frac_pos_r_{b0}bp'], sign=False):>6}"
        line += f"{_fmt(r[f'mean_r_{b1}bp']):>13}{_fmt(r[f'mean_ret_{b1}bp_pct']):>9}"
        lines.append(line)
    return "\n".join(lines)


def write_outputs(out_dir, candidates, summary, settings, title, context, chart_config=None, horizon=30):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("candidate_*.png"):  # a reused directory must not show an earlier run's charts
        stale.unlink()
    candidates.to_parquet(out / "candidates.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
    plot_chart(summary, out / "chart.png", title, context, chart_config, horizon)
    return out


def plot_chart(summary, path, title, context, chart_config=None, horizon=30):
    """Mean forward return by horizon for one configuration (both segments after a
    split); with several configurations and none singled out, mean return at
    `horizon` per configuration instead. Every mean is labeled with its count."""
    if "pattern" in summary:  # one row per side: pattern groups are in summary.csv
        summary = summary[summary["pattern"] == "all"]
    rows = summary[summary["config"] == chart_config] if chart_config else summary
    kind = "directional forward return" if "side" in rows else "forward return"
    if rows["config"].nunique() == 1 and len(rows) <= len(SERIES):
        fig = _plot_horizons(rows)
        what = f"Mean {kind} by minutes elapsed after the trigger bar completed; n = outcomes available"
    else:
        fig = _plot_configs(rows, horizon)
        what = f"Mean {horizon}-minute {kind} per configuration; n = {horizon}-minute outcomes available"
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / height, title, color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.45 / height, f"{textwrap.fill(context, 125)}\n{what}", color=INK_2, fontsize=8, va="top",
             linespacing=1.5)
    fig.text(0.02, 0.1 / height, textwrap.fill(CAVEAT, 130), color=MUTED, fontsize=7, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _canvas(height, left, bottom, top=1.05):
    """8-inch-wide figure with fixed margins in inches, so header, axes, legend and footer never overlap."""
    fig = plt.figure(figsize=(8, height), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([left / 8, bottom / height, 1 - (left + 0.3) / 8, 1 - (bottom + top) / height])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=8)
    return fig, ax


def _label_bar(ax, pos, value, n, horizontal=False):
    """Count label just beyond the bar end (at the baseline when there is no data)."""
    text = f"n={n}" if not np.isnan(value) else f"n={n} (no data)"
    value = 0.0 if np.isnan(value) else value
    side = 1 if value >= 0 else -1
    if horizontal:
        ax.annotate(text, (value, pos), xytext=(4 * side, 0), textcoords="offset points", color=INK_2,
                    fontsize=7.5, ha="left" if side > 0 else "right", va="center")
    else:
        ax.annotate(text, (pos, value), xytext=(0, 3 * side), textcoords="offset points", color=INK_2,
                    fontsize=7.5, ha="center", va="bottom" if side > 0 else "top")


def _plot_horizons(rows):
    legend = len(rows) > 1
    height, bottom = 5.0, (1.45 if legend else 0.95)
    fig, ax = _canvas(height, left=0.95, bottom=bottom)
    x = np.arange(len(HORIZONS))
    width = 0.22 if legend else 0.3
    offsets = (np.arange(len(rows)) - (len(rows) - 1) / 2) * (width + 0.04)
    for i, (r, off) in enumerate(zip(rows.to_dict("records"), offsets)):
        means = np.array([r[f"mean_{h}m_pct"] for h in HORIZONS], dtype=float)
        label = (f"{r['side'] + ' · ' if 'side' in r else ''}{r['segment']} ({r['role']}): {r['first_session']} "
                 f"to {r['last_session']}, {r['n_candidates']} candidates")
        ax.bar(x + off, np.nan_to_num(means), width, color=SERIES[i], label=label, zorder=3)
        for pos, h, m in zip(x + off, HORIZONS, means):
            _label_bar(ax, pos, m, r[f"n_{h}m"])
    ax.axhline(0, color=BASELINE, lw=1, zorder=2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(x, [f"{h} min" for h in HORIZONS])
    ax.set_ylabel("Mean directional forward return (%)" if "side" in rows else "Mean forward return (%)",
                  color=INK_2, fontsize=8)
    ax.margins(y=0.15)
    if legend:
        fig.legend(loc="upper left", bbox_to_anchor=(0.02, (bottom - 0.4) / height), frameon=False,
                   fontsize=8, labelcolor=INK_2)
    return fig


def _plot_configs(rows, horizon):
    several_segments = rows["segment"].nunique() > 1
    labels = []
    for r in rows.to_dict("records"):
        label = f"{r['config']} · {r['side']}" if "side" in r else r["config"]
        labels.append(f"{label} [{r['segment']}]" if several_segments else label)
    height = 2.35 + 0.34 * len(rows)
    fig, ax = _canvas(height, left=max(2.3, 0.2 + 0.065 * max(map(len, labels))), bottom=1.0)
    y = np.arange(len(rows))[::-1]
    means = rows[f"mean_{horizon}m_pct"].to_numpy(float)
    ax.barh(y, np.nan_to_num(means), 0.55, color=SERIES[0], zorder=3)
    for pos, m, n in zip(y, means, rows[f"n_{horizon}m"]):
        _label_bar(ax, pos, m, n, horizontal=True)
    ax.axvline(0, color=BASELINE, lw=1, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(y, labels)
    kind = "directional forward return" if "side" in rows else "forward return"
    ax.set_xlabel(f"Mean {horizon}-minute {kind} (%)", color=INK_2, fontsize=8)
    ax.margins(x=0.2)
    return fig


def plot_candidates(out_dir, candidates, bars, ticker, limit=6):
    """Inspection charts for the first `limit` candidates by signal time (never chosen by outcome)."""
    out = Path(out_dir)
    paths = []
    for n, c in enumerate(candidates.sort_values("signal_time", kind="stable").head(limit).to_dict("records"), 1):
        path = out / f"candidate_{n}_{c['session']:%Y-%m-%d}_{c['side']}.png"
        _plot_candidate(bars[bars["session"] == c["session"]], c, ticker, path)
        paths.append(path)
    return paths


def _px(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def _et(ts):
    return "—" if pd.isna(ts) else ts.tz_convert(NY).strftime("%H:%M")


def _plot_candidate(day, c, ticker, path):
    """5-minute candles for one session with the opening box, signal bar and optional levels."""
    minute = pd.Timedelta(minutes=1)
    open_ = c["session_open"]
    x = lambda ts: (ts - open_) / minute  # noqa: E731 - minutes since the session open
    has_exit = "exit_time" in c and not pd.isna(c["exit_time"])
    session_len = x(day["session_close"].iloc[0])
    right = min(session_len, max(120.0, x(c["exit_time"]) + 15 if has_exit else 0.0))
    shown = day[(day["bar_start"] - open_) / minute < right]

    height = 6.9
    fig = plt.figure(figsize=(8, height), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0.75 / 8, 2.45 / height, 1 - (0.75 + 1.55) / 8, (height - 2.45 - 1.05) / height])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

    ax.add_patch(plt.Rectangle((0, c["or_low"]), 15, c["or_range"], facecolor=OPENING_TINT, edgecolor=BASELINE,
                               lw=0.8, zorder=1))
    sig_start, sig_end = x(c["bar_start"]), x(c["signal_time"])
    ax.axvspan(sig_start, sig_end, color=SIGNAL_TINT, lw=0, zorder=1)
    for level in (c["or_high"], c["or_low"]):
        ax.plot([15, right], [level, level], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.axvline(90, color=BASELINE, lw=0.8, ls=(0, (4, 3)), zorder=1)

    for b in shown.itertuples():
        mid = x(b.bar_start) + 2.5
        is_signal = b.bar_start == c["bar_start"]
        edge = SERIES[0] if is_signal else INK_2
        up = b.close >= b.open
        ax.plot([mid, mid], [b.low, b.high], color=edge, lw=1.1 if is_signal else 0.7, zorder=3)
        ax.add_patch(plt.Rectangle((mid - 1.7, min(b.open, b.close)), 3.4, max(abs(b.close - b.open), 1e-9),
                                   facecolor=SURFACE if up else edge, edgecolor=edge, lw=1.1 if is_signal else 0.7,
                                   zorder=4))

    labels = [(c["or_high"], f"opening high {_px(c['or_high'])}", INK_2),
              (c["or_low"], f"opening low {_px(c['or_low'])}", INK_2)]
    if "entry_price" in c and not pd.isna(c["entry_price"]):
        end = x(c["exit_time"]) if has_exit else right
        for level, color, style, name in ((c["entry_price"], INK, (0, (1, 1.5)), "entry"),
                                          (c["stop_price"], SERIES[1], (0, (4, 2)), "stop"),
                                          (c["target_price"], TARGET, (0, (4, 2)), "target")):
            ax.plot([sig_end, end], [level, level], color=color, lw=1.4, ls=style, zorder=5)
            labels.append((level, f"{name} {_px(level)}", INK))
        if has_exit:
            ax.plot(x(c["exit_time"]), c["exit_price"], "o", ms=6, color=INK, mec=SURFACE, mew=1.5, zorder=6)
    elif "stop_price" in c and not pd.isna(c["stop_price"]):
        labels += [(c["stop_price"], f"stop {_px(c['stop_price'])} (no entry)", INK_2)]

    ax.set_xlim(-3, right + 2)
    ax.margins(y=0.12)
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.05
    placed = []
    for level, text, color in sorted(labels):
        y = max([level] + [p + gap for p in placed[-1:]])
        placed.append(y)
        ax.annotate(text, (right + 2, level), xytext=(right + 6, y), textcoords="data", color=color, fontsize=7,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if abs(y - level) > gap / 4 else None)
    ax.annotate(f"signal known {_et(c['signal_time'])}", (sig_end, hi), xytext=(3, -2), textcoords="offset points",
                color=INK_2, fontsize=7, va="top")
    ax.annotate("search ends", (90, hi), xytext=(3, -12), textcoords="offset points", color=MUTED, fontsize=7,
                va="top")
    ticks = np.arange(0, right + 1, 30)
    ax.set_xticks(ticks, [_et(open_ + t * minute) for t in ticks])
    ax.set_xlabel("New York time (5-minute bars; shaded box = first 15 minutes)", color=INK_2, fontsize=7.5)

    title = f"{ticker} · {c['session']:%Y-%m-%d} · {c['side']} · {c['pattern']} · {c['config']}"
    fig.text(0.02, 1 - 0.15 / height, title, color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.45 / height, "Chosen as one of the first candidates by time, not by outcome. The blue bar is "
             "the signal bar; it is known only when it completes.", color=INK_2, fontsize=8, va="top")

    pair = lambda a, b: f"{_px(c[a])} / {_px(c[b])}"  # noqa: E731
    columns = [("Opening range", [
        ("open / high", pair("or_open", "or_high")),
        ("low / close", pair("or_low", "or_close")),
        ("range, known", f"{_px(c['or_range'])}, {_et(c['opening_end'])}"),
        ("prior ATR14", _px(c["prev_atr_14"])),
        ("range / ATR", f"{c['range_atr_ratio']:.3f} (gate {c['threshold']:.2f})"),
    ]), ("Signal bar", [
        ("bar (ET)", f"{_et(c['bar_start'])} - {_et(c['signal_time'])}"),
        ("open / high", pair("signal_open", "signal_high")),
        ("low / close", pair("signal_low", "ref_close")),
        ("pattern", c["pattern"]),
        ("30m directional", "—" if pd.isna(c["fwd_ret_30m_pct"]) else f"{c['fwd_ret_30m_pct']:+.3f}%"),
    ])]
    if "entry_status" in c:
        r0, r1 = (c[f"barrier_r_{b}bp"] for b in COST_BPS)
        columns.append(("Barrier comparison", [
            ("entry", f"{c['entry_status']} {_et(c['entry_time'])} {_px(c['entry_price'])}"),
            ("stop / target", pair("stop_price", "target_price")),
            ("exit", f"{c['exit_reason'] or '—'} {_et(c['exit_time'])} {_px(c['exit_price'])}"),
            ("ambiguous", "yes (stop first)" if c["ambiguous"] else "no"),
            (f"R @{COST_BPS[0]} / {COST_BPS[1]} bp", "—" if pd.isna(r0) else f"{r0:+.2f} / {r1:+.2f}"),
        ]))
    for col, (heading, rows) in enumerate(columns):
        left = 0.02 + col * 0.33
        fig.text(left, 1.95 / height, heading, color=INK, fontsize=8, fontweight="bold", va="top")
        for i, (k, v) in enumerate(rows):
            y = (1.95 - 0.24 * (i + 1)) / height
            fig.text(left, y, k, color=MUTED, fontsize=7, va="top")
            fig.text(left + 0.115, y, v, color=INK_2, fontsize=7, va="top")
    fig.text(0.02, 0.1 / height, textwrap.fill(BARRIER_CAVEAT if "entry_status" in c else CAVEAT, 130),
             color=MUTED, fontsize=7, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
