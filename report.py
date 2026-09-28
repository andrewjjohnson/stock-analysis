"""Comparison tables, terminal summary, output files and one static chart."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from outcomes import EXCURSION_MINUTES, HORIZONS  # noqa: E402

# Chart colors: categorical slots 1-2 (validated pair) plus neutral ink and chrome.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

CAVEAT = ("Signal study, not P&L: the reference price is the completed trigger bar's close, not a fill. "
          "Overlapping candidate windows are not independent observations.")


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


def _fmt(value, n=None, sign=True):
    text = "—" if np.isnan(value) else (f"{value:+.3f}" if sign else f"{value:.2f}")
    return text if n is None else f"{text} ({n})"


def format_table(summary):
    """Compact comparison table: means with their available counts in parentheses."""
    e = EXCURSION_MINUTES
    header = f"{'config':<33}{'segment':<11}{'cands':>6}" + "".join(f"{f'{h}m % (n)':>16}" for h in HORIZONS)
    header += f"{'>0 @30m':>9}{f'MFE{e} %':>9}{f'MAE{e} %':>9}"
    lines = [header]
    for r in summary.to_dict("records"):
        line = f"{r['config']:<33}{r['segment']:<11}{r['n_candidates']:>6}"
        line += "".join(f"{_fmt(r[f'mean_{h}m_pct'], r[f'n_{h}m']):>16}" for h in HORIZONS)
        line += f"{_fmt(r['frac_pos_30m'], sign=False):>9}"
        line += f"{_fmt(r[f'mean_mfe_{e}m_pct']):>9}{_fmt(r[f'mean_mae_{e}m_pct']):>9}"
        lines.append(line)
    return "\n".join(lines)


def write_outputs(out_dir, candidates, summary, settings, title, context, chart_config=None, horizon=30):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out / "candidates.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
    plot_chart(summary, out / "chart.png", title, context, chart_config, horizon)
    return out


def plot_chart(summary, path, title, context, chart_config=None, horizon=30):
    """Mean forward return by horizon for one configuration (both segments after a
    split); with several configurations and none singled out, mean return at
    `horizon` per configuration instead. Every mean is labeled with its count."""
    rows = summary[summary["config"] == chart_config] if chart_config else summary
    if rows["config"].nunique() == 1:
        fig = _plot_horizons(rows)
        what = "Mean forward return by minutes elapsed after the trigger bar completed; n = outcomes available"
    else:
        fig = _plot_configs(rows, horizon)
        what = f"Mean {horizon}-minute forward return per configuration; n = {horizon}-minute outcomes available"
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / height, title, color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.45 / height, f"{context}\n{what}", color=INK_2, fontsize=8, va="top", linespacing=1.5)
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
        label = (f"{r['segment']} ({r['role']}): {r['first_session']} to {r['last_session']}, "
                 f"{r['n_candidates']} candidates")
        ax.bar(x + off, np.nan_to_num(means), width, color=SERIES[i], label=label, zorder=3)
        for pos, h, m in zip(x + off, HORIZONS, means):
            _label_bar(ax, pos, m, r[f"n_{h}m"])
    ax.axhline(0, color=BASELINE, lw=1, zorder=2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(x, [f"{h} min" for h in HORIZONS])
    ax.set_ylabel("Mean forward return (%)", color=INK_2, fontsize=8)
    ax.margins(y=0.15)
    if legend:
        fig.legend(loc="upper left", bbox_to_anchor=(0.02, (bottom - 0.4) / height), frameon=False,
                   fontsize=8, labelcolor=INK_2)
    return fig


def _plot_configs(rows, horizon):
    labels = [f"{r['config']} [{r['segment']}]" if rows["segment"].nunique() > 1 else r["config"]
              for r in rows.to_dict("records")]
    height = 2.35 + 0.34 * len(rows)
    fig, ax = _canvas(height, left=2.3, bottom=1.0)
    y = np.arange(len(rows))[::-1]
    means = rows[f"mean_{horizon}m_pct"].to_numpy(float)
    ax.barh(y, np.nan_to_num(means), 0.55, color=SERIES[0], zorder=3)
    for pos, m, n in zip(y, means, rows[f"n_{horizon}m"]):
        _label_bar(ax, pos, m, n, horizontal=True)
    ax.axvline(0, color=BASELINE, lw=1, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"Mean {horizon}-minute forward return (%)", color=INK_2, fontsize=8)
    ax.margins(x=0.2)
    return fig
