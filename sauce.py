"""VWAP + Sauce on Massive minute bars: a small trade simulation, one ticker per run.

  uv run python sauce.py --start 2024-01-01 --end 2024-12-31
  uv run python sauce.py --ticker QQQ --start 2022-04-01 --end 2024-12-31 --compare
  uv run python sauce.py --start 2024-01-01 --end 2024-12-31 --continuation --vwap-to-vwap

run.py is a signal study: forward outcomes from each trigger close. This script instead
simulates the trades that the VWAP + Sauce rules describe (strategies/vwap_sauce.py):
entries at the next bar's open, moving band targets, stops, and one position at a time.
Results are gross and per share of the underlying: no options, costs or slippage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import download
import features
import outcomes
import report
import strategies.vwap_sauce as vs
from run import iso_date, timed

# --compare: the two open questions the brief asks to test, crossed (the base configuration first).
COMPARE = ({}, {"vwap_lookback_sessions": 4}, {"require_slow_parallel": False},
           {"vwap_lookback_sessions": 4, "require_slow_parallel": False})


def config_label(params):
    """'baseline' (the brief's defaults), or the settings that differ from them."""
    show = lambda v: ("on" if v else "off") if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else v  # noqa: E731
    defaults = {**vs.DEFAULTS, **vs.CHOICES}
    return " ".join(f"{k}={show(v)}" for k, v in params.items() if v != defaults[k]) or "baseline"


def make_configs(params, compare=False):
    variants = [{**params, **v} for v in COMPARE] if compare else [params]
    return [{"label": config_label(v), "params": {**vs.DEFAULTS, **vs.CHOICES, **v}} for v in variants]


def run_sauce(minutes, sessions, configs, min_coverage=0.5, timings=None):
    """Features -> one simulation per configuration -> setup log, summary and monthly counts. No file I/O."""
    timings = {} if timings is None else timings
    with timed(timings, "features"):
        rth, bars, _, info = features.build_features(minutes, sessions, vs.BAR_MINUTES,
                                                     (vs.FAST_PERIOD, vs.SLOW_PERIOD), min_coverage=min_coverage)
        lookbacks = sorted({c["params"]["vwap_lookback_sessions"] for c in configs})
        with_vwap = {n: vs.add_indicators(bars, sessions, n) for n in lookbacks}
    study = bars["in_study"].to_numpy(bool)
    info["study_bars_missing_vwap"] = {n: int(b.loc[study, "vwap"].isna().sum()) for n, b in with_vwap.items()}
    months = sessions.index[sessions["in_study"]].strftime("%Y-%m").nunique()
    logs, summary, monthly = [], [], []
    for c in configs:
        params = c["params"]
        with timed(timings, "simulate"):
            log = vs.simulate(with_vwap[params["vwap_lookback_sessions"]], **params)
        with timed(timings, "reporting"):
            enabled = {"A"} | ({"A_cont"} if params["enable_continuation"] else set()) | (
                {"B"} if params["enable_vwap_to_vwap"] else set())
            log.insert(0, "config", c["label"])
            summary += [{"config": c["label"], **r} for r in vs.summarize(log, months) if r["setup"] in enabled]
            m = vs.monthly(log)
            m.insert(0, "config", c["label"])
            logs.append(log)
            monthly.append(m)
    return {"log": pd.concat(logs, ignore_index=True), "summary": pd.DataFrame(summary),
            "monthly": pd.concat(monthly, ignore_index=True), "bars": with_vwap, "minutes": rth, "info": info,
            "months": months, "configs": configs}


BASELINE_GROUPS = ("Setup A, actual rules", "Setup A entries, fixed bracket", "random time, same day/side/bracket",
                   "same moment, opposite side")
BASELINE_DEFINITION = (
    "Is Setup A's entry better than chance? Each Setup A trade keeps its side and its target and stop distances in "
    "sigma at entry (target_price_at_entry, stop_price); trades whose target or stop was already beyond the entry are "
    "left out. Every entry below is exited the same way: outcomes.barrier_exits on one-minute bars with that fixed "
    "target and stop (first touch; a later open beyond a level fills at that open for the stop, at the level for the "
    "target; both in one minute = stop) or the session close. Random entries: bar opens drawn uniformly from the same "
    "session (never its first bar), same side, distances scaled by sigma at the previous bar's close. Opposite side: "
    "the same entry and distances, the other direction. Gross, one share.")


def baseline_entries(log, bars, minutes, draws, seed=0):
    """Setup A's entries and their matches, one row per entry: group, trade, entry_time, side, sigma, open, target, stop.

    Rows come trade by trade: the "setup" and "flipped" rows in trade order, then the
    "random" rows, `draws` per trade in trade order.
    """
    t = log[(log["setup"] == "A") & log["trade_no"].notna()]
    side = np.where(t["side"] == "long", 1, -1)
    entry, sig = t["entry_price"].to_numpy(float), t["sigma_at_entry"].to_numpy(float)
    reward = side * (t["target_price_at_entry"].to_numpy(float) - entry) / sig
    risk = side * (entry - t["stop_price"].to_numpy(float)) / sig
    ok = (reward > 0) & (risk > 0)
    t, side, entry, sig, reward, risk = t[ok], side[ok], entry[ok], sig[ok], reward[ok], risk[ok]
    n = len(t)
    # Random entries: `draws` bar opens per trade in the same session, never its first bar (sigma from the bar before).
    rng = np.random.default_rng(seed)
    sess = bars["session"].to_numpy()
    cuts = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1], True])
    span = {pd.Timestamp(sess[a]): (a, z) for a, z in zip(cuts[:-1], cuts[1:])}
    lo, hi = np.array([span[d] for d in t["session"]], dtype=int).reshape(-1, 2).T
    pos = rng.integers(lo[:, None] + 1, hi[:, None], size=(n, draws)).ravel()
    starts = pd.DatetimeIndex(bars["bar_start"].to_numpy()[pos])
    ts = vs._ns(minutes["ts"])
    k = np.minimum(np.searchsorted(ts, vs._ns(starts)), len(ts) - 1)
    r_open = np.where(ts[k] == vs._ns(starts), minutes["open"].to_numpy(float)[k], np.nan)  # NaN: no entry minute

    def frame(group, trade, times, sides, sigmas, opens, rewards, risks):
        return pd.DataFrame({"group": group, "trade": trade, "entry_time": pd.DatetimeIndex(times).as_unit("ns"),
                             "side": sides, "sigma": sigmas, "open": opens,
                             "target": opens + sides * rewards * sigmas, "stop": opens - sides * risks * sigmas})

    ids, rep = np.arange(n), (lambda a: np.repeat(a, draws))  # noqa: E731
    return pd.concat([frame("setup", ids, t["entry_time"], side, sig, entry, reward, risk),
                      frame("flipped", ids, t["entry_time"], -side, sig, entry, reward, risk),
                      frame("random", rep(ids), starts, rep(side), bars["sigma"].to_numpy(float)[pos - 1], r_open,
                            rep(reward), rep(risk))], ignore_index=True)


def entry_baseline(log, bars, minutes, end, draws, seed=0):
    """Setup A entries against matched random entries, every one exited by the same fixed bracket.

    Returns (rows, pct, band): one summary row per group in BASELINE_GROUPS; the share of
    2,000 random portfolios (one random draw per Setup A trade) whose average P&L in sigma
    is below Setup A's fixed-bracket average; and the 5th-95th percentile of those averages.
    """
    e = baseline_entries(log, bars, minutes, draws, seed)
    close = bars.groupby("session")["session_close"].first()
    day = e["entry_time"].dt.tz_convert(features.NY).dt.tz_localize(None).dt.normalize()
    x = outcomes.barrier_exits(minutes, e["entry_time"], e["side"], e["stop"], e["target"],
                               close.reindex(day).to_numpy(), end, cost_bps=(0,))
    done = ((x["entry_status"] == "ok") & x["exit_reason"].isin(["target", "stop", "stop_gap", "close"])).to_numpy()
    e["usd"] = np.where(done, e["side"] * (x["exit_price"] - x["entry_price"]), np.nan)
    e["pnl_sigma"], e["reason"] = e["usd"] / e["sigma"], x["exit_reason"].to_numpy()

    def row(group, g, reasons=True):
        done = g["usd"].notna()
        out = {"group": group, "trades": int(done.sum()), "win_%": (g["usd"][done] > 0).mean() * 100 if done.any()
               else np.nan, "avg_sigma": g["pnl_sigma"].mean(), "avg_usd": g["usd"].mean()}
        if reasons:
            out |= {f"%_{r}": g["reason"][done].isin(names).mean() * 100 if done.any() else np.nan
                    for r, names in (("target", ["target"]), ("stop", ["stop", "stop_gap"]), ("close", ["close"]))}
        return out

    t = log[(log["setup"] == "A") & log["trade_no"].notna()]
    kept = t.iloc[np.flatnonzero(np.isin(t["entry_time"], e.loc[e["group"] == "setup", "entry_time"]))]
    rows = [row(BASELINE_GROUPS[0], kept.rename(columns={"pnl_usd": "usd"}), reasons=False)]
    rows += [row(name, e[e["group"] == g]) for name, g in zip(BASELINE_GROUPS[1:], ("setup", "random", "flipped"))]
    n = int((e["group"] == "setup").sum())
    if not n:
        return rows, np.nan, (np.nan, np.nan)
    grid = e.loc[e["group"] == "random", "pnl_sigma"].to_numpy().reshape(n, draws)
    picks = grid[np.arange(n), np.random.default_rng(seed + 1).integers(draws, size=(2000, n))]
    with np.errstate(invalid="ignore"):
        averages = np.nanmean(picks, axis=1)
    setup_avg = e.loc[e["group"] == "setup", "pnl_sigma"].mean()
    return rows, (averages < setup_avg).mean() * 100, tuple(np.nanpercentile(averages, [5, 95]))


def format_baseline(result):
    rows, pct, (low, high) = result
    lines = [f"{'':<38}{'trades':>8}{'win %':>7}{'avg σ':>9}{'avg $':>9}{'target %':>10}{'stop %':>8}{'close %':>9}"]
    for r in rows:
        lines.append(f"{r['group']:<38}{r['trades']:>8}{_num(r['win_%'], '.0f'):>7}{_num(r['avg_sigma'], '+.3f'):>9}"
                     f"{_num(r['avg_usd'], '+.3f'):>9}" + "".join(f"{_num(r.get(f'%_{k}', np.nan), '.0f'):>{w}}"
                                                          for k, w in (("target", 10), ("stop", 8), ("close", 9))))
    if not np.isnan(pct):
        lines.append(f"Setup A's timing beat {pct:.0f}% of 2,000 random sets of the same size (50% = no better than "
                     f"random timing on the same days); 90% of random sets averaged {low:+.3f}σ to {high:+.3f}σ. "
                     "Beating random is not the same as making money: compare the averages.")
    return "\n".join(lines)


def settings_for(args, sessions, source, results):
    study = sessions[sessions["in_study"]]
    info = results["info"]
    return {
        "ticker": args.ticker.upper(),
        "tool": "VWAP + Sauce trade simulation (sauce.py, strategies/vwap_sauce.py)",
        "study_sessions": [str(study.index[0].date()), str(study.index[-1].date()), len(study)],
        "calendar_months": results["months"],
        "data_source": source,
        "prices": ("synthetic random walk (demo only; no adjustment applies)" if source.startswith("SYNTHETIC") else
                   "Massive split-adjusted (adjusted=true); NOT dividend-adjusted, so returns are price returns"),
        "sessions": "XNYS regular hours only (exchange_calendars: holidays, early closes); timestamps UTC",
        "bar_minutes": vs.BAR_MINUTES,
        "min_bar_coverage": args.min_coverage,
        "warmup_sessions": args.warmup_sessions,
        "indicators": {"vwap": "typical price (H+L+C)/3 x volume, anchored at the open of the session "
                               "vwap_lookback_sessions - 1 sessions back, re-anchored every session",
                       "sigma": "volume-weighted standard deviation of typical price around VWAP, same window",
                       "bands": "VWAP +/- 1, 1.5, 2 sigma (U1/U1.5/U2, L1/L1.5/L2)",
                       "fast": f"TA-Lib EMA{vs.FAST_PERIOD} of close", "slow": f"TA-Lib EMA{vs.SLOW_PERIOD} of close",
                       "continuity": "EMAs run over the usable bars of warm-up and study, never reset overnight"},
        "configs": {c["label"]: c["params"] for c in results["configs"]},
        "brief_assumption_defaults": vs.DEFAULTS,
        "switchable_choice_defaults": vs.CHOICES,
        "implementation_choices": vs.IMPLEMENTATION,
        "pnl": report.SAUCE_CAVEAT,
        "interpretation": "exploratory / in-sample: every configuration is fixed in advance and nothing is selected",
        "random_baseline": None if not args.random_baseline else {
            "draws_per_trade": args.random_baseline, "seed": 0, "definition": BASELINE_DEFINITION},
        "coverage": {k: info[k] for k in ("rth_minutes_expected", "rth_minutes_present", "bars_usable",
                                          "bars_dropped_low_coverage", "study_bars_missing", "study_bars_missing_vwap")}
                    | {"sessions_without_data_count": len(info["sessions_without_data"]),
                       "sessions_partial_count": len(info["sessions_partial"]),
                       "first_20_sessions_without_data": info["sessions_without_data"][:20],
                       "first_20_sessions_partial": info["sessions_partial"][:20]},
    }


def chart_dates(log, config, limit, requested, study):
    """Requested study sessions, then the first `limit` sessions (by time) with a Setup A setup or a trade."""
    rows = log[log["config"] == config]
    picked = rows.loc[(rows["setup"] == "A") | rows["trade_no"].notna(), "session"].drop_duplicates().sort_values()
    asked = [pd.Timestamp(d) for d in requested]
    for d in (d for d in asked if d not in study):
        print(f"note: --chart-dates {d.date()} is not a study session; skipped")
    return list(dict.fromkeys([d for d in asked if d in study] + list(picked.head(limit))))


def _num(value, fmt, missing="—"):
    return missing if pd.isna(value) else format(value, fmt)


def format_summary(summary):
    header = (f"{'setup':<18}{'side':<7}{'setups':>7}{'/month':>8}{'trades':>7}{'/month':>8}{'win %':>7}"
              f"{'avg win $':>10}{'avg loss $':>11}{'exp. $':>8}{'exp. σ':>8}{'total $':>9}{'max DD $':>9}"
              f"{'max DD σ':>9}{'invalid %':>10}")
    lines = [header]
    for r in summary.to_dict("records"):
        name = r["setup"] if r["subset"] == "all" else f"  {r['subset']}"
        invalid = r["pct_invalidated_slow_cross"] if r["setup"] == "A" else r["pct_head_fake"]
        lines.append(f"{name:<18}{r['side']:<7}{r['setups']:>7}{r['setups_per_month']:>8.2f}{r['trades']:>7}"
                     f"{r['trades_per_month']:>8.2f}{_num(r['win_rate'] * 100, '.0f'):>7}"
                     f"{_num(r['avg_win_usd'], '+.3f'):>10}{_num(r['avg_loss_usd'], '+.3f'):>11}"
                     f"{_num(r['expectancy_usd'], '+.3f'):>8}{_num(r['expectancy_sigma'], '+.3f'):>8}"
                     f"{_num(r['total_usd'], '+.2f'):>9}{_num(r['max_drawdown_usd'], '.2f'):>9}"
                     f"{_num(r['max_drawdown_sigma'], '.2f'):>9}{_num(invalid, '.0f'):>10}")
    return "\n".join(lines)


def format_details(summary):
    """Exit reasons, entry types and how the setups ended, for the both-sides rows."""
    lines = []
    for r in summary[(summary["side"] == "both") & (summary["subset"] == "all")].to_dict("records"):
        ended = {"A": f"invalidated by SLOW crossing {r['n_invalidated_slow_cross']}, FAST back without parallel "
                      f"{r['n_no_parallel']}, without trendline {r['n_no_trendline']}",
                 "B": f"head fakes {r['n_head_fake']}",
                 "A_cont": f"excursion over without a pullback-and-reject entry {r['n_excursion_ended']}"}[r["setup"]]
        lines.append(f"{r['setup']:<7}exits: target {r['n_exit_target']}, structure {r['n_exit_structure_stop']}, "
                     f"price stop {r['n_exit_price_stop']}, time {r['n_exit_time_stop']}"
                     + (f", end of data {r['n_exit_end_of_data']}" if r["n_exit_end_of_data"] else "")
                     + f" (ambiguous bars {r['n_ambiguous']}, exit at the entry open {r['n_exit_at_entry_open']}); "
                     f"entries: standard {r['n_entry_standard']}, fade {r['n_entry_fade']}, reentry "
                     f"{r['n_entry_reentry']}; median bars held {_num(r['median_bars_held'], '.0f')}")
        lines.append(f"{'':<7}setups ended: {ended}; blocked by an open position {r['n_blocked']}; still open at "
                     f"the session end {r['n_session_end']}")
    return "\n".join(lines)


def print_report(args, sessions, source, results, timings, out_dir, charts):
    info = results["info"]
    study = sessions[sessions["in_study"]]
    missing = info["rth_minutes_expected"] - info["rth_minutes_present"]
    prices = "SYNTHETIC prices" if source.startswith("SYNTHETIC") else "split-adjusted, not dividend-adjusted"
    print(f"\n{args.ticker.upper()} · VWAP + Sauce · {study.index[0].date()} to {study.index[-1].date()} "
          f"({len(study)} sessions, {results['months']} months) · {vs.BAR_MINUTES}-min bars · regular hours · {prices}")
    print(f"data      {source}")
    print(f"coverage  {len(sessions)} sessions incl. {len(sessions) - len(study)} warm-up: {info['rth_minutes_present']:,} "
          f"of {info['rth_minutes_expected']:,} regular-session minutes present ({missing:,} missing); sessions "
          f"without data: {len(info['sessions_without_data'])}, partial: {len(info['sessions_partial'])}")
    vwap_gaps = ", ".join(f"{n}-session VWAP {k}" for n, k in info["study_bars_missing_vwap"].items())
    print(f"bars      {info['bars_usable']:,} usable (dropped {info['bars_dropped_low_coverage']} with under "
          f"{args.min_coverage:.0%} of their minutes); study bars missing {vwap_gaps}, "
          + ", ".join(f"{k.replace('ema_', 'EMA')} {v}" for k, v in info["study_bars_missing"].items()))
    for c in results["configs"]:
        s = results["summary"][results["summary"]["config"] == c["label"]]
        print(f"\n{c['label']} (in-sample; fixed in advance, nothing selected)")
        print(format_summary(s))
        print(format_details(s))
        if c["label"] in results.get("baseline", {}):
            print(f"\nEntry check, {args.random_baseline} random entries per Setup A trade (gross; see "
                  "settings.json random_baseline)")
            print(format_baseline(results["baseline"][c["label"]]))
    print(f"\n{report.SAUCE_CAVEAT} Setup A is rare by design: the counts above are the honest frequency.")
    print("timings   " + " · ".join(f"{k} {timings.get(k, 0):.2f}s"
                                   for k in ("data", "features", "simulate", "baseline", "reporting")
                                   if k != "baseline" or args.random_baseline))
    print(f"outputs   {out_dir}/: setup_log.csv ({len(results['log'])} rows), summary.csv, monthly.csv, "
          f"{'entry_baseline.csv, ' if args.random_baseline else ''}settings.json, {len(charts)} session chart(s)")


def execute(args, minutes, sessions, source, timings):
    """Simulate on already-loaded minutes, write the outputs and print the summary."""
    configs = make_configs(args.params, args.compare)
    results = run_sauce(minutes, sessions, configs, args.min_coverage, timings)
    ticker = args.ticker.upper()
    if args.random_baseline:
        end = sessions.loc[sessions["in_study"], "close"].iloc[-1]
        with timed(timings, "baseline"):
            results["baseline"] = {
                c["label"]: entry_baseline(results["log"][results["log"]["config"] == c["label"]],
                                           results["bars"][c["params"]["vwap_lookback_sessions"]], results["minutes"],
                                           end, args.random_baseline) for c in configs}
    with timed(timings, "reporting"):
        for k in ("log", "summary", "monthly"):
            results[k].insert(0, "ticker", ticker)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("session_*.png"):  # a reused directory must not show an earlier run's charts
            stale.unlink()
        results["log"].to_csv(out / "setup_log.csv", index=False)
        results["summary"].to_csv(out / "summary.csv", index=False)
        results["monthly"].to_csv(out / "monthly.csv", index=False)
        stale = out / "entry_baseline.csv"
        if args.random_baseline:
            pd.DataFrame([{"ticker": ticker, "config": label, **r, "pct_random_sets_beaten": pct, "random_p5": lo,
                           "random_p95": hi} for label, (rows, pct, (lo, hi)) in results["baseline"].items()
                          for r in rows]).to_csv(stale, index=False)
        elif stale.exists():
            stale.unlink()
        (out / "settings.json").write_text(json.dumps(settings_for(args, sessions, source, results), indent=2,
                                                      default=str) + "\n")
        first = configs[0]
        dates = chart_dates(results["log"], first["label"], args.charts, args.chart_dates,
                            sessions.index[sessions["in_study"]])
        charts = report.plot_sauce_sessions(out, results["log"][results["log"]["config"] == first["label"]],
                                            results["bars"][first["params"]["vwap_lookback_sessions"]], sessions,
                                            ticker, first["label"], first["params"], dates)
    print_report(args, sessions, source, results, timings, out, charts)
    return results


def parse_args(argv=None):
    d = vs.DEFAULTS
    on_off = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="VWAP + Sauce: simulated trades on 2-minute bars (gross, per share of "
                                            "the underlying, not options P&L). Defaults are the brief's assumptions.")
    p.add_argument("--ticker", default="SPY", help="one US stock/ETF per run (default SPY)")
    p.add_argument("--start", type=iso_date, required=True, help="first study date, YYYY-MM-DD")
    p.add_argument("--end", type=iso_date, required=True, help="last study date, YYYY-MM-DD (inclusive)")
    p.add_argument("--instrument", choices=["underlying"], default=d["instrument"],
                   help="simulated instrument (only the underlying for now)")
    p.add_argument("--vwap-lookback-sessions", type=int, default=d["vwap_lookback_sessions"],
                   help="sessions in the anchored VWAP window, today included (default 3; the brief also names 4)")
    p.add_argument("--parallel-distance-sigma", type=float, default=d["parallel_distance_sigma"],
                   help="SLOW within this many sigma of the 2-sigma band counts as near (default 0.25)")
    p.add_argument("--parallel-lookback", type=int, default=d["parallel_lookback"],
                   help="bars in the SLOW-to-band slope (default 5)")
    p.add_argument("--slope-threshold", type=float, default=d["slope_threshold"],
                   help="|slope| below this many sigma per bar counts as flat (default 0.02)")
    p.add_argument("--require-slow-parallel", action=on_off, default=d["require_slow_parallel"],
                   help="Setup A needs SLOW to go parallel before entry (default on)")
    p.add_argument("--trendline-confirm", dest="use_trendline_confirm", action=on_off,
                   default=d["use_trendline_confirm"], help="also need price to break a line fitted to SLOW (default off)")
    p.add_argument("--trendline-lookback", type=int, default=d["trendline_lookback"],
                   help="SLOW values in that line (default 10)")
    p.add_argument("--fade-entry", dest="allow_fade_entry", action=on_off, default=d["allow_fade_entry"],
                   help="enter once SLOW goes parallel, before FAST crosses back, if the day is green (default off)")
    p.add_argument("--target-level", type=float, default=d["target_level"],
                   help="Setup A target band in sigma (default 1.5; 1 = L1/U1, 0 = VWAP)")
    p.add_argument("--extended-target", dest="allow_extended_target", action=on_off,
                   default=d["allow_extended_target"], help="target the 1-sigma band instead (default off)")
    p.add_argument("--structure-stop", action=on_off, default=d["structure_stop"],
                   help="exit when FAST closes back outside the 2-sigma band (default on)")
    p.add_argument("--price-stop", action=on_off, default=d["price_stop"],
                   help="exit when price trades beyond setup_extreme (default on)")
    p.add_argument("--time-stop", action=on_off, default=d["time_stop"],
                   help="flat at each session's last bar (default on; off = overnight holds)")
    p.add_argument("--reentry", dest="allow_reentry", action=on_off, default=d["allow_reentry"],
                   help="re-enter on a pullback to FAST after a structure stop (default off)")
    p.add_argument("--max-reentries", type=int, default=d["max_reentries"], help="per setup (default 2)")
    p.add_argument("--continuation", dest="enable_continuation", action=on_off, default=d["enable_continuation"],
                   help="also trade A-continuation while FAST is outside the band (default off)")
    p.add_argument("--vwap-to-vwap", dest="enable_vwap_to_vwap", action=on_off, default=d["enable_vwap_to_vwap"],
                   help="also trade Setup B, VWAP to VWAP (default off)")
    p.add_argument("--latch-slow-parallel", action=on_off, default=vs.CHOICES["latch_slow_parallel"],
                   help="not from the brief: once SLOW goes parallel it counts until entry (default on); off = SLOW "
                        "must be parallel on the entry bar itself")
    p.add_argument("--ambiguous-fill", choices=["stop", "target"], default=vs.CHOICES["ambiguous_fill"],
                   help="not from the brief: a bar touching both the price stop and the target counts as this "
                        "(default stop)")
    p.add_argument("--stop-buffer-sigma", type=float, default=vs.CHOICES["stop_buffer_sigma"],
                   help="not from the brief: move the price stop this many sigma further away (default 0)")
    p.add_argument("--random-baseline", type=int, default=0, metavar="N",
                   help="also compare Setup A's entries with N random entries per trade (same session, side and "
                        "stop/target distances; fixed-bracket exits); e.g. 200")
    p.add_argument("--compare", action="store_true",
                   help="run four fixed variants on shared data: VWAP lookback 3 and 4 x slow-parallel required "
                        "on and off (other options apply to all four)")
    p.add_argument("--warmup-sessions", type=int, default=120, help="sessions loaded before --start (default 120)")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="min fraction of a 2-minute bar's minutes present (default 0.5 = one traded minute)")
    p.add_argument("--charts", type=int, default=12,
                   help="chart the first N sessions with a Setup A setup or a trade (default 12)")
    p.add_argument("--chart-dates", type=iso_date, nargs="+", default=[], help="also chart these sessions")
    p.add_argument("--out", default="output/sauce", help="output directory")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    args = p.parse_args(argv)
    args.params = {k: getattr(args, k) for k in (*d, *vs.CHOICES) if hasattr(args, k)}
    try:
        vs.validate({**d, **vs.CHOICES, **args.params})
    except ValueError as e:
        p.error(str(e))
    if args.warmup_sessions < 0 or not 0 < args.min_coverage <= 1 or args.charts < 0 or args.random_baseline < 0:
        p.error("need --warmup-sessions >= 0, 0 < --min-coverage <= 1, --charts >= 0 and --random-baseline >= 0")
    if args.compare and (args.vwap_lookback_sessions != d["vwap_lookback_sessions"]
                         or args.require_slow_parallel != d["require_slow_parallel"]):
        p.error("--compare sets the VWAP lookback and slow-parallel requirement itself; leave those at defaults")
    return args


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    with timed(timings, "data"):
        sessions = features.trading_sessions(args.start, args.end, args.warmup_sessions)
        # Never study or cache today's session: it may be open or its data still incomplete.
        today = sessions.index >= pd.Timestamp.now(tz=features.NY).tz_localize(None).normalize()
        if today.any():
            print(f"note: skipping {int(today.sum())} session(s) from today onward (data may be incomplete)")
            sessions = sessions[~today]
        if not sessions["in_study"].any():
            raise SystemExit("No completed sessions in the requested study range.")
        first, last = sessions.index[0].date(), sessions.index[-1].date()
        minutes, source = download.load_minute_bars(args.ticker, str(first), str(last), cache_dir=args.cache_dir,
                                                    refresh=args.refresh, final_close=sessions["close"].iloc[-1])
    execute(args, minutes, sessions, source, timings)


if __name__ == "__main__":
    main()
