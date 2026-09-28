"""Quick intraday signal study on Massive minute bars: single runs, small sweeps, optional split.

Pipeline: minute bars -> session-anchored bars + TA-Lib features -> vectorized trigger
mask -> rows and forward outcomes for triggered bars only -> summary, files, chart.

  uv run python run.py --start 2025-04-01 --end 2026-03-31
  uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30
  uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --split-date 2025-10-01
"""

import argparse
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import download
import features
import outcomes
import report
from strategies.spy_ema import DAILY_EMA_PERIOD, spy_ema

# CLI name -> plain strategy function. To add a strategy, write a function with the
# same shape (features, **params) -> (mask, keep) and list it here.
STRATEGIES = {"spy_ema": spy_ema}


@contextmanager
def timed(timings, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


def make_configs(fasts, slows, daily_filter):
    """Explicit fast x slow grid. Pairs with fast >= slow are rejected, not swapped."""
    configs, rejected = [], []
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                rejected.append(f"fast={fast} slow={slow}")
                continue
            label = f"fast={fast} slow={slow} daily_filter={'on' if daily_filter else 'off'}"
            configs.append({"label": label, "params": {"fast": fast, "slow": slow, "daily_filter": daily_filter}})
    return configs, rejected


def make_segments(sessions, split_date=None):
    """One in-sample segment, or an earlier (selection) and later (evaluation) segment.

    `end` is the last session close in the segment: no outcome may extend past it.
    """
    study = sessions[sessions["in_study"]]

    def segment(name, role, s):
        return {"name": name, "role": role, "first": s.index[0], "last": s.index[-1], "end": s["close"].iloc[-1]}

    if split_date is None:
        return [segment("in_sample", "exploratory", study)]
    split = pd.Timestamp(split_date)
    earlier, later = study[study.index < split], study[study.index >= split]
    if earlier.empty or later.empty:
        raise SystemExit(f"--split-date {split_date} must leave study sessions on both sides.")
    return [segment("earlier", "selection", earlier), segment("later", "evaluation", later)]


def build_candidate_rows(bars, idx, strategy, config, segment):
    """Feature snapshot rows for the triggered bars only (idx = positions in `bars`)."""
    b = bars.iloc[idx].reset_index(drop=True)
    rows = pd.DataFrame({"strategy": strategy, "config": config["label"], **config["params"],
                         "segment": segment["name"]}, index=b.index)
    rows["session"] = b["session"]
    rows["bar_start"] = b["bar_start"]
    rows["signal_time"] = b["bar_end"]  # trigger bar completion: when the signal is known
    rows["ref_close"] = b["close"]
    for name, values in config["keep"].items():
        rows[name] = values[idx]
    return rows


def empty_candidates(config):
    """A zero-trigger result: the expected columns and no rows."""
    utc = "datetime64[ns, UTC]"
    dtypes = {"strategy": "str", "config": "str", **{k: np.asarray(v).dtype for k, v in config["params"].items()},
              "segment": "str", "session": "datetime64[ns]", "bar_start": utc, "signal_time": utc,
              "ref_close": float, **{k: float for k in config["keep"]}, **{k: float for k in outcomes.OUTCOME_COLUMNS}}
    return pd.DataFrame({k: pd.Series(dtype=d) for k, d in dtypes.items()})


def evaluate_segment(segment, configs, bars, rth, strategy, timings):
    """Candidate rows + outcomes for each config within one segment, and their summaries.

    Outcomes are computed once for the union of triggered bars across configs, then
    shared: they depend only on the trigger bar, never on the configuration.
    """
    tables, summaries = [], []
    with timed(timings, "outcomes"):
        union = np.unique(np.concatenate([c["idx"][segment["name"]] for c in configs]))
        if union.size:
            shared = outcomes.forward_outcomes(
                rth, bars["bar_end"].iloc[union], bars["close"].iloc[union],
                bars["session_close"].iloc[union], segment["end"])
            shared.index = union
        for c in configs:
            idx = c["idx"][segment["name"]]
            if idx.size:
                rows = build_candidate_rows(bars, idx, strategy, c, segment)
                table = pd.concat([rows, shared.loc[idx].reset_index(drop=True)], axis=1)
            else:
                table = empty_candidates(c)
            tables.append(table)
    with timed(timings, "reporting"):
        for c, table in zip(configs, tables):
            summaries.append({"strategy": strategy, "config": c["label"], **c["params"], "segment": segment["name"],
                              "role": segment["role"], "first_session": segment["first"].date(),
                              "last_session": segment["last"].date(), **report.summarize(table)})
    return tables, summaries


def select_config(summaries, horizon, min_labeled):
    """Highest mean return at `horizon` among configs with >= min_labeled available outcomes, else None."""
    qualified = [s for s in summaries if s[f"n_{horizon}m"] >= max(min_labeled, 1)]
    return max(qualified, key=lambda s: s[f"mean_{horizon}m_pct"])["config"] if qualified else None


def run_study(minutes, sessions, *, strategy="spy_ema", fasts=(9,), slows=(21,), daily_filter=True,
              bar_minutes=5, min_coverage=0.8, split_date=None, min_labeled=30, select_horizon=30, timings=None):
    """Features -> triggers -> candidate outcomes -> summaries. No file I/O."""
    timings = {} if timings is None else timings
    configs, rejected = make_configs(fasts, slows, daily_filter)
    if not configs:
        raise SystemExit(f"No valid configurations: every pair has fast >= slow ({', '.join(rejected)}).")

    with timed(timings, "features"):
        periods = sorted({c["params"]["fast"] for c in configs} | {c["params"]["slow"] for c in configs})
        rth, bars, daily, info = features.build_features(minutes, sessions, bar_minutes, periods,
                                                         DAILY_EMA_PERIOD, min_coverage)
    segments = make_segments(sessions, split_date)

    with timed(timings, "triggers"):
        in_segment = {s["name"]: bars["session"].between(s["first"], s["last"]).to_numpy() for s in segments}
        for c in configs:
            mask, c["keep"] = STRATEGIES[strategy](bars, **c["params"])
            c["idx"] = {name: np.flatnonzero(mask & inside) for name, inside in in_segment.items()}

    tables, summaries = evaluate_segment(segments[0], configs, bars, rth, strategy, timings)
    selected = None
    if split_date is not None:
        selected = select_config(summaries, select_horizon, min_labeled)
        if selected is not None:
            chosen = [c for c in configs if c["label"] == selected]
            later_tables, later_summaries = evaluate_segment(segments[1], chosen, bars, rth, strategy, timings)
            tables += later_tables
            summaries += later_summaries

    non_empty = [t for t in tables if len(t)]
    candidates = pd.concat(non_empty, ignore_index=True) if non_empty else empty_candidates(configs[0])
    return {"candidates": candidates, "summary": pd.DataFrame(summaries), "selected": selected,
            "configs": [c["label"] for c in configs], "rejected": rejected, "segments": segments, "info": info}


def settings_for(args, sessions, source, results):
    study = sessions[sessions["in_study"]]
    info = results["info"]
    return {
        "ticker": args.ticker.upper(),
        "strategy": args.strategy,
        "study_sessions": [str(study.index[0].date()), str(study.index[-1].date()), len(study)],
        "data_source": source,
        "prices": ("synthetic random walk (demo only; no adjustment applies)" if source.startswith("SYNTHETIC") else
                   "Massive split-adjusted (adjusted=true); NOT dividend-adjusted, so returns are price returns"),
        "sessions": "XNYS regular hours only (exchange_calendars: holidays, early closes); timestamps UTC",
        "bar_minutes": args.bar_minutes,
        "min_bar_coverage": args.min_coverage,
        "warmup_sessions": args.warmup_sessions,
        "daily_ema_period": DAILY_EMA_PERIOD,
        "daily_features": "from the previous completed session, used for the whole current session",
        "configs": results["configs"],
        "rejected_configs_fast_ge_slow": results["rejected"],
        "reference_price": "close of the completed trigger bar (signal study, not a fill price)",
        "horizons_minutes": list(outcomes.HORIZONS),
        "excursion_minutes": outcomes.EXCURSION_MINUTES,
        "split_date": args.split_date,
        "selection": None if args.split_date is None else {
            "metric": f"mean_{args.select_horizon}m_pct on the earlier segment",
            "min_labeled_candidates": args.min_labeled, "selected": results["selected"]},
        "interpretation": ("exploratory / in-sample: no split date" if args.split_date is None else
                           "select on earlier segment, evaluate the pick once on the later segment; a later "
                           "period inspected repeatedly is not a pristine holdout"),
        "coverage": {k: info[k] for k in ("rth_minutes_expected", "rth_minutes_present", "bars_usable",
                                          "bars_dropped_low_coverage", "study_sessions_missing_daily")}
                    | {"sessions_without_data_count": len(info["sessions_without_data"]),
                       "sessions_partial_count": len(info["sessions_partial"]),
                       "first_20_sessions_without_data": info["sessions_without_data"][:20],
                       "first_20_sessions_partial": info["sessions_partial"][:20]},
    }


def print_report(args, sessions, source, results, timings, out_dir):
    info, summary, segs = results["info"], results["summary"], results["segments"]
    study = sessions[sessions["in_study"]]
    missing = info["rth_minutes_expected"] - info["rth_minutes_present"]
    ema_gaps = ", ".join(f"{k}: {v}" for k, v in info["study_bars_missing"].items())
    prices = "SYNTHETIC prices" if source.startswith("SYNTHETIC") else "split-adjusted, not dividend-adjusted"
    print(f"\n{args.ticker.upper()} · {args.strategy} · {study.index[0].date()} to {study.index[-1].date()} "
          f"({len(study)} sessions) · {args.bar_minutes}-min bars · regular hours · {prices}")
    print(f"data      {source}")
    print(f"          {info['outside_regular_hours_dropped']:,} bars outside regular hours ignored, "
          f"{info['duplicates_dropped']:,} duplicate timestamps dropped")
    print(f"coverage  {len(sessions)} sessions incl. {len(sessions) - len(study)} warm-up: {info['rth_minutes_present']:,} of "
          f"{info['rth_minutes_expected']:,} regular-session minutes present ({missing:,} missing); "
          f"sessions without data: {len(info['sessions_without_data'])}, partial: {len(info['sessions_partial'])}")
    if info["sessions_without_data"]:
        print(f"          no data: {', '.join(info['sessions_without_data'][:8])}"
              f"{' ...' if len(info['sessions_without_data']) > 8 else ''}")
    print(f"bars      {info['bars_usable']:,} usable (dropped {info['bars_dropped_low_coverage']} below "
          f"{args.min_coverage:.0%} minute coverage); study bars missing {ema_gaps}")
    print(f"daily     previous-session EMA{DAILY_EMA_PERIOD} unavailable for {info['study_sessions_missing_daily']} "
          f"of {len(study)} study sessions" + (" (daily filter off)" if not args.daily_filter else ""))
    if results["rejected"]:
        print(f"rejected  fast >= slow: {', '.join(results['rejected'])}")

    first = summary[summary["segment"] == segs[0]["name"]]
    if args.split_date is None:
        print(f"\nExploratory / in-sample (no --split-date): {len(first)} configuration(s)")
        print(report.format_table(first))
    else:
        print(f"\nSelection on earlier segment {segs[0]['first'].date()} to {segs[0]['last'].date()}: highest mean "
              f"{args.select_horizon}m return with >= {args.min_labeled} available {args.select_horizon}m outcomes")
        print(report.format_table(first))
        if results["selected"] is None:
            print(f"\nNo configuration qualified, so nothing was selected and the later segment "
                  f"({segs[1]['first'].date()} to {segs[1]['last'].date()}) was not evaluated.")
        else:
            print(f"\nSelected {results['selected']}; evaluated once on later segment "
                  f"{segs[1]['first'].date()} to {segs[1]['last'].date()}:")
            print(report.format_table(summary[summary["segment"] == "later"]))
            print("A later period you keep re-inspecting stops being a clean holdout.")
    print(f"\n{report.CAVEAT}")
    print("timings   " + " · ".join(f"{k} {timings.get(k, 0):.2f}s"
                                   for k in ("data", "features", "triggers", "outcomes", "reporting")))
    print(f"outputs   {out_dir}/: candidates.parquet ({len(results['candidates'])} rows), summary.csv, "
          "settings.json, chart.png")


def execute(args, minutes, sessions, source, timings):
    """Run the study on already-loaded minutes, write outputs, print the summary."""
    results = run_study(minutes, sessions, strategy=args.strategy, fasts=args.fast, slows=args.slow,
                        daily_filter=args.daily_filter, bar_minutes=args.bar_minutes, min_coverage=args.min_coverage,
                        split_date=args.split_date, min_labeled=args.min_labeled,
                        select_horizon=args.select_horizon, timings=timings)
    with timed(timings, "reporting"):
        # Chart one configuration when there is one to show; otherwise compare them all.
        configs, selected = results["configs"], results["selected"]
        chart_config = selected or (configs[0] if len(configs) == 1 else None)
        title = f"{args.ticker.upper()} · {args.strategy} · {chart_config or f'{len(configs)} configurations'}"
        if selected:
            context = "Selected on the earlier segment only, then evaluated once on the later segment."
        elif args.split_date:
            context = "Split date given but no configuration qualified: earlier (selection) segment only."
        else:
            context = "In-sample / exploratory: no split date."
        out = report.write_outputs(args.out, results["candidates"], results["summary"],
                                   settings_for(args, sessions, source, results), title, context,
                                   chart_config, args.select_horizon)
    print_report(args, sessions, source, results, timings, out)
    return results


def iso_date(text):
    return str(pd.Timestamp(text).date())  # ValueError -> argparse usage error


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quick intraday signal study on Massive minute bars "
                                            "(a signal study, not a backtest).")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="spy_ema")
    p.add_argument("--ticker", default="SPY", help="one US stock/ETF per run (default SPY)")
    p.add_argument("--start", type=iso_date, required=True, help="first study date, YYYY-MM-DD")
    p.add_argument("--end", type=iso_date, required=True, help="last study date, YYYY-MM-DD (inclusive)")
    p.add_argument("--bar-minutes", type=int, default=5, help="intraday bar size, anchored to each session open")
    p.add_argument("--fast", type=int, nargs="+", default=[9], help="fast EMA period(s); several values = sweep")
    p.add_argument("--slow", type=int, nargs="+", default=[21], help="slow EMA period(s); several values = sweep")
    p.add_argument("--daily-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="require previous session close > its daily EMA50 (default on)")
    p.add_argument("--split-date", type=iso_date, help="select on sessions before this date, evaluate the pick on sessions from it")
    p.add_argument("--min-labeled", type=int, default=30,
                   help="min candidates with an available selection-horizon outcome to qualify (default 30)")
    p.add_argument("--select-horizon", type=int, choices=outcomes.HORIZONS, default=30,
                   help="selection metric = mean forward return at this horizon (default 30)")
    p.add_argument("--warmup-sessions", type=int, default=120, help="sessions loaded before --start for indicators")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="min fraction of a bar's (or session's) minutes that must be present (default 0.8)")
    p.add_argument("--out", default="output/latest", help="output directory")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    args = p.parse_args(argv)
    if args.bar_minutes < 1 or args.warmup_sessions < 0 or not 0 < args.min_coverage <= 1:
        p.error("need --bar-minutes >= 1, --warmup-sessions >= 0 and 0 < --min-coverage <= 1")
    if min(args.fast + args.slow) < 2:
        p.error("EMA periods must be >= 2")
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
                                                    refresh=args.refresh, expect_through=last)
    execute(args, minutes, sessions, source, timings)


if __name__ == "__main__":
    main()
