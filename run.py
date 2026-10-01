"""Quick intraday signal study on Massive minute bars: single runs, small sweeps, optional split.

Pipeline: minute bars -> session-anchored bars + TA-Lib features -> vectorized trigger
mask -> rows and forward outcomes for triggered bars only -> summary, files, chart.

  uv run python run.py --start 2025-04-01 --end 2026-03-31
  uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30
  uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --split-date 2025-10-01
  uv run python run.py --strategy opening_range_reversal --start 2024-01-01 --end 2024-12-31 --barriers
  uv run python run.py --strategy auction_reclaim --start 2024-01-01 --end 2024-12-31 --barriers
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
import strategies.auction_reclaim as ar
import strategies.opening_range_reversal as orr
import volume_profile
from strategies.spy_ema import DAILY_EMA_PERIOD, spy_ema

ORR = "opening_range_reversal"
AR = "auction_reclaim"
# CLI name -> plain strategy function. To add a strategy, write a function with the
# same shape (features, **params) -> (mask, keep) and list it here. auction_reclaim is a
# small state machine instead: (features, minutes, day, **params) -> (mask, triggers, funnel).
STRATEGIES = {"spy_ema": spy_ema, ORR: orr.opening_range_reversal, AR: ar.auction_reclaim}


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
    if "triggers" in config:  # per-trigger state the strategy recorded when each signal fired
        rows = pd.concat([rows, config["triggers"].loc[idx].reset_index(drop=True)], axis=1)
    return rows


def empty_candidates(config, barriers=False):
    """A zero-trigger result: the expected columns and no rows."""
    utc = "datetime64[ns, UTC]"
    dtypes = {"strategy": "str", "config": "str", **{k: np.asarray(v).dtype for k, v in config["params"].items()},
              "segment": "str", "session": "datetime64[ns]", "bar_start": utc, "signal_time": utc, "ref_close": float}
    columns = {k: pd.Series(dtype=d) for k, d in dtypes.items()}
    columns |= {k: pd.Series(v[:0]) for k, v in config["keep"].items()}
    columns |= {k: pd.Series(dtype=d) for k, d in (config["triggers"].dtypes.items() if "triggers" in config else ())}
    columns |= {k: pd.Series(dtype=float) for k in outcomes.OUTCOME_COLUMNS}
    if barriers:
        columns |= {k: pd.Series(dtype=d) for k, d in outcomes.BARRIER_COLUMNS.items()}
    return pd.DataFrame(columns)


def evaluate_segment(segment, configs, bars, rth, strategy, timings, opening=None, barriers=False):
    """Candidate rows + outcomes for each config within one segment, and their summaries.

    Forward outcomes are computed once for the union of triggered bars across configs,
    then shared: they depend only on the trigger bar and its side, never on the
    configuration. The opening-range reversal's barrier levels are shared the same way;
    auction_reclaim's frozen stop/target differ by variant, so its barrier paths are per
    configuration. `opening` (per-session table) switches the summary to one row per side
    and pattern with session counts; auction_reclaim uses one row per side with its setup
    funnel.
    """
    is_ar = strategy == AR
    tables, summaries = [], []
    with timed(timings, "outcomes"):
        union = np.unique(np.concatenate([c["idx"][segment["name"]] for c in configs]))
        if union.size:
            at = bars.iloc[union]
            if is_ar:  # a bar can trigger only one side (long needs close > open, short close < open)
                side = pd.concat([c["triggers"]["side_sign"] for c in configs]).groupby(level=0).first()
                side = side.loc[union].to_numpy()
            else:
                side = at["side_sign"].to_numpy() if "side_sign" in bars else None
            shared = outcomes.forward_outcomes(rth, at["bar_end"], at["close"], at["session_close"], segment["end"], side)
            if barriers and not is_ar:
                stop, target = orr.barrier_levels(bars, union)
                exits = outcomes.barrier_exits(rth, at["bar_end"], side, stop, target, at["session_close"],
                                               segment["end"])
                shared = pd.concat([shared, exits], axis=1)
            shared.index = union
        for c in configs:
            idx = c["idx"][segment["name"]]
            if idx.size:
                rows = build_candidate_rows(bars, idx, strategy, c, segment)
                parts = [rows, shared.loc[idx].reset_index(drop=True)]
                if barriers and is_ar:
                    parts.append(outcomes.barrier_exits(
                        rth, rows["signal_time"], rows["side_sign"], rows["frozen_stop"], rows["frozen_target"],
                        bars["session_close"].iloc[idx], segment["end"],
                        min_reward_risk=ar.RULES[c["params"]["rules"]]["min_reward_risk"]))
                table = pd.concat(parts, axis=1)
            else:
                table = empty_candidates(c, barriers)
            tables.append(table)
    with timed(timings, "reporting"):
        for c, table in zip(configs, tables):
            base = {"strategy": strategy, "config": c["label"], **c["params"], "segment": segment["name"],
                    "role": segment["role"], "first_session": segment["first"].date(),
                    "last_session": segment["last"].date()}
            if is_ar:
                counts = ar.funnel_counts(c["funnel"].loc[segment["first"]:segment["last"]])
                summaries += [{**base, **row} for row in report.summarize_sides(table, counts, ar.PATTERNS, barriers)]
            elif opening is None:
                summaries.append({**base, **report.summarize(table)})
            else:
                counts = orr.opening_counts(opening.loc[segment["first"]:segment["last"]], c["params"]["threshold"])
                summaries += [{**base, **row} for row in report.summarize_sides(table, counts, orr.PATTERNS, barriers)]
    return tables, summaries


def select_config(summaries, horizon, min_labeled):
    """Highest mean return at `horizon` among configs with >= min_labeled available outcomes, else None."""
    qualified = [s for s in summaries if s[f"n_{horizon}m"] >= max(min_labeled, 1)]
    return max(qualified, key=lambda s: s[f"mean_{horizon}m_pct"])["config"] if qualified else None


def run_study(minutes, sessions, *, strategy="spy_ema", fasts=(9,), slows=(21,), daily_filter=True,
              thresholds=(orr.BASELINE_THRESHOLD,), barriers=False, bar_minutes=5, min_coverage=0.8,
              split_date=None, min_labeled=30, select_horizon=30, location=ar.BASELINE["location"],
              profile_bins=ar.BASELINE["profile_bins"], rvol_filter=True, compare=False, rules="baseline",
              timings=None):
    """Features -> triggers -> candidate outcomes -> summaries. No file I/O.

    spy_ema uses fasts/slows/daily_filter; opening_range_reversal uses thresholds and
    barriers, with 5-minute bars and one summary row per side and pattern. auction_reclaim
    uses rules/location/profile_bins/rvol_filter (or the five fixed `compare` variants of one
    rule set) and barriers, with one summary row per side; nothing is selected after a split.
    """
    timings = {} if timings is None else timings
    is_orr, is_ar = strategy == ORR, strategy == AR
    if is_orr:
        configs, rejected = orr.make_configs(thresholds), []
    elif is_ar:
        configs, rejected = ar.make_configs(location, profile_bins, rvol_filter, compare, rules), []
    else:
        configs, rejected = make_configs(fasts, slows, daily_filter)
        if not configs:
            raise SystemExit(f"No valid configurations: every pair has fast >= slow ({', '.join(rejected)}).")

    opening = day = None
    with timed(timings, "features"):
        periods = ([] if is_orr or is_ar else
                   sorted({c["params"]["fast"] for c in configs} | {c["params"]["slow"] for c in configs}))
        rth, bars, daily, info = features.build_features(minutes, sessions, bar_minutes, periods,
                                                         DAILY_EMA_PERIOD, min_coverage, orr.ATR_PERIOD)
        if is_orr:
            bars, opening = orr.add_features(bars, rth, daily, sessions)
        elif is_ar:  # computed once and shared by every variant
            bars, rth, day = ar.add_features(bars, rth, daily, sessions, [c["params"]["profile_bins"] for c in configs])
            study_minutes = rth[rth["session"].isin(sessions.index[sessions["in_study"]])]
            approx = ~study_minutes.groupby("session")["vwap_vendor_ok"].min().astype(bool)
            info.update(vwap_approx_sessions=[str(d.date()) for d in approx.index[approx]],
                        profile_status=day.loc[day["in_study"], "profile_status"].value_counts().to_dict())
    segments = make_segments(sessions, split_date)
    if is_ar and split_date is not None:
        segments[0]["role"] = "comparison"  # compared, never used to pick a winner

    with timed(timings, "triggers"):
        in_segment = {s["name"]: bars["session"].between(s["first"], s["last"]).to_numpy() for s in segments}
        for c in configs:
            if is_ar:
                mask, c["triggers"], c["funnel"] = ar.auction_reclaim(bars, rth, day, **c["params"])
                c["keep"] = {}
            else:
                mask, c["keep"] = STRATEGIES[strategy](bars, **c["params"])
            c["idx"] = {name: np.flatnonzero(mask & inside) for name, inside in in_segment.items()}

    tables, summaries = evaluate_segment(segments[0], configs, bars, rth, strategy, timings, opening, barriers)
    selected = None
    if split_date is not None and is_orr:
        # Long and short stay separate: each side picks its own threshold from its own earlier rows.
        selected = {side: select_config([r for r in summaries if r["side"] == side and r["pattern"] == "all"],
                                        select_horizon, min_labeled) for side in orr.SIDES}
        chosen = [c for c in configs if c["label"] in selected.values()]
        if chosen:
            later_tables, later_summaries = evaluate_segment(segments[1], chosen, bars, rth, strategy, timings,
                                                             opening, barriers)
            tables += [t[t["side"].map(selected).eq(t["config"])] for t in later_tables]
            summaries += [r for r in later_summaries if selected[r["side"]] == r["config"]]
    elif split_date is not None and is_ar:
        # No selection: the variants are compared on the earlier segment only, and the configuration fixed
        # in advance (the baseline, or the single one requested) is the only one evaluated on the later segment.
        selected = configs[0]["label"]
        later_tables, later_summaries = evaluate_segment(segments[1], configs[:1], bars, rth, strategy, timings,
                                                         barriers=barriers)
        tables += later_tables
        summaries += later_summaries
    elif split_date is not None:
        selected = select_config(summaries, select_horizon, min_labeled)
        if selected is not None:
            chosen = [c for c in configs if c["label"] == selected]
            later_tables, later_summaries = evaluate_segment(segments[1], chosen, bars, rth, strategy, timings)
            tables += later_tables
            summaries += later_summaries

    non_empty = [t for t in tables if len(t)]
    candidates = pd.concat(non_empty, ignore_index=True) if non_empty else empty_candidates(configs[0], barriers)
    return {"candidates": candidates, "summary": pd.DataFrame(summaries), "selected": selected,
            "configs": [c["label"] for c in configs], "rejected": rejected, "segments": segments, "info": info,
            "bars": bars, "minutes": rth, "day": day}


ORR_DEFINITION = {
    "adaptation": "deterministic research adaptation of https://www.youtube.com/watch?v=XFtayhPIdEs, not a replay; "
                  "numeric definitions, completed-candle signal, first-signal limit and fixed exits are our assumptions",
    "opening_range": f"first {orr.OPENING_MINUTES} regular-session minutes, all present; known at open+{orr.OPENING_MINUTES}",
    "size_gate": f"opening high - low >= threshold x daily ATR({orr.ATR_PERIOD}) from completed regular-session "
                 "daily bars through the previous session",
    "direction": "opening close < open: long only; > open: short only; equal: skipped",
    "signal_bars": f"complete {orr.BAR_MINUTES}-minute bars anchored to the open, starting at/after open+"
                   f"{orr.OPENING_MINUTES} and completing strictly before open+{orr.WINDOW_MINUTES}",
    "patterns": "TA-Lib (installed version): long CDLHAMMER > 0 or CDLENGULFING > 0; short CDLSHOOTINGSTAR < 0 or "
                "CDLENGULFING < 0; previous bar complete, contiguous and same-session",
    "outside": "long: low and close strictly below opening low; short: high and close strictly above opening high",
    "signal_limit": "earliest qualifying bar per ticker/session/threshold",
    "filters": "none beyond the above (no trend, volume, RSI or EMA filter)",
    "outcomes": "gross directional signal response from the signal bar close (side x raw return); not strategy P&L",
}
ORR_BARRIERS = {
    "entry": "open of the minute starting at signal completion (idealized bar-based fill)",
    "stop": "signal low (long) / high (short); engulfing: extreme across both candles",
    "target": "opening high (long) / opening low (short)",
    "exit": "first touch in time order; later open beyond stop exits at that open; beyond target exits at target; "
            "both in one minute = ambiguous, stop first; otherwise regular-session close; gaps in data = unresolved",
    "friction_bps_per_side": list(outcomes.COST_BPS),
    "friction_note": "illustrative sensitivity, not calibrated costs; excludes borrow costs",
}
AR_DEFINITION = {
    "inspiration": "failed-auction reversion model in Fabio Valentini's published Auction Market playbook "
                   "(https://www.chartfanatics.com/strategies/auction-market-strategy); our own deterministic "
                   "approximation, not his method, and no claim about his performance",
    "thresholds": "every numeric threshold is a research default, not a verified rule of his",
    "order_flow": "NOT observed: relative volume is total-volume intensity and the candle filter is a shape rule; "
                  "neither measures aggressive buyers/sellers, absorption or cumulative delta",
    "profile": f"{volume_profile.PROFILE_METHOD} (not traded volume at price): the previous XNYS session's complete "
               "regular-session minutes (all expected minutes present), equal bins from its low to its high, each "
               "minute's volume spread uniformly over its [low, high]; POC = highest bin's center (ties: lower); "
               f"{volume_profile.VALUE_AREA_FRACTION:.0%} value area grown from POC by the larger adjacent bin (ties: "
               "lower first); frozen for the session; an unavailable profile is never replaced by an older session",
    "atr": f"A = TA-Lib ATR({ar.ATR_PERIOD}) of completed regular-session daily bars through the previous session; "
           f"b = {ar.BUFFER_ATR} A, d = {ar.RETEST_ATR} A; A missing or <= 0 skips the session",
    "vwap": "session VWAP reset at each open: cumulative Massive minute vwap x volume while every positive-volume "
            "minute so far has it; otherwise HLC3 x volume over the whole observed prefix, labeled "
            f"{ar.APPROX_VWAP}; read at the signal bar's completion and exactly 15 minutes earlier with one method",
    "relative_volume": f"5-minute bar volume / median volume of the same session-relative slot over the "
                       f"{ar.RVOL_SESSIONS} preceding sessions (today excluded; >= {ar.RVOL_MIN_OBS} complete "
                       "observations; positive median); total-volume intensity, not signed flow",
    "balance": f">= {ar.BALANCE_MIN_INSIDE} of the {ar.BALANCE_BARS} complete consecutive same-session bars before "
               "the excursion close inside [VAL, VAH]",
    "excursion": "long: close < VAL - b after a close that was not (fresh break); short: close > VAH + b likewise; "
                 "the extreme is tracked from the excursion bar through the reclaim bar",
    "reclaim": f"first bar within the next {ar.RECLAIM_BARS}: long close in (VAL + b, POC) and close > open; short "
               "close in (POC, VAH - b) and close < open; POC reached first expires the setup; no signal on it",
    "retest": "first bar within the rule set's retest window after the reclaim overlapping [edge - d, edge + d] "
              "(or the frozen reclaim LVN), closing back inside (VAL + b, POC) / (POC, VAH - b), passing every filter",
    "invalidation": "checked first on every bar after the reclaim, including the trigger bar: extreme breached, POC "
                    "touched, or close beyond the excursion threshold",
    "confirmation": "direction of the trade, |close - open| / range >= body_min, (close - low) / range (long) or "
                    f"(high - close) / range (short) >= close_location_min, relative volume >= {ar.RVOL_MIN} "
                    "(unless disabled); zero-range candles and unavailable inputs skip",
    "vwap_filter": f"long: close < VWAP and VWAP - VWAP 15 min earlier >= -{ar.VWAP_SLOPE_ATR} A; short: close > "
                   f"VWAP and that change <= {ar.VWAP_SLOPE_ATR} A (experimental)",
    "levels": f"stop = excursion extreme -/+ {ar.STOP_ATR} A; target = previous-session POC; positive risk and "
              "reward with reward/risk >= min_reward_risk at the signal close; frozen at emission",
    "signal_window": "signal bars complete from 09:50 ET through the rule set's last time, inclusive; one candidate "
                     "per side per session regardless of outcome; one active setup at a time; missing/incomplete "
                     "bar, invalidation or expiry resets and evaluation resumes on the following bar",
    "rules": {name: {"signal_bars_complete_et": f"09:50-{pd.Timestamp('09:30') + r['signal_last']:%H:%M}",
                     **{k: v for k, v in r.items() if k != "signal_last"}} for name, r in ar.RULES.items()},
    "loose_rules_note": "fixed after seeing only 2024 signal counts under candidate relaxations, never outcomes",
    "location": f"value_edge: the band [edge - d, edge + d]; reclaim_lvn: at the reclaim only, a {ar.LVN_BINS}-bin "
                "bar-approximated profile of the minutes from the excursion-extreme bar through the reclaim; LVN = "
                "positive volume, [1,2,1]/4-smoothed volume below both neighbors and <= half the smaller of the "
                "largest smoothed volumes on each side (first/last two bins excluded), center inside prior value "
                "and behind the reclaim close; nearest the reclaimed edge (ties lower); no node = no setup",
    "outcomes": "gross directional signal response from the signal bar close (side x raw return); not P&L",
}
AR_BARRIERS = {
    "entry": "open of the minute starting at signal completion (idealized; never the retest low/high)",
    "levels": "frozen stop and target from the signal; entry_status invalid when the actual entry breaks the "
              "geometry or gives reward/risk below the rule set's min_reward_risk (the candidate is kept, nothing "
              "replaces it)",
    "exit": "stop, target or regular-session close; first touch in time order; later open beyond stop exits at that "
            "open; beyond target exits at target; both in one minute = ambiguous, stop first; gaps = unresolved; no "
            "stop moves, trailing, scaling or compounding",
    "friction_bps_per_side": list(outcomes.COST_BPS),
    "friction_note": "illustrative sensitivity, not calibrated costs; short borrow feasibility and cost not modeled",
}


def settings_for(args, sessions, source, results):
    study = sessions[sessions["in_study"]]
    info = results["info"]
    is_orr, is_ar = args.strategy == ORR, args.strategy == AR
    if is_orr:
        strategy = {"thresholds": args.threshold, "baseline_threshold": orr.BASELINE_THRESHOLD,
                    "definition": ORR_DEFINITION, "barriers": ORR_BARRIERS if args.barriers else None}
    elif is_ar:
        strategy = {"variants": {c["label"]: c["params"] for c in ar.make_configs(
                        args.location, args.profile_bins, args.rvol_filter, args.compare, args.rules)},
                    "baseline": ar.BASELINE, "profile_method": volume_profile.PROFILE_METHOD,
                    "definition": AR_DEFINITION, "barriers": AR_BARRIERS if args.barriers else None,
                    "profile_status_study_sessions": info["profile_status"],
                    "vwap_approximation_sessions": info["vwap_approx_sessions"]}
    else:
        strategy = {"daily_ema_period": DAILY_EMA_PERIOD}
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
        **strategy,
        "daily_features": "from the previous completed session, used for the whole current session",
        "configs": results["configs"],
        "rejected_configs_fast_ge_slow": results["rejected"],
        "reference_price": "close of the completed trigger bar (signal study, not a fill price)",
        "horizons_minutes": list(outcomes.HORIZONS),
        "excursion_minutes": outcomes.EXCURSION_MINUTES,
        "split_date": args.split_date,
        "selection": None if args.split_date is None else {
            "metric": "none: variants are compared on the earlier segment only, and the configuration fixed in "
                      "advance is the one evaluated on the later segment" if is_ar else
                      f"mean_{args.select_horizon}m_pct on the earlier segment"
                      + (", chosen separately for long and short" if is_orr else ""),
            "min_labeled_candidates": None if is_ar else args.min_labeled,
            ("evaluated_later" if is_ar else "selected"): results["selected"]},
        "interpretation": ("exploratory / in-sample: no split date" if args.split_date is None else
                           "earlier segment: diagnostic comparison; later segment: the fixed configuration only; a "
                           "later period inspected repeatedly is not a pristine holdout" if is_ar else
                           "select on earlier segment, evaluate the pick once on the later segment; a later "
                           "period inspected repeatedly is not a pristine holdout"),
        "coverage": {k: info[k] for k in ("rth_minutes_expected", "rth_minutes_present", "bars_usable",
                                          "bars_dropped_low_coverage", "study_sessions_missing_daily",
                                          "study_sessions_missing_atr")}
                    | {"sessions_without_data_count": len(info["sessions_without_data"]),
                       "sessions_partial_count": len(info["sessions_partial"]),
                       "first_20_sessions_without_data": info["sessions_without_data"][:20],
                       "first_20_sessions_partial": info["sessions_partial"][:20]},
    }


def print_report(args, sessions, source, results, timings, out_dir):
    info = results["info"]
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
    if args.strategy == ORR:
        print(f"bars      {info['bars_usable']:,} usable (dropped {info['bars_dropped_low_coverage']} below "
              f"{args.min_coverage:.0%} minute coverage); signal bars, their previous bar and the opening range "
              "must be complete")
        print(f"daily     prior-session ATR{orr.ATR_PERIOD} unavailable for {info['study_sessions_missing_atr']} "
              f"of {len(study)} study sessions")
        print_orr_report(args, results)
    elif args.strategy == AR:
        status = ", ".join(f"{k} {v}" for k, v in sorted(info["profile_status"].items()))
        approx = info["vwap_approx_sessions"]
        print(f"bars      {info['bars_usable']:,} usable (dropped {info['bars_dropped_low_coverage']} below "
              f"{args.min_coverage:.0%} minute coverage); a setup needs every 5-minute bar complete (5 of 5 minutes)")
        print(f"daily     prior-session ATR{ar.ATR_PERIOD} unavailable for {info['study_sessions_missing_atr']} "
              f"of {len(study)} study sessions")
        print(f"profile   previous-session {volume_profile.PROFILE_METHOD} (minute volume spread over each minute's "
              f"range; not traded volume at price): {status}")
        print(f"vwap      Massive minute vwap, except {ar.APPROX_VWAP} (from the first minute lacking it) in "
              f"{len(approx)} of {len(study)} study sessions"
              + (f": {', '.join(approx[:8])}{' ...' if len(approx) > 8 else ''}" if approx else ""))
        print_ar_report(args, results)
    else:
        print(f"bars      {info['bars_usable']:,} usable (dropped {info['bars_dropped_low_coverage']} below "
              f"{args.min_coverage:.0%} minute coverage); study bars missing {ema_gaps}")
        print(f"daily     previous-session EMA{DAILY_EMA_PERIOD} unavailable for {info['study_sessions_missing_daily']} "
              f"of {len(study)} study sessions" + (" (daily filter off)" if not args.daily_filter else ""))
        if results["rejected"]:
            print(f"rejected  fast >= slow: {', '.join(results['rejected'])}")
        print_ema_report(args, results)
    print(f"\n{report.CAVEAT}")
    print("timings   " + " · ".join(f"{k} {timings.get(k, 0):.2f}s"
                                   for k in ("data", "features", "triggers", "outcomes", "reporting")))
    charts = f", {len(results.get('candidate_charts', []))} candidate chart(s)" if args.strategy in (ORR, AR) else ""
    stability = ", stability.csv" if args.strategy == AR else ""
    print(f"outputs   {out_dir}/: candidates.parquet ({len(results['candidates'])} rows), summary.csv, "
          f"settings.json{stability}, chart.png{charts}")


def print_ema_report(args, results):
    summary, segs = results["summary"], results["segments"]
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


def print_orr_report(args, results):
    summary, segs, selected = results["summary"], results["segments"], results["selected"]
    print("\nSession funnel (counts only; one signal at most per session and threshold)")
    print(report.format_counts(summary))
    first = summary[summary["segment"] == segs[0]["name"]]
    if args.split_date is None:
        print("\nExploratory / in-sample (no --split-date). Gross directional signal response from the signal bar "
              "close, NOT strategy P&L; long and short kept separate")
    else:
        print(f"\nSelection on earlier segment {segs[0]['first'].date()} to {segs[0]['last'].date()}, separately "
              f"per side: highest mean directional {args.select_horizon}m return with >= {args.min_labeled} "
              f"available {args.select_horizon}m outcomes. Gross signal response, NOT strategy P&L")
    print(report.format_table(first))
    if args.barriers:
        print(f"\nBarrier comparison ({segs[0]['name']}): {report.BARRIER_CAVEAT}")
        print(report.format_barrier_table(first))
    if args.split_date is not None:
        later = summary[summary["segment"] == "later"]
        for side, pick in selected.items():
            print(f"{side}: " + (f"selected {pick}; evaluated once on {segs[1]['first'].date()} to "
                                 f"{segs[1]['last'].date()}" if pick else "no threshold qualified; later "
                                 "segment not evaluated for this side"))
        if len(later):
            print(report.format_table(later))
            if args.barriers:
                print(report.format_barrier_table(later))
            print("A later period you keep re-inspecting stops being a clean holdout.")
    print("Short-side rows do not establish borrow availability or executable short returns. SPY and QQQ are "
          "correlated: separate runs are not independent evidence.")


def print_ar_report(args, results):
    summary, segs = results["summary"], results["segments"]
    print("\nSetup funnel (aggregate counts; at most one signal per side and session)")
    print(report.format_setup_funnel(summary))
    first = summary[summary["segment"] == segs[0]["name"]]
    if args.split_date is None:
        print("\nExploratory / in-sample (no --split-date). Gross directional signal response from the signal bar "
              "close, NOT strategy P&L; long and short kept separate")
    else:
        print(f"\nDiagnostic comparison on the earlier segment {segs[0]['first'].date()} to "
              f"{segs[0]['last'].date()} (nothing is selected). Gross signal response, NOT strategy P&L")
    print(report.format_table(first))
    if args.barriers:
        print(f"\nBarrier comparison ({segs[0]['name']}): {report.BARRIER_CAVEAT} "
              "entry invalid = broken geometry or reward/risk below the rule set's minimum at the actual entry.")
        print(report.format_barrier_table(first))
    if args.split_date is not None:
        later = summary[summary["segment"] == "later"]
        print(f"\n{results['selected']} (fixed in advance, not selected) on the later segment "
              f"{segs[1]['first'].date()} to {segs[1]['last'].date()}:")
        print(report.format_table(later))
        if args.barriers:
            print(report.format_barrier_table(later))
        print("A later period you keep re-inspecting stops being a clean holdout.")
    shown = results["configs"][0]
    stability = results["stability"]
    print(f"\nStability for {shown} (periods with candidates; small counts are noise, not evidence)")
    print(report.format_stability(stability[stability["config"] == shown]))
    print(f"Profile levels are a {volume_profile.PROFILE_METHOD}, not traded volume at price. Relative volume and "
          "candle shape are not order flow (no aggressor side, absorption or delta). Short-side rows do not "
          "establish borrow availability or executable short returns. SPY and QQQ are correlated: separate runs "
          "are not independent evidence.")


def execute(args, minutes, sessions, source, timings):
    """Run the study on already-loaded minutes, write outputs, print the summary."""
    results = run_study(minutes, sessions, strategy=args.strategy, fasts=args.fast, slows=args.slow,
                        daily_filter=args.daily_filter, thresholds=args.threshold, barriers=args.barriers,
                        bar_minutes=args.bar_minutes, min_coverage=args.min_coverage, split_date=args.split_date,
                        min_labeled=args.min_labeled, select_horizon=args.select_horizon, location=args.location,
                        profile_bins=args.profile_bins, rvol_filter=args.rvol_filter, compare=args.compare,
                        rules=args.rules, timings=timings)
    ticker = args.ticker.upper()
    with timed(timings, "reporting"):
        results["candidates"].insert(0, "ticker", ticker)
        results["summary"].insert(0, "ticker", ticker)
        # Chart one configuration when there is one to show; otherwise compare them all.
        configs, selected = results["configs"], results["selected"]
        if args.strategy == ORR:
            chart_config = configs[0] if len(configs) == 1 and args.split_date is None else None
            picked = bool(selected) and any(selected.values())
        elif args.strategy == AR:
            chart_config = configs[0] if len(configs) == 1 or args.split_date else None
            picked = False
        else:
            chart_config = selected or (configs[0] if len(configs) == 1 else None)
            picked = bool(selected)
        title = f"{ticker} · {args.strategy} · {chart_config or f'{len(configs)} configurations'}"
        if picked:
            context = "Selected on the earlier segment only, then evaluated once on the later segment."
        elif args.split_date and args.strategy == AR:
            context = (f"{configs[0]}, fixed in advance (nothing selected): earlier (comparison) and later "
                       "(evaluation) segments.")
        elif args.split_date:
            context = "Split date given but no configuration qualified: earlier (selection) segment only."
        else:
            context = "In-sample / exploratory: no split date."
        if args.strategy in (ORR, AR):
            context += " Gross directional signal response, not P&L; long and short separate."
        out = report.write_outputs(args.out, results["candidates"], results["summary"],
                                   settings_for(args, sessions, source, results), title, context,
                                   chart_config, args.select_horizon)
        if args.strategy == ORR:
            # First candidates in time for the baseline (else first) threshold; never picked by outcome.
            shown = next((c for c in configs if c == orr.config_label(orr.BASELINE_THRESHOLD)), configs[0])
            cands = results["candidates"]
            results["candidate_charts"] = report.plot_candidates(out, cands[cands["config"] == shown],
                                                                 results["bars"], ticker)
        elif args.strategy == AR:
            cands = results["candidates"]
            results["stability"] = report.stability(cands, args.barriers)
            results["stability"].to_csv(out / "stability.csv", index=False)
            # First candidates in time for the baseline (else the single) configuration; never picked by outcome.
            results["candidate_charts"] = report.plot_auction_candidates(
                out, cands[cands["config"] == configs[0]], results["bars"], results["minutes"], ticker)
    print_report(args, sessions, source, results, timings, out)
    return results


def iso_date(text):
    return str(pd.Timestamp(text).date())  # ValueError -> argparse usage error


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quick intraday signal study on Massive minute bars "
                                            "(a signal study, not a backtest).")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="spy_ema")
    p.add_argument("--ticker", help="one US stock/ETF per run (default SPY; QQQ for auction_reclaim)")
    p.add_argument("--start", type=iso_date, required=True, help="first study date, YYYY-MM-DD")
    p.add_argument("--end", type=iso_date, required=True, help="last study date, YYYY-MM-DD (inclusive)")
    p.add_argument("--bar-minutes", type=int, default=5, help="intraday bar size, anchored to each session open")
    p.add_argument("--fast", type=int, nargs="+", default=[9], help="fast EMA period(s); several values = sweep")
    p.add_argument("--slow", type=int, nargs="+", default=[21], help="slow EMA period(s); several values = sweep")
    p.add_argument("--daily-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="require previous session close > its daily EMA50 (default on)")
    p.add_argument("--threshold", type=float, nargs="+", default=[orr.BASELINE_THRESHOLD],
                   help="opening_range_reversal: opening range / prior ATR14 gate(s); baseline 0.25, "
                        "small sweep 0.20 0.25 0.30")
    p.add_argument("--barriers", action="store_true",
                   help="opening_range_reversal / auction_reclaim: add the fixed stop/target comparison "
                        "(idealized, not P&L)")
    p.add_argument("--rules", choices=sorted(ar.RULES), default=ar.BASELINE["rules"],
                   help="auction_reclaim: rule set (default baseline; loose = signals until 15:30, 6-bar retest, "
                        "milder candle, reward/risk >= 1.0)")
    p.add_argument("--location", choices=ar.LOCATIONS, default=ar.BASELINE["location"],
                   help="auction_reclaim: retest location (default value_edge)")
    p.add_argument("--profile-bins", type=int, default=ar.BASELINE["profile_bins"],
                   help="auction_reclaim: previous-session profile bins (default 48)")
    p.add_argument("--rvol-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="auction_reclaim: require relative volume >= 1.20 on the retest bar (default on)")
    p.add_argument("--compare", action="store_true",
                   help="auction_reclaim: run the five fixed diagnostic variants of the chosen rule set (its "
                        "base, reclaim_lvn, rvol filter off, 32 and 64 bins) on shared data")
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
    if args.strategy in (ORR, AR) and args.bar_minutes != orr.BAR_MINUTES:
        p.error(f"{args.strategy} uses {orr.BAR_MINUTES}-minute signal bars (--bar-minutes {orr.BAR_MINUTES})")
    if args.strategy not in (ORR, AR) and args.barriers:
        p.error(f"--barriers applies only to --strategy {ORR} or {AR}")
    defaults = {k: v for k, v in ar.BASELINE.items() if k != "rules"}
    ar_options = {"location": args.location, "profile_bins": args.profile_bins, "rvol_filter": args.rvol_filter}
    if args.strategy != AR and (args.compare or ar_options != defaults or args.rules != ar.BASELINE["rules"]):
        p.error(f"--rules, --location, --profile-bins, --no-rvol-filter and --compare apply only to --strategy {AR}")
    if args.compare and ar_options != defaults:
        p.error("--compare runs the fixed variants; leave --location, --profile-bins and --rvol-filter at defaults")
    if args.profile_bins < 3:
        p.error("--profile-bins must be >= 3")
    args.ticker = args.ticker or ("QQQ" if args.strategy == AR else "SPY")
    if min(args.threshold) <= 0:
        p.error("--threshold values must be > 0")
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
