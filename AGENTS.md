# AGENTS.md

Guide for AI coding agents working in this repository. Humans: start with README.md.
Detailed schemas, time semantics and design rationale are in `docs/design.md`.

## What this is

A personal proof of concept: plain Python scripts that test an intraday stock signal on
Massive one-minute bars and measure what happened afterwards. It is a **signal study, not
a backtest**. The reference price is the trigger bar's close, not a fill, and there is no
P&L. The priorities are a short edit–run–inspect loop, readable code and minimal
infrastructure.

Boundaries from the original brief:
- **Quant Forge** is the user's separate trading-research system (sibling checkout
  `../quant-forge`). Never import from it, modify it, recreate its architecture or read its
  data; it holds reserved holdout data.
- **One provider, one indicator library:** Massive is the only data provider (the
  official `massive` client, called directly). TA-Lib is the only indicator library
  (called directly, with no fallback implementation if it fails to install).
- **Plain code only:** functions, dicts, arrays and DataFrames. No adapter layers, plugin
  registries, dependency injection, strategy class hierarchies, databases, services or web
  UI, Docker, job queues, audit trails or checkpoint/resume.
- **Deliberately left out of v1:** portfolio accounting, order execution, commissions,
  target/stop simulation, ML training, walk-forward orchestration and equity curves. If a
  task needs one, keep it as small as the rest, and never present signal-study returns as
  P&L.
- **Prices** are Massive split-adjusted (`download.ADJUSTED`) but not dividend-adjusted,
  so returns are price returns, never total returns.
- **Speed claims:** the CLI prints measured per-stage timings. Don't claim a speedup
  without measuring it.

## Commands

Everything runs through uv (Python 3.12+, locked in `uv.lock`). There is no build step and
no linter or formatter config.

```bash
uv sync                          # install/update dependencies, including pytest
uv run pytest -q                 # all tests: about 1 s, no API key, no network
uv run pytest tests/test_outcomes.py::test_split_selects_on_earlier_segment_and_outcomes_stay_inside_segments -q
uv run pytest -k lookahead -q    # select by keyword
uv run python demo.py            # offline end-to-end run on SYNTHETIC data -> output/demo/{single,sweep,split}
uv run python run.py --help
uv run python run.py --start 2025-04-01 --end 2026-03-31 --out output/x                              # needs MASSIVE_API_KEY
uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --split-date 2025-10-01 --out output/y
```

- **Checking a change:** `demo.py` is the quickest end-to-end check without credentials,
  though its numbers mean nothing. Inspect results in `output/<name>/`: `summary.csv`,
  `candidates.parquet` (load it with pandas), `settings.json` and `chart.png`.
- **The API key** comes from the environment. It can also be kept in the git-ignored
  `.env` and loaded with `uv run --env-file .env ...`.
- **Minute bars are cached** in `data/cache/`, and an identical request never hits the
  network. Use `--refresh` to re-download.

## Architecture

Data flow, one ticker per run:

1. `run.main` calls `features.trading_sessions` to get XNYS sessions from
   `exchange_calendars`, plus `--warmup-sessions` sessions before `--start`. It then calls
   `download.load_minute_bars`, which reads the Parquet cache or else Massive `list_aggs`.
2. `run.execute` is shared by the CLI and `demo.py`. It calls `run.run_study`, which does
   no file I/O and is the entry point tests use:
   - `features.build_features` returns regular-session minutes (`rth`, used only for
     outcomes), usable intraday bars with indicator columns (`bars`), daily bars and a
     coverage `info` dict. Indicators are computed over warm-up plus study; the
     `bars.in_study` flag does the trimming.
   - `make_segments` returns one `in_sample` segment, or `earlier` (selection) and `later`
     (evaluation) when `--split-date` is given.
   - For each config, `STRATEGIES[name](bars, **params)` returns a boolean mask plus `keep`
     arrays. Candidates are `np.flatnonzero(mask & in_segment)`.
   - `evaluate_segment` takes the union of candidate indices across configs and makes one
     `outcomes.forward_outcomes` call for it. The outcomes depend only on the trigger bar,
     so they are shared across configs. It then runs `build_candidate_rows` per config,
     then `report.summarize`.
   - With a split, `select_config` picks from the earlier segment's summaries, and only
     the pick is then evaluated on the later segment.
3. `report.write_outputs` writes the output files, and `run.print_report` prints the
   terminal summary with timings.

**Strategy contract** (`strategies/spy_ema.py` is the example):
`fn(features, **params) -> (mask, keep)`.
- `mask` is a bool array aligned with all rows of `bars`, including warm-up rows.
- `keep` maps output column names to full-length arrays; `keep[name][idx]` becomes a
  candidate column.
- Strategies see feature data only: `bars` never contains outcomes.

## Invariants: each has a guarding test

| Rule | Test |
|---|---|
| **Candidate-only work:** after the vectorized mask, rows and outcomes are built only for triggered indices, with no dense forward-return columns and no per-bar logs. With zero triggers, `forward_outcomes` and `build_candidate_rows` are never called; the result is an empty table with the expected columns plus a zero-count summary. | `test_zero_triggers_never_build_rows_or_outcomes`, `test_known_trigger_timestamp_and_outcomes_only_for_triggers` |
| **No lookahead:** Massive timestamps are bar starts, and a bar's close/high/low are usable only at `bar_end`. Daily features come from the previous session. Intraday EMAs are causal and run continuously across sessions, with no morning reset. | `test_future_prices_cannot_change_earlier_features_or_triggers` |
| **Outcomes:** looked up by elapsed minutes from T = `bar_end`. A value exists only if every minute in [T, T+h) is present and T+h is no later than the session close and the segment end. Otherwise it is NaN and the candidate is kept. MFE ≥ 0 ≥ MAE, and the trigger bar's own minutes are excluded. | `test_missing_minutes_close_and_segment_end_give_nan_but_keep_rows`, `test_known_trigger_timestamp_and_outcomes_only_for_triggers` |
| **Split:** rank only on earlier-segment summaries; a config qualifies with ≥ `--min-labeled` available outcomes at `--select-horizon`. Only the pick is evaluated later, nothing is picked if none qualifies, and without a split everything is labeled exploratory/in-sample. | `test_split_selects_on_earlier_segment_and_outcomes_stay_inside_segments` |
| **Data:** an identical request reads the cache without creating a Massive client. Every page is fetched. A fresh download that doesn't reach the final session's last regular-hours minute raises an error and is not cached; checking only the date would accept a download cut off mid-session or holding only pre-market bars. Synthetic data is never a fallback for a failed request. | `tests/test_download.py` |

Missing minutes are never filled. Gaps are dropped and reported (in the coverage lines and
`settings.json`), and rows whose required inputs are NaN never trigger.

## Extending

- **New strategy:**
  1. Add a function in `strategies/`.
  2. Add it to `STRATEGIES` in `run.py`; this also adds it to `--strategy`.
  3. Give it a parameter grid in `make_configs`, which currently holds only spy_ema's
     fast × slow grid.
- **New indicator:** add a TA-Lib column in `features.build_features` over the usable-bar
  series. It is computed once per distinct parameter and reused by every sweep config.
- **Horizons:** `outcomes.HORIZONS` and `EXCURSION_MINUTES` drive the outcome, summary,
  chart and `--select-horizon` choices. The terminal table's `>0 @30m` column and the
  default `--select-horizon 30` assume 30 stays in the list.
- **Docs:** when behaviour changes, update README.md (user-facing assumptions) and
  `docs/design.md`.

## Gotchas

- **pandas 3 timestamps:** `to_datetime(unit="ms")` gives `datetime64[ms]`, and
  timezone-aware `.to_numpy()` returns Timestamp objects. Minute timestamps are normalised
  to UTC nanoseconds in `features.regular_session_minutes`. Use `outcomes._ns` for integer
  lookups.
- **pandas 3 copy-on-write:** chained assignment doesn't write through.
- **TA-Lib NaN propagation:** a NaN in the middle of a series makes every later output
  NaN. That's why indicators run only over usable bars and sessions; never insert NaN rows
  into an indicator input. Input shorter than the period returns all NaN.
- **Massive client quirks:**
  - `massive.RESTClient`'s default `api_key` is read when the module is imported, so
    `download.py` passes the key explicitly.
  - `list_aggs` follows `next_url`; `get_aggs` does not.
  - The SDK's paginator stops silently if a page fails to decode, hence the truncation
    guard.
  - `retries=10` stretches the SDK's backoff to about 100 s in total, enough to ride out
    per-minute 429 responses.
- **No bars for quiet minutes:** Massive emits no bar for a minute without trades, so
  illiquid tickers get many unavailable outcomes under the full-window rule.
- **Calendar range:** `exchange_calendars` defaults to about 20 years of history, so
  `trading_sessions` builds the calendar with an explicit start that covers the warm-up.
- **Today's session** is skipped in `run.main`, because its data may be incomplete.
- **Charts:** `report.py` forces matplotlib's Agg backend.

## Workflow

- **Branches:** the default branch is `main`. Work on a feature branch and open a PR.
- **Public repo:** the GitHub repo is public. `data/`, `output/` and `.env` are
  git-ignored.
- **No CI:** run `uv run pytest -q` and `uv run python demo.py` before pushing.
- **Commit identity:** git has no global identity on the user's machine. Commit with
  `git -c user.name=… -c user.email=…`, using the identity from
  `git log -1 --format='%an <%ae>'`.
- **Live data:** real Massive downloads have worked since 2026-09-29 (SPY, QQQ and NVDA
  runs). Minute history on the current plan appears to start around 2021-09-30: NVDA
  requested from 2020 comes back starting then, and the missing earlier sessions are
  reported as sessions without data.
