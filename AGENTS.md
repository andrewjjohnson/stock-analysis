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
  P&L. There are two trade simulations. `sauce.py` (VWAP + Sauce) has next-open fills, band
  targets, stops and one position at a time; its results are gross, per share of the
  underlying, and labeled that way, never as options P&L. `alert_spreads.py` trades $1 SPY
  credit spreads on Massive option minute bars; its results are dollars per spread after
  the `--slippage`/`--commission` settings (default 0), always shown with wider-slippage
  rows and next to baselines with identical exits.
- **Prices** are Massive split-adjusted (`download.ADJUSTED`) but not dividend-adjusted,
  so returns are price returns, never total returns.
- **Speed claims:** the CLI prints measured per-stage timings. Don't claim a speedup
  without measuring it.

## Commands

Everything runs through uv (Python 3.12+, locked in `uv.lock`). There is no build step and
no linter or formatter config.

```bash
uv sync                          # install/update dependencies, including pytest
uv run pytest -q                 # all tests: about 7 s, no API key, no network
uv run pytest tests/test_outcomes.py::test_split_selects_on_earlier_segment_and_outcomes_stay_inside_segments -q
uv run pytest -k lookahead -q    # select by keyword
uv run python demo.py            # offline end-to-end run on SYNTHETIC data -> output/demo/<run>/
uv run python run.py --help
uv run python run.py --start 2025-04-01 --end 2026-03-31 --out output/x                              # needs MASSIVE_API_KEY
uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --split-date 2025-10-01 --out output/y
uv run python sauce.py --ticker SPY --start 2022-04-01 --end 2024-12-31 --compare --out output/z   # cached SPY: no key needed
uv run --env-file .env python alerts.py --csv data-files/<log>.csv --out output/alerts   # alert-log check
uv run --env-file .env python alert_spreads.py --csv data-files/<log>.csv --out output/alert_spreads   # needs options data
uv run --env-file .env python meanrev.py --out output/meanrev     # SPY 1-5 day mean reversion (add --final-test for the holdout)
uv run --env-file .env python dip_spreads.py --out output/dip_spreads   # put credit spreads on the dip signals
uv run --env-file .env python stock_dip_spreads.py --out output/stock_dip_spreads   # the same on single stocks
uv run --env-file .env python gap_recovery.py --out output/gap_recovery   # how often gap-down opens fill (no options)
uv run --env-file .env python scalp_meanrev.py --out output/scalp_meanrev   # intraday mean reversion (design period)
uv run --env-file .env python intraday_momentum.py --out output/intraday_momentum   # morning move vs the last half hour
uv run --env-file .env python trend_etfs.py --out output/trend_etfs   # ETF trend following next to SPY dip buying
uv run --env-file .env python call_spreads.py --out output/call_spreads   # SPY call credit spreads 30-45 days out by delta
uv run --env-file .env python iron_condors.py --out output/iron_condors   # SPY iron condors 30-45 days out
uv run --env-file .env python red_bars.py --out output/red_bars   # 3 red bars in a row, 5 min to daily
uv run --env-file .env python bar_patterns.py --out output/bar_patterns   # red x3-green-red bottoms and mirrored tops
uv run --env-file .env python levels.py --out output/levels   # do support/resistance levels hold? (--chart DATE draws them)
uv run --env-file .env python swings.py --out output/swings   # 5-minute swing lines and swing timing vs random copies
uv run --env-file .env python breakouts.py --out output/breakouts   # breakout trades at key levels (TSLA + ETFs)
uv run --env-file .env python breakout_check.py --out output/breakout_check   # TSLA breakouts, shares + options, one-second fills (dry run; --final-test = 2025-26)
uv run --env-file .env python kalman_supertrend.py --out output/kalman_supertrend   # the user's Pine Kalman SuperTrend strategy, rebuilt
uv run --env-file .env python supertrend_0dte.py --out output/supertrend_0dte   # the same trades as 0DTE at-the-money SPY options
uv run --env-file .env python confluence_scalper.py --out output/confluence_scalper   # five-indicator scalper (design period; --final-test = 2025-26)
uv run --env-file .env python green_goose.py --out output/green_goose   # Green Goose overnight options: stock stats, then SPY options
uv run --env-file .env python goose_indicators.py --out output/goose_indicators   # which 15:50 indicators help Green Goose (screen 2021-24, check 2024-26)
uv run --env-file .env python goose_spreads.py --out output/goose_spreads   # Green Goose's direction as $1 SPY credit spreads
uv run --env-file .env python breakout_flow.py --out output/breakout_flow   # TSLA breakouts with an order-flow (buying-pressure) filter
uv run --env-file .env python band_scalp.py --out output/band_scalp   # Bollinger + RSI scalp with trap filters (SPY, QQQ, 12 mega caps)
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

**VWAP + Sauce** is a separate trade simulation with its own CLI, because it needs entries,
exits and one position at a time rather than candidates and forward outcomes:
1. `sauce.main` loads sessions and minutes the same way as `run.main`.
2. `sauce.run_sauce`, which does no file I/O, calls `features.build_features` with 2-minute
   bars and EMA 8/48. `vwap_sauce.add_indicators` then adds the anchored VWAP and sigma,
   once per distinct `vwap_lookback_sessions`.
3. `vwap_sauce.simulate(bars, **params)` runs one per-bar state machine over the study bars
   and returns the setup log: one row per setup instance and trade. `vwap_sauce.summarize`
   and `vwap_sauce.monthly` compute the statistics from that log.
4. `sauce.execute` writes `setup_log.csv`, `summary.csv`, `monthly.csv`, `settings.json`
   and `session_<date>.png` (`report.plot_sauce_sessions`). With `--random-baseline N`,
   `sauce.entry_baseline` also compares Setup A's entries with matched random ones under
   identical fixed-bracket exits (`outcomes.barrier_exits`) and writes `entry_baseline.csv`.

**Alert check** (`alerts.py`) scores a CSV of posted SPY calls (the private log lives in the
git-ignored `data-files/`; never commit it or its results). `run_study` does no file I/O:
`check_rows` validates the log, `add_outcomes` looks up the reference price and each horizon
by explicit timestamp (`outcomes._ns`), with pending and unavailable statuses, and `Results`
collects one statistic per row. `main` loads sessions and minutes like `run.main`, plus
ex-dividend dates from Massive's `list_dividends`.

**Alert spreads** (`alert_spreads.py`) trades the intraday alerts as same-day $1 vertical
spreads. `prepare_trade` picks the legs (`spread_legs`), loads each contract-day's minute bars
through `option_loader` (cached in `data/cache/options/`; the client is created only on a
miss) and keeps only minutes in which both legs traded. `simulate` applies one exit rule.
`run_study` runs the user's rule, a take-profit x stop grid and the baselines (always bullish,
always bearish, shuffled calls) on the same days, plus a no-alert context over every session
the data plan covers. The plan has option minute bars from 2024-10-01 (a rolling two years)
but no quotes or trades, so fills are approximations from traded prices.

**Mean reversion** (`meanrev.py`) is a daily SPY signal study. `daily_table` turns minutes into
one row per session, including a snapshot 10 minutes before the close; `ticker_features`
computes each day's indicators from that snapshot plus earlier full sessions (a loop over days,
so nothing later can leak in); `forward_outcomes` measures snapshot-to-snapshot returns.
`run_study` does no file I/O. The holdout (2025-01-01 on) is read only with `--final-test`, which
was run once on 2026-10-01: SPY 2025+ is no longer clean for daily SPY signal ideas.
`dip_spreads.py` prices its dip signals as put credit spreads (`spread_trade`, `simulate`) with
per-contract multi-session bars (`contract_loader`), against the same spread on every session.
`stock_dip_spreads.py` reuses that pricing (`dip_spreads.price_spread`) for single stocks: strikes
and weekly expiries come from the options contracts reference (`load_chain`, `pick_strikes`,
`pick_expiry`), and spreads open over earnings are skipped (`earnings_days`, inferred from
volume and gaps). `gap_recovery.py` is a price study of gap-down opens (`gap_events` per ticker,
`summarize` per group); `run_study` does no file I/O, and `load_ticker` fills sessions a renamed
ticker lacks from its old ticker (`FORMER`). `scalp_meanrev.py` tests intraday mean-reversion signals on 5- and
2-minute bars (`signal_masks`, `round2_masks`, `fast_masks`), with outcomes for every bar in the
10:30-15:00 window (`bar_outcomes`) so the excess can use a time-of-day baseline; the 2025+ holdout is
read only with `--final-test`. `intraday_momentum.py` (`day_table`, `signal_outcomes`) and `trend_etfs.py` (`trend_signal`,
`risk_weights`, `sleeve_returns`, `dip_positions`) are small daily studies; both do no file I/O in
`run_study`. `call_spreads.py` picks strikes by delta from implied volatility
(`plan_entries`: parity forward, Black-76) and prices call spreads with `dip_spreads.price_spread`
(trade `"right": "C"`). `iron_condors.py` adds the put side (`call_spreads.choose_shorts` with
"P", put deltas through parity) and combines both spreads into one value path (`condor_path`). `levels.py` tests levels
known before the open (`level_table`, `real_levels`): the first touch and a race from the level (`races`,
`measure`), against fake levels at matched distances (`fake_levels`; `near_levels` on the same session), with
a session bootstrap (`compare`). `run_study` does no file I/O; `--chart` draws sessions with their levels
(`plot_session`). `swings.py` finds 5-minute swings with a vectorised zigzag (`zigzag`), tests them as
lines drawn from 11:30 through `levels.measure` (`swing_levels`, `random_fakes`, `near_fakes`), and compares leg
timing (`legs`, `timing_stats`, `hazard`) with random-direction copies of each session (`flipped`). `breakouts.py`
trades through the levels (`rule_levels`, `simulate`, `run_trades`) with stop-order fills, stops, targets and
costs, against the same trades at random fake levels; every result has a worst and a best bound for the order
inside the fill minute (`simulate(best=...)`). `breakout_check.py` replays the TSLA breakout trades on
one-second bars (`second_loader`, `replay`) and prices one at-the-money option per trade from one-second option
trades (`option_leg`, `option_trades`); its 2025-26 run is the one-time check. `kalman_supertrend.py` rebuilds the user's Pine
strategy on 5-minute extended-hours bars (`five_minute_bars`, `indicators`) and trades it minute by minute
(`trade_from`, `strategy`) against random entries with the same exits (`random_trades`). `supertrend_0dte.py` prices
those trades as same-day at-the-money options from one-minute option trades (`contract`, `price_at`,
`option_trades`). `confluence_scalper.py` combines reimplementations of MACD,
Squeeze Momentum, SuperTrend AI (`supertrend_ai`, `kmeans3`), swing structure and the volatility waves into one
entry rule, traded with `kalman_supertrend.py`'s exits or a SuperTrend AI exit (`trend_strategy`). `green_goose.py`
tests an overnight RSI(2)/ADX-DMI direction rule on stock moves (`daily_signals`, `morning_moves`) and as SPY
options picked by Black-76 delta and theta (`choose_contract`, `option_trades`, `exit_v1`, `exit_v2`).
`goose_indicators.py` screens 32 conditions known at 15:50 (`indicator_frame`, `conditions`) on SPY's move to the
next open against slid copies of each (`slides`, `edge_test`, `screen`), lets the kept ones vote (`rule_direction`)
and checks the rule from 2024-10 as stock moves (`stock_check`) and as Green Goose's options (`option_pnl`).
`goose_spreads.py` sells Green Goose's direction as $1 next-session spreads (`spread_trade`, with `alert_spreads.spread_legs`
placement and prices from both-legs minutes via `value_at`), exited at 9:35, by an 80% take profit or at expiry
(`trade_pnl`, `settle`).
`breakout_flow.py` adds an order-flow filter to `breakout_check.py`'s TSLA trades: buying versus selling volume
from one-second bars with volume (`second_loader(volume=True)`, its own cache) before each fill (`pressure`,
`flow_columns`), tested with session-clustered regressions (`cluster_ols`, `split_test`).
`band_scalp.py` tests a Bollinger + RSI scalp on 5-minute bars (`five_minute_bars`, `find_setups`) with trend, band-width,
volume-profile level (`profile_levels`) and order-flow filters (`bar_deltas`, `cvd_flags`), traded on minute bars
(`simulate`).

Every rule the source brief marked as an assumption is a key in `vwap_sauce.DEFAULTS`, and
a CLI flag. The choices the brief leaves open are listed in `vwap_sauce.IMPLEMENTATION`.
Don't add filters beyond those two lists.

## Invariants: each has a guarding test

| Rule | Test |
|---|---|
| **Candidate-only work:** after the vectorized mask, rows and outcomes are built only for triggered indices, with no dense forward-return columns and no per-bar logs. With zero triggers, `forward_outcomes` and `build_candidate_rows` are never called; the result is an empty table with the expected columns plus a zero-count summary. | `test_zero_triggers_never_build_rows_or_outcomes`, `test_known_trigger_timestamp_and_outcomes_only_for_triggers` |
| **No lookahead:** Massive timestamps are bar starts, and a bar's close/high/low are usable only at `bar_end`. Daily features come from the previous session. Intraday EMAs are causal and run continuously across sessions, with no morning reset. | `test_future_prices_cannot_change_earlier_features_or_triggers` |
| **Outcomes:** looked up by elapsed minutes from T = `bar_end`. A value exists only if every minute in [T, T+h) is present and T+h is no later than the session close and the segment end. Otherwise it is NaN and the candidate is kept. MFE ≥ 0 ≥ MAE, and the trigger bar's own minutes are excluded. | `test_missing_minutes_close_and_segment_end_give_nan_but_keep_rows`, `test_known_trigger_timestamp_and_outcomes_only_for_triggers` |
| **Split:** rank only on earlier-segment summaries; a config qualifies with ≥ `--min-labeled` available outcomes at `--select-horizon`. Only the pick is evaluated later, nothing is picked if none qualifies, and without a split everything is labeled exploratory/in-sample. | `test_split_selects_on_earlier_segment_and_outcomes_stay_inside_segments` |
| **Sauce, no lookahead:** VWAP and sigma at a bar use bars through its close only, and EMAs are causal. Changing later minutes leaves earlier indicators and trades unchanged. Targets and price stops fill against the level known at the previous close. | `test_future_prices_cannot_change_earlier_indicators_or_trades`, `test_target_level_is_the_one_known_at_the_previous_close`, `test_anchored_vwap_and_sigma_match_a_direct_computation` |
| **Sauce fills and sessions:** a decision made at a close fills at the next bar's open in the same session. Setups end with their session, and a position carries overnight only with `--no-time-stop`. There is one position at a time. | `test_setup_a_long_arms_goes_parallel_enters_next_open_and_exits_at_the_moving_target`, `test_sessions_are_independent_unless_the_time_stop_is_off`, `test_one_position_at_a_time_blocks_other_entries` |
| **Sauce symmetry:** the upper band is the exact mirror of the lower band. | `test_upper_band_is_the_exact_mirror_of_the_lower_band` |
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
- **Today's session** is skipped in `run.main` and `sauce.main`, because its data may be incomplete.
- **2-minute coverage:** `sauce.py` defaults to `--min-coverage 0.5`, where one traded minute
  makes a bar, as on a chart. At 0.8, every 2-minute bar with a quiet minute would be
  dropped.
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
