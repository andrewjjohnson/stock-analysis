# Design reference

Contracts, schemas and the reasons behind non-obvious choices. `AGENTS.md` has the
overview and invariants, and `README.md` has the user-facing assumptions.

## Time model

- **Timestamps** are timezone-aware UTC everywhere. America/New_York is used only to name
  sessions: `session` is the local date at midnight (naive), matching `exchange_calendars`
  session labels.
- **Minute bars:** a minute bar's `ts` is the start of that minute (Massive's `t`). Its
  close is the price at `ts + 1 min`.
- **Intraday bars:** bar k of a session has `bar_start = open + k * size` and
  `bar_end = min(bar_start + size, close)`. The last bar of an early-close or odd-sized
  session can be short. A bar's values become known at `bar_end`.
- **Signal time:** a candidate's `signal_time` is the trigger bar's `bar_end`, called T.

Worked example with 5-minute bars:
- The 10:20–10:25 ET bar is built from the minutes starting 10:20 through 10:24, and its
  `ref_close` is the close of the 10:24 minute. T is 10:25.
- The 30-minute return uses the close of the minute starting 10:54, which is the price at
  10:55.
- The 60-minute MFE/MAE window is the minutes starting 10:25 through 11:24.

## Stage contracts

**`features.trading_sessions(start, end, warmup_sessions)`** returns a frame indexed by
session date:
- Columns are `open` and `close` in UTC (early closes included) and `in_study`.
- It holds `warmup_sessions` sessions before the first study session.
- `run.main` drops today's session and later ones before downloading.

**`download.load_minute_bars(ticker, start, end, *, cache_dir, refresh, adjusted, expect_through, client)`**
returns `(minutes, source)`.
- `minutes` holds the raw provider rows `ts, open, high, low, close, volume, vwap,
  transactions`, including extended hours.
- The cache file is `<cache_dir>/<TICKER>_1min_<start>_<end>_<splitadj|unadjusted>.parquet`.
- `client` exists so tests can pass a stand-in; production code leaves it `None`.

**`features.build_features(minutes, sessions, bar_minutes, ema_periods, daily_ema_period, min_coverage)`**
returns `(rth, bars, daily, info)`:
- `rth`: sorted, de-duplicated regular-session minutes, plus `session`, `session_open` and
  `session_close`. Only outcomes read it.
- `bars`: usable bars only, covering warm-up plus study. Columns are `bar_start, bar_end,
  session, session_close, open, high, low, close, volume, n_minutes, coverage, ema_<p>…,
  prev_close, prev_ema_<daily>, in_study`.
- `daily`: one row per calendar session, including sessions with no data. Columns are
  `open, high, low, close, volume, n_minutes, expected_minutes, usable, ema_<daily>,
  prev_close, prev_ema_<daily>`.
- `info`: the counts behind the terminal coverage lines. These cover duplicates, dropped
  extended-hours rows, sessions without data or with partial data, expected and present
  minutes, dropped bars, and study rows missing each indicator.

**`outcomes.forward_outcomes(minutes, signal_time, ref_close, session_close, segment_end)`**
returns one row per candidate, in the order given. Columns are `fwd_ret_{10,30,60,120}m_pct`,
`mfe_60m_pct` and `mae_60m_pct`, in percent, with NaN where unavailable.

**`run.run_study(minutes, sessions, *, strategy, fasts, slows, daily_filter, bar_minutes, min_coverage, split_date, min_labeled, select_horizon, timings)`**
returns a dict with `candidates`, `summary`, `selected` (a config label or `None`),
`configs`, `rejected`, `segments` and `info`.

**Segments:**
- `in_sample` (role `exploratory`) when there is no split.
- `earlier` (role `selection`, sessions before `--split-date`) and `later` (role
  `evaluation`) when there is one.
- A segment's `end` is its last session's close, and no outcome may reach past it.

## Output schemas

**`candidates.parquet`** columns, in order:
- `strategy`, `config` and the params (for spy_ema: `fast`, `slow`, `daily_filter`)
- `segment`, `session`, `bar_start`, `signal_time`, `ref_close`
- the strategy's `keep` columns (for spy_ema: `fast_ema`, `slow_ema`,
  `prev_session_close`, `prev_session_ema50`)
- the outcome columns

With zero triggers the file still has these columns, just no rows.

**`summary.csv`** has one row per config and segment:
- Identity columns: `strategy`, `config`, the params, `segment`, `role`, `first_session`,
  `last_session`, then `n_candidates`.
- Per horizon h: `n_{h}m`, `unavailable_{h}m`, `mean_{h}m_pct`, `median_{h}m_pct` and
  `frac_pos_{h}m` (the fraction strictly above zero).
- Excursions: `n_excursion_60m`, `unavailable_excursion_60m`, and the mean and median of
  `mfe_60m_pct` and `mae_60m_pct`.

A statistic with no data is NaN, which is blank in the CSV, never 0.

**`settings.json`** holds only what's needed to interpret the files: dates, data source,
price adjustment, bar and coverage settings, configs, horizons, split and selection
result, and coverage counts.

## Decisions and why

- **Coverage threshold 0.8** (`--min-coverage`, used for both intraday bars and daily
  sessions). It tolerates an odd missing minute without inventing prices. Bars below it
  are dropped from the series rather than NaN-filled, because TA-Lib propagates NaN. After
  a dropped bar, the next crossover compares against the previous usable bar.
- **Daily EMA over usable sessions only.** A session below coverage gives the next session
  NaN daily features rather than stale ones.
- **Outcomes need every minute of the window,** not just the endpoint. The rule is strict
  and easy to state, and excursions need the full path anyway.
- **Outcomes are shared across configs.** An outcome depends only on the trigger bar (T,
  reference close, session close, segment end), so one call over the union of every
  config's candidates is exact.
- **Split by date.** Earlier outcomes therefore end by the last earlier session's close,
  strictly before the split. The explicit segment-end check keeps that true if segment
  rules ever change.
- **Only the selected config touches the later segment.** Other configs' later results are
  never computed or seen, which keeps the later period as clean as it can be.
- **Truncation guard.** The SDK's paginator returns silently if a page fails to decode.
  Caching that short result would quietly shorten every later run.
- **Today's session is never studied or cached.** It may still be open, or a delayed feed
  may still be filling in.
- **Cache key = ticker, first and last session (warm-up included) and adjustment.**
  Changing the dates or `--warmup-sessions` means a new download. There is no incremental
  ingestion, by design.

## How correctness was checked (2026-09-28)

- **Independent recomputation:** 60 sampled demo candidates had their outcomes recomputed
  from raw minutes with pandas timestamp slicing, and none differed.
- **Mutation check:** each injected bug below made at least one test fail. If you weaken
  or rewrite a test, re-run this kind of check.
  - daily features taken from the same session's close or EMA
  - signal time at the bar start
  - outcome clock starting at the bar start
  - excursions including the trigger bar
  - horizons counted as the next h rows
  - ignoring the session close or segment end
  - unclipped MAE
  - outcomes computed for every bar
  - ignoring the cache
  - removing the truncation guard
  - picking a winner when none qualifies
  - triggering on every bar above instead of on the cross
  - resetting intraday EMAs each session
- **Pagination** was exercised through the real `massive` client with its HTTP layer
  stubbed (`tests/test_download.py`).
- **Not yet verified:** any live Massive request.
