# stock-analysis

A personal proof of concept for quickly testing an intraday stock signal on Massive
one-minute bars and comparing what happened afterwards. It is a **signal study**, not a
backtest: no orders, fills, costs, position sizing or P&L (the opening-range reversal's
optional `--barriers` comparison is the one narrow, clearly labeled exception). It is
independent of Quant Forge and shares no code or data with it.

## Setup

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.12+ if needed.

```bash
uv sync
```

The TA-Lib wheels on PyPI include the TA-Lib C library, so there is no separate
`brew install ta-lib`. Without uv: create a venv, run
`pip install massive TA-Lib exchange-calendars pandas numpy pyarrow matplotlib pytest`,
and drop `uv run` from the commands below.

## API key

`MASSIVE_API_KEY` is read from the environment:

```bash
export MASSIVE_API_KEY=your-key
```

Alternatively, keep the key in the git-ignored `.env` (`cp .env.example .env`, then edit
it) and add `--env-file .env` after `uv run` in the commands below.

## Run

Single configuration: SPY, EMA 9/21 on 5-minute bars, daily EMA50 filter on.

```bash
uv run python run.py --start 2025-04-01 --end 2026-03-31 --out output/spy_single
```

Sweep of nine configurations (any fast >= slow pair is rejected and listed):

```bash
uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --out output/spy_sweep
```

Chronological split: rank configurations on sessions before the split date, then evaluate
only the selected one on the later sessions.

```bash
uv run python run.py --start 2025-04-01 --end 2026-03-31 --fast 5 9 12 --slow 20 21 30 --split-date 2025-10-01 --min-labeled 30 --out output/spy_split
```

Other options include `--ticker QQQ`, `--bar-minutes 15`, `--no-daily-filter`,
`--select-horizon 60`, `--warmup-sessions 120`, `--min-coverage 0.8` and `--refresh`;
see `uv run python run.py --help`. The first run downloads the study range plus 120
warm-up sessions to `data/cache/<TICKER>_1min_<first>_<last>_splitadj.parquet`, and
repeating the same request reads that file without network access. That is why the
three commands above share one date range: the sweep and split reuse the first download.
On a rate-limited plan, a long first download can take a minute or two. If Massive
returns less data than the calendar expects, the run stops and nothing is cached.

Offline demo on clearly labeled SYNTHETIC random-walk data (no key, no network); it runs
the EMA single, sweep and split runs and two opening-range reversal runs into `output/demo/`:

```bash
uv run python demo.py
```

Tests (synthetic fixtures, no key needed):

```bash
uv run pytest -q
```

## Opening-range reversal (`--strategy opening_range_reversal`)

Fades an unusually large first 15 minutes. Baseline on SPY, then QQQ as a separate run
(one ticker per run; any single stock such as NVDA works the same way). No dates in this
project are designated exploratory and none are documented as a reserved holdout, so
calendar 2024 is used as an **explicitly exploratory** demonstration period:

```bash
uv run python run.py --strategy opening_range_reversal --ticker SPY --start 2024-01-01 --end 2024-12-31 --barriers --out output/orr_spy
uv run python run.py --strategy opening_range_reversal --ticker QQQ --start 2024-01-01 --end 2024-12-31 --barriers --out output/orr_qqq
```

Optional small threshold sweep with a chronological split (each side picks its threshold
on the earlier half only, by mean directional 30-minute return with `--min-labeled`
available outcomes; the pick is evaluated once on the later half):

```bash
uv run python run.py --strategy opening_range_reversal --ticker SPY --start 2024-01-01 --end 2024-12-31 --threshold 0.20 0.25 0.30 --split-date 2024-07-01 --barriers --out output/orr_spy_split
```

Definitions (in `strategies/opening_range_reversal.py`), all per ticker and session:

- **Opening range**: minutes [open, open+15), all 15 present; known at open+15 (09:45 on
  a normal day). Its high minus low must be >= threshold (0.25 baseline) x the daily
  ATR(14) from completed regular-session daily bars **through the previous session**.
- **Direction**: opening close below its open: long only; above: short only; equal: skipped.
- **Signal bar**: a complete (5 of 5 minutes) 5-minute bar anchored to the open that starts
  at or after open+15 and completes strictly before open+90, so 09:50 is the first
  possible signal and a bar completing at 11:00 is not accepted.
- **Pattern** (installed TA-Lib): long `CDLHAMMER > 0` or `CDLENGULFING > 0`; short
  `CDLSHOOTINGSTAR < 0` or `CDLENGULFING < 0` (engulfing returns ±100, or ±80 when an edge
  is equal). All three compare with the previous candle, so that bar must also be complete,
  contiguous and in the same session. TA-Lib's shooting star also requires the body to gap
  above the previous body, which is uncommon on continuous 5-minute bars, so expect few
  shooting-star signals. Trailing candle averages use the continuous bar history.
- **Outside**: long low and close strictly below the opening low; short high and close
  strictly above the opening high. The candle may straddle the boundary.
- **One signal**: the earliest qualifying bar per session and threshold; no other filter.

Outcomes are the tool's usual 10/30/60/120-minute returns and 60-minute MFE/MAE from the
signal bar's close, **signed by side** (short = -1 x raw return): a gross directional
signal response, not strategy P&L. `--barriers` adds an idealized fixed-exit comparison per
candidate: entry at the next minute's open (at signal completion), stop at the signal low
(long) or high (short) or across both candles for engulfing, target at the opposite
opening-range boundary, first touch in time order, a later open beyond the stop exits at
that open and beyond the target at the target, both levels in one minute = ambiguous and
stop first, otherwise the regular-session close. Invalid geometry or a missing entry minute
keeps the candidate (no later signal replaces it); missing minutes before an exit leave it
unresolved. R is signed price change / initial stop distance; 0 and 1 bp per side
(`rate x (entry + exit)` per share) is an illustrative friction sensitivity, not calibrated
costs, and excludes borrow. Short rows do not show that shares could be borrowed. SPY and
QQQ are correlated, so their separate results are not independent evidence.

**Deliberate departures from the video** (engineering assumptions, not claims from it):
completed candles only (the video sometimes enters before a candle closes); TA-Lib's
pattern definitions instead of visual judgment; a strict "close outside the range" rule;
only the first signal per session; fixed stop/target with no trailing, partial or
discretionary exits; each ETF's own prices and ATR (no index points, futures or CFD
mapping). Its performance claims and its explanation of institutional intent are not
assumed.

## Output (in `--out`)

- `candidates.parquet`: one row per trigger, with configuration, segment, `signal_time`
  (trigger-bar completion, UTC), `ref_close`, the strategy's feature snapshot,
  `fwd_ret_{10,30,60,120}m_pct`, `mfe_60m_pct` and `mae_60m_pct`. NaN means unavailable.
- `summary.csv`: one row per configuration and segment, with candidate count and, per
  horizon, available/unavailable counts, mean, median and fraction strictly above zero,
  plus MFE/MAE summaries. Metrics with no data are blank, not 0.
- `settings.json`: the settings needed to interpret the files.
- `chart.png`: mean forward return by horizon for the single or selected configuration
  (earlier vs later after a split), or the mean 30-minute return per configuration for
  a sweep. Every bar shows its count.

Every run also adds a `ticker` column to both tables. For `opening_range_reversal`:

- `candidates.parquet` adds `side`, `session_open`, `opening_end` (when the range is
  known), `or_open/high/low/close`, `or_range`, `prev_atr_14`, `range_atr_ratio`, the
  side-appropriate pattern flags `hammer` / `shooting_star` / `engulfing`, `pattern`,
  `signal_open/high/low` (`ref_close` is the signal close) and, with `--barriers`,
  `entry_time`, `entry_status` (ok / unavailable / invalid), `entry_price`, `stop_price`,
  `target_price`, `exit_reason` (target / stop / stop_gap / close / unresolved),
  `exit_time`, `exit_price`, `ambiguous`, `holding_minutes`, `barrier_ret_{0,1}bp_pct`
  and `barrier_r_{0,1}bp`.
- `summary.csv` has one row per threshold, segment and side (`pattern` = `all`, with the
  session funnel: sessions, opening incomplete, ATR unavailable, eligible sessions, flat
  openings skipped, qualifying openings) and one per mutually exclusive pattern group
  (hammer, engulfing, both; shooting_star, engulfing, both), which add up to the side row.
  With `--barriers` it adds entry, exit-reason and ambiguity counts and R statistics.
- `candidate_<n>_<session>_<side>.png`: the first six baseline candidates by time (never
  picked by outcome), with 5-minute candles, the opening box, the signal bar, entry/stop/
  target and a small feature/level table for checking alignment.

## Editing or adding a strategy

`spy_ema()` in `strategies/spy_ema.py` is a plain function. It receives the feature
table and returns a boolean trigger mask plus the arrays to keep for each trigger. The
table holds only completed, adequately covered bars and no outcomes. Its columns are
`bar_start`, `bar_end`, OHLCV, `session`, `ema_<p>` for each requested period,
`prev_close` and `prev_ema_50`. Edit the conditions there directly. To add a strategy:

1. Write `strategies/my_idea.py` with `def my_idea(features, **params): return mask, keep`.
2. In `run.py`, import it, add it to `STRATEGIES` (this also adds it to `--strategy`),
   and give it a parameter grid in `make_configs`.
3. For a new indicator, add a TA-Lib column in `features.build_features`; it is computed
   once and reused by every configuration.

## Assumptions that affect interpretation

- **Prices** use Massive `adjusted=true`, which is split-adjusted only. Returns are
  price returns, not dividend-adjusted total returns.
- **Sessions** are XNYS regular hours only, with holidays and early closes from
  `exchange_calendars`. Timestamps are UTC, and America/New_York only names sessions.
  Extended-hours bars are ignored. Today's session is never studied or cached.
- **Bars** are anchored at each session open. Massive timestamps are bar starts; a bar's
  close is usable only at its end. A bar is used only if at least 80% of its minutes
  exist (`--min-coverage`); others are dropped and counted. Missing prices are never
  filled.
- **Indicators**: intraday EMAs (and the TA-Lib candle patterns) run over the continuous
  history of usable bars and are not reset each morning. The daily EMA50 and ATR(14) use
  regular-session daily bars built from the same minutes; sessions below the coverage rule
  are left out, and the next session gets no value. Daily values from the previous
  completed session apply to the whole current session.
- **Warm-up**: indicators are computed over 120 prior sessions plus the study range and
  then trimmed to the study range. If history is too short, the affected features are
  missing, triggers that need them are skipped, and the missing counts are printed.
- **Outcomes**: the reference price is the trigger bar's close at its completion time T.
  This describes the signal; it does not claim an order could fill there. The return at
  h minutes uses the close of the minute bar ending at T+h, measured in elapsed minutes,
  not rows. MFE/MAE come from minute highs/lows in [T, T+60) and exclude the trigger
  bar's own path. An outcome is NaN unless every minute in [T, T+h) exists and T+h is
  within both the session and the evaluation segment. The candidate row is always kept.
- **Overlap**: candidates close together share forward windows, so their outcomes are not
  independent observations, and the counts overstate the evidence.
- **In-sample vs split**: without `--split-date`, every comparison is exploratory and
  in-sample. With a split, only configurations with at least `--min-labeled` available
  30-minute outcomes before the split qualify. If none qualify, nothing is picked. Earlier
  outcomes end before the split, and later outcomes stay in the later segment; later
  indicators may still warm up on earlier bars. A later period you keep re-checking while
  iterating stops being a clean holdout.
