# stock-analysis

A personal proof of concept for quickly testing an intraday stock signal on Massive
one-minute bars and comparing what happened afterwards. It is a **signal study**, not a
backtest: no orders, fills, costs, position sizing or P&L. It is independent of Quant
Forge and shares no code or data with it.

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
single, sweep and split into `output/demo/`:

```bash
uv run python demo.py
```

Tests (synthetic fixtures, no key needed):

```bash
uv run pytest -q
```

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
- **Indicators**: intraday EMAs run over the continuous history of usable bars and are
  not reset each morning. The daily EMA50 uses regular-session daily bars built from the
  same minutes. Daily values from the previous completed session apply to the whole
  current session.
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
