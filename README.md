# stock-analysis

A personal proof of concept for quickly testing an intraday stock signal on Massive
one-minute bars and comparing what happened afterwards. It is a **signal study**, not a
backtest: no orders, fills, costs, position sizing or P&L (the optional `--barriers`
comparison of the opening-range reversal and auction reclaim is the one narrow, clearly
labeled exception). It is
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
On a rate-limited plan, a long first download can take a minute or two. If a download
stops before the last regular-session minute of the final requested day, the run stops
and nothing is cached. Gaps elsewhere are reported in the coverage lines, never filled.

Offline demo on clearly labeled SYNTHETIC random-walk data (no key, no network); it runs
the EMA single, sweep and split runs, two opening-range reversal runs and the auction
reclaim comparison into `output/demo/`:

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

## Auction reclaim (`--strategy auction_reclaim`)

Our own deterministic approximation of the failed-auction **reversion** model in Fabio
Valentini's published Auction Market playbook
([Chart Fanatics](https://www.chartfanatics.com/strategies/auction-market-strategy)):
prior-session value, a failed auction outside it, a reclaim, a pullback, confirmation and
a return toward the prior point of control. It is not his method and claims nothing about
his performance. Every number below is a research default, not a verified rule of his.
There is no continuation strategy.

What minute OHLCV cannot see: **order flow**. Relative volume is total-volume intensity,
and the candle rule is a shape. Neither measures aggressive buyers or sellers, absorption
or cumulative delta. The profile is a `bar_approximated_volume_profile`: each minute's
volume is spread evenly over its [low, high], which is **not traded volume at price**.

Default ticker QQQ; run SPY separately (equity/ETF minute data only, no futures or order
book). No dates are designated exploratory and none are documented as a reserved holdout,
so calendar 2024 is the explicitly exploratory period, with the split at 2024-07-01 that
the opening-range reversal already uses:

```bash
uv run python run.py --strategy auction_reclaim --ticker QQQ --start 2024-01-01 --end 2024-12-31 --barriers --out output/ar_qqq
uv run python run.py --strategy auction_reclaim --ticker SPY --start 2024-01-01 --end 2024-12-31 --barriers --out output/ar_spy
```

Diagnostic comparisons (`--compare`): the baseline plus four variants that each change one
setting (`reclaim_lvn` location, relative-volume filter off, 32 bins, 64 bins), all run on
one shared set of features. With `--split-date`, the variants are compared on the earlier
segment only. Nothing is selected: the baseline, fixed in advance, is the only
configuration evaluated on the later segment.

```bash
uv run python run.py --strategy auction_reclaim --ticker QQQ --start 2024-01-01 --end 2024-12-31 --compare --split-date 2024-07-01 --barriers --out output/ar_qqq_compare
uv run python run.py --strategy auction_reclaim --ticker SPY --start 2024-01-01 --end 2024-12-31 --compare --split-date 2024-07-01 --barriers --out output/ar_spy_compare
```

Single settings: `--rules baseline|loose`, `--location value_edge|reclaim_lvn`,
`--profile-bins N`, `--no-rvol-filter`. `--compare` applies its four one-change variants to
the chosen rule set.

**Rule sets.** `baseline` is defined below. `loose` changes only five settings: signal bars
may complete until 15:30 ET, the retest window is 6 bars, the candle needs a body >= 30% of
its range with the close in the top (long) or bottom (short) 40%, and reward/risk >= 1.0
(also at the actual entry for `--barriers`). It was fixed after seeing only 2024 signal
counts under candidate relaxations, never any outcomes.

**Periods for auction_reclaim.** Massive gives about five years of minute history (from
2021-09-30 when this was set up), so with 120 warm-up sessions the first study date is
2022-04-01.

- **Exploration**: 2022-04-01 to 2024-12-31 on QQQ, SPY, IWM and DIA.
- **Held back**: 2025-01-01 onward. Run it once, and only with a configuration chosen
  before looking at it; after that it is no longer a clean check.

```bash
uv run --env-file .env python run.py --strategy auction_reclaim --ticker IWM --start 2022-04-01 --end 2024-12-31 --rules loose --compare --barriers --out output/ar_explore/iwm_loose
```

DIA has missing minutes on many sessions (348 of 811 in 2021-10..2024-12). The rule that
the previous session must be complete then removes almost half of its profiles, and an
incomplete 5-minute bar resets a setup.

Definitions (in `strategies/auction_reclaim.py` and `volume_profile.py`), per ticker and
session. The long side is shown; the short side mirrors it:

- **Frozen at the open**:
  - **Profile**: from the immediately preceding XNYS session, which must have every expected
    regular-session minute and valid OHLCV. It uses 48 equal bins from that session's low to
    its high, with the last bin including its right edge. POC is the center of the largest
    bin (ties: lower). The value area starts at POC and adds the larger adjacent bin (ties:
    lower first) until it holds 70% of the volume; VAL and VAH are its edges. An unusable
    previous session skips the day. An older session is never substituted.
  - **ATR and offsets**: A = TA-Lib ATR(14) of completed daily bars through the previous
    session, with b = 0.02 A and d = 0.03 A.
- **VWAP**: reset at each open. It is the cumulative Massive minute `vwap` x volume while
  every positive-volume minute so far has that field. After that, HLC3 x volume over the
  whole prefix, labeled `hlc3_approximation`. It is read at the signal bar's completion and
  exactly 15 minutes earlier, both with the method valid at the signal, so a later switch
  never revises an earlier signal.
- **Relative volume**: 5-minute volume / median of the same session-relative slot over the
  20 preceding sessions (today excluded, at least 10 complete observations).
- **Sequence** (complete 5-minute bars; a missing or incomplete bar resets):
  1. **Balance**: at least 3 of the 4 bars before the excursion close in [VAL, VAH].
  2. **Excursion**: a close < VAL - b after a close that was not (a fresh break).
  3. **Reclaim**: the first bar within the next 6 with a close in (VAL + b, POC) and
     close > open. A high at POC first expires the setup. The reclaim never signals.
  4. **Retest**: the first bar within the next 3 that overlaps [VAL - d, VAL + d] and closes
     in (VAL + b, POC).
  5. **Invalidation**: checked first on every bar after the reclaim, including the trigger:
     a low below the excursion low (tracked through the reclaim bar), a high at POC, or a
     close < VAL - b.
- **Confirmation**:
  - **Candle**: close > open, body >= 50% of the range, close in the top 25% of the range.
  - **Relative volume**: >= 1.20.
  - **VWAP**: close < VWAP, and VWAP minus VWAP 15 minutes earlier >= -0.10 A.
  - **Reward/risk**: stop = excursion low - 0.01 A and target = previous POC, with
    reward/risk >= 1.25 at the signal close.
- **Limits**: signal bars complete 09:50-11:30 ET inclusive (`loose`: until 15:30). Each
  session allows one candidate per side, regardless of outcome, and one active setup at a time.
- **`reclaim_lvn`**: at the reclaim only, it builds a 16-bin approximate profile of the
  minutes from the excursion-extreme bar through the reclaim, which needs at least 10
  minutes. Volume is smoothed with [1,2,1]/4, and the first and last two bins are never
  candidates. A node has positive volume, smoothed volume below both neighbors, and
  smoothed volume at most half of the smaller of the peaks on each side. Its center must be
  inside prior value and below the reclaim close. The node nearest VAL wins (ties: lower),
  and its bin replaces the edge band; the retest must close above it. With no node there is
  no setup, the edge is never substituted, and the count is reported as `no LVN`.

`--barriers` reuses the idealized fixed-exit comparison. Entry is the open of the minute
starting at signal completion, never the retest extreme. Stop and target stay frozen, and
the exit is the stop, the target or the session close, with the same gap, ambiguity and
missing-data rules. `entry_status` is `invalid` when the actual entry breaks the geometry
or gives reward/risk < 1.25; the candidate is kept and nothing replaces it. Long and short
stay separate. Short borrow feasibility and costs are not modeled.

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

For `auction_reclaim`:

- `candidates.parquet` adds the variant (`location`, `profile_bins`, `rvol_filter`),
  `side`, `profile_method`, `prev_session`, `val`/`poc`/`vah`, `prev_atr_14`, `buffer_b`,
  `retest_tol_d`, session-relative 5-minute slots and end times for the excursion, extreme,
  reclaim and signal, `excursion_extreme`, `reclaim_close`, the retest location
  (`location_lo`/`location_hi`: the edge band or the frozen LVN bin), candle fractions,
  `rvol`/`rvol_base`, `vwap`, `vwap_15m_ago`, `vwap_change_15m`, `vwap_method`,
  `frozen_stop`, `frozen_target`, `signal_reward_risk` and, with `--barriers`, the barrier
  columns listed above.
- `summary.csv` has one row per variant, segment and side, with the aggregate setup funnel.
  It covers sessions, eligible sessions, missing profile or ATR, excursions, reclaims
  (including POC-first and expired), `lvn_no_node`, invalidations by reason, expired
  retests, and resets for missing bars and for the 11:30 cutoff. No per-bar records are
  kept.
- `stability.csv`: candidates and mean 30-minute directional return (and barrier R) per
  variant, segment, side and year, then month.
- `candidate_<n>_<session>_<side>.png`: the first six candidates of the baseline (or single)
  configuration by time. Each shows the previous-session profile, candles, VAL/POC/VAH,
  VWAP, the excursion, reclaim and signal bars, the retest location and the stop/target.

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

`auction_reclaim` is the exception: it needs a small per-session state machine rather than
a vectorized mask. It returns `(mask, triggers, funnel)`: the snapshot rows are built only
when a signal fires, and the funnel holds aggregate counts.

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
