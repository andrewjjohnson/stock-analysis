# stock-analysis

A personal proof of concept for quickly testing an intraday stock signal on Massive
one-minute bars and comparing what happened afterwards. It is a **signal study**, not a
backtest: no orders, fills, costs, position sizing or P&L. There are two narrow, clearly
labeled exceptions: the optional `--barriers` comparison of the opening-range reversal and
auction reclaim, and `sauce.py`, which simulates the VWAP + Sauce trades gross on the
underlying. It is independent of Quant Forge and shares no code or data with it.

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
the EMA single, sweep and split runs, two opening-range reversal runs, the auction reclaim
comparison and two VWAP + Sauce runs into `output/demo/`:

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

## VWAP + Sauce (`sauce.py`)

One trader's 2-minute-chart method: a multi-day anchored VWAP with standard-deviation bands,
plus an 8/48 EMA pair ("sauce"). Unlike the studies above it is a **trade simulation**, so it
has its own script: entries at the next bar's open, moving band targets, stops, and one
position at a time. Results are gross price moves per share of the **underlying**, with no
options, costs or slippage. They are not the trader's options P&L.

```bash
uv run python sauce.py --ticker SPY --start 2022-04-01 --end 2024-12-31 --out output/sauce_spy
uv run python sauce.py --ticker SPY --start 2022-04-01 --end 2024-12-31 --compare --out output/sauce_spy_compare
uv run python sauce.py --ticker SPY --start 2022-04-01 --end 2024-12-31 --continuation --vwap-to-vwap --out output/sauce_spy_all
```

With the default 120 warm-up sessions, a 2022-04-01 start reuses the auction_reclaim cache
files. `--compare` runs the two questions the brief asks to test, crossed: VWAP lookback 3
and 4 sessions, with the slow-parallel requirement on and off. Nothing is selected. Like
auction_reclaim, 2025 onward has not been run; keep it back until the rules are fixed.

**Indicators** (read when a bar completes; `strategies/vwap_sauce.py`):

- **Bars**: 2-minute bars anchored to each session open, built from the one-minute bars. A
  bar needs one of its two minutes (`--min-coverage 0.5`). Massive emits no bar for a
  minute without trades, so that is what a chart shows. Regular hours only.
- **VWAP**: typical price (H+L+C)/3 x volume, anchored at the open of the session
  `vwap_lookback_sessions - 1` sessions ago and re-anchored at every open. sigma is the
  volume-weighted standard deviation of typical price around VWAP over the same window.
  Bands are VWAP +/- 1, 1.5 and 2 sigma (U1, U1.5, U2 and L1, L1.5, L2). The value is blank
  if an earlier session in the window has no data.
- **FAST / SLOW**: TA-Lib EMA 8 / EMA 48 of the 2-minute close, continuous across sessions.

**Setup A**, the band-extension reversal, on the lower band (long). The upper band mirrors
it as a short, and a test checks that the two sides are exact mirrors.

1. **Arm**: a close with FAST below L2 (price alone below L2 does not arm). `setup_extreme`
   is the lowest low while armed.
2. **SLOW goes parallel**: 0 <= SLOW - L2 <= 0.25 sigma, and the least-squares slope of
   SLOW - L2 over the last 5 bars is >= 0 or smaller in size than 0.02 sigma per bar.
   **Invalidation**: SLOW closes below L2.
3. **Optional trendline**: a line fitted to the last 10 SLOW values; confirmation is a
   close crossing it in the reversal direction.
4. **Entry**: the first close with FAST back above L2, if SLOW went parallel, fills long at
   the next bar's open. The optional **fade** enters once SLOW is parallel, while FAST is
   still below L2, but only when the session's realized P&L is above zero.
5. **Target**: the moving L1.5 band, re-read every bar (`--extended-target`: L1).
6. **Stops**: FAST closes back below L2 (the structure stop, filled at the next open);
   price trades below `setup_extreme` (the price stop); the session's last bar (the time
   stop).
7. **Optional re-entry**: after a structure stop, a bar that touches FAST and closes above
   it re-enters at the next open, at most twice per setup.

**A-continuation** (`--continuation`): while FAST is below L2, a close above FAST and then a
later close below it go short at the next open. Its price stop is the high of that bounce
bar (the latest bar that closed above FAST). The trade exits on that stop, when FAST closes
back above L2 (which hands off to the Setup A long at the same open), or on the time stop.

**B, VWAP to VWAP** (`--vwap-to-vwap`): FAST closing across a ladder level (L2, L1.5, L1,
VWAP, U1, U1.5, U2) sets a bias toward the next level. The first later bar that touches FAST
and closes back in the bias direction enters at the next open, with the next level as the
target. FAST closing back across the crossed level is a head fake: no trade, or an exit at
the next open. Crosses of the center VWAP are reported separately.

**One position per symbol.** An open position blocks new entries, which are counted as
blocked. On the same bar, Setup A comes before the continuation, which comes before B.

**The brief's assumptions.** These rules were not stated by the trader. Each is a flag,
with the brief's default:

| Brief assumption | Flag | Default |
|---|---|---|
| regular hours only | none (extended hours are not supported) | on |
| instrument | `--instrument` | `underlying` |
| VWAP window, today included | `--vwap-lookback-sessions` | 3 (also test 4) |
| "near" distance | `--parallel-distance-sigma` | 0.25 |
| slope window / flat threshold | `--parallel-lookback` / `--slope-threshold` | 5 bars / 0.02 sigma per bar |
| slow-parallel required | `--[no-]require-slow-parallel` | on |
| trendline confirmation | `--trendline-confirm` / `--trendline-lookback` | off / 10 |
| fade entry, only on a green day | `--fade-entry` | off |
| target band / extended target | `--target-level` (0 = VWAP) / `--extended-target` | 1.5 / off |
| structure, price and time stops | `--[no-]structure-stop`, `--[no-]price-stop`, `--[no-]time-stop` | all on |
| re-entry / maximum re-entries | `--reentry` / `--max-reentries` | off / 2 |
| continuation module | `--continuation` | off |
| VWAP-to-VWAP module | `--vwap-to-vwap` | off |

**Choices the brief leaves open.** They are made here, listed in
`vwap_sauce.IMPLEMENTATION`, and written to `settings.json`:

- **Intraday setups**: a setup ends at its session's last bar. A signal on that bar has no
  next open, so it does not trade.
- **One setup per excursion**: after a setup ends, FAST must close back inside the band
  before that side can arm again. Each session starts fresh.
- **Latching**: slow-parallel and the trendline confirmation stay true once seen while
  armed. If FAST comes back inside before them, the setup ends without a trade.
  `--no-latch-slow-parallel` instead needs SLOW to be parallel on the entry bar itself.
- **Contiguous windows**: the slope and trendline windows must be contiguous and in one
  session, so neither exists in a session's first 4 bars (9 for the trendline).
- **Fade and re-entry trades** start with FAST outside the band, so their structure stop
  applies only after FAST has closed back inside.
- **Fills**: targets and price stops fill inside the bar against the level known at the
  previous close. They fill at that level, or at the open if the bar opens beyond it. An
  entry that opens beyond its target therefore exits at once, for zero; the summary counts
  these as "exit at the entry open". If one bar touches both the stop and the target, the
  stop is counted (the ambiguous count); `--ambiguous-fill target` gives the target instead.
- **Re-entry** follows only a structure stop. While flat, the setup ends if SLOW crosses
  the band, price trades beyond `setup_extreme` or the target is touched.
- **Continuation stop**: `setup_extreme` lies on the continuation's favorable side, so its
  price stop is the bounce bar's high instead (low, for a long). Its FAST-back-inside exit
  is logged as `structure_stop`.
- **Experiments, not from the brief**: `--stop-buffer-sigma X` moves the price stop X sigma
  further away, and `--target-level 0` targets VWAP itself.
- **B**: when several levels are crossed at once, the outermost counts. A new cross
  replaces a bias still waiting for its pullback. A cross beyond U2 or L2 has no next level
  and makes no setup.

**Outputs** (in `--out`):

- `setup_log.csv`: one row per setup instance and trade; a setup without a trade has one
  row with empty trade columns. Columns:
  - identity: config, setup (`A` / `A_cont` / `B`), band, side, session, instance
  - setup times: `arm_time`, `setup_extreme`, `slow_parallel_time`,
    `trendline_confirm_time`, `pullback_time`
  - B only: `level_crossed`, `center_cross`
  - how it ended: `invalidation_time` / `invalidation_reason`, `outcome`, `n_blocked`
  - per trade: `trade_no`, `signal_time`, `entry_time`, `entry_price`, `entry_type`
    (standard / fade / reentry), `target_level`, `target_price_at_entry`, `stop_price`,
    `sigma_at_entry`, `exit_time`, `exit_price`, `exit_reason` (target / structure_stop /
    price_stop / time_stop), `ambiguous`, `pnl_usd` (per share), `pnl_sigma`, `pnl_pct`,
    `bars_held`
- `summary.csv`: one row per configuration, setup and side, and for B also center and
  off-center. It has setups and trades per month, win rate, average win and loss,
  expectancy, total and max drawdown (each in $ and in sigma), exit reasons, entry types
  and how the setups ended. Setup A adds the % of armed setups invalidated by SLOW crossing
  the band before any entry; B adds the % of head fakes.
- `monthly.csv`: setups, trades, wins and P&L per setup and month.
- `entry_baseline.csv` (with `--random-baseline`): the four comparison rows per
  configuration, with the share of random sets beaten and their 5th-95th percentile.
- `session_<date>.png`: an overlay for checking trades against the rules. It shows
  2-minute candles, VWAP, the six bands (the target band in green), FAST and SLOW, with
  markers for arm, SLOW parallel, invalidation, entry and exit, and one line per setup
  underneath. The charted sessions are the first `--charts` (default 12) with a Setup A
  setup or a trade, by time and never by outcome; add others with `--chart-dates`.

**Is the entry better than chance?** `--random-baseline 200` compares Setup A's entries
with matched random ones and writes `entry_baseline.csv`. Each Setup A trade keeps its side
and its target and stop distances (in sigma). The comparison has four groups:

- Setup A under its actual rules, for reference.
- Setup A's entries exited by a fixed bracket at those distances (or the session close).
- 200 random entries per trade: bar opens in the same session, same side, same bracket.
- The same entries taken in the opposite direction.

Every bracket is exited the same way, with `outcomes.barrier_exits` on one-minute bars, so
only the entry differs. The run also reports what share of 2,000 random sets of the same
size Setup A's average beats. Beating random entries is not the same as making money:
compare the averages too.

Setup A is rare by design; nothing is loosened to make more trades. On cached SPY and QQQ
minutes for 2022-04-01 to 2024-12-31, about 15 setups arm per month and 4.5 to 5 of them
trade with the defaults. The run above prints this frequency every time. Max drawdown is
over one share per trade, in trade order: a check on the sequence, not portfolio accounting.

## Alert check (`alerts.py`)

Measures how often a log of posted SPY calls was right, using independent Massive minutes. The
log is a CSV with one row per call (`date, weekday, strategy, time_et, direction, action,
ticker, ref_price, score, flag, alerts_posted, notes, source_line`): `intraday` calls at 10:00
ET and `overnight` calls at 15:55 ET, each BULLISH, BEARISH or NEUTRAL. Keep the log in the
git-ignored `data-files/`. It is a signal study of the underlying, not option P&L.

```bash
uv run --env-file .env python alerts.py --csv data-files/<log>.csv --out output/alerts
```

- **Checks:** row consistency (weekday, action, score, ticker, duplicates), alert times
  against the schedule and the XNYS session, missing rows (reported, never filled), and
  `ref_price` against nearby minute bars, flagging gaps over 0.1%.
- **Reference:** the Massive price known at the alert time, the close of the minute bar
  ending then. The CSV's `ref_price` is a sensitivity check.
- **Horizons:** intraday to the same close (primary) and the next close; overnight to the
  next open (primary: the open of its first regular minute), 10:00 the next session and the
  next close. Every regular minute from the reference bar to the end point must exist,
  otherwise the outcome is unavailable. One that ends in today's session or later is pending.
  Both are kept in the table and left out of the statistics.
- **Statistics:** hit rate (signed return > 0) with a Wilson interval next to the base rate
  (SPY up over the same windows, which an always-bullish rule scores), the paired difference
  with a bootstrap interval over dates, 10,000 shuffles of the calls across dates keeping the
  bullish/bearish counts, and a coin-flip binomial test. Exploratory sections cover the other
  horizons, premium-selling distances, overnight move sizes, ex-dividend windows (dates from
  Massive's dividends endpoint), NEUTRAL move sizes, the overnight score, naive momentum rules,
  halves and months, flagged rows and the `ref_price` reference.
- **Output:** `alerts_with_outcomes.csv`, `summary.csv` (one statistic per row),
  `report.md`, `hit_rates.png` and `overnight_score.png`.

## Alert spreads (`alert_spreads.py`)

Trades the intraday alerts from the same log as same-day $1 SPY vertical spreads, on Massive
option minute bars (an options data plan is required). BULLISH sells the put at the first
strike at or above SPY at 10:00 ET and buys the put $1 below; BEARISH sells the call at the
first strike at or below and buys the call $1 above. The user's rule (`--take-profit`, `--stop`)
defaults to a take profit at 80% of the credit with no stop, otherwise an exit at 15:30 ET.
`--rule-since` is the date that rule was adopted: the report keeps the trades before it (which
chose the rule, so they flatter it) apart from the trades since, which are the real test.

```bash
uv run --env-file .env python alert_spreads.py --csv data-files/<log>.csv --out output/alert_spreads
```

- **Baselines with identical exits, on the same days:** the bull put spread every day, the
  bear call spread every day, and the alerts' calls shuffled across the days (10,000 times).
- **Grid:** take profit (10–90% of the credit, or none) x stop (10–90% of the max loss, or
  none), all with the 15:30 exit. Exploratory; every cell is reported.
- **Context** (`--context`, default on): both directions on every session the data plan
  covers, with no alerts, to show what the structure and exits do on their own.
- **Fills:** the plan has no bid/ask quotes, so the spread trades only in minutes where both
  legs traded. Entry and the time exit use the opens of the first such minute (within 5
  minutes); take profits and stops use minute closes. Costs come from `--slippage` ($/share
  per leg per fill) and `--commission` ($/contract per leg per fill), both 0 by default
  (a commission-free broker; replaying actual fills at traded prices matched them within
  about $1 a trade). The user's rule is also shown with +$0.01 and +$0.03 slippage, which
  decide the sign of the result. Early assignment is ignored.
- **Consistency:** every grid cell gets a consistency score (average trade ÷ its standard
  error), win rate, profit factor, worst trade, drawdown and profitable months, scored with
  the base costs and with +$0.01 slippage. A month-by-month check re-picks the most
  consistent rule without each month and trades it in that month, to show whether choosing
  the best cell works on data it didn't see.
- **Strike placement:** the same $1 spread moved −3 to +3 dollars (+ = deeper in the money:
  more credit, smaller max loss, needs the move). Each placement is run under the 50% and 80%
  take profits on the alert days (with shuffled calls and wider slippage) and, with context,
  every session without alerts. Placements deeper in the money are priced from their
  out-of-the-money twins by put-call parity ($1 minus the other right's spread on the same
  strikes), because thin in-the-money prints showed profits in both directions without any
  alerts; the own-print result is shown beside it. Days without usable prices at a placement
  are counted, with what the offset-0 spread made on them. `--no-offsets` skips it (the first
  run downloads about 12 more contracts per session).
- **Breakeven stop:** the user's rule and the 50%/80% rules are also run with a stop at
  breakeven that arms once profit reaches 40% of the credit, on the alert days and every
  context session, trade by trade against the same spreads without it.
- **Scaling and drawdowns:** the user's rule resampled in runs of 5 consecutive trades
  (a block bootstrap, so losing clusters survive) over 3 and 12 months, under four
  scenarios: as measured, an honest estimate (the month-by-month re-picking average, when
  the rule was chosen on these trades), +$0.01 slippage, and no edge. Results per spread
  and at each `--contracts` size (default 1 5 10 20), plus what a bad-case drawdown looks
  like inside (trades, losers, winners, longest streak).
- **Output:** `trades.csv`, `grid.csv`, `monthly_check.csv`, `offsets.csv`, `scaling.csv`, `scaling.png`,
  `context_grid.csv`, `report.md`, `equity.png`, `grid.png`, `consistency.png`,
  `offsets.png` and `context_grid.png`. Dollars are per one spread, before taxes.

## Mean reversion (`meanrev.py`)

A daily signal study of whether SPY mean-reverts over 1–5 sessions, to choose what to try with
credit spreads. It uses SPY plus context ETFs (QQQ, IWM, TLT, HYG, GLD, UUP and VIXY as a VIX
proxy) from Massive minute bars, back-adjusted for cash dividends.

```bash
uv run --env-file .env python meanrev.py --out output/meanrev                # design period only
uv run --env-file .env python meanrev.py --final-test --out output/meanrev   # also the holdout, once
```

- **Timing:** every indicator uses prices up to 10 minutes before the close (15:50 ET) plus
  earlier full sessions, so a signal can be acted on the same afternoon. Outcomes run from that
  15:50 price to the 15:50 price 1, 2, 3 or 5 sessions later, plus the low/high in between.
- **Periods:** design = sessions through 2024-12-31 (outcomes must end by then); holdout =
  2025-01-01 onward, read only by `--final-test`. Every choice (52 indicators, dip and rally
  events, classic rules, models, thresholds) is fixed in the code; the three primary holdout
  hypotheses are picked from design results by `select_primary`.
- **Analyses:** rank correlation of each indicator with forward returns (block-bootstrap
  intervals, Benjamini-Hochberg q-values); dip and rally event studies with credit-spread odds
  (how often a strike k% beyond the entry would have expired worthless); regime splits; classic
  rules (Connors RSI(2), IBS, 3 down days, Bollinger) traded one position at a time; and
  purged walk-forward logistic regression and boosted trees, next to a simple oversold score.
- **Output:** `report.md`, `indicators_*.csv`, `events_*.csv`, `regimes_*.csv`, `rules_*.csv`,
  `ml.csv`, `ml_importance.csv`, `features.parquet`, `indicators.png`, `event_paths.png`,
  `ml.png`. A signal study of SPY, not option P&L.

## Put spreads after dips (`dip_spreads.py`)

Prices the mean-reversion study's dip signals as SPY put credit spreads with real option prices
(Massive option minute bars, from 2024-10-01), and compares each signal with the same spread
opened on every session.

```bash
uv run --env-file .env python dip_spreads.py --out output/dip_spreads
```

- **Primary test (fixed before any option price was seen):** "3+ down days in a row" at 15:50;
  short put at the $1 strike at or below 1% under SPY's 15:50 price, long put $5 lower,
  expiring 5 sessions later, held to expiry; holdout sessions (2025-01-01 on). The Connors
  RSI(2) entry is the second pre-registered test.
- **Exploratory grid:** expiries 1/2/3/5 sessions; short strike just in the money (the first
  strike above SPY, as in the alert spreads), at the money, or 0.25% / 0.5% / 1% / 2% below;
  $1 or $5 wide (compared by return on risk, since a $1 spread risks far less), held to expiry or a 50%/80% take profit, each with and without a breakeven
  stop (armed once profit reaches 40% of the credit), closed at 15:50 on the first session SPY is
  up (at most 5 sessions; also combined with the 80% take profit), other dip signals, and Oct-Dec 2024.
- **Fills:** both legs must print in the same minute; entry at the first such minute from
  15:50; settlement at intrinsic value from SPY's actual close on the expiry day. Strikes use
  SPY's unadjusted price. Costs default to 0 (`--slippage`, `--commission`) and are always shown
  with +$0.01 and +$0.03 slippage.
- **Output:** `report.md`, `spreads.csv` (every spread, exit and cost), `stats.csv`,
  `one_at_a_time.csv`, `signals.png`, `grid.png`, `equity.png`. Contract bars are cached per
  contract in `data/cache/options_multi/` (about 11,000 contracts on the first run).

## Put spreads after dips in single stocks (`stock_dip_spreads.py`)

Runs the same put-spread test on single stocks (MSFT, AAPL, AMZN and META by default, `--tickers`
to change), with SPY's spread alongside, and adds RSI(2) < 10 as a second trigger.

```bash
uv run --env-file .env python stock_dip_spreads.py --out output/stock_dip_spreads
```

- **Primary test (fixed before any stock option price was seen):** "3+ down days in a row" at
  15:50; short put at the listed strike at or below 1% under the 15:50 price, long put at the
  listed strike nearest 1.5% of the price lower; the first listed expiry at least 2 sessions out
  (weeklies expire on Fridays, so 2-6 sessions); 80% take profit, else held to expiry; spreads
  open over earnings skipped. Compared with the same spread on every session. The second test
  adds RSI(2) < 10 as an alternative trigger.
- **Exploratory:** at the money and 2% below, holding to expiry, RSI(2) < 10 alone, keeping the
  earnings trades, and all tickers traded together (one spread at a time per ticker).
- **Earnings:** not in the data plan, so in each late-month reporting window the session with the
  biggest stock-specific volume jump and the one with the biggest stock-specific overnight gap
  are flagged, with a session either side. The windows fit companies that report in late
  Jan/Apr/Jul/Oct; check the dates listed in the report for other tickers, or pass exact dates
  with `--earnings-csv` (columns `ticker,date`, the first session after each report).
- **Fills and costs:** as in `dip_spreads.py`, shown at traded prices and with +$0.03, +$0.05
  and +$0.10 slippage per leg (single-stock options trade wider than SPY's). Strikes and expiries
  come from Massive's options contracts reference; a split inside the option window stops the
  run.
- **Output:** `report.md`, `spreads.csv`, `stats.csv`, `tickers.png`, `combined.png`.

## Gap-down recoveries (`gap_recovery.py`)

A price study, no options: when a ticker opens at least 1% below the previous close (SPY, MSFT,
AAPL, AMZN and META by default), how often does the gap fill, when, and how much comes back if it
doesn't?

```bash
uv run --env-file .env python gap_recovery.py --out output/gap_recovery
uv run --env-file .env python gap_recovery.py --min-gap 0.25 --out output/gap_recovery_small   # small gaps too
```

- **Gap:** the first regular minute's open against the previous session's last regular close; on
  an ex-dividend morning the payout is taken out of the gap. A missing session never makes a gap.
- **Measured per gap:** whether and when the price trades back at the previous close that session;
  the best recovery and the close as a % of the gap; how far it fell first; whether it fell another
  gap's worth below the open, and which of the two came first; fills within 1, 2, 3, 5 and 10
  sessions.
- **Splits:** gap size (0.25-0.5%, 0.5-1%, 1-2%, 2-3%, 3-5%, 5%+; buckets below `--min-gap` stay
  empty) and kind: earnings (inferred, see `stock_dip_spreads.py`), with the market (SPY also opened
  1%+ lower, or `--min-gap` if that is smaller) or the stock alone.
- **Filter tests** (fixed before any filtered result was seen): skip earnings; then skip the first 30
  minutes and measure from the 10:00 price (gaps already filled by 10:00 count as missed); then split
  by whether the 10:00 price is above or below the open. The yardstick is the race: did the price reach
  the previous close before falling as far again below the entry price? About 50% means no edge.
  A higher fill rate alone is not an edge, because a stock that has already bounced has less ground
  to cover and a closer stop.
- **Renamed tickers:** sessions Massive files only under a former ticker are filled from it (META's
  2022-01-31 to 2022-06-08 sessions come from FB).
- **Output:** `report.md`, `gaps.csv` (one row per gap), `summary.csv`, `outcomes.png`, `later.png`,
  `filters.png`.

## Intraday mean reversion for a scalper (`scalp_meanrev.py`)

A signal study on stock minute bars, no options: after a stretched 5-minute bar, does price snap
back over the next 5-30 minutes by more than a scalper's costs? Signals only from 10:30 to 15:00 ET.

```bash
uv run --env-file .env python scalp_meanrev.py --out output/scalp_meanrev                 # design period
uv run --env-file .env python scalp_meanrev.py --final-test --out output/scalp_meanrev    # adds the holdout
```

- **Tickers:** SPY, QQQ, IWM, MSFT, AAPL, AMZN, META; each alone and pooled (index ETFs, stocks).
- **Round 1 (fixed before any result):** 27 signals, each long (buy the stretch below) and short (fade
  the stretch above): 9/20 EMA stretches in ATRs, VWAP bands, RSI(2) and RSI(14) extremes, Bollinger
  bands, runs of same-colour bars, sharp 15-minute moves, and seven combinations (with or against the
  20/50 EMA trend, with the daily trend, a Bollinger hook, RSI(2) plus a VWAP band, a run plus VWAP).
- **Round 2 (fixed after round 1 came back empty):** stretches on 2x normal volume or on quiet volume,
  on volatile days, after 3+ daily down (up) days, on 0.5%+ gap days, and the core signals on
  2-minute bars. Each round has its own multiple-testing correction.
- **Outcome:** the move in bps over 5, 10, 15 (primary) and 30 minutes from the signal bar's close, in
  the signal's direction; the excess over any bar in the same half hour; and a +1 / -1 ATR bracket
  within 30 minutes. Uncertainty is clustered by day.
- **Selection:** design 2022-2024; a signal qualifies with enough events, a positive excess with BH
  q < 0.10, and an average move above an illustrative round-trip cost (1 bp ETFs, 2 bps stocks). The
  2025+ holdout is read only with `--final-test`, for the qualifiers.
- **Output:** `report.md`, `stats.csv`, `design.csv`, `events.parquet`, `design.png` (and with
  `--final-test`, `holdout.csv`, `holdout.png`).

## Intraday momentum (`intraday_momentum.py`)

Does the morning's move predict the last half hour (published for SPY)? At 15:30, long after an up
morning (previous close to 10:00, net of a dividend paid that morning), short after a down one, to the
close; also the 15:00-15:30 move and both agreeing; volatile mornings split out. SPY (primary), QQQ,
IWM; design to 2024, the 2025+ holdout only with `--final-test`.

```bash
uv run --env-file .env python intraday_momentum.py --out output/intraday_momentum
```

Output: `report.md`, `stats.csv`, `days.csv`, `momentum.png`.

## Trend following on ETFs (`trend_etfs.py`)

Long in uptrends, short (or flat) in downtrends, for SPY, QQQ, IWM, TLT, GLD, UUP and HYG at equal risk
(10% a year each, from 60-session volatility, at most 2x), after 2 bps per unit traded. Rules: 1- and
3-month momentum and the 50- and 100-session average, long/short and long/flat; decisions at 15:50
from 2022-03-01. Compared with buying and holding SPY or the seven ETFs, with SPY dip buying (meanrev's
"3 down days; exit on the first up day"), and with half trend, half dip buying.

```bash
uv run --env-file .env python trend_etfs.py --out output/trend_etfs
```

Output: `report.md`, `rules.csv`, `daily.csv`, `trend.png`. Five years hold one bear market, so the
longer classic rules (6-12 months, 200 days) need more history.

## Call credit spreads, 30-45 days out (`call_spreads.py`)

Sells a SPY call credit spread every session at 15:50 with real option prices (from 2024-10-01): the
weekly expiry closest to 30 or 45 days out (more than 21), the short call at the $5 strike whose delta
is closest to 0.20, 0.30 or 0.40, the long call $5 or $10 higher, entered as a working order (the first
minute both legs trade, through 10:30 the next morning).

```bash
uv run --env-file .env python call_spreads.py --out output/call_spreads
```

- **Delta:** from each strike's own implied volatility (Black-76 on the forward from the at-the-money
  call and put, last trades from 14:50 to 15:50); no quotes or Greeks are in the data plan.
- **Exits:** held to expiry, 50% take profit, closed at 21 days to expiry, 50% or 21 days, and 50%
  with a stop at a loss of 2x the credit. Primary: 45 days, 0.30 delta, $5 wide, 50% or 21 days.
- **Entry filters:** after 3+ up days, RSI(2) above 90, below the 50-day average, volatile periods.
- **Costs:** traded prices, +$0.03 (main tables) and +$0.05 per share per leg per fill.
- **Output:** `report.md`, `spreads.csv`, `stats.csv`, `entries.csv`, `grid.png`, `equity.png`.

## Iron condors, 30-45 days out (`iron_condors.py`)

The direction-neutral version of the call spreads: a call credit spread above and a put credit spread
below, same expiry, both short strikes at the same delta (0.20, 0.30 or 0.40), $5 or $10 wings, 30 or 45
days out, entered every session like `call_spreads.py`. Exits apply to the condor's total value; filters
are volatile, calm and below the 50-day average.

```bash
uv run --env-file .env python iron_condors.py --out output/iron_condors
```

Output: `report.md`, `condors.csv`, `stats.csv`, `grid.png`, `equity.png`; the report also splits each
condor's result into its call and put sides.

## Three red bars by timeframe (`red_bars.py`)

Does the "3 down days" bounce carry to shorter bars? After 3+ red bars in a row on 5-minute, 15-minute,
1-hour, 4-hour and daily bars (SPY, QQQ, IWM), the move over the next 1, 3 and 5 bars against any bar of
the same size, plus a +/-1 ATR race. Red means a red candle (close below open) or a down close (below the
previous close). Design to 2024; the 2025+ holdout only with `--final-test`, for qualifiers.

```bash
uv run --env-file .env python red_bars.py --out output/red_bars
```

Output: `report.md`, `stats.csv`, `design.csv`, `red_bars.png`.

## Bottoming and topping candle patterns (`bar_patterns.py`)

Red-red-red-green-red (bought) and green-green-green-red-green (faded), each also with a higher low /
lower high, on 5-minute, 15-minute and 1-hour bars for SPY, QQQ and IWM, measured like `red_bars.py`
(next 1/3/5 bars in the signal's direction against any bar, a +/-1 ATR race; design to 2024, holdout only
with `--final-test`).

```bash
uv run --env-file .env python bar_patterns.py --out output/bar_patterns
```

Output: `report.md`, `stats.csv`, `design.csv`, `patterns.png`.

## Support and resistance levels (`levels.py`)

Do widely watched levels hold? Levels known before the open: yesterday's high, low and close, last week's
high and low, today's pre-market high and low, yesterday's volume POC and value area (bar-approximated),
and the nearest whole dollar and multiple of $5 below and above the open. For the first touch of each level
during regular hours, a race from the level: it held if price moved 0.1 (or 0.25) daily ATR back away
before going as far through it. Every real level is compared with fake ones measured the same way: 10 per
level and session at distances from the open drawn from the same level's other days, plus (a check added
after the first results) fakes 0.2-0.4 ATR either side of the real level on the same session. SPY, QQQ and
IWM; design to 2024, the 2025+ holdout only with `--final-test`, for qualifiers.

```bash
uv run --env-file .env python levels.py --out output/levels
uv run --env-file .env python levels.py --chart 2024-08-05 2024-03-04:2024-03-08 --chart-ticker SPY --out output/levels
```

Output: `report.md`, `stats.csv`, `design.csv`, `pooled.csv`, `near.csv`, `events.parquet`, `levels.png`.
`--chart` skips the study and writes `charts/<ticker>_<date>.png` (the session's 5-minute candles with every
level, after yesterday's session and today's pre-market in grey, marking each level's first touch and whether
it held or broke) plus `charts/<ticker>_levels.csv`.

## Swings on a 5-minute chart (`swings.py`)

Swings come from a zigzag on 5-minute bars that restarts each session (a turn needs a reversal of 0.15
daily ATR, about $0.90 on SPY). Part 1: do the session's own swing highs and lows act as support and
resistance? Each swing becomes a line when it is confirmed, but not before 11:30 ET, and is measured like
`levels.py` (first touch, the +/-0.1 ATR race, random and near fake lines). Part 2: is the time between peaks
and valleys more regular, or more predictable, than in random-direction copies of the same sessions (each
5-minute bar keeps its size but is flipped up or down at random, so the volatility pattern stays and any
memory goes)? SPY, QQQ and IWM; design to 2024, the 2025+ holdout only with `--final-test`.

```bash
uv run --env-file .env python swings.py --out output/swings
uv run --env-file .env python swings.py --chart 2024-03-05 --chart-ticker SPY --out output/swings
```

Output: `report.md`, `stats.csv`, `design.csv`, `near.csv`, `timing.csv`, `events.parquet`, `swings.png`.
`--chart` skips the study and writes `charts/<ticker>_<date>.png`: the session's 5-minute candles, its zigzag
with minutes per leg, and each swing line from the time it is drawn, with its first touch and outcome.

## Breakout trades at key levels (`breakouts.py`)

A trade simulation of the `levels.py` finding that price runs through some levels: a stop order at
yesterday's high or last week's high (long, when the session opens below it) or at yesterday's close
(either side), filled at its first touch, with a stop back through the level and a target beyond it
(0.1 ATR primary; 0.25 ATR and stop-only exits too), 0/1/2 bps per side. Yesterday's and last week's low
(short breakdowns) are reported as a check. TSLA, SPY, QQQ and IWM; the same trades at random fake levels
as the baseline; design to 2024, the 2025+ holdout only with `--final-test`, for qualifiers.

Minute bars can't show whether a dip to the stop inside the fill minute came before or after the fill, so
every result has two bounds: *worst* (fixed in advance; it makes even random levels lose several bps a
trade) and *best* (random levels come out near zero).

```bash
uv run --env-file .env python breakouts.py --out output/breakouts
```

Output: `report.md`, `stats.csv`, `tests.csv`, `trades.parquet`, `breakouts.png`. Results are per share of
the stock or ETF, gross of commissions, never options P&L.

## TSLA breakout check with shares and options (`breakout_check.py`)

The one-time 2025-26 check of the TSLA breakout trades from `breakouts.py`: long through yesterday's high,
through yesterday's close (either side) and short through yesterday's low, with a 0.1 ATR stop and target.
Fills are replayed on Massive one-second bars, so the order inside a minute is known. Each share trade also
buys one at-the-money option (a call for longs, a put for shorts, the first expiry after the trade date), priced
at the first option trades after the share fill and exit, plus $0.02/$0.05/$0.10 per share of slippage per fill
(option prices are trades; the plan has no quotes). Without `--final-test` it is a dry run on 2021-24, with
options for 2024-10 to 2024-12 only (the option data starts 2024-10-01).

```bash
uv run --env-file .env python breakout_check.py --out output/breakout_check                # dry run
uv run --env-file .env python breakout_check.py --final-test --out output/breakout_check   # 2025-26, once
```

Output in `design/` or `holdout/`: `report.md`, `shares.parquet`, `options.parquet`, `share_stats.csv`,
`option_stats.csv`, `tests.csv`, `settings.json`, `check.png`. One-second bars are cached in
`data/cache/seconds/`.

## Kalman SuperTrend + ADX volatility waves (`kalman_supertrend.py`)

The user's Pine Script strategy (v3.7) rebuilt from its source: 5-minute bars with extended hours, a Kalman-
smoothed close (in effect a 9-bar EMA) as the centre of a SuperTrend (ATR 7 x 2), BUY/SELL on flips during
regular hours, and the script's exits (1.5 ATR stop, 2 ATR target, breakeven after 1 ATR, the outer volatility
wave, a 12-bar time stop). Fills a minute after each signal, stops and targets on minute bars, 1-tick slippage.
Compared with random entries that use the same exits, and with the move after each flag. SPY, plus QQQ, IWM and
TSLA for reading; variants flat by the close and on a regular-hours chart.

```bash
uv run --env-file .env python kalman_supertrend.py --out output/kalman_supertrend
```

Output: `report.md`, `stats.csv`, `random_compare.csv`, `flag_moves.csv`, `years.csv`, `trades.parquet`,
`strategy.png`. Results are per share of the ETF or stock, never options P&L.

## The Kalman SuperTrend as 0DTE options (`supertrend_0dte.py`)

The trades from `kalman_supertrend.py` (SPY, flat by the close), taken the way the script's author says they
trade it: a same-day at-the-money call for a BUY or put for a SELL, at the $1 strike nearest the share fill,
bought and sold when the share trade enters and exits. Option prices are Massive one-minute trades (no quotes on
the plan) plus $0.01/$0.02/$0.05 a share of slippage per fill; per contract, no commissions. Only the sessions the
option plan covers (a rolling two years).

```bash
uv run --env-file .env python supertrend_0dte.py --out output/supertrend_0dte
```

Output: `report.md`, `stats.csv`, `trades.parquet`, `options.png`.

## Five-indicator scalper (`confluence_scalper.py`)

A strategy built on the five indicators a Reddit author lists for their 0DTE scalper, each reimplemented: MACD
(ChrisMoody, SMA signal), Squeeze Momentum (LazyBear), SuperTrend AI (LuxAlgo's k-means clustering of SuperTrend
factors), a Larry Williams swing-structure stand-in for Pure Price Action (LuxAlgo), and the ADX Volatility Waves
approximation. BUY when SuperTrend AI is bullish, MACD is above its signal, squeeze momentum is positive and rising,
structure is bullish and price is below the upper wave, all together for the first time; SELL the mirror. SPY
5-minute bars (QQQ, IWM, TSLA for reading), flat by the close; exit A = `kalman_supertrend.py`'s bracket exits,
exit B = hold until SuperTrend AI turns. Compared with plain SuperTrend AI flips and with random entries taken
while SuperTrend AI points the same way. Design to 2024; 2025-26 only with `--final-test`.

```bash
uv run --env-file .env python confluence_scalper.py --out output/confluence_scalper
```

Output: `report.md`, `stats.csv`, `random_compare.csv`, `tests.csv`, `trades.parquet`, `strategy.png`.

## Green Goose (`green_goose.py`)

An overnight options strategy from a Substack post: at 15:50 buy an at-the-money call or put, sell the next
morning. Direction: Wilder RSI(2) above 85 -> puts, below 15 -> calls, otherwise with the day's candle; overridden
by ADX moving between the DI lines (puts if -DI leads) and then by RSI(2) crossing into that zone; no trade when
ADX is above 60 (version 1: ADX/DMI 5, version 2: 6). First the stock moves from 15:50 to the next open, 9:35,
9:40, 10:00 and 11:00 in the signal's direction (SPY, QQQ, IWM, AAPL, META, five years), against calls and puts in
the same mix at random. Then SPY options from the plan's option data: delta 0.47-0.53 and theta below -0.12 by
Black-76 from traded prices, expiry 5+ days out (or the next session), version 2 out at 9:35, version 1 with its
opening-price rules and a 10% trailing stop until 11:00.

```bash
uv run --env-file .env python green_goose.py --out output/green_goose
uv run --env-file .env python green_goose.py --skip-options --out output/green_goose   # stocks only
```

Output: `report.md`, `stocks.csv`, `options.csv`, `option_trades.parquet`, `tests.csv`, `green_goose.png`.

## Green Goose indicator search (`goose_indicators.py`)

Which conditions known at 15:50 tell how SPY moves overnight, so Green Goose can keep the pieces that help and add
new ones. It tests Green Goose's own pieces (RSI(2) extremes, the candle, ADX entering the DI zone, RSI(2) stabs, ADX
above 60) and 23 new ones (IBS, Bollinger bands, closing streaks, big days, MFI, the 50-day average, MACD, the last
half hour, the opening gap, VIXY moves, volume, the turn of the month, weekends). Each is judged on the move from
the 15:50 price to the next open over 2021-10 to 2024-09, against the same condition slid to other dates, and must
point the same way on QQQ and IWM. The kept ones vote for calls, puts or no trade. The resulting rule is checked once
on 2024-10 to 2026-09, as stock moves and as Green Goose's SPY options (same contracts, version 1 exits).

```bash
uv run --env-file .env python goose_indicators.py --out output/goose_indicators
```

Output: `report.md`, `screen.csv`, `stocks.csv`, `options.csv`, `per_condition.csv`, `tests.csv`, `screen.png`,
`options.png`.

## Green Goose as $1 credit spreads (`goose_spreads.py`)

Green Goose's 15:50 direction sold as a $1 SPY vertical expiring the next session, instead of buying an at-the-money
option: a put spread when it's bullish, a call spread when it's bearish. The main strike placement is the one in
`alert_spreads.py` (short leg at or just in the money); $1 and $2 further out of the money are shown for reading.
Exits: buy back at 9:35, take profit at 80% of the credit (else buy back at 15:30), or hold to expiry. Compared with
selling a put or call spread every night and with the opposite of Green Goose, at $0-$0.03 slippage per leg per fill.

```bash
uv run --env-file .env python goose_spreads.py --out output/goose_spreads
```

Output: `report.md`, `stats.csv`, `tests.csv`, `trades.parquet`, `spreads.png`.

## TSLA breakouts with a buying-pressure filter (`breakout_flow.py`)

The two TSLA breakout rules from `breakout_check.py` (long through yesterday's high, short through yesterday's low),
taken only when order flow before the touch points the trade's way. Order flow is approximated from one-second bars:
each second's volume counts as buying or selling by its price change, and pressure is (buying - selling) / total over
the 5 minutes before the fill (1 and 15 minutes for reading). Tested on 2021-24. 2025-26 was already used for the rules'
one-time check, so it is read only with `--final-test`.

```bash
uv run --env-file .env python breakout_flow.py --out output/breakout_flow
```

Output (in `design/` or `holdout/`): `report.md`, `stats.csv`, `splits.csv`, `trades.parquet`, `settings.json`,
`pressure.png`.

## Bollinger + RSI scalp with trap filters (`band_scalp.py`)

A user-supplied mean-reversion scalp, tested as written on 5-minute bars of SPY, QQQ and 12 mega caps (NVDA, MSFT,
AAPL, GOOGL, AMZN, META, AVGO, TSLA, JPM, WMT, LLY, V).
- **Entry:** a bar pierces a Bollinger band (20, 2) with RSI(14) beyond 30/70, and a later bar closes back inside.
- **Filters:** the 1-hour 200 EMA, a cap on band-width growth, a previous-session POC/VAH/VAL at the pierce, and
  order flow from one-second bars (divergence or absorption, no accelerating pressure).
- **Exits:** a stop beyond the pierce wick, the middle band as the target, out after 10 bars.
- **Study design:** the choices the spec leaves open are fixed in the script's docstring. The report adds the filters
  one at a time. Design is 2022-24; 2025-26 is read only with `--final-test`.

```bash
uv run --env-file .env python band_scalp.py --out output/band_scalp
```

Output (in `design/` or `holdout/`): `report.md`, `stats.csv`, `tests.csv`, `setups.parquet` and `trades.parquet`
(every setup's and trade's indicator and filter states), `settings.json`, `filters.png`.

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
