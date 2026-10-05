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

**`download.load_minute_bars(ticker, start, end, *, cache_dir, refresh, adjusted, final_close, client)`**
returns `(minutes, source)`.
- `minutes` holds the raw provider rows `ts, open, high, low, close, volume, vwap,
  transactions`, including extended hours.
- `final_close` is the UTC close of the last requested session. A fresh download must
  reach that session's last regular minute, or it raises and is not cached.
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
- **Truncation guard.** The SDK's paginator returns silently if a page fails to decode,
  and pages arrive in time order. So a download counts as complete only if it reaches the
  final session's last regular minute. The date alone isn't enough: the cut can land
  mid-session, and pre-market bars already carry the final date. Caching a short result
  would quietly shorten every later run.
- **Today's session is never studied or cached.** It may still be open, or a delayed feed
  may still be filling in.
- **Cache key = ticker, first and last session (warm-up included) and adjustment.**
  Changing the dates or `--warmup-sessions` means a new download. There is no incremental
  ingestion, by design.

## VWAP + Sauce (`sauce.py`, `strategies/vwap_sauce.py`)

A trade simulation, kept apart from `run.py` because candidates and forward outcomes can't
express its rules: entries at the next open, moving targets, stops, re-entries and one
position at a time. README.md has the rules, the table of the brief's assumptions and the
choices made where the brief is silent.

**Contracts**

- **`vwap_sauce.anchored_vwap(bars, sessions, lookback)`** returns `(vwap, sigma)` arrays
  aligned with `bars`.
  - Each value holds per-session sums of v, tp x v and tp^2 x v over the previous
    `lookback - 1` calendar sessions, plus the current session's cumulative sums through the
    bar itself. sigma^2 = E[tp^2] - VWAP^2 (population, volume-weighted).
  - It is NaN until enough earlier sessions exist, or when any of them has no usable bars.
- **`vwap_sauce.simulate(bars, **params)`** takes usable bars with `session, bar_start,
  bar_end, open, high, low, close, fast, slow, vwap, sigma, in_study`. It returns the setup
  log with columns `vwap_sauce.LOG_COLUMNS`, one row per instance and trade.
  - It simulates only `in_study` rows. Warm-up rows supply only the previous bar's levels.
  - Tests call it directly with hand-built indicator columns.
- **`sauce.run_sauce(minutes, sessions, configs, min_coverage, timings)`** returns `log`,
  `summary`, `monthly`, `bars` (keyed by lookback), `minutes`, `info`, `months` and
  `configs`. It does no file I/O.

**Per-bar order** (bar i, in time order):
1. Orders placed at the previous close fill at this bar's open, the exit before the entry.
2. The price stop and the target are checked inside the bar, against the levels known at
   bar i-1's close. A bar that opens beyond a level fills at its open; otherwise the fill
   is at the level. If both are touched, the stop wins and the trade is flagged ambiguous.
3. The setup states update at the close, and the entry signals are collected.
4. A close-based exit (structure stop, continuation exit or B head fake) is scheduled for
   the next open.
5. At most one entry is scheduled for the next open: A before the continuation before B,
   and only if the book will be flat by then. Other signals are counted as blocked.
6. On the session's last bar, the time stop exits at the close, and every setup that
   isn't holding the position ends as `session_end`.

**Time semantics:**
- `arm_time`, `slow_parallel_time`, `invalidation_time` and `signal_time` are the bar
  ends at which each condition became known.
- `entry_time` is the entry bar's start, since it fills at the open.
- `exit_time` is the bar start for a fill at an open, and the bar end for a touch or the
  time stop.
- `bars_held` counts the bars the position was exposed to. An entry that exits at its own
  open has 0.

**Decisions and why:**
- **Separate script.** Wiring it into `run_study` would add a third special case to
  segments, selection and outcomes. None of them apply, because the configurations are
  fixed in advance and nothing is selected.
- **2-minute bars from the minute cache**, not a second download. They share the cache
  key, and the anchoring and coverage rules are the same as every other bar size. The
  coverage default is 0.5 because a minute without trades has no Massive bar, and one
  traded minute is what the trader's chart would show.
- **The level for an intrabar fill is the previous close's.** The level at bar i includes
  bar i's own volume and close, so using it would be lookahead.
- **Intraday setups, re-armed per excursion.** Without the re-arm rule, a setup
  invalidated while FAST stays outside would re-arm on the next bar and be invalidated
  again, many times per excursion.
- **Everything the brief leaves open is a named choice** (`IMPLEMENTATION`), not a hidden
  default, so the user can argue with each one.

**How correctness was checked (2026-09-30):**
- Hand-built paths test every rule: arm, parallel, invalidation, re-arm, each entry and
  exit type, the fills, re-entry, continuation hand-off, B, the one-position rule and
  sessions.
- A mirror test checks that the short side reflects the long side exactly.
- VWAP and sigma were compared with a direct window computation for lookbacks 3 and 4.
- Each injected bug below made at least one test in `tests/test_vwap_sauce.py` fail:
  - the target level taken from the current bar
  - VWAP ignoring earlier sessions
  - a scaled sigma
  - no SLOW-crossing invalidation
  - re-arming without FAST coming back inside
  - an ambiguous bar counted as the target
  - the flat test ignoring its threshold
  - entries filled at the signal bar
  - no exits on the entry bar
  - a structure stop filled at the signal close
  - an upper-band sign error
  - `setup_extreme` updating after the signal
  - a continuation entry without a pullback
  - no one-position rule
  - a fade without a green day
  - a last-bar signal filling next session
  - re-entries off by one
  - B crosses detected across sessions
- A real run on cached SPY and QQQ minutes (2022-04-01 to 2024-12-31) was checked by eye
  against the session charts.

## Alert check (`alerts.py`)

- **Reference price:** the close of the minute bar starting at alert time - 1 min, the price
  known at the alert time. The CSV's `ref_price` is compared with that bar and its
  neighbours. Massive closes can sit on a half cent, so "exact" means equal to the cent.
- **End points:** "close" and "HH:MM" use the price known then; "next open" uses the open of
  the next session's first regular minute. The next session comes from the XNYS calendar,
  so holidays are skipped and early closes respected.
- **Availability:** an outcome needs every regular minute in [reference bar, end point],
  counted against the calendar (`minutes_expected`). An end point in today's session or
  later is `pending`. Pending and unavailable rows keep NaN outcomes and are counted, not
  scored.
- **Ex-dividend:** a next-session window spans an ex-date D when session < D <= next
  session. The same-day intraday window never does: the drop happens at the open.
- **Statistics:** Wilson intervals for rates; bootstrap over dates (rows) for means, medians
  and the paired hit-minus-base difference; shuffles reassign the same calls across the same
  dates, so the bullish/bearish counts stay fixed. Each test seeds its own generator from
  `--seed`, so results don't depend on call order.
- **Primary results:** the intraday same-day-close and overnight next-open hit rates, each
  against its base rate. Everything else is labeled exploratory in `summary.csv`.

## Alert spreads (`alert_spreads.py`)

- **Legs:** `spread_legs` rounds SPY's 10:00 price to the cent; bullish shorts the put at its
  ceiling strike and buys $1 below, bearish shorts the call at its floor strike and buys $1
  above. Tickers follow `O:SPY<yymmdd><P|C><strike x 1000, 8 digits>`, same-day expiry.
- **Minutes:** one contract-day per request, which fits in one page, so the SDK's silent
  paging stop can't truncate it. Empty results are cached (the contract didn't trade); an
  out-of-plan date raises `NotInPlan` and is reported, never cached.
- **Spread value** exists only in minutes where both legs printed: the difference of their
  opens (fills) or closes (checks). Nothing is carried forward. Entry needs such a minute in
  [10:00, 10:05) with a credit strictly between 0 and the $1 width; the time exit needs one
  in [15:30, 15:35), or 30 minutes before an early close. A day missing either is left out
  of both directions so every comparison uses the same days.
- **Exits:** take profit X% buys back at credit x (1 - X%) once a close is at least the
  two-leg slippage below it; stop Y% triggers when a close reaches credit + Y% x (width -
  credit) and pays that close (capped at the width) plus slippage; otherwise the time exit
  pays its open (clipped to [0, width]) plus slippage. Commission is charged on 4 leg fills.
- **Costs:** `cost_tiers(slippage, commission)` gives `base` (the CLI settings, default 0 and
  0) plus +$0.01 and +$0.03 slippage tiers, and `gross` when base has costs. Zero slippage
  was checked against a set of real fills: replayed at traded prices, the rule-following
  trades matched them within about $1 a trade.
- **Statistics:** totals and means per spread, bootstrap intervals over days, Wilson
  intervals for win rates, and the shuffled-call p-value (share of shuffles with a total at
  least the alerts'). The user's rule is the main result; the grid is exploratory.
- **The user's rule** comes from `--take-profit`/`--stop` (default 80%, no stop; it was 50%
  until 2026-10-01) and must be a grid cell so it can be ranked. Because it was chosen on the
  trades in the log, `--rule-since` (default 2026-10-01) splits its results into the trades
  that chose it and the trades since; only the latter are out of sample. The month-by-month
  check then compares the re-picked rules with the 50% reference, since the user's rule is
  in-sample in every month.
- **Strike placement:** `spread_legs(side, spot, offset)` moves both legs `offset` dollars deeper
  in the money (bullish: short put at ceil(spot) + offset; bearish: short call at floor(spot) -
  offset). `strike_offsets` runs each offset on the alert days where both directions have
  prices at that offset, so offsets can differ in days; it records how many offset-0 days are
  missing and what offset 0 made on them, because missing deep in-the-money prices cluster on
  big-move days.
- **Parity pricing:** offsets > 0 put both legs in the money, where same-day options print
  thinly. Priced from their own prints they showed profits in both directions on every
  session without alerts (bull put and bear call alike), a sign of print bias rather than
  edge. `prepare_trade(..., parity=True)` instead values the spread as width minus the
  same-strike spread in the other right (`parity_legs`), whose out-of-the-money legs trade
  far more; the own-print result stays in `offsets.csv` (`mean_own_prints`) as a check.
- **Scaling:** `resample_paths` is a moving-block bootstrap (blocks of 5 consecutive trades,
  wrapping), because independent resampling would break up the losing clusters that make
  drawdowns. Scenarios shift the user's trade sequence to a target average (honest
  estimate, $0) or use the +$0.01-slippage sequence; dollars scale linearly with contracts,
  which assumes size doesn't change fills. `drawdown_anatomy` reports the trades, losers,
  winners and longest losing streak between a path's peak and its lowest point after it.
- **Consistency:** `consistency(pnl)` = mean / (sample std / sqrt(n)), the t-statistic of the
  average trade, chosen before looking because it rewards steady profits rather than win
  rate or a few large wins. `leave_one_month_out` picks the highest-scoring cell on all other
  months and records its P&L in the held-out month next to the user's rule; the report also
  shows the 3x3 neighbourhood average (plateaus over peaks) and the same exits in the
  no-alert context.

## Mean reversion (`meanrev.py`)

- **Snapshot:** the decision time is the session close minus 10 minutes (15:50, or 12:50 on an
  early close). `snap` is the close of the last minute bar starting before it (the price known
  then), `snap_high/low/volume` cover the session up to it and `post_high/low` the rest of the
  session. A day's indicators use `[earlier full closes..., today's snap]`; tests check that
  changing anything from 15:50 on leaves that day's and earlier days' indicators unchanged.
- **Dividends:** prices before each ex-date are multiplied by 1 - cash / previous close, so
  snapshot-to-snapshot returns are total returns (cached as `<T>_cash_dividends_*.parquet`).
- **Outcomes:** `fwd_h` = snap(t+h) / snap(t) - 1; `low_h`/`high_h` use day t after the decision,
  full days in between and day t+h up to its decision. Design outcomes end by 2024-12-31.
- **Inference:** a circular moving-block bootstrap over sessions (blocks of 10) for every
  interval, because 1-5 session windows overlap and signals cluster; "clusters" counts events
  more than 5 sessions apart. The indicator table adds Benjamini-Hochberg q-values.
- **ML:** the ML swing study's fixed models and an expanding walk-forward in session positions;
  a block's training rows are those whose label window (t + h) ended before the block begins.
- **Pre-registration:** events, rules, models and the selection rule for the three primary
  hypotheses were fixed before the holdout run; a few exploratory additions (double-oversold
  events, the score's top/bottom 10%, a mean-reversion-only logistic model) were added after the
  design run but before the holdout was opened, and are labeled as such in the code.

## Put spreads after dips (`dip_spreads.py`)

- **Strikes:** `put_strikes` takes the $1 strike at or below `distance`% under SPY's actual
  (unadjusted) 15:50 price, and the long leg `width` dollars lower. `ITM` (-1) instead takes the
  first strike at or above the price, as `alert_spreads.spread_legs` does for bullish alerts. Contracts expire h sessions
  after entry (SPY lists every weekday).
- **Data:** one request per contract covering the five sessions before its expiry (a single
  page), cached as `options_multi/<contract>_<start>_<end>.parquet`; the window depends only on
  the expiry, so every entry that uses the contract shares one file.
- **Path:** regular-hours minutes of the entry-to-expiry sessions in which both legs printed.
  Entry = opens of the first such minute at or after the decision time (close - 10 minutes) and
  before that session's close. Take profits check minute closes from the entry minute to the
  expiry close. Otherwise the spread settles at min(max(short strike - SPY close, 0), width).
- **Breakeven stop** (both spread scripts, `breakeven_at`): arms at the first minute close with
  value <= credit x (1 - 40%); exits at the first later close >= the credit received, filled at
  that close (capped at the width) plus slippage, so jumps past breakeven still lose. A take
  profit reached first still wins. Scratches at breakeven count as non-winners.
- **First-up-day exit** (`first_up_exits`, `simulate(exit_at=...)`): the exit time is 15:50 on the first
  session after entry whose meanrev `ret_1d` (15:50 price vs the previous close) is up, or on the 5th
  session after entry. The spread is bought back at the open of the first both-legs minute at or after
  that time, plus 2 x slippage (capped at the width); a take profit reached earlier wins; with no print
  before the expiry close the spread settles. "next print" marks exits filled after that day's close.
- **Comparison:** every structure is priced on every session, so a signal's average P&L per
  spread is compared with the every-session average for the same structure (excess), with the
  meanrev block bootstrap over entry sessions. One-at-a-time trading takes a signal only after
  the previous spread from that signal has exited.

## Put spreads after dips in single stocks (`stock_dip_spreads.py`)

- **Chain:** `load_chain` lists each week's expiring puts with `list_options_contracts`
  (`expired=True`, one week per query so each answer is a single page), cached as
  `options_chains/<ticker>_puts_<first>_<last>.parquet`.
- **Expiry and strikes:** `pick_expiry` takes the first listed expiry at least 2 sessions after
  entry (none past 7). `pick_strikes` takes the highest listed strike at or below `distance`%
  under the 15:50 price, and the long leg at the listed strike nearest 1.5% of the price lower
  (at least one strike lower; ties go to the lower strike). Widths therefore vary with the grid
  ($2.50 or $5 steps).
- **Pricing:** `dip_spreads.price_spread`, the same entry, path and settlement code as SPY. Legs are
  fetched once per contract from 7 sessions before its expiry, in parallel (`prefetch`).
- **Prices:** split-adjusted Massive minutes, used as actual prices; `check_splits` stops the run
  if a split falls inside the option window. Signals use dividend-adjusted closes as in meanrev.
- **Earnings:** `earnings_days` flags, in each reporting window the data fully covers (22nd of
  Jan/Apr/Jul/Oct to the 8th of the next month), the session with the largest stock-specific
  volume ratio (volume / prior 20-session median, over SPY's same ratio) and the one with the
  largest |stock gap - SPY gap|, each with one session either side. A spread is skipped when a
  flagged session falls in (entry day, expiry]. Checked by hand against the 20 reports from Oct
  2024 to Oct 2025: either signal alone missed some (news days such as tariff headlines or the
  election had bigger jumps), but the flagged sessions covered every report. `--earnings-csv`
  replaces the flags with exact dates.
- **Combined view:** each ticker is traded one spread at a time (primary structure; SPY uses
  its own 3-session, 0.5%-below, $5 spread from dip_spreads.py), pooled by exit date for totals,
  monthly results and drawdown.

## Gap-down recoveries (`gap_recovery.py`)

- **Gap:** open = the first regular minute's open; target = the previous calendar session's last
  regular close minus any cash dividend going ex that morning; gap % = open / target - 1. The daily
  table keeps a NaN row for every session without data, so a gap is never measured across a missing
  session.
- **Same session:** fill = the first minute whose high >= target (`fill_minutes` counts to that
  bar's end); best = (session high - open) / (target - open); close_rec likewise with the last
  minute's close; further_drop = the lowest low up to and including the fill bar (the whole session if
  unfilled) against the open; extension = a low <= open - (target - open). `first` compares the first
  fill bar with the first extension bar ("same minute" when one bar does both).
- **Later sessions:** filled_kd uses daily highs over the gap day and the next k sessions; unknown
  (NaN) when the data ends first or a session in the window is missing before any fill.
- **Kinds:** for stocks, earnings = a session flagged by `stock_dip_spreads.earnings_days(margin=0)`;
  otherwise "with the market" when SPY's own gap that morning is at least min(1%, `--min-gap`) down;
  otherwise "stock alone".
- **Sizes:** `SIZES` buckets from 0.25% up; the filter tables group them with `size_groups(min_gap)`
  (the first two buckets at or above the threshold on their own, the rest together).
- **Race:** of gaps that either filled or extended first, the share that filled first, with a Wilson
  interval. A driftless random walk from the open gives about 50%.
- **From 10:00** (`after_wait`, `WAIT` = 30 minutes): price_10 = the close of the last bar ending by
  10:00; for gaps not filled by then, the same measures over the bars ending after 10:00, with
  target - price_10 as the yardstick (best_10, close_10, and the race against a low <= price_10 -
  (target - price_10)). `FILTERS` lists the five filter tests and where each one's measures start.
- **Renamed tickers:** `FORMER` maps a ticker to its old one; missing sessions inside the span of
  missing sessions are taken from the old ticker's cached minutes (META <- FB, 90 sessions in 2022).

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
  - checking only the final session's date rather than its last regular minute (added
    2026-09-29 after a review finding)
  - picking a winner when none qualifies
  - triggering on every bar above instead of on the cross
  - resetting intraday EMAs each session
- **Pagination** was exercised through the real `massive` client with its HTTP layer
  stubbed (`tests/test_download.py`).
- **Not yet verified:** any live Massive request.

## Intraday mean reversion (`scalp_meanrev.py`)

- **Bars and indicators:** `features.build_features` 5-minute (and 2-minute) bars with EMA 9/20/50;
  `add_indicators` adds ATR(14), RSI(2)/RSI(14), Bollinger(20, 2) z, EMA distances in ATRs, runs of
  same-direction closes and the 15-minute move over its 50-bar volatility (both within the session),
  and the session VWAP with its volume-weighted standard deviation from minute bars (each minute's own
  VWAP). `add_context` adds relative volume for the bar's time of day (previous 20 sessions), the daily
  run of closes through the previous session, today's gap and the previous daily ATR %.
- **Events:** `first_of_run`: the first bar of each run in which a condition holds (per session), with
  the bar ending from 10:30 to 15:00 ET and the session on or after 2022-01-03.
- **Outcomes** (`bar_outcomes`, every window bar): long-signed bps from the bar close to the close of
  the minute ending at T+h; available only if every minute in [T, T+h) is present and T+h <= the
  session close. The bracket uses the bar's ATR: +1 if a minute high reaches close + ATR before a minute
  low reaches close - ATR within 30 minutes, -1 for the reverse, 0 for neither, NaN for the same minute.
- **Excess:** the signed move minus the signed average of every window bar in the same half hour and
  period (design or holdout) for that ticker.
- **Statistics:** means with standard errors clustered by day (pooled groups cluster across tickers);
  BH q-values within each round of design tests; selection and the holdout rule as in the module
  docstring. A check on 2026-10-02 found clustered and naive errors nearly equal (events rarely bunch
  within a day).

## Intraday momentum (`intraday_momentum.py`)

- **Prices:** `price_at` takes the close of the minute bar ending at a time (starting a minute earlier);
  a missing bar leaves the day out. Morning = previous session's last-minute close (less a dividend going
  ex that morning) to the bar ending at open + 30 minutes; late = close - 60 to close - 30 minutes;
  last = close - 30 minutes to the close (early closes included).
- **Signals and statistics:** sign of the morning or late move (or both agreeing) times the last move,
  one observation per day; t-tests, Wilson hit-rate intervals and an OLS slope of last on morning.

## Trend following on ETFs (`trend_etfs.py`)

- **Prices:** meanrev's dividend-adjusted 15:50 snapshots; returns snapshot to snapshot.
- **Signals:** `trend_signal` = sign(price / price n sessions ago - 1) or sign(price - n-session
  average), from snapshots up to and including the session. Long/flat clips at 0.
- **Sizing and costs:** `risk_weights` = min(10% / (60-session daily-return volatility x sqrt 252), 2).
  `sleeve_returns`: the position decided at t earns the return to t+1, less cost_bps x the change in
  position decided at t (opening from flat included). The portfolio is the average of the sleeves.
- **Dip buying benchmark:** `dip_positions` enters at a snapshot with a streak of 3+ down days and exits at
  the first later snapshot whose move from the previous close is up, or after 5 sessions (re-entering at
  once if the streak still qualifies), on SPY bought outright.

## Call credit spreads (`call_spreads.py`)

- **Expiries:** `weekly_chain` lists, per week, the call strikes of the later of the Thursday and Friday
  expiries (Friday, or Thursday in a holiday week). `pick_expiry` takes the one closest to t + 30 or 45
  calendar days that is more than 21 days out and expires by the last session.
- **Delta:** at 15:50 on day t, the at-the-money strike (nearest the 15:50 price) gives the forward
  F = K0 + C0 - P0 from the call's and put's last trades in [14:50, 15:50) (no discounting; only $5
  strikes are used, since 30-45-day SPY options trade mostly there); the
  at-the-money volatility comes from Black-76 on F. Candidate strikes for deltas 0.10-0.50 at that
  volatility are priced the same way; each gets its own implied volatility and delta N(d1); the short
  strike is the candidate closest to the target delta within 0.06. Rates and dividends enter only through
  the parity forward.
- **Legs and pricing:** long = the first listed strike at least $5 / $10 above the short. Each contract is
  fetched once from 56 days before its expiry. `dip_spreads.price_spread` with `right="C"` settles at
  min(max(SPY close - short, 0), width). Entry is a working order: the first both-legs minute from 15:50
  through 10:30 the next session (`entry_until`), since these legs seldom print together in the last ten
  minutes alone.
- **Exits** (`simulate`): take profit at credit x (1 - 50%) as in dip_spreads; the stop fills at the first
  minute close >= credit x 3 (a loss of 2x the credit), capped at the width, plus slippage; the 21-day exit
  at the open of the first both-legs minute from 15:50 on the first session with (expiry - session) <= 21
  calendar days; the earliest rule wins; otherwise settlement.
- **Statistics:** entries every session overlap, so intervals resample 20-session blocks of entry days;
  one-at-a-time takes a new entry only after the previous spread's exit session.

## Iron condors (`iron_condors.py`)

- **Put strikes:** `call_spreads.choose_shorts(right="P")`: candidates at the strike where the call delta is
  1 - target; each put's implied volatility comes from the equivalent call price C = P + F - K (Black-76
  parity), and its delta is 1 - N(d1). Long put = the highest listed strike at or below short - width.
- **Combining:** each spread is priced and entered separately (working orders as in call_spreads.py); the
  condor starts at the later entry, its credit is the sum of the two, and its value at every minute either
  spread traded is the sum of each spread's latest traded value (`condor_path`). Settlement is the sum of
  the two settlements; max loss = the wider wing minus the credit.
- **Exits** (`simulate`): as in call_spreads.py on the total, with four legs of slippage each way.

## Three red bars by timeframe (`red_bars.py`)

- **Bars:** `features.resample_bars` (anchored at the open, 80% coverage) for 5/15/60/240 minutes; daily bars
  from the regular-session minutes. A red candle is close < open; a down close is close < the previous bar's
  close. `red_runs` counts the current run; 5- and 15-minute runs restart each session (and a session's first
  down close, which would include the overnight gap, does not count).
- **Outcomes** (`forward`): close h bars later over the signal close, in bps; for 5/15 minutes only inside the
  session. The race uses bar highs/lows of the next 5 bars against +/- ATR(14) of that bar size; a bar doing
  both is left out.
- **Statistics:** signal = run >= 3; excess = event move minus the mean move of every bar (same ticker, bar
  size, definition and period), with standard errors clustered by session date; BH across the 30 design
  tests on the 3-bar excess.

## Candle patterns (`bar_patterns.py`)

- **Matching** (`pattern_mask`): candle colours (green close > open, red close < open, a doji breaks the
  pattern) of the last five bars equal the pattern, inside one session for 5/15 minutes. The strict versions
  need the fifth bar's low above the lowest low of the first three bars (bottoms) or its high below their
  highest high (tops). Outcomes and statistics reuse `red_bars.forward` and `red_bars.summarize(sig, side)`,
  with side = -1 for the faded tops.

## Support and resistance levels (`levels.py`)

- **Levels** (`level_table`, one row per session, all known by 9:30 ET): yesterday's regular-hours high, low
  and close and last week's (the previous calendar week's) high and low, each NaN when a source session is
  below 80% minute coverage (an older session is never substituted); today's pre-market high and low from the
  4:00 ET to 9:30 minute bars (NaN with fewer than 30); yesterday's POC/VAH/VAL from
  `auction_reclaim.previous_session_levels` (48 bins); and the nearest multiples of $1 and $5 strictly below
  and above the first regular minute's open. ATR = `features.daily_features` prev_atr_14.
- **Events** (`real_levels`, `races`, `measure`): a level below the open is support, above it resistance;
  levels within 0.05 ATR of the open are skipped. The touch is the first regular-hours minute reaching the
  level that starts at least 30 minutes before the close. The race uses minute highs/lows to the close: a
  break counts from the touching minute on (price came from the open's side, so it reached the level first);
  a hold counts only from the next minute, since the touching minute's high/low order is unknown. A later
  minute that does both is NaN; neither by the close is 0.
- **Controls:** `fake_levels` draws, for each real level-day, 10 distances (in ATR) from the same ticker,
  level, side and period on other sessions, and places them on the same session. `near_levels` (added after
  the first results) puts fakes 0.2/0.3/0.4 ATR above and below the real level on the same session, kept on
  the same side of the open, because the random fakes match the distances only on average and can be touched
  on a different mix of days.
- **Statistics** (`compare`): held % among resolved touches (held + broke), real minus fake, with a standard
  error from 2,000 bootstrap resamples of the period's sessions (`boot_weights`, shared by every comparison
  in a period). The primary test (ticker x level at 0.1 ATR, random fakes) uses BH across the 36 design tests
  and needs the same sign at 0.25 ATR. Without `--final-test`, holdout sessions are not measured at all.

## Swings on a 5-minute chart (`swings.py`)

- **Bars** (`grid`): `features.resample_bars` 5-minute bars (80% coverage) as a sessions x 78 array, NaN for
  missing bars and after early closes. Sessions under 80% minute coverage, or without an ATR yet, are skipped.
- **Zigzag** (`zigzag`, vectorised over rows): R = 0.15 x prev_atr_14 per session, restarting each session.
  Undecided rows track both extremes; a peak is confirmed when a bar's low is <= the running high - R and that
  bar did not itself set the high (unknown high/low order), a valley mirrored; the next leg starts from the
  confirming bar. The swing is known at the confirming bar's end.
- **Swing lines** (`swing_levels`): start = max(confirming bar's end, open + 2 h); reference = the close of the
  last minute starting before that; side and distance (ATR) from it; dropped within 0.05 ATR or when the start is
  less than 30 minutes before the close. Touches and races are `levels.measure` with a per-level `start`
  (`levels.races(start=...)`). `random_fakes` draw distances from the same type, origin (drawn at 11:30 or later),
  side and period on other sessions (never their own); `near_fakes` sit 0.2/0.3/0.4 ATR either side. Statistics
  are `levels.statistics`; BH across the 6 design tests.
- **Timing** (`legs`, `timing_stats`, `hazard`): a leg runs from one swing's extreme bar to the next one's;
  each session's last, unconfirmed leg counts only as still running (for the hazard). Spread = SD of log leg
  minutes; lag-1 = Spearman correlation of consecutive legs in a session. Null (`flipped`): each 5-minute bar's
  open/high/low/close relative to the previous close (the first bar's own open) is kept or mirrored with
  probability 1/2, chained from the session's open; the same zigzag runs on 100 copies per session (built 25 at a
  time). Monte Carlo p-values (two-sided from the null mean; one-sided for the holdout), BH across the 6 tests.

## Breakout trades (`breakouts.py`)

- **Levels and rules** (`rule_levels`): `levels.real_levels` rows for prev_high and week_high on the resistance
  side (traded long), prev_close on both sides, and prev_low / week_low on the support side (short, a check).
  Trade direction = -side (through the level).
- **Fills and exits** (`simulate`, vectorised per session over levels x minutes): entry at the first minute that
  reaches the level within the touch window (starting at least 30 minutes before the close), filled at the
  level or at that minute's open if it is already through it. Stop and target are measured from the level, in
  ATR. A minute reaching both counts as stopped; a later minute opening beyond the stop fills at its open; a
  target reached in the fill minute when the fill is already beyond it exits at the fill. Unresolved trades exit
  at the session's last minute close.
- **Inside the fill minute** (`best`): worst = a stop touch there counts (the rule fixed in advance); best = a
  target touch there comes first, and a stop touch counts only if the minute closed beyond the stop. A one-off
  check of TSLA with one-second bars (outside the repo) landed close to the best case.
- **Statistics** (`trade_stats`): net = gross - 2 x cost bps; mean with a session-clustered standard error, one-sided
  p for mean > 0; % a year = summed net bps / 100 / years. BH across the 12 primary tests (worst case).

## TSLA breakout check (`breakout_check.py`)

- **Entries** (`candidates`): `breakouts.run_trades` (worst case) gives each real level's first touch minute;
  the fakes are a seeded sample of `levels.fake_levels` rows.
- **Second-by-second replay** (`replay`): the fill minute and every later minute whose bar reaches the stop or
  target are read from one-second bars (`second_loader`, cached per stock and per stock's options). Fill = the
  first second reaching the level, at the level or that second's open if already through it; then the first
  second reaching the stop or target exits (both in one second, or a stop in the fill second, counts as stopped;
  a stop gapped through fills at that second's open). If the seconds never show the minute bar's touch, the trade
  is left out and counted.
- **Options** (`option_leg`, `option_trades`): the first listed expiry after the trade date and the listed strike
  nearest the level (`load_chain`: puts, expired and live, one query per week; calls share the strikes). Entry =
  the open of the first option second at or after the share fill within 60 s (else skipped); exit = the first at
  or after the share exit within 60 s, else the close of the last one in the 60 s before. P&L = 100 x (exit - entry
  - 2 x slippage) per contract; % of premium uses entry + slippage.
- **Tests:** shares at 1 bp per side and options at $0.05 per fill, one-sided mean > 0 per rule with
  session-clustered errors, BH within each set of three.

## Kalman SuperTrend (`kalman_supertrend.py`)

- **Bars** (`five_minute_bars`): clock-aligned 5-minute bars from 04:00 to 20:00 ET on XNYS sessions (regular hours
  only for the regular-hours chart), each with its minute index range; `rth` = the bar starts inside the
  session, `last_rth` = it ends at the session close.
- **Indicators** (`indicators`): the script's scalar Kalman filter (`kalman`), TA-Lib ATR(7), the SuperTrend band
  recurrence on the Kalman centre (`supertrend`, ratcheting bands, flips on a close beyond the previous band),
  flags only on regular-hours bars; VWMA(50) from rolling sums; TA-Lib STDDEV(20) x 1.5 x (1 + 0.8 ADX(14)/100),
  smoothed with Pine's RMA (`rma`, seeded with the first full-window mean); outer waves = VWMA +/- 2 x width.
- **Trades** (`trade_from`, `strategy`): fill at the open of the first minute after the signal bar plus a tick;
  stop/target from the signal close and its ATR, checked minute by minute (stop first; an open beyond either fills
  there, including the fill minute); breakeven from the bar after one whose high/low reached 1 ATR; at each bar
  close, in order: regular close (variant), opposite flag, wave, time (12 bars after the signal). Market exits fill
  at the next minute's open less a tick. A same-direction flag while a trade is open at the bar's close is
  ignored.
- **Baselines:** `random_trades` (20 per real trade, same session and direction, random regular-hours bar, same
  exits); `flag_moves` (fill to +15/30/60 minutes, against every regular-hours bar at the same New York time).

## Kalman SuperTrend as 0DTE options (`supertrend_0dte.py`)

- **Contract** (`contract`): `alert_spreads.option_ticker` for the trade day, "C" for longs and "P" for shorts,
  strike = floor(share fill + 0.5).
- **Prices** (`price_at`, `option_trades`): contract-day minute bars from `alert_spreads.option_loader`. Entry = the
  open of the bar starting at the share fill minute; exit = the close of the exit minute's bar for stop, target,
  breakeven and the regular close, the open of the exit minute's bar for exits decided at a bar close. A missing
  bar falls back to the next bar within 2 minutes (open), then the last bar in the 5 minutes before (close);
  otherwise the trade is counted as unpriced. `asp.NotInPlan` marks sessions outside the rolling option window.
- **Statistics** (`stats`): P&L per contract = 100 x (exit - entry - 2 x slippage), session-clustered mean and
  one-sided p, % of premium (entry + slippage), dollars a year and the worst run at one contract a trade.

## Five-indicator scalper (`confluence_scalper.py`)

- **Indicators:** `cm_macd` (EMA 12 - EMA 26, SMA 9 signal); `squeeze_momentum` (LINEARREG 20 of close minus the
  mean of the 20-bar midrange and SMA 20; squeeze = SMA 20 +/- 1.5 sd inside SMA 20 +/- 1.5 SMA(TR)); `supertrend_ai`
  (per factor 1.0-5.0 step 0.5 on ATR 10: trend from the previous bands, ratcheting bands, performance = EMA(10-
  memory) of the next move signed by the previous close's side of the line; per bar `kmeans3` on the nine scores,
  started at the 25/50/75th percentiles; the top cluster's mean factor drives the final band, whose trend uses the
  updated bands); `swing_structure` (short-term swings known a bar later, intermediate ones when the next short-term
  swing on that side is known; state flips on a close beyond the last unbroken intermediate level).
- **Signals** (`indicators`): the first regular-hours bar on which all five long (short) conditions hold; plain
  SuperTrend AI flips for comparison.
- **Trades:** exit A reuses `kalman_supertrend.strategy(flat_by_close=True)`; exit B (`trend_trade_from`,
  `trend_strategy`) holds until SuperTrend AI points against the trade at a bar's close or the regular close.
  `random_entries` draws same-session bars where SuperTrend AI already points the trade's way.

## Green Goose (`green_goose.py`)

- **Signals** (`daily_signals`, `decide`): per session, TA-Lib RSI(2), ADX(n), PLUS_DI(n) and MINUS_DI(n) over up to
  200 earlier dividend-adjusted daily bars plus today's bar to 15:50 (`meanrev.daily_table` snapshot); yesterday's
  values are the full bars'. Base direction from RSI(2) (> 85 puts, < 15 calls) or the 15:50 price against the open;
  then the ADX-enters-the-DI-zone override, then the RSI(2)-crosses-into-the-zone override; ADX > 60 vetoes.
- **Stocks** (`morning_moves`, `stock_stats`): entry = the close of the last minute before 15:50, less a cash
  dividend with the next session's ex-date; exits at the next first regular minute's open and at the last minute
  close before open + 5/10/30/90 minutes. Baseline = (share of calls - share of puts) x the mean move.
- **Options** (`choose_contract`, `option_trades`): forward from the call and put at the strike nearest the 15:50
  price (prices = last close before 15:50), Black-76 implied volatility (puts through parity), delta and theta per
  calendar day (`b76`); the bracketing strike in the delta band with theta < -0.12. Expiry = the first session at
  least 5 (or 1) calendar days out. Entry = the open of the 15:50 bar; `exit_v2` = the 9:35 price; `exit_v1` = the
  opening-price rules with a trailing stop at 90% of the highest price since the open, checked before each bar's
  high updates it, until 11:00.

## Green Goose indicator search (`goose_indicators.py`)

- **Inputs** (`indicator_frame`):
  - `green_goose.daily_signals`: RSI(2) and ADX/DI(5), today's and yesterday's.
  - `meanrev.ticker_features`: IBS, Bollinger z, closing streak, the day's move, distance from the 50-day average,
    gap and volume ratio.
  - `mfi_macd`: MFI(14) and the MACD(12, 26, 9) histogram, over up to 200 earlier full bars plus today's bar to
    15:50.
  - The 15:20-15:50 move, from a second daily table with a 40-minute decision time.
  - VIXY's move from its last close to 15:50.
  - The calendar: the next session's turn-of-month flag and the calendar days until it.
- **Conditions** (`conditions`): the inputs become 32 sets of sessions. A missing input makes a condition false and
  marks the session unusable.
- **Screen** (`edge_test`, `slides`, `screen`):
  - Edge = the mean move to the next open on the condition's sessions, minus the mean over every usable design
    session.
  - Null = the condition circularly shifted by at least 20 sessions. A shift is skipped if, afterwards, more than
    halfway from the chance overlap to a full overlap of its sessions are still its sessions. That covers a weekly
    or monthly pattern landing back on itself, and a persistent condition that has barely moved.
  - p is two-sided, from the centred null.
  - QQQ and IWM edges use each ticker's own indicators.
- **Rule** (`rule_direction`): calls if call signals > put signals + warnings; puts if put signals > call signals.
- **Checks:**
  - `stock_check` slides the rule's call and put sessions the same way as the screen.
  - `option_pnl` takes the `green_goose.option_trades` rows that match each session's direction (version 1 exit,
    slippage per fill).
  - `diff_interval` resamples sessions to compare the new rule with Green Goose.

## Green Goose as $1 credit spreads (`goose_spreads.py`)

- **Legs:** `alert_spreads.spread_legs(side, 15:50 price, offset)`, expiring the next session. Both days'
  contract-days are fetched up front.
- **Prices** (`value_at`): the opens of the first minute within 5 minutes of the time in which both legs traded.
  Failing that, each leg's first open in those minutes; failing that, missing. Prices are taken at:
  - Entry, 15:50. The credit must be between 0 and 1.
  - 9:35 the next morning.
  - 15:30, or 30 minutes before an early close.
- **Exits** (`trade_pnl`):
  - 9:35 and the take profit go through `alert_spreads.simulate`, with slippage on every fill. The take profit
    watches the both-legs minute closes from after the entry through the minute before the 15:30 exit.
  - Hold to expiry = credit - entry slippage - `settle`. `settle` is the strike's distance from SPY's last
    regular-hours close, capped at 0-1.
- **Sample:** a spread counts only when its entry, 9:35 and 15:30 prices all exist, so every exit uses the same
  trades. The report repeats hold to expiry on every entered spread as a check.

## TSLA breakouts with a buying-pressure filter (`breakout_flow.py`)

- **Trades:** `breakout_check.candidates` and `replay_all` for the two kept rules, real levels only, at 1 bp per side.
- **Order flow** (`pressure`, `flow_columns`):
  - One request per trade: one-second bars with volume from 15 minutes (plus one second) before the fill, or from the
    session open, up to the millisecond before the fill second. `breakout_check.second_loader(volume=True)` keeps
    these in its own `<ticker>_volume.pkl` cache.
  - Each second is classed by the sign of its close minus the previous close, with flat seconds carrying the last
    sign. Volume before the first change is unclassed.
  - Pressure over a window = (buy - sell) / (buy + sell). The window starts at the later of the fill minus W minutes
    and 9:30:01, so the opening-auction second only sets the starting price. At least 10 seconds with trades.
  - The sign is flipped for shorts.
- **Statistics:**
  - `cluster_ols` is OLS with standard errors clustered by session.
  - The primary test (`split_test`) regresses net bps on an indicator for pressure > 0.
  - The "beyond the price move" line regresses net bps on pressure and the signed bps move over the same window.

## Bollinger + RSI scalp with trap filters (`band_scalp.py`)

- **Bars and indicators** (`five_minute_bars`):
  - `features.resample_bars` builds the 5-minute and 1-hour bars, each needing 80% of its minutes.
  - TA-Lib BBANDS(20, 2, population sd) and RSI(14) on 5-minute closes, run continuously across sessions.
  - Band width = (upper - lower) / middle, and its growth over 3 bars.
  - The 1-hour 200 EMA is joined with `merge_asof` on bar ends, so a 5-minute bar sees only hourly bars completed by
    its end.
- **Setups** (`find_setups`):
  - A pierce bar arms a setup; the first bar within 6 that closes back inside, in the same session, is the entry bar.
  - Later pierce bars inside that span don't start new setups.
  - The wick is the extreme from the pierce bar to the entry bar.
- **Levels** (`profile_levels`): levels.py's 48-bin profile of the previous session (`volume_profile.bar_profile` /
  `value_area`), with that session needing 80% of its minutes, plus `daily_features`' previous ATR(14).
- **Order flow** (`bar_deltas`, `cvd_flags`):
  - Fetched only for setups passing the trend, expansion and level filters, as one request per setup. The request
    runs from one second before the first lookback bar (up to 12 bars earlier in the session) to the end of the entry
    bar, via `breakout_check.second_loader(volume=True)`.
  - Seconds are classed by the tick rule, with the opening-auction second excluded.
- **Trades** (`simulate`):
  - Each filter row is simulated on its own, in time order, one position at a time per ticker.
  - Entry at the open of the first minute at or after the entry bar's end.
  - Exits are checked minute by minute until 50 minutes after entry. Within a minute the stop is checked before the
    target; a stop gapped through fills at the open. The time exit is the last minute's close.

