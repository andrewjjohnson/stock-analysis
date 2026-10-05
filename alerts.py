"""How often were a log of posted SPY calls right? A signal study on Massive minute bars.

  uv run --env-file .env python alerts.py --csv data-files/<log>.csv --out output/alerts

The CSV holds one row per posted call: `intraday` at 10:00 ET (BULLISH = sell puts, BEARISH =
sell calls) and `overnight` at 15:55 ET (BULLISH = buy calls, BEARISH = buy puts). This script
checks every row against Massive SPY minutes, measures what SPY did afterwards, signed by the
call, and compares hit rates with an always-long baseline, shuffled calls, a coin flip and
naive momentum rules. It studies the underlying only, not option P&L: the CSV has no strikes,
expiries or premiums.

Time conventions (docs/design.md):
- Massive timestamps are bar starts, so the price known at time T is the close of the minute
  bar starting at T - 1 min. "Close" is the close of the last regular minute (15:59-16:00 on
  a full day), and "next open" is the open of the next session's first regular minute.
- An outcome needs every regular-session minute from the reference bar to its end point;
  otherwise it is unavailable (NaN). One whose end point falls in today's session or later
  is pending. Both stay in the table and out of every statistic.
- Prices are split-adjusted, not dividend-adjusted, so a window that crosses an ex-dividend
  date includes the mechanical drop. Those rows are flagged.
"""

import argparse
import math
import os
import re
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from massive import RESTClient  # noqa: E402

import download  # noqa: E402
import features  # noqa: E402
import outcomes  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
MINUTE = pd.Timedelta(minutes=1)
NS_MIN = outcomes.NS_PER_MINUTE
TICKER = "SPY"

SIDES = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
SCHEDULE = {"intraday": "10:00", "overnight": "15:55"}
ACTIONS = {"intraday": {"BULLISH": "SELL PUTS", "BEARISH": "SELL CALLS", "NEUTRAL": "NO PLAY"},
           "overnight": {"BULLISH": "BUY CALLS", "BEARISH": "BUY PUTS", "NEUTRAL": "NO PLAY"}}
POSTED = {"BULLISH": 1, "BEARISH": 2, "NEUTRAL": 3}  # the usual overnight alerts_posted pattern

# Horizon -> (session, end point). "close" and "HH:MM" use the price known then; "open" uses
# the open of that session's first regular minute.
HORIZONS = {"close": ("same", "close"), "next_open": ("next", "open"), "next_1000": ("next", "10:00"),
            "next_close": ("next", "close")}
STRATEGY_HORIZONS = {"intraday": ("close", "next_close"), "overnight": ("next_open", "next_1000", "next_close")}
PRIMARY = {"intraday": "close", "overnight": "next_open"}
END_LABELS = {"close": "same-day close", "next_open": "next open", "next_1000": "10:00 next session",
              "next_close": "next session's close"}

PREMIUM_PCT = (0.0, 0.25, 0.5, 1.0)  # intraday: how far against the call SPY may close
MOVE_PCT = (0.25, 0.5)               # overnight: how far in the call's direction SPY moved
MAX_REF_DIFF_PCT = 0.1
# Equal to the cent: Massive minute closes can sit on a half cent (sub-penny prints), which a
# two-decimal ref_price rounds; the tolerance keeps those exact cases from failing on float noise.
EXACT_DOLLARS = 0.005 + 1e-9
# Candidate ref_price conventions: (short code, description, bar start relative to the alert time, field).
CONVENTIONS = (
    ("T-2", "close of the bar ending 2 min before the alert", -3, "close"),
    ("T-1", "close of the bar ending 1 min before the alert", -2, "close"),
    ("T", "close of the bar ending at the alert time", -1, "close"),
    ("open T", "open of the bar starting at the alert time", 0, "open"),
    ("T+1", "close of the bar ending 1 min after the alert", 0, "close"),
    ("T+2", "close of the bar ending 2 min after the alert", 1, "close"),
    ("vwap T", "VWAP of the bar ending at the alert time", -1, "vwap"),
)
KNOWN_AT = {"T-2": -2, "T-1": -1, "T": 0, "open T": 0, "T+1": 1, "T+2": 2}  # minutes after the alert
ALTERNATIVE = re.compile(r"bot (?:sent|also posted) (BULLISH|BEARISH|NEUTRAL)")
OFF_SCHEDULE = re.compile(r"off-schedule at (\d{1,2}:\d{2}) ET \((BULLISH|BEARISH|NEUTRAL)\)")
MONTHS = "January February March April May June July August September October November December".split()


# ---------------------------------------------------------------- the CSV

def load_alerts(path):
    """The alert CSV plus `session`, UTC `alert_time` and the call's `side` (+1 / -1 / 0)."""
    a = pd.read_csv(path, dtype={"time_et": str, "flag": str, "notes": str})
    a[["flag", "notes"]] = a[["flag", "notes"]].fillna("")
    unknown = sorted(set(a["direction"]) - set(SIDES))
    if unknown or not a["strategy"].isin(SCHEDULE).all():
        raise SystemExit(f"Unexpected direction or strategy values in {path}: {unknown}")
    a["session"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    local = pd.to_datetime(a["date"] + " " + a["time_et"]).astype("datetime64[ns]")
    a["alert_time"] = local.dt.tz_localize(NY).dt.tz_convert("UTC")
    a["side"] = a["direction"].map(SIDES)
    return a


def check_rows(alerts, sessions):
    """Internal consistency and schedule checks. Adds `alt_direction` (the call from `notes` on
    flagged rows, else the recorded one) and `issues` (empty when the row is clean)."""
    a = alerts.copy()
    issues = [[] for _ in range(len(a))]

    def note(mask, text):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            issues[i].append(text)

    overnight = (a["strategy"] == "overnight").to_numpy()
    score = a["score"].to_numpy(float)
    implied = np.select([score >= 2, score <= -2], ["BULLISH", "BEARISH"], "NEUTRAL")
    bounds = sessions.reindex(a["session"])
    t = outcomes._ns(a["alert_time"])
    after_open, by_close = t > outcomes._ns(bounds["open"]), t <= outcomes._ns(bounds["close"])
    note(~a["session"].isin(sessions.index), "not an XNYS session")
    note(a["weekday"] != a["session"].dt.strftime("%a"), "weekday does not match the date")
    note(a["ticker"] != TICKER, f"ticker is not {TICKER}")
    note(a["action"] != [ACTIONS[s][d] for s, d in zip(a["strategy"], a["direction"])],
         "action does not match direction")
    note(overnight & (np.isnan(score) | (score != np.round(score)) | (implied != a["direction"])),
         "score does not match direction")
    note(~overnight & ~np.isnan(score), "intraday row has a score")
    note(overnight & (a["alerts_posted"] != a["direction"].map(POSTED)).to_numpy(),
         "alerts_posted breaks the usual 1/2/3 pattern")
    note(a["time_et"] != a["strategy"].map(SCHEDULE), "time is off the posting schedule")
    note(~(after_open & by_close), "time is outside the session's regular hours")
    note(a.duplicated(["date", "strategy"], keep=False), "duplicate date and strategy")
    a["issues"] = ["; ".join(x) for x in issues]

    alt = [m.group(1) if f and (m := ALTERNATIVE.search(n)) else d
           for f, n, d in zip(a["flag"], a["notes"], a["direction"])]
    a["alt_direction"] = alt
    return a


def missing_rows(alerts, sessions):
    """Sessions from the first to the last alert date with no row, per strategy."""
    window = sessions.index[(sessions.index >= alerts["session"].min()) & (sessions.index <= alerts["session"].max())]
    have = {s: set(alerts.loc[alerts["strategy"] == s, "session"]) for s in SCHEDULE}
    return {s: [str(d.date()) for d in window if d not in have[s]] for s in SCHEDULE}


def off_schedule_alerts(alerts):
    """Extra alerts mentioned in `notes` that have no row of their own."""
    return [{"date": r.date, "strategy": r.strategy, "time_et": t, "direction": d}
            for r in alerts.itertuples() for t, d in OFF_SCHEDULE.findall(r.notes)]


# ---------------------------------------------------------------- market lookups

def bar_value(minutes, starts, field):
    """`field` of the minute bar starting at each time; NaN where that bar is missing."""
    t = outcomes._ns(minutes["ts"])
    want = outcomes._ns(starts)
    i = np.searchsorted(t, want)
    found = i < len(t)
    found[found] = t[i[found]] == want[found]
    out = np.full(len(want), np.nan)
    out[found] = minutes[field].to_numpy(float)[i[found]]
    return out


def price_known_at(minutes, times):
    """Close of the minute bar ending at each time (it starts one minute earlier)."""
    return bar_value(minutes, pd.DatetimeIndex(times) - MINUTE, "close")


def minutes_present(minutes, start, end):
    """Regular-session minute bars starting in [start, end) (UTC ns arrays)."""
    t = outcomes._ns(minutes["ts"])
    return np.searchsorted(t, end) - np.searchsorted(t, start)


def minutes_expected(sessions, start, end):
    """Regular-session minutes the calendar has in [start, end) (UTC ns arrays)."""
    o, c = outcomes._ns(sessions["open"])[None, :], outcomes._ns(sessions["close"])[None, :]
    overlap = np.minimum(c, np.asarray(end)[:, None]) - np.maximum(o, np.asarray(start)[:, None])
    return np.clip(overlap, 0, None).sum(axis=1) // NS_MIN


def price_conventions(alerts, minutes):
    """How closely each candidate convention reproduces ref_price, over rows with that bar."""
    rows = []
    for code, text, offset, field in CONVENTIONS:
        px = bar_value(minutes, alerts["alert_time"] + offset * MINUTE, field)
        ok = ~np.isnan(px)
        diff = alerts["ref_price"].to_numpy(float)[ok] - px[ok]
        pct = np.abs(diff / px[ok]) * 100
        rows.append({"code": code, "convention": text, "n": int(ok.sum()),
                     "exact": int((np.abs(diff) <= EXACT_DOLLARS).sum()),
                     "median_abs_diff_pct": np.median(pct) if ok.any() else np.nan,
                     "mean_abs_diff_pct": pct.mean() if ok.any() else np.nan,
                     "max_abs_diff_pct": pct.max() if ok.any() else np.nan,
                     "rows_over_limit": int((pct > MAX_REF_DIFF_PCT).sum())})
    return pd.DataFrame(rows)


def horizon_end(alerts, sessions, horizon):
    """(UTC end time, end session) of one horizon for every alert."""
    which, when = HORIZONS[horizon]
    following = pd.Series(sessions.index[1:], index=sessions.index[:-1])
    day = alerts["session"] if which == "same" else alerts["session"].map(following)
    day = pd.DatetimeIndex(day)
    if when in ("open", "close"):
        end = pd.DatetimeIndex(sessions[when].reindex(day))
    else:
        end = (day + pd.Timedelta(f"{when}:00")).tz_localize(NY).tz_convert("UTC")
    return end, day


def horizon_returns(alerts, minutes, sessions, today, ref):
    """SPY's raw return (%, positive = up) from `ref` to each applicable horizon, plus its status:
    ok, pending (ends in today's session or later) or unavailable (a needed minute is missing)."""
    ref = np.asarray(ref, dtype=float)
    start = outcomes._ns(alerts["alert_time"]) - NS_MIN  # the reference bar
    out = {}
    for h, (_, when) in HORIZONS.items():
        end, day = horizon_end(alerts, sessions, h)
        if day.isna().any():
            raise ValueError("The session calendar must extend past every alert's next session.")
        e = outcomes._ns(end)
        if when == "open":
            px, path_end = bar_value(minutes, end, "open"), e + NS_MIN
        else:
            px, path_end = price_known_at(minutes, end), e
        complete = minutes_present(minutes, start, path_end) == minutes_expected(sessions, start, path_end)
        inside = e <= outcomes._ns(sessions["close"].reindex(day))
        ok = complete & inside & ~np.isnan(px) & ~np.isnan(ref)
        status = np.where(day >= today, "pending", np.where(ok, "ok", "unavailable"))
        applies = alerts["strategy"].map(lambda s: h in STRATEGY_HORIZONS[s]).to_numpy(bool)
        out[f"ret_{h}_pct"] = np.where(applies & (status == "ok"), (px / ref - 1) * 100, np.nan)
        out[f"status_{h}"] = np.where(applies, status, "")
    return pd.DataFrame(out, index=alerts.index)


def path_extremes(minutes, start, end, rows):
    """Lowest low and highest high of the minute bars in [start, end), for the selected rows."""
    t = outcomes._ns(minutes["ts"])
    lo, hi = minutes["low"].to_numpy(float), minutes["high"].to_numpy(float)
    a, b = np.searchsorted(t, outcomes._ns(start)), np.searchsorted(t, outcomes._ns(end))
    low, high = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for i in np.flatnonzero(rows):
        low[i], high[i] = lo[a[i]:b[i]].min(), hi[a[i]:b[i]].max()
    return low, high


def fetch_ex_dividends(ticker, start, end, client=None):
    """Cash dividends with an ex-date in [start, end], from Massive's dividends endpoint."""
    if client is None:
        key = os.environ.get("MASSIVE_API_KEY")
        if not key:
            raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python alerts.py ...")
        client = RESTClient(api_key=key, retries=10)
    rows = [(d.ex_dividend_date, d.cash_amount, d.pay_date, d.dividend_type)
            for d in client.list_dividends(ticker=ticker, ex_dividend_date_gte=str(start),
                                           ex_dividend_date_lte=str(end), limit=1000)]
    divs = pd.DataFrame(rows, columns=["ex_dividend_date", "cash_amount", "pay_date", "dividend_type"])
    return divs.sort_values("ex_dividend_date", ignore_index=True)


def add_outcomes(alerts, minutes, sessions, today, dividends):
    """Price checks, horizon outcomes and naive-rule inputs for every alert (minutes: regular hours)."""
    a = alerts.copy()
    a["massive_price"] = price_known_at(minutes, a["alert_time"])
    a["ref_diff"] = a["ref_price"] - a["massive_price"]
    a["ref_diff_pct"] = a["ref_diff"] / a["massive_price"] * 100
    a["ref_off_0.1pct"] = a["ref_diff_pct"].abs() > MAX_REF_DIFF_PCT
    matches = {code: np.abs(a["ref_price"] - bar_value(minutes, a["alert_time"] + off * MINUTE, field))
               <= EXACT_DOLLARS for code, _, off, field in CONVENTIONS}
    a["ref_matches"] = [",".join(c for c in matches if matches[c].iloc[i]) for i in range(len(a))]

    following = pd.Series(sessions.index[1:], index=sessions.index[:-1])
    previous = pd.Series(sessions.index[:-1], index=sessions.index[1:])
    a["next_session"] = a["session"].map(following)

    out = horizon_returns(a, minutes, sessions, today, a["massive_price"])
    vs_ref = horizon_returns(a, minutes, sessions, today, a["ref_price"])
    side = a["side"].to_numpy(float)
    directional = side != 0
    for h in HORIZONS:
        a[f"ret_{h}_pct"] = out[f"ret_{h}_pct"]
        a[f"signed_{h}_pct"] = np.where(directional, side * out[f"ret_{h}_pct"], np.nan)
        a[f"status_{h}"] = out[f"status_{h}"]
    for h in HORIZONS:
        a[f"ret_{h}_pct_vs_ref_price"] = vs_ref[f"ret_{h}_pct"]

    # Intraday path from the alert to the close: how far SPY went against (and with) the call.
    rows = ((a["strategy"] == "intraday") & (a["status_close"] == "ok")).to_numpy()
    close = pd.DatetimeIndex(sessions["close"].reindex(a["session"]))
    low, high = path_extremes(minutes, a["alert_time"], close, rows)
    a["low_to_close_pct"] = (low / a["massive_price"] - 1) * 100
    a["high_to_close_pct"] = (high / a["massive_price"] - 1) * 100
    a["adverse_to_close_pct"] = np.select([side > 0, side < 0], [a["low_to_close_pct"], -a["high_to_close_pct"]],
                                          np.nan)
    signed_close = a["signed_close_pct"]
    for k in PREMIUM_PCT:
        known = signed_close.notna()
        a[f"closed_within_{k:g}pct"] = (signed_close >= -k).where(known).astype("boolean")
        a[f"touched_{k:g}pct"] = (a["adverse_to_close_pct"] < -k).where(known).astype("boolean")

    # Naive rules: intraday = prior close -> 10:00, overnight = today's open -> 15:55.
    prior_close = price_known_at(minutes, pd.DatetimeIndex(sessions["close"].reindex(a["session"].map(previous))))
    day_open = bar_value(minutes, pd.DatetimeIndex(sessions["open"].reindex(a["session"])), "open")
    a["prior_close"], a["session_open_price"] = prior_close, day_open
    start = np.where(a["strategy"] == "intraday", prior_close, day_open)
    a["momentum_side"] = np.sign(a["massive_price"] - start)

    a["ex_div_date"], a["ex_div_cash"] = "", np.nan
    if dividends is not None:
        for d in dividends.itertuples():
            ex = pd.Timestamp(d.ex_dividend_date)
            crosses = (a["session"] < ex) & (a["next_session"] >= ex)
            a.loc[crosses, "ex_div_date"], a.loc[crosses, "ex_div_cash"] = d.ex_dividend_date, d.cash_amount
    a["crosses_ex_div"] = (a["ex_div_date"] != "") if dividends is not None else pd.NA
    return a


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.959964):
    """95% Wilson interval for k successes in n, in percent."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half) * 100, (centre + half) * 100


def bootstrap(n, stat, reps, seed):
    """95% percentile interval of stat(index array of shape (reps, n)) over resampled dates."""
    if n == 0:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, n, (reps, n))
    lo, hi = np.percentile(stat(idx), [2.5, 97.5])
    return lo, hi


def shuffled_orders(n, reps, seed):
    return np.argsort(np.random.default_rng(seed).random((reps, n)), axis=1)


def binomial_tail(k, n, p=0.5):
    """P(at least k successes in n) for a coin with success chance p."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def binomial_head(k, n, p=0.5):
    """P(at most k successes in n)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def shuffle_test(side, raw, reps, seed):
    """Reassign the same calls across the same dates `reps` times. Returns (p_better, p_worse,
    percentile): the shares of shuffles scoring at least / at most the real hits (with the +1
    correction), and the percent scoring strictly fewer."""
    if len(side) == 0:
        return np.nan, np.nan, np.nan
    real = int((side * raw > 0).sum())
    sims = (side[shuffled_orders(len(side), reps, seed)] * raw > 0).sum(axis=1)
    return (1 + (sims >= real).sum()) / (reps + 1), (1 + (sims <= real).sum()) / (reps + 1), (sims < real).mean() * 100


def spearman_rows(x, y):
    """Spearman rank correlation of each row of x with the same row of y (ties get average ranks)."""
    rx = pd.DataFrame(np.atleast_2d(x)).rank(axis=1).to_numpy()
    ry = pd.DataFrame(np.atleast_2d(y)).rank(axis=1).to_numpy()
    rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))


def spearman_test(x, y, reps, seed):
    """Rank correlation, its date-bootstrap interval and a two-sided shuffle p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    rho = spearman_rows(x, y)[0]
    sims = spearman_rows(np.broadcast_to(x, (reps, len(x))), y[shuffled_orders(len(y), reps, seed)])
    p = (1 + (np.abs(sims) >= abs(rho) - 1e-12).sum()) / (reps + 1)
    idx = np.random.default_rng(seed).integers(0, len(x), (reps, len(x)))
    lo, hi = np.nanpercentile(spearman_rows(x[idx], y[idx]), [2.5, 97.5])  # NaN: a resample with one score
    return rho, lo, hi, p


def median_gap_test(values, is_neutral, reps, seed):
    """Median |move| on directional days minus NEUTRAL days, the one-sided shuffle p-value for
    NEUTRAL days moving less, and a two-sided one for any difference."""
    v, g = np.asarray(values, float), np.asarray(is_neutral, bool)
    real = np.median(v[~g]) - np.median(v[g])
    labels = g[shuffled_orders(len(g), reps, seed)]
    sims = np.nanmedian(np.where(~labels, v, np.nan), axis=1) - np.nanmedian(np.where(labels, v, np.nan), axis=1)
    centre = sims.mean()
    return (real, (1 + (sims >= real - 1e-12).sum()) / (reps + 1),
            (1 + (np.abs(sims - centre) >= abs(real - centre) - 1e-12).sum()) / (reps + 1))


class Results:
    """Long-format summary rows: one statistic per row."""

    def __init__(self, reps, seed):
        self.rows, self.reps, self.seed = [], reps, seed

    def add(self, section, strategy, horizon, subset, metric, value, n=None, count=None, lo=np.nan, hi=np.nan,
            p=np.nan, unit="%", role="exploratory", note=""):
        self.rows.append({"section": section, "strategy": strategy, "horizon": horizon, "subset": subset,
                          "metric": metric, "role": role, "n": n, "count": count, "value": value, "ci_low": lo,
                          "ci_high": hi, "p_value": p, "unit": unit, "note": note})

    def rate(self, section, strategy, horizon, subset, metric, hits, role="exploratory", note=""):
        hits = np.asarray(hits, bool)
        n, k = len(hits), int(hits.sum())
        lo, hi = wilson(k, n)
        self.add(section, strategy, horizon, subset, metric, k / n * 100 if n else np.nan, n, k, lo, hi,
                 role=role, note=note)

    def hits(self, strategy, horizon, subset, rows, ret_col="ret", role="exploratory", brief=False,
             section="hit_rate"):
        """Hit rate of the calls in `rows` against always-long on the same rows, with intervals,
        a coin-flip p-value and the shuffle test. Pending and unavailable rows are counted only."""
        directional = rows[rows["side"] != 0]
        status = directional[f"status_{horizon}"]
        raw_col = f"ret_{horizon}_pct" if ret_col == "ret" else f"ret_{horizon}_pct_vs_ref_price"
        d = directional[status == "ok"]
        side, raw = d["side"].to_numpy(float), d[raw_col].to_numpy(float)
        n, signed = len(d), side * raw
        hit, up = signed > 0, raw > 0
        k, ups = int(hit.sum()), int(up.sum())
        key = (section, strategy, horizon, subset)
        lo, hi = wilson(k, n)
        counts = f"{int((side > 0).sum())} bullish, {int((side < 0).sum())} bearish; " \
                 f"{int((status == 'pending').sum())} pending, {int((status == 'unavailable').sum())} unavailable"
        self.add(*key, "hit_rate", k / n * 100 if n else np.nan, n, k, lo, hi,
                 p=binomial_tail(k, n) if n else np.nan, role=role, note=counts + "; p = coin flip, one-sided")
        lo, hi = wilson(ups, n)
        self.add(*key, "base_rate_spy_up", ups / n * 100 if n else np.nan, n, ups, lo, hi, role=role,
                 note="always-bullish hit rate on the same rows")
        if brief:
            return
        lo, hi = bootstrap(n, lambda i: (hit[i].mean(axis=1) - up[i].mean(axis=1)) * 100, self.reps, self.seed)
        self.add(*key, "hit_minus_base", (hit.mean() - up.mean()) * 100 if n else np.nan, n, None, lo, hi,
                 unit="points", role=role, note="paired bootstrap over dates")
        self.add(*key, "coin_flip_p_worse", binomial_head(k, n) if n else np.nan, n, k,
                 p=binomial_head(k, n) if n else np.nan, unit="p", note="chance of this few hits or fewer")
        p, p_worse, pct = shuffle_test(side, raw, self.reps, self.seed)
        self.add(*key, "shuffled_calls_percentile", pct, n, None, p=p, role=role,
                 note=f"{self.reps:,} shuffles keeping the bullish/bearish counts; p = share at least as good")
        self.add(*key, "shuffled_calls_p_worse", p_worse, n, None, p=p_worse, unit="p",
                 note="share of shuffles at most as good")
        for name, values in (("mean_signed_ret", signed), ("median_signed_ret", signed), ("mean_long_ret", raw)):
            fn = np.mean if name.startswith("mean") else np.median
            lo, hi = bootstrap(n, lambda i, v=values, f=fn: f(v[i], axis=1), self.reps, self.seed)
            self.add(*key, name, fn(values) if n else np.nan, n, None, lo, hi, unit="% return",
                     note="bootstrap over dates")

    def frame(self):
        return pd.DataFrame(self.rows)


def run_study(alerts, minutes, sessions, today, dividends=None, *, reps=10_000, seed=0, timings=None):
    """Checks, outcomes and every statistic. No file I/O.

    sessions: XNYS calendar from one session before the first alert through each alert's next
    session. minutes: raw Massive minute bars (any hours) for the completed sessions.
    today: naive session date; outcomes ending in it or later are pending.
    """
    timings = {} if timings is None else timings
    today = pd.Timestamp(today)
    with timed(timings, "outcomes"):
        rth, counts = features.regular_session_minutes(minutes, sessions)
        done = sessions[sessions.index < today]
        have = rth.groupby("session").size().reindex(done.index, fill_value=0)
        expected = ((done["close"] - done["open"]) / MINUTE).astype(int)
        coverage = {**counts, "sessions": len(done), "first_session": str(done.index[0].date()),
                    "last_session": str(done.index[-1].date()), "minutes_expected": int(expected.sum()),
                    "minutes_present": int(have.sum()),
                    "sessions_missing_minutes": {str(d.date()): int(expected[d] - have[d])
                                                 for d in done.index[have < expected]}}
        a = check_rows(alerts, sessions)
        a = add_outcomes(a, rth, sessions, today, dividends)
        conventions = price_conventions(a[a["session"] < today], rth)

    with timed(timings, "statistics"):
        res = Results(reps, seed)
        variants = {"as recorded": a,
                    "flagged rows excluded": a[a["flag"] == ""],
                    "alternative calls from notes": a.assign(direction=a["alt_direction"],
                                                             side=a["alt_direction"].map(SIDES))}
        mid = done.index[(done.index >= a["session"].min())]
        split = mid[len(mid) // 2]
        for strategy, horizons in STRATEGY_HORIZONS.items():
            s = a[a["strategy"] == strategy]
            primary = PRIMARY[strategy]
            for h in horizons:
                res.hits(strategy, h, "as recorded", s, role="primary" if h == primary else "exploratory")
                res.hits(strategy, h, "ref_price as reference", s, ret_col="ref_price")
                if s["crosses_ex_div"].notna().all() and h != "close":
                    res.hits(strategy, h, "ex-dividend windows excluded", s[~s["crosses_ex_div"].astype(bool)])
            for name, v in variants.items():
                if name != "as recorded":
                    res.hits(strategy, primary, name, v[v["strategy"] == strategy])
            for direction in ("BULLISH", "BEARISH"):
                res.hits(strategy, primary, f"{direction} calls only", s[s["direction"] == direction], brief=True)
            res.hits(strategy, primary, f"first half (before {split.date()})", s[s["session"] < split], brief=True)
            res.hits(strategy, primary, f"second half (from {split.date()})", s[s["session"] >= split], brief=True)
            for month, m in s.groupby(s["session"].dt.strftime("%Y-%m")):
                res.hits(strategy, primary, month, m, brief=True)
            neutral_tests(res, strategy, s)
            naive_rules(res, strategy, s)

        intraday = a[(a["strategy"] == "intraday") & (a["side"] != 0) & (a["status_close"] == "ok")]
        for k in PREMIUM_PCT:
            sub = f"k = {k:g}%"
            res.rate("premium_distance", "intraday", "close", sub, "closed_within_k",
                     intraday[f"closed_within_{k:g}pct"].to_numpy(bool), note="signed close >= -k")
            res.rate("premium_distance", "intraday", "close", sub, "touched_k",
                     intraday[f"touched_{k:g}pct"].to_numpy(bool), note="some minute traded beyond -k before the close")
            res.rate("premium_distance", "intraday", "close", sub, "long_closed_within_k",
                     (intraday["ret_close_pct"] >= -k).to_numpy(), note="always bullish (sell puts daily), same rows")
            res.rate("premium_distance", "intraday", "close", sub, "long_touched_k",
                     (intraday["low_to_close_pct"] < -k).to_numpy(), note="always bullish, same rows")

        overnight = a[a["strategy"] == "overnight"]
        for h in STRATEGY_HORIZONS["overnight"]:
            d = overnight[(overnight["side"] != 0) & (overnight[f"status_{h}"] == "ok")]
            for k in MOVE_PCT:
                res.rate("move_size", "overnight", h, f"k = {k:g}%", "moved_k_with_call",
                         (d[f"signed_{h}_pct"] > k).to_numpy())
                res.rate("move_size", "overnight", h, f"k = {k:g}%", "long_moved_k_up",
                         (d[f"ret_{h}_pct"] > k).to_numpy(), note="always bullish, same rows")

        scored = overnight[overnight["status_next_open"] == "ok"]
        rho, lo, hi, p = spearman_test(scored["score"], scored["ret_next_open_pct"], reps, seed)
        res.add("score", "overnight", "next_open", "all overnight rows", "spearman_score_vs_return", rho,
                len(scored), None, lo, hi, p, unit="rho", note="SPY's next-open return (positive = up); "
                "bootstrap interval, two-sided shuffle p")
        for level in (2, 3, 4):
            res.hits("overnight", "next_open", f"|score| = {level}", overnight[overnight["score"].abs() == level],
                     brief=True, section="score")
    return {"table": a, "summary": res.frame(), "conventions": conventions, "coverage": coverage,
            "gaps": missing_rows(alerts, sessions), "off_schedule": off_schedule_alerts(alerts),
            "dividends": dividends, "split": split, "today": today, "reps": reps}


def neutral_tests(res, strategy, rows):
    """NEUTRAL implies a small or unclear move: compare |move| on NEUTRAL and directional days."""
    subsets = {"as recorded": rows}
    if rows["crosses_ex_div"].notna().all() and rows["crosses_ex_div"].astype(bool).any():
        subsets["ex-dividend windows excluded"] = rows[~rows["crosses_ex_div"].astype(bool)]
    for h in STRATEGY_HORIZONS[strategy]:
        for name, s in subsets.items():
            if name != "as recorded" and h == "close":
                continue
            d = s[s[f"status_{h}"] == "ok"]
            move = d[f"ret_{h}_pct"].abs().to_numpy()
            neutral = (d["side"] == 0).to_numpy()
            for label, mask in (("neutral", neutral), ("directional", ~neutral)):
                v = move[mask]
                lo, hi = bootstrap(len(v), lambda i, v=v: np.median(v[i], axis=1), res.reps, res.seed)
                res.add("neutral", strategy, h, name, f"median_abs_move_{label}", np.median(v) if len(v) else np.nan,
                        len(v), None, lo, hi, unit="% |return|", note="bootstrap over dates")
                res.add("neutral", strategy, h, name, f"mean_abs_move_{label}", v.mean() if len(v) else np.nan,
                        len(v), None, unit="% |return|")
            if neutral.any() and (~neutral).any():
                gap, p, p_two = median_gap_test(move, neutral, res.reps, res.seed)
                res.add("neutral", strategy, h, name, "median_gap_directional_minus_neutral", gap, len(move), None,
                        p=p, unit="% |return|", note="one-sided shuffle p for NEUTRAL days moving less")
                res.add("neutral", strategy, h, name, "median_gap_p_two_sided", p_two, len(move), None, p=p_two,
                        unit="p", note="two-sided shuffle p for any difference")


def naive_rules(res, strategy, rows):
    """Naive rules on the system's own directional rows: their hit rate and how often the system agrees."""
    h = PRIMARY[strategy]
    d = rows[(rows["side"] != 0) & (rows[f"status_{h}"] == "ok") & (rows["momentum_side"].fillna(0) != 0)]
    rules = {"intraday": (("prior close to 10:00 direction", 1), ("reverse of prior close to 10:00", -1)),
             "overnight": (("open to 15:55 direction", 1),)}[strategy]
    raw = d[f"ret_{h}_pct"].to_numpy(float)
    for name, flip in rules:
        rule = flip * d["momentum_side"].to_numpy(float)
        res.rate("naive_rule", strategy, h, name, "rule_hit_rate", rule * raw > 0,
                 note="on the sessions where the system made a directional call")
        res.rate("naive_rule", strategy, h, name, "system_agrees", rule == d["side"].to_numpy(float))


# ---------------------------------------------------------------- outputs

def pick(summary, section, strategy, horizon, subset, metric):
    m = summary[(summary["section"] == section) & (summary["strategy"] == strategy) & (summary["horizon"] == horizon)
                & (summary["subset"] == subset) & (summary["metric"] == metric)]
    return m.iloc[0] if len(m) else None


def plot_hit_rates(summary, path):
    """Hit rate against the always-long base rate, with Wilson intervals, per strategy and horizon."""
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    widths = [len(v) for v in STRATEGY_HORIZONS.values()]
    grid = fig.add_gridspec(1, 2, width_ratios=widths, left=0.08, right=0.98, bottom=0.25, top=0.77, wspace=0.08)
    for g, (strategy, horizons) in enumerate(STRATEGY_HORIZONS.items()):
        ax = fig.add_subplot(grid[g])
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7.5)
        ns = [int(pick(summary, "hit_rate", strategy, h, "as recorded", "hit_rate")["n"]) for h in horizons]
        for j, (metric, name) in enumerate((("hit_rate", "Alert calls"), ("base_rate_spy_up", "SPY rose (always bullish)"))):
            for x, h in enumerate(horizons):
                r = pick(summary, "hit_rate", strategy, h, "as recorded", metric)
                pos = x + (j - 0.5) * 0.34
                ax.bar(pos, r["value"], 0.3, color=SERIES[j], zorder=3, label=name if (g, x) == (0, 0) else None)
                ax.errorbar(pos, r["value"], yerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=4)
                ax.annotate(f"{r['value']:.0f}%", (pos, r["ci_high"]), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=INK_2)
        labels = [f"to {END_LABELS[h]}{' (primary)' if h == PRIMARY[strategy] else ''}\nn = {n}"
                  for h, n in zip(horizons, ns)]
        ax.set_xticks(range(len(horizons)), labels)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        if g:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Share of calls (%)", color=INK_2, fontsize=8)
        ax.set_title(f"{strategy.capitalize()} calls (posted {SCHEDULE[strategy]} ET)", color=INK, fontsize=9,
                     loc="left")
    fig.text(0.02, 0.97, "How often SPY moved the way the alert said", color=INK, fontsize=11, fontweight="bold",
             va="top")
    fig.text(0.02, 0.915, "Hit rate of BULLISH/BEARISH calls next to how often SPY simply rose over the same "
             "windows.\nWhiskers: 95% Wilson intervals. Dashed line: 50%, a coin flip.", color=INK_2, fontsize=7.5,
             va="top", linespacing=1.5)
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.98, 0.03, "Signal study of SPY, not option P&L", color=MUTED, fontsize=7, ha="right")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_score(table, summary, path):
    """Overnight next-open return by score, every row, with the mean per score."""
    d = table[(table["strategy"] == "overnight") & (table["status_next_open"] == "ok")]
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0.09, 0.17, 0.88, 0.6])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=8)
    ax.axvspan(-1.5, 1.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(0, 1.01, "NEUTRAL (no play)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
            color=MUTED)
    ax.axhline(0, color=BASELINE, lw=1, zorder=1)
    rng = np.random.default_rng(0)  # jitter only
    scores = range(-4, 5)
    for s in scores:
        v = d.loc[d["score"] == s, "ret_next_open_pct"].to_numpy(float)
        if not len(v):
            continue
        ax.scatter(s + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=SERIES[0], alpha=0.7, lw=0, zorder=3)
        ax.plot([s - 0.3, s + 0.3], [v.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.annotate(f"n={len(v)}", (s, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                    textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(list(scores), [f"{s:+d}" if s else "0" for s in scores])
    ax.set_xlim(-4.6, 4.6)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel("SPY 15:55 → next open (%)", color=INK_2, fontsize=8)
    r = pick(summary, "score", "overnight", "next_open", "all overnight rows", "spearman_score_vs_return")
    fig.text(0.02, 0.97, "Overnight score against what SPY did by the next open", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Each dot is one overnight alert; black bars are the mean per score.\nRank correlation "
             f"{r['value']:+.2f} (95% interval {r['ci_low']:+.2f} to {r['ci_high']:+.2f}, shuffle p = "
             f"{r['p_value']:.2f}, n = {int(r['n'])}).", color=INK_2, fontsize=7.5, va="top", linespacing=1.5)
    fig.text(0.02, 0.03, "Score ≥ +2 = BULLISH (buy calls), ≤ −2 = BEARISH (buy puts). Signal study of SPY, not "
             "option P&L.", color=MUTED, fontsize=7)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def table_columns(table):
    """alerts_with_outcomes.csv: the CSV's columns, then checks, then outcomes."""
    t = table.assign(alert_time_utc=table["alert_time"].dt.strftime("%Y-%m-%dT%H:%MZ"),
                     next_session=table["next_session"].dt.strftime("%Y-%m-%d"))
    first = ["date", "weekday", "strategy", "time_et", "direction", "action", "ticker", "ref_price", "score", "flag",
             "alerts_posted", "notes", "source_line", "alert_time_utc", "issues", "alt_direction", "side",
             "massive_price", "ref_diff", "ref_diff_pct", "ref_off_0.1pct", "ref_matches", "next_session",
             "ex_div_date", "ex_div_cash", "crosses_ex_div"]
    outcome = [c for h in HORIZONS for c in (f"status_{h}", f"ret_{h}_pct", f"signed_{h}_pct")]
    path = ["low_to_close_pct", "high_to_close_pct", "adverse_to_close_pct"] + \
           [f"{kind}_{k:g}pct" for k in PREMIUM_PCT for kind in ("closed_within", "touched")]
    rest = ["prior_close", "session_open_price", "momentum_side"] + [f"ret_{h}_pct_vs_ref_price" for h in HORIZONS]
    return t[first + outcome + path + rest]


def write_outputs(out_dir, result, source):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_columns(result["table"]).to_csv(out / "alerts_with_outcomes.csv", index=False)
    result["summary"].to_csv(out / "summary.csv", index=False)
    plot_hit_rates(result["summary"], out / "hit_rates.png")
    plot_score(result["table"], result["summary"], out / "overnight_score.png")
    (out / "report.md").write_text(render_report(result, source))
    return out


# ---------------------------------------------------------------- report.md

def pct(v, digits=1, sign=False):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def interval(r, digits=1, unit="%"):
    return "n/a" if pd.isna(r["ci_low"]) else f"{r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f}{unit}"


def p_text(p):
    return "n/a" if pd.isna(p) else "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def window_label(strategy, h):
    return f"{SCHEDULE[strategy]} to {END_LABELS[h]}"


def render_report(result, source):
    a, s, cov = result["table"], result["summary"], result["coverage"]
    today = result["today"]
    hit = lambda strategy, h, subset="as recorded", metric="hit_rate", section="hit_rate": pick(  # noqa: E731
        s, section, strategy, h, subset, metric)
    lines = []
    w = lines.append

    w(f"# Alert log against the market: SPY, {a['date'].min()} to {a['date'].max()}\n")
    w(f"Generated by `alerts.py` on {today.date()} from {len(a)} alert rows and Massive one-minute SPY bars. "
      "This is a signal study of what SPY did after each alert. It is **not option P&L**: the alert log has no "
      "strikes, expiries or premiums, so whether a trade made money also depended on things measured nowhere "
      "here (premium paid or received, time decay, volatility, fills).\n")
    w("## How to read this\n")
    w("- **Hit:** SPY moved the way the call said over the window. BULLISH needs SPY higher at the end of the "
      "window than at the alert, BEARISH needs it lower. NEUTRAL (NO PLAY) rows make no directional call, so "
      "they are left out of hit rates and checked separately.")
    w("- **Base rate:** how often SPY simply rose over the same windows on the same days. Someone who called "
      "BULLISH every time would have scored exactly this, so a hit rate only means something next to it. In a "
      "rising market, 60% can be unremarkable.")
    w("- **95% interval:** the range of true hit rates that fit the data. It is wide with few calls: with "
      "about 50 calls, a hit rate's interval is roughly ±14 points, so 56% is consistent with anything from "
      "about 42% to 70%.")
    w("- **Shuffle test:** keep the system's own mix of BULLISH and BEARISH calls but hand them out to the same "
      f"dates at random, {result['reps']:,} times. If the real calls beat most shuffles, their timing carried information. The p-value is the share "
      "of shuffles that did at least as well: small (below 0.05) means hard to explain by luck.")
    w("- **Coin flip:** random BULLISH/BEARISH calls are right 50% of the time whatever the market does.")
    w("- **Primary vs exploratory:** exactly two results were fixed in advance as the main test (section 2). "
      "Everything in section 3 is exploratory: with dozens of comparisons, a few will look notable by chance.\n")

    # ------------------------------------------------ 1. data checks
    w("## 1. Data checks\n")
    w("### The alert log\n")
    counts = a.groupby(["strategy", "direction"]).size().unstack(fill_value=0)
    w(md_table(["Strategy", "Rows", "BULLISH", "BEARISH", "NEUTRAL"],
               [[st, int(counts.loc[st].sum()), *(int(counts.loc[st].get(d, 0)) for d in SIDES)]
                for st in counts.index]))
    w("")
    issue_types = pd.Series([i for x in a["issues"] for i in x.split("; ") if i]).value_counts()
    consistent = ["weekday does not match the date", "action does not match direction",
                  "score does not match direction", "intraday row has a score", f"ticker is not {TICKER}",
                  "duplicate date and strategy", "not an XNYS session"]
    bad = [c for c in consistent if c in issue_types]
    if bad:
        w("Consistency problems: " + "; ".join(f"{c} ({issue_types[c]} rows)" for c in bad) + ". Rows: " +
          ", ".join(f"{r.date} {r.strategy}" for r in a.itertuples() if any(c in r.issues for c in bad)) + ".")
    else:
        w("Every row is internally consistent: each date is an XNYS session and matches its weekday, the ticker "
          "is always SPY, every action matches its direction, every overnight score maps to its direction "
          "(≥ +2 BULLISH, ≤ −2 BEARISH, otherwise NEUTRAL), intraday rows carry no score, and no date has two "
          "rows for the same strategy.")
    posted = a[a["issues"].str.contains("alerts_posted")]
    w(f"\n`alerts_posted` breaks the usual overnight pattern (BULLISH 1, BEARISH 2, NEUTRAL 3) on {len(posted)} "
      "rows: " + ", ".join(f"{r.date} ({r.direction}, {r.alerts_posted} posted{', ' + r.flag if r.flag else ''})"
                           for r in posted.itertuples()) + ". It otherwise just restates the direction, so it "
      "adds no independent information and is not analysed.\n")
    flagged = a[a["flag"] != ""]
    w("Flagged rows (the row holds the recorded call; the alternative comes from `notes`):\n")
    w(md_table(["Date", "Strategy", "Flag", "Recorded call", "Alternative in notes"],
               [[r.date, r.strategy, r.flag, r.direction, r.alt_direction] for r in flagged.itertuples()]))
    off = result["off_schedule"]
    if off:
        w(f"\n`notes` also mention {len(off)} off-schedule alert(s) with no row of their own, not analysed: " +
          ", ".join(f"{o['date']} {o['time_et']} ET {o['direction']}" for o in off) + ".")
    w("\n### Alert times\n")
    off_time = a[a["issues"].str.contains("posting schedule")]
    outside = a[a["issues"].str.contains("regular hours")]
    w(f"Every alert time was checked against its strategy's schedule (intraday {SCHEDULE['intraday']}, overnight "
      f"{SCHEDULE['overnight']} ET) and against its session's XNYS hours, early closes included. "
      f"Outside regular hours: {len(outside) or 'none'}. Off schedule: {len(off_time) or 'none'}.")
    for r in off_time.itertuples():
        clock = pd.Timestamp(r.alert_time).tz_convert(NY)
        times = sorted({(clock + KNOWN_AT[c] * MINUTE).strftime("%H:%M") for c in r.ref_matches.split(",")
                        if c in KNOWN_AT})
        w(f"\n- {r.date} {r.strategy} at {r.time_et} ET: its ref_price ({r.ref_price:.2f}) equals the Massive price "
          f"at {' and '.join(times) or 'none of the minutes checked'} ET" +
          (f", not {r.time_et}. It is most likely the {SCHEDULE[r.strategy]} alert stamped a minute early. "
           if times and r.time_et not in times and SCHEDULE[r.strategy] in times else ". ") +
          f"Outcomes use the recorded {r.time_et} time; section 3.10 shows the results with ref_price instead.")
    w("\n### Missing rows\n")
    w("A missing row is not a NO PLAY: these sessions are simply absent and are not filled.\n")
    for st, days in result["gaps"].items():
        note = " (today: the alert may not have been posted yet when this ran)" if str(today.date()) in days else ""
        w(f"- **{st}** ({len(days)}): {', '.join(days) or 'none'}{note}")
    w("\n### Does `ref_price` match the market?\n")
    conv = result["conventions"]
    best = conv.sort_values(["median_abs_diff_pct", "exact"], ascending=[True, False]).iloc[0]
    n_checked = int(conv["n"].max())
    w(f"`ref_price` was compared with Massive minute bars around each alert time ({n_checked} rows; rows from "
      "today's session are not loaded yet). *Exact* means equal to the cent: Massive prices can carry sub-penny "
      "trades (e.g. 748.165), which a two-decimal `ref_price` rounds.\n")
    w(md_table(["Massive price", "Exact matches", "Median gap", "Largest gap", "Rows off > 0.1%"],
               [[c.convention, f"{c.exact}/{c.n}", pct(c.median_abs_diff_pct, 4), pct(c.max_abs_diff_pct, 3),
                 c.rows_over_limit] for c in conv.itertuples()]))
    checked = a[a["massive_price"].notna()]
    off_rows = checked[checked["ref_off_0.1pct"]]
    inexact = checked[(checked["ref_diff"].abs() > EXACT_DOLLARS)].sort_values("ref_diff_pct", key=abs,
                                                                                ascending=False)
    exact = checked["ref_diff"].abs() <= EXACT_DOLLARS
    half_cent = int((exact & ((checked["massive_price"] * 1000).round() % 10 == 5)).sum())
    w(f"\nBest match: **{best.convention}**, the price known at the alert time ({best.exact} of {best.n} exact, "
      f"{half_cent} of them a half-cent Massive price rounded to the cent). This study uses that price as its "
      "reference. Against it, " +
      (f"**{len(off_rows)} rows are off by more than 0.1%**: " + ", ".join(
          f"{r.date} {r.strategy} ({pct(r.ref_diff_pct, 3, True)})" for r in off_rows.itertuples())
       if len(off_rows) else "no row is off by more than 0.1%") +
      f". {len(inexact)} rows differ from it by more than half a cent" + (":\n" if len(inexact) else "."))
    if len(inexact):
        w(md_table(["Date", "Strategy", "Time", "ref_price", "Massive at alert time", "Gap", "ref_price equals"],
                   [[r.date, r.strategy, r.time_et, f"{r.ref_price:.2f}", f"{r.massive_price:.4f}",
                     f"{r.ref_diff:+.2f} ({pct(r.ref_diff_pct, 3, True)})",
                     ", ".join(dict(c[:2] for c in CONVENTIONS)[x] for x in r.ref_matches.split(",") if x)
                     or "none of the prices checked"]
                    for r in inexact.itertuples()]))
    w("\n### Market data\n")
    missing = cov["sessions_missing_minutes"]
    w(f"- Source: {source}. SPY one-minute bars, split-adjusted, not dividend-adjusted.")
    w(f"- {cov['sessions']} completed XNYS sessions, {cov['first_session']} to {cov['last_session']} (one session "
      "before the first alert, so every alert has a prior close). Today's session is not loaded.")
    w(f"- Regular-hours minutes: {cov['minutes_present']:,} of {cov['minutes_expected']:,} expected. " +
      ("No minute is missing." if not missing else
       "Missing minutes by session: " + ", ".join(f"{d} ({n})" for d, n in missing.items()) + "."))
    unavailable = {f"{st} {window_label(st, h)}": int((a.loc[a['strategy'] == st, f'status_{h}'] == 'unavailable').sum())
                   for st, hs in STRATEGY_HORIZONS.items() for h in hs}
    n_unavailable = sum(unavailable.values())
    w(f"- Outcomes unavailable because a needed minute is missing: {n_unavailable}" +
      ("." if not n_unavailable else " (" + ", ".join(f"{k}: {v}" for k, v in unavailable.items() if v) + ")."))
    w("\n### Ex-dividend dates\n")
    divs = result["dividends"]
    if divs is None:
        w("Massive's dividends endpoint could not be read, so ex-dividend crossings are **unavailable**. "
          "Overnight windows that span an ex-date include a drop of about the dividend.")
    else:
        crossing = a[a["crosses_ex_div"].astype(bool)]
        w("From Massive's dividends endpoint: " + ", ".join(
            f"ex-date {d.ex_dividend_date}, ${d.cash_amount:.4f} ({d.dividend_type})" for d in divs.itertuples()) +
          ". Prices are not dividend-adjusted, so SPY opens roughly the dividend (about 0.25%) lower on an "
          "ex-date. Windows that span one: " + (", ".join(
              f"{r.date} {r.strategy} ({r.direction})" for r in crossing.itertuples()) or "none") +
          ". The intraday same-day window never spans one (the drop happens at the open, before 10:00).")
    w("\n### Pending\n")
    pend = [(r.date, r.strategy, [window_label(r.strategy, h) for h in STRATEGY_HORIZONS[r.strategy]
                                  if getattr(r, f"status_{h}") == "pending"]) for r in a.itertuples()]
    pend = [p for p in pend if p[2]]
    w("Outcomes that end in today's session or later are pending and left out of every statistic:\n")
    for date, st, hs in pend:
        w(f"- {date} {st}: {', '.join(hs)}")

    # ------------------------------------------------ 2. primary results
    w("\n## 2. Primary results\n")
    w("Two results were fixed in advance: the intraday hit rate from 10:00 to the same day's close, and the "
      "overnight hit rate from 15:55 to the next open. Each is shown next to its base rate.\n")
    rows = []
    for st in STRATEGY_HORIZONS:
        h = PRIMARY[st]
        r, b, d, sh = (hit(st, h, metric=m) for m in ("hit_rate", "base_rate_spy_up", "hit_minus_base",
                                                       "shuffled_calls_percentile"))
        rows.append([f"{st} ({window_label(st, h)})", f"{int(r['n'])} ({r['note'].split(';')[0]})",
                     f"**{pct(r['value'])}** ({int(r['count'])}/{int(r['n'])}; {interval(r)})",
                     f"{pct(b['value'])} ({int(b['count'])}/{int(b['n'])}; {interval(b)})",
                     f"{d['value']:+.1f} pts ({d['ci_low']:+.1f} to {d['ci_high']:+.1f})",
                     f"p = {p_text(sh['p_value'])}", f"p = {p_text(r['p_value'])}"])
    w(md_table(["Calls", "Directional calls", "Hit rate (95% interval)", "SPY rose: always-bullish score",
                "Hit rate minus base rate", "Shuffle test", "vs coin flip"], rows))
    w("")
    for st in STRATEGY_HORIZONS:
        w(primary_sentence(st, hit))
    w("\n![Hit rate against base rate](hit_rates.png)\n")

    # ------------------------------------------------ 3. exploratory
    w("## 3. Exploratory findings\n")
    w("Not planned as the main test. Many comparisons follow, so treat any single striking number with "
      "suspicion unless it lines up with the primary results.\n")
    w("### 3.1 Every horizon\n")
    rows = []
    for st, hs in STRATEGY_HORIZONS.items():
        for h in hs:
            r, b, d, sh, m, ml = (hit(st, h, metric=x) for x in (
                "hit_rate", "base_rate_spy_up", "hit_minus_base", "shuffled_calls_percentile", "mean_signed_ret",
                "mean_long_ret"))
            rows.append([st, window_label(st, h) + (" (primary)" if h == PRIMARY[st] else ""),
                         f"{pct(r['value'])} ({int(r['count'])}/{int(r['n'])}; {interval(r)})",
                         f"{pct(b['value'])}", f"{d['value']:+.1f} ({d['ci_low']:+.1f} to {d['ci_high']:+.1f})",
                         p_text(sh["p_value"]),
                         f"{m['value']:+.3f}% ({m['ci_low']:+.3f} to {m['ci_high']:+.3f})", f"{ml['value']:+.3f}%"])
    w(md_table(["Strategy", "Window", "Hit rate (95% interval)", "Base rate", "Minus base (pts)",
                "Shuffle p (as good or better)",
                "Mean move with the call (95% interval)", "Mean SPY move (always long)"], rows))
    w("\nMean move with the call is SPY's return signed by the call (positive = SPY went the called way), "
      "averaged over directional calls, with a bootstrap interval over dates. Medians are in `summary.csv`.\n")
    w("By call direction (primary windows). A BEARISH call is right when SPY fell, so its always-bullish "
      "score on the same days is the complement.\n")
    rows = []
    for st in STRATEGY_HORIZONS:
        for direction in ("BULLISH", "BEARISH"):
            r = hit(st, PRIMARY[st], f"{direction} calls only")
            rows.append([st, direction, f"{pct(r['value'])} ({int(r['count'])}/{int(r['n'])}; {interval(r)})"])
    w(md_table(["Strategy", "Call", "Hit rate (95% interval)"], rows))
    w("")

    w("### 3.2 Intraday: premium-selling distances\n")
    w("Selling puts (BULLISH) or calls (BEARISH) at 10:00 pays as long as SPY doesn't finish too far against "
      "the call. *Closed within k*: SPY's close was no more than k against the call. *Touched*: some minute "
      "between 10:00 and the close traded beyond k against the call. Always-bullish = selling puts every one "
      "of the same days.\n")
    rows = []
    for k in PREMIUM_PCT:
        sub = f"k = {k:g}%"
        c, t, lc, lt = (pick(s, "premium_distance", "intraday", "close", sub, m) for m in (
            "closed_within_k", "touched_k", "long_closed_within_k", "long_touched_k"))
        rows.append([f"{k:g}%", f"{pct(c['value'])} ({int(c['count'])}/{int(c['n'])}; {interval(c)})",
                     pct(lc["value"]), f"{pct(t['value'])} ({int(t['count'])}/{int(t['n'])})", pct(lt["value"])])
    w(md_table(["k against the call", "Alerts: closed within k", "Always bullish: closed within k",
                "Alerts: touched k before close", "Always bullish: touched"], rows))

    w("\n### 3.3 Overnight: was the move big enough?\n")
    w("Buying calls or puts at 15:55 needs a move large enough to beat the premium. Share of directional "
      "overnight calls where SPY moved more than k in the called direction, next to always-bullish (SPY up "
      "more than k).\n")
    rows = []
    for h in STRATEGY_HORIZONS["overnight"]:
        for k in MOVE_PCT:
            m = pick(s, "move_size", "overnight", h, f"k = {k:g}%", "moved_k_with_call")
            lm = pick(s, "move_size", "overnight", h, f"k = {k:g}%", "long_moved_k_up")
            rows.append([window_label("overnight", h), f"{k:g}%",
                         f"{pct(m['value'])} ({int(m['count'])}/{int(m['n'])}; {interval(m)})", pct(lm["value"])])
    w(md_table(["Window", "Move more than", "Alerts", "Always bullish"], rows))

    w("\n### 3.4 Ex-dividend windows\n")
    if divs is None:
        w("Unavailable: the dividends endpoint could not be read.")
    else:
        crossing = a[a["crosses_ex_div"].astype(bool)]
        directional_crossing = crossing[crossing["side"] != 0]
        rows = []
        for st, hs in STRATEGY_HORIZONS.items():
            for h in hs:
                ex = hit(st, h, "ex-dividend windows excluded")
                if ex is not None:
                    r = hit(st, h)
                    rows.append([st, window_label(st, h), f"{pct(r['value'])} (n={int(r['n'])})",
                                 f"{pct(ex['value'])} (n={int(ex['n'])})"])
        w(f"{len(crossing)} alert windows span an ex-date, {len(directional_crossing)} of them with a directional "
          "call" + (": " + ", ".join(f"{r.date} {r.strategy} {r.direction}" for r in directional_crossing.itertuples())
                    if len(directional_crossing) else "") +
          ". A call whose window spans an ex-date gets the mechanical drop for free if BEARISH and against it if "
          "BULLISH. NEUTRAL rows only affect the NEUTRAL comparison (3.5), which is also shown without them. Hit "
          "rates with the spanning rows excluded:\n")
        w(md_table(["Strategy", "Window", "As recorded", "Ex-dividend windows excluded"], rows))

    w("\n### 3.5 NEUTRAL / NO PLAY days\n")
    w("NEUTRAL implies no clear move. If that call means anything, SPY should move less (in either direction) "
      "on NEUTRAL days than on days with a directional call. Median absolute move:\n")
    rows = []
    for st, hs in STRATEGY_HORIZONS.items():
        for h in hs:
            for sub in ("as recorded", "ex-dividend windows excluded"):
                nm = pick(s, "neutral", st, h, sub, "median_abs_move_neutral")
                if nm is None:
                    continue
                dm = pick(s, "neutral", st, h, sub, "median_abs_move_directional")
                gap = pick(s, "neutral", st, h, sub, "median_gap_directional_minus_neutral")
                two = pick(s, "neutral", st, h, sub, "median_gap_p_two_sided")
                rows.append([st, window_label(st, h) + ("" if sub == "as recorded" else ", ex-div excluded"),
                             f"{pct(nm['value'], 3)} (n={int(nm['n'])}; {interval(nm, 3)})",
                             f"{pct(dm['value'], 3)} (n={int(dm['n'])}; {interval(dm, 3)})",
                             p_text(gap["p_value"]) if gap is not None else "n/a",
                             p_text(two["p_value"]) if two is not None else "n/a"])
    w(md_table(["Strategy", "Window", "NEUTRAL days", "Directional days", "p: NEUTRAL moved less",
                "p: any difference"], rows))
    w("")
    w(neutral_sentence(s))

    w("\n### 3.6 Overnight score\n")
    r = pick(s, "score", "overnight", "next_open", "all overnight rows", "spearman_score_vs_return")
    w(f"Rank correlation between the overnight score (−4 to +4, every row including NEUTRAL) and SPY's return "
      f"from 15:55 to the next open: **{r['value']:+.2f}** (95% interval {r['ci_low']:+.2f} to "
      f"{r['ci_high']:+.2f}; shuffle p = {p_text(r['p_value'])}; n = {int(r['n'])}). +1 would mean higher scores "
      "always came with higher returns; 0 means no relationship." + (
          " Here it is negative and its interval excludes zero: more bullish scores came before weaker openings, "
          "the opposite of what the score claims." if r["ci_high"] < 0 else
          " Here it is positive and its interval excludes zero: higher scores came before stronger openings."
          if r["ci_low"] > 0 else " Its interval includes zero.") + "\n")
    rows = []
    for level in (2, 3, 4):
        h_ = pick(s, "score", "overnight", "next_open", f"|score| = {level}", "hit_rate")
        b_ = pick(s, "score", "overnight", "next_open", f"|score| = {level}", "base_rate_spy_up")
        rows.append([f"±{level}", int(h_["n"]), f"{pct(h_['value'])} ({int(h_['count'])}/{int(h_['n'])}; "
                     f"{interval(h_)})", pct(b_["value"])])
    w(md_table(["|score|", "Calls", "Hit rate (95% interval)", "Base rate"], rows))
    w("\n![Overnight return by score](overnight_score.png)\n")

    w("### 3.7 Naive rules: is it just following momentum?\n")
    w("Each rule is scored on the same sessions as the system's directional calls (sessions where the rule "
      "has no direction are dropped). *Agrees* is how often the system made the same call.\n")
    rows = []
    for st in STRATEGY_HORIZONS:
        h = PRIMARY[st]
        sysr = hit(st, h)
        rows.append([st, "the alerts", f"{pct(sysr['value'])} ({int(sysr['count'])}/{int(sysr['n'])})", "—"])
        for name in sorted(set(s[(s["section"] == "naive_rule") & (s["strategy"] == st)]["subset"])):
            rr = pick(s, "naive_rule", st, h, name, "rule_hit_rate")
            ag = pick(s, "naive_rule", st, h, name, "system_agrees")
            rows.append([st, name, f"{pct(rr['value'])} ({int(rr['count'])}/{int(rr['n'])}; {interval(rr)})",
                         f"{pct(ag['value'])} ({int(ag['count'])}/{int(ag['n'])})"])
    w(md_table(["Strategy", "Rule", "Hit rate", "System agrees"], rows))
    w("\n" + momentum_sentence(s))

    w("\n### 3.8 Stability over time\n")
    w(f"Primary hit rates by half (split at {result['split'].date()}) and by month, with the base rate for the "
      "same rows. Small groups swing a lot; a month with 10 calls has an interval of about ±25 points.\n")
    subsets = [x for x in dict.fromkeys(s.loc[(s["section"] == "hit_rate") & s["subset"].str.match(
        r"(first|second) half|\d{4}-\d{2}$"), "subset"])]
    rows = []
    for sub in subsets:
        row = [sub if not re.match(r"\d{4}-\d{2}$", sub) else MONTHS[int(sub[5:]) - 1] + " " + sub[:4]]
        for st in STRATEGY_HORIZONS:
            r, b = hit(st, PRIMARY[st], sub), hit(st, PRIMARY[st], sub, "base_rate_spy_up")
            row.append("—" if r is None or not r["n"] else
                       f"{pct(r['value'], 0)} ({int(r['count'])}/{int(r['n'])}) vs {pct(b['value'], 0)}")
        if any(c != "—" for c in row[1:]):
            rows.append(row)
    w(md_table(["Period", "Intraday hit rate vs base", "Overnight hit rate vs base"], rows))

    w("\n### 3.9 Flagged rows\n")
    rows = []
    for sub in ("as recorded", "flagged rows excluded", "alternative calls from notes"):
        row = [sub]
        for st in STRATEGY_HORIZONS:
            r, b = hit(st, PRIMARY[st], sub), hit(st, PRIMARY[st], sub, "base_rate_spy_up")
            row.append(f"{pct(r['value'])} ({int(r['count'])}/{int(r['n'])}) vs {pct(b['value'])}")
        rows.append(row)
    w(md_table(["Rows", "Intraday hit rate vs base", "Overnight hit rate vs base"], rows))
    w("\nSwapping in a NEUTRAL alternative removes that row from the directional calls; swapping a NEUTRAL "
      "record for a directional alternative adds one.")

    w("\n### 3.10 Using the CSV's ref_price as the reference\n")
    rows = []
    for st, hs in STRATEGY_HORIZONS.items():
        for h in hs:
            r, rp = hit(st, h), hit(st, h, "ref_price as reference")
            rows.append([st, window_label(st, h), f"{pct(r['value'])} ({int(r['count'])}/{int(r['n'])})",
                         f"{pct(rp['value'])} ({int(rp['count'])}/{int(rp['n'])})"])
    w(md_table(["Strategy", "Window", "Massive price at alert time", "CSV ref_price"], rows))
    changed = 0
    for h in HORIZONS:
        d = a[(a["side"] != 0) & a[f"ret_{h}_pct"].notna()]
        changed += int(((d["side"] * d[f"ret_{h}_pct"] > 0) != (d["side"] * d[f"ret_{h}_pct_vs_ref_price"] > 0)).sum())
    w(f"\nAcross all horizons, {changed} directional outcome(s) flip between hit and miss when ref_price is used.")

    # ------------------------------------------------ 4. caveats
    w("\n## 4. Caveats\n")
    n_calls = {st: int(hit(st, PRIMARY[st])["n"]) for st in STRATEGY_HORIZONS}
    w(f"- **Small samples.** {n_calls['intraday']} intraday and {n_calls['overnight']} overnight directional calls "
      f"(95% intervals of about ±{100 / math.sqrt(n_calls['intraday']):.0f} and "
      f"±{100 / math.sqrt(n_calls['overnight']):.0f} points). A 10-point gap between a hit rate and its base rate "
      "is well inside the noise.")
    w(f"- **One market regime.** {a['date'].min()} to {a['date'].max()}, under five months of a single market. "
      "A rule that worked or failed here may behave differently in another regime.")
    w("- **Not option P&L.** A hit says SPY moved the right way, not that the option trade made money. Premium "
      "sellers can be right on direction and still lose to a large adverse swing intraday, and option buyers "
      "need the move to beat the premium and time decay.")
    w("- **Reference prices.** Outcomes start from the Massive price known at the alert time, not a fill. The "
      "next open is the first regular-hours minute's open, and the close is the last regular minute's close; "
      "both can differ by a few cents from the official auction prints.")
    w("- **Overlapping windows.** The next-session horizons of consecutive days overlap, so those results are "
      "not fully independent.")
    w("- **Two primary tests.** Requiring p < 0.025 for either would keep the chance of a false alarm across "
      "both at 5%.")
    w("- **Unverifiable here:** the original message log (`source_line` points into it), whether the recorded "
      "call or the alternative is the one followers actually saw first on flagged rows, and anything about "
      "option prices.")

    # ------------------------------------------------ 5. bottom line
    w("\n## 5. Bottom line\n")
    w(bottom_line(hit, n_calls))
    return "\n".join(lines) + "\n"


def primary_sentence(strategy, hit):
    h = PRIMARY[strategy]
    r, b, d, sh, shw, cw = (hit(strategy, h, metric=m) for m in (
        "hit_rate", "base_rate_spy_up", "hit_minus_base", "shuffled_calls_percentile", "shuffled_calls_p_worse",
        "coin_flip_p_worse"))
    verdict = ("The interval is entirely above zero: the calls beat always-bullish." if d["ci_low"] > 0 else
               "The interval is entirely below zero: the calls did worse than always-bullish." if d["ci_high"] < 0
               else "The interval includes zero: the calls are not distinguishable from always-bullish.")
    better = r["value"] >= 50
    p = sh["p_value"] if better else shw["p_value"]
    strength = ("That clears even the stricter 0.025 bar for two primary tests." if p < 0.025 else
                "That is borderline: it would not clear the stricter 0.025 bar for two primary tests." if p < 0.05
                else "That is well within what luck produces.")
    if better:
        luck = (f"Handing the same calls to random dates did at least as well in {p * 100:.1f}% of shuffles "
                f"(p = {p_text(p)}), and a coin flip would get {int(r['count'])}/{int(r['n'])} or better with "
                f"probability {p_text(r['p_value'])}. {strength}")
    else:
        luck = (f"Handing the same calls to random dates did this badly or worse in only {p * 100:.1f}% of shuffles "
                f"(p = {p_text(p)}), so the timing looks worse than random. {strength} A coin flip would do this "
                f"badly with probability {p_text(cw['p_value'])}.")
    return (f"- **{strategy.capitalize()}** ({window_label(strategy, h)}): the alerts were right {int(r['count'])} "
            f"times out of {int(r['n'])} ({pct(r['value'])}). SPY rose in {int(b['count'])} of those same windows "
            f"({pct(b['value'])}), so simply calling BULLISH every time would have scored {pct(b['value'])}. "
            f"The gap is {d['value']:+.1f} points (95% interval {d['ci_low']:+.1f} to {d['ci_high']:+.1f}). "
            f"{verdict} {luck}")


def neutral_sentence(s):
    out = []
    for st in STRATEGY_HORIZONS:
        g = pick(s, "neutral", st, PRIMARY[st], "as recorded", "median_gap_directional_minus_neutral")
        if g is None:
            continue
        two = pick(s, "neutral", st, PRIMARY[st], "as recorded", "median_gap_p_two_sided")
        smaller = g["value"] > 0
        if smaller and g["p_value"] < 0.05:
            verdict = "evidence that NEUTRAL flags quieter windows."
        elif not smaller and two["p_value"] < 0.05:
            verdict = ("the opposite of what NO PLAY implies, and a gap this large rarely comes from shuffling "
                       f"(two-sided p = {p_text(two['p_value'])}).")
        else:
            verdict = "no clear evidence either way."
        out.append(f"- {st.capitalize()} ({window_label(st, PRIMARY[st])}): NEUTRAL windows moved "
                   f"{'less' if smaller else 'more'} than directional ones, by {abs(g['value']):.3f} percentage "
                   f"points of median |move|: {verdict}")
    return "\n".join(out)


def momentum_sentence(s):
    out = []
    for st in STRATEGY_HORIZONS:
        for r in s[(s["section"] == "naive_rule") & (s["strategy"] == st) & (s["metric"] == "system_agrees")
                   ].itertuples():
            if r.subset.startswith("reverse"):
                continue
            out.append(f"The {st} calls match the '{r.subset}' rule {pct(r.value, 0)} of the time "
                       f"({int(r.count)}/{int(r.n)}).")
    return " ".join(out) + (" Agreement near 100% would mean the alert mostly restates the recent move; near "
                            "50% means it is unrelated to it; near 0% means it fades it.")


def bottom_line(hit, n_calls):
    parts = []
    for st in STRATEGY_HORIZONS:
        h = PRIMARY[st]
        r, d, sh, shw = (hit(st, h, metric=m) for m in ("hit_rate", "hit_minus_base", "shuffled_calls_percentile",
                                                         "shuffled_calls_p_worse"))
        shuffle = (f"Shuffled calls did at least as well {sh['p_value'] * 100:.0f}% of the time" if r["value"] >= 50
                   else f"Shuffled calls did this badly or worse only {shw['p_value'] * 100:.1f}% of the time")
        coin = "yes" if r["ci_low"] > 50 else "no; it did worse" if r["ci_high"] < 50 else "no evidence"
        long_ = "yes" if d["ci_low"] > 0 else "no; it did worse" if d["ci_high"] < 0 else "no evidence"
        parts.append(f"- **{st.capitalize()}** ({window_label(st, h)}): hit rate {pct(r['value'])} (95% interval "
                     f"{interval(r)}, {int(r['n'])} calls). Beats a coin flip: **{coin}** (the interval "
                     f"{'includes' if r['ci_low'] <= 50 <= r['ci_high'] else 'excludes'} 50%). Beats simply being "
                     f"long SPY: **{long_}** (hit rate minus base rate {d['value']:+.1f} points, interval "
                     f"{d['ci_low']:+.1f} to {d['ci_high']:+.1f}). {shuffle}.")
    any_edge = any(hit(st, PRIMARY[st], metric="hit_minus_base")["ci_low"] > 0 for st in STRATEGY_HORIZONS)
    any_coin = any(hit(st, PRIMARY[st])["ci_low"] > 50 for st in STRATEGY_HORIZONS)
    worse = [st for st in STRATEGY_HORIZONS if hit(st, PRIMARY[st], metric="hit_minus_base")["ci_high"] < 0]
    summary = ("**Answer:** " + (
        "at least one call type shows evidence of beating being long SPY over this period; see the caveats on "
        "sample size before relying on it." if any_edge else
        "on this sample there is no evidence that the calls beat simply being long SPY" +
        (", and no evidence they beat a coin flip either." if not any_coin else
         ". They beat a coin flip, but not the always-bullish rule on the same days.")) +
        (f" The {' and '.join(worse)} calls did worse than being long SPY over the same windows: following them "
         "was a handicap, not an edge" + (
             ", though the margin is borderline once two primary tests are allowed for (shuffle p above 0.025)."
             if any(hit(st, PRIMARY[st], metric="shuffled_calls_p_worse")["p_value"] >= 0.025 for st in worse)
             else ".") if worse else "") +
        f" With {n_calls['intraday']} intraday and {n_calls['overnight']} overnight calls, only a large edge "
        "(10 to 15 points or more) could have shown up; a small real edge cannot be ruled out, and only a longer "
        "log could find one.")
    return "\n".join(parts) + "\n\n" + summary


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Alert-log check against Massive SPY minutes "
                                            "(a signal study of the underlying, not option P&L).")
    p.add_argument("--csv", required=True, help="alert log (date, strategy, time_et, direction, ref_price, ...)")
    p.add_argument("--out", default="output/alerts", help="output folder (default output/alerts)")
    p.add_argument("--cache-dir", default="data/cache", help="minute-bar Parquet cache (default data/cache)")
    p.add_argument("--refresh", action="store_true", help="re-download minute bars even if cached")
    p.add_argument("--reps", type=int, default=10_000, help="shuffles and bootstrap resamples (default 10,000)")
    p.add_argument("--seed", type=int, default=0, help="random seed for shuffles and bootstraps (default 0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    alerts = load_alerts(args.csv)
    if (alerts["ticker"] != TICKER).any():
        raise SystemExit(f"Only {TICKER} alerts are supported.")
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    with timed(timings, "data"):
        # One session before the first alert (its prior close) through the session after the last one.
        sessions = features.trading_sessions(alerts["session"].min(), alerts["session"].max() + pd.Timedelta(days=10),
                                             warmup_sessions=1)
        done = sessions[sessions.index < today]  # never load today's session: its data may be incomplete
        first, last = done.index[0].date(), done.index[-1].date()
        minutes, source = download.load_minute_bars(TICKER, str(first), str(last), cache_dir=args.cache_dir,
                                                    refresh=args.refresh, final_close=done["close"].iloc[-1])
        try:
            dividends = fetch_ex_dividends(TICKER, first, sessions.index[-1].date())
        except Exception as e:  # reported as unavailable, never guessed
            print(f"warning: could not read Massive dividends ({type(e).__name__}); ex-dividend checks unavailable")
            dividends = None
    result = run_study(alerts, minutes, sessions, today, dividends, reps=args.reps, seed=args.seed, timings=timings)
    with timed(timings, "outputs"):
        out = write_outputs(args.out, result, source)

    s = result["summary"]
    print(f"Alert log: {len(alerts)} rows, {alerts['date'].min()} to {alerts['date'].max()} "
          f"(data: {source})")
    for st in STRATEGY_HORIZONS:
        h = PRIMARY[st]
        r, b = pick(s, "hit_rate", st, h, "as recorded", "hit_rate"), pick(s, "hit_rate", st, h, "as recorded",
                                                                           "base_rate_spy_up")
        print(f"  {st:9s} {window_label(st, h):24s} hit {int(r['count'])}/{int(r['n'])} = {r['value']:.1f}% "
              f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]  SPY up {b['value']:.1f}%")
    print("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
    print(f"wrote {out}/: alerts_with_outcomes.csv, summary.csv, report.md, hit_rates.png, overnight_score.png")


if __name__ == "__main__":
    main()
