"""Does SPY mean-revert over 1-5 days? Indicators, dip/rally events, classic rules and ML.

  uv run --env-file .env python meanrev.py --out output/meanrev                 # design period only
  uv run --env-file .env python meanrev.py --final-test --out output/meanrev    # also the holdout, once

A daily signal study on Massive minute bars (SPY plus context ETFs), built to choose what to try
with credit spreads next. It is a signal study of the underlying, not option P&L.

Time and data conventions:
- The decision time is 10 minutes before each session's close (15:50 ET on a full day). Every
  feature uses today's prices up to then (the "snapshot": price known at 15:50, high/low/volume
  so far) and earlier sessions' full regular-hours bars, so a signal can be acted on the same day.
- Outcomes run from the 15:50 price to the 15:50 price h sessions later (h = 1, 2, 3, 5), plus
  the lowest low / highest high in between for credit-spread odds.
- Prices are split-adjusted by Massive and back-adjusted here for cash dividends, so returns are
  total returns and an ex-dividend drop is not a fake dip.
- Design period: sessions through 2024-12-31; its outcomes must also end by then. Sessions from
  2025-01-01 are the holdout (the repo's convention for SPY studies): exploration runs never
  read their outcomes, and --final-test evaluates the pre-registered candidates on them once.

Everything below that is a choice (features, events, rules, models, thresholds) was fixed before
looking at any result. Primary hypotheses for the holdout are picked from design results by
the rule in select_primary(); every other number is exploratory.
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402
from massive import RESTClient  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import alerts  # noqa: E402  (Wilson intervals, markdown tables, p-value text)
import download  # noqa: E402
import features  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, TARGET  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKER = "SPY"
CONTEXT = ("QQQ", "IWM", "TLT", "HYG", "GLD", "UUP", "VIXY")
START, END = "2021-10-04", "2026-09-30"   # the data plan's five years when this was written
DESIGN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
DECISION_MINUTES = 10                      # decide 10 minutes before the close
HORIZONS = (1, 2, 3, 5)                    # sessions
PRIMARY_HORIZON = 5
CREDIT_K = (0.5, 1.0, 2.0)                 # % moves for credit-spread odds
BLOCK = 10                                 # sessions per bootstrap block (windows overlap, signals cluster)
FDR = 0.10                                 # Benjamini-Hochberg false discovery rate for the indicator table
MIN_EVENTS = 30                            # a primary candidate needs at least this many design events

# ---------------------------------------------------------------- pre-registered tests

# Dips (bullish mean reversion: long, or sell put spreads) and their mirrors.
DIPS = {
    "1-day drop ≥ 1.5%": lambda f: f["ret_1d"] <= -1.5,
    "1-day drop ≥ 2σ": lambda f: f["retz_1d"] <= -2,
    "3-day drop ≥ 3%": lambda f: f["ret_3d"] <= -3,
    "5-day drop ≥ 4%": lambda f: f["ret_5d"] <= -4,
    "RSI(2) < 10": lambda f: f["rsi_2"] < 10,
    "RSI(2) < 10, above 200-day": lambda f: (f["rsi_2"] < 10) & (f["above_sma200"] == 1),
    "Below lower Bollinger band": lambda f: f["bb_z20"] < -2,
    "3+ down days in a row": lambda f: f["streak"] <= -3,
    "IBS < 0.2 (near the day's low)": lambda f: f["ibs"] < 0.2,
    "New 10-day low": lambda f: f["low_10"] == 1,
    "5%+ below 20-day high": lambda f: f["drawdown_20"] <= -5,
    "VIXY up ≥ 10% today": lambda f: f["vixy_ret_1d"] >= 10,
    # Added after the design run, before the holdout was opened (exploratory):
    "RSI(2) < 10 and IBS < 0.2": lambda f: (f["rsi_2"] < 10) & (f["ibs"] < 0.2),
    "3+ down days and RSI(2) < 10": lambda f: (f["streak"] <= -3) & (f["rsi_2"] < 10),
    "Oversold score in its top 10%": lambda f: f["oversold_top10"] == 1,
}
RALLIES = {
    "1-day rise ≥ 1.5%": lambda f: f["ret_1d"] >= 1.5,
    "1-day rise ≥ 2σ": lambda f: f["retz_1d"] >= 2,
    "3-day rise ≥ 3%": lambda f: f["ret_3d"] >= 3,
    "5-day rise ≥ 4%": lambda f: f["ret_5d"] >= 4,
    "RSI(2) > 90": lambda f: f["rsi_2"] > 90,
    "RSI(2) > 90, below 200-day": lambda f: (f["rsi_2"] > 90) & (f["above_sma200"] == 0),
    "Above upper Bollinger band": lambda f: f["bb_z20"] > 2,
    "3+ up days in a row": lambda f: f["streak"] >= 3,
    "IBS > 0.8 (near the day's high)": lambda f: f["ibs"] > 0.8,
    "New 10-day high": lambda f: f["high_10"] == 1,
    "5%+ above 20-day low": lambda f: f["runup_20"] >= 5,
    "VIXY down ≥ 5% today": lambda f: f["vixy_ret_1d"] <= -5,
    # Added after the design run, before the holdout was opened (exploratory):
    "RSI(2) > 90 and IBS > 0.8": lambda f: (f["rsi_2"] > 90) & (f["ibs"] > 0.8),
    "3+ up days and RSI(2) > 90": lambda f: (f["streak"] >= 3) & (f["rsi_2"] > 90),
    "Oversold score in its bottom 10%": lambda f: f["oversold_bottom10"] == 1,
}
NEVER = lambda f: pd.Series(False, index=f.index)  # noqa: E731
# Classic rules, unchanged from their usual published form: (name, side, entry, exit, max sessions held).
RULES = [
    ("Connors: RSI(2) < 10 above 200-day; exit above 5-day avg", 1,
     lambda f: (f["rsi_2"] < 10) & (f["above_sma200"] == 1), lambda f: f["dist_sma5"] > 0, 10),
    ("IBS < 0.2; hold 1 session", 1, lambda f: f["ibs"] < 0.2, NEVER, 1),
    ("3 down days; exit on first up day", 1, lambda f: f["streak"] <= -3, lambda f: f["ret_1d"] > 0, 5),
    ("Below lower Bollinger; exit at 20-day avg", 1, lambda f: f["bb_z20"] < -2, lambda f: f["dist_sma20"] > 0, 10),
    ("1-day drop ≥ 1.5%; hold 3 sessions", 1, lambda f: f["ret_1d"] <= -1.5, NEVER, 3),
    ("Short: RSI(2) > 90 below 200-day; exit below 5-day avg", -1,
     lambda f: (f["rsi_2"] > 90) & (f["above_sma200"] == 0), lambda f: f["dist_sma5"] < 0, 10),
    ("Short: IBS > 0.8; hold 1 session", -1, lambda f: f["ibs"] > 0.8, NEVER, 1),
    ("Short: 3 up days; exit on first down day", -1, lambda f: f["streak"] >= 3, lambda f: f["ret_1d"] < 0, 5),
    ("Short: above upper Bollinger; exit at 20-day avg", -1, lambda f: f["bb_z20"] > 2,
     lambda f: f["dist_sma20"] < 0, 10),
    ("Short: 1-day rise ≥ 1.5%; hold 3 sessions", -1, lambda f: f["ret_1d"] >= 1.5, NEVER, 3),
]
# Composite oversold score: the average of design-standardized, sign-flipped stretch measures.
SCORE_PARTS = ("ret_3d", "rsi_2", "bb_z20", "ibs", "dist_sma5")

FEATURE_GROUPS = {
    "Recent return": ["ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "retz_1d", "retz_3d", "retz_5d"],
    "Oscillator": ["rsi_2", "rsi_3", "rsi_5", "rsi_14", "stoch_10"],
    "Trend distance": ["dist_sma5", "dist_sma10", "dist_sma20", "dist_sma50", "dist_sma200", "bb_z20"],
    "Today's bar": ["ibs", "ret_intraday", "gap", "range_atr"],
    "Streaks and extremes": ["streak", "low_10", "high_10", "drawdown_20", "drawdown_60", "runup_20"],
    "Volatility and volume": ["atr_pct", "rv5", "rv20", "rv_ratio", "vol_ratio"],
    "Regime": ["above_sma200", "sma50_above_200"],
    "Cross-asset": ["qqq_ret_1d", "iwm_rel_1d", "iwm_rel_5d", "tlt_ret_1d", "tlt_ret_5d", "hyg_ret_1d", "hyg_ret_5d",
                    "gld_ret_5d", "uup_ret_5d", "vixy_ret_1d", "vixy_ret_5d", "vixy_dist_sma10"],
    "Calendar": ["dow", "turn_of_month", "sessions_to_month_end"],
}
ML_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
LABELS = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}

# Fixed in advance, as in the ML swing study: never tuned on results.
MIN_TRAIN_SESSIONS, TEST_SESSIONS, THREADS = 250, 63, 4
MODELS = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "boosted_trees": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=8,
                                                             min_samples_leaf=50, l2_regularization=1.0,
                                                             early_stopping=False, random_state=0),
}
ML_HORIZONS = (1, 5)
# Exploratory extra model (added before the holdout was opened): logistic on the stretch measures only.
MR_FEATURES = ["ret_1d", "ret_3d", "ret_5d", "rsi_2", "rsi_3", "rsi_5", "bb_z20", "ibs", "dist_sma5", "dist_sma20",
               "dist_sma50", "dist_sma200", "streak", "rv20", "above_sma200"]
MR_MODEL = "logistic, mean-reversion indicators only"
# Pre-registered follow-up (written before the holdout was opened): whatever the holdout shows, the options
# step prices these with real option prices on 2024-10 onward, so the choice is not made on holdout results.
OPTIONS_FOLLOW_UP = ("the primary dip signal as a put credit spread", "the Connors RSI(2) rule as a put credit spread",
                     "the primary rally signal as a call credit spread")

# ---------------------------------------------------------------- data

def load_dividends(ticker, first, last, cache_dir="data/cache", refresh=False):
    """Cash dividends with an ex-date in (first, last], summed per ex-date, cached as Parquet."""
    path = Path(cache_dir) / f"{ticker}_cash_dividends_{first}_{last}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set. Run with: uv run --env-file .env python meanrev.py ...")
    client = RESTClient(api_key=key, retries=10)
    rows = [(d.ex_dividend_date, d.cash_amount, d.currency) for d in client.list_dividends(
        ticker=ticker, ex_dividend_date_gt=str(first), ex_dividend_date_lte=str(last), limit=1000)]
    df = pd.DataFrame(rows, columns=["ex_date", "cash_amount", "currency"])
    if not df.empty and set(df["currency"].dropna()) - {"USD"}:
        raise RuntimeError(f"{ticker} dividends are not all in USD.")
    df = (df.assign(ex_date=pd.to_datetime(df["ex_date"]).astype("datetime64[ns]"))
          .groupby("ex_date", as_index=False)["cash_amount"].sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


PRICES = ["open", "high", "low", "close", "snap", "snap_high", "snap_low", "post_high", "post_low"]


def daily_table(minutes, sessions, decision_minutes=DECISION_MINUTES):
    """One row per session from regular-hours minutes: the full bar, the snapshot at the decision
    time (price known then = close of the last bar ending by then; high/low/volume so far) and the
    high/low after it. Sessions without minutes are NaN."""
    rth, counts = features.regular_session_minutes(minutes, sessions)
    decision = rth["session_close"] - pd.Timedelta(minutes=decision_minutes)
    before = rth["ts"] < decision  # a bar that starts before the decision time has ended by then
    g, pre, post = rth.groupby("session"), rth[before].groupby("session"), rth[~before].groupby("session")
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "minutes": g.size(),
        "snap": pre["close"].last(), "snap_high": pre["high"].max(), "snap_low": pre["low"].min(),
        "snap_volume": pre["volume"].sum(), "snap_ts": pre["ts"].last(),
        "post_high": post["high"].max(), "post_low": post["low"].min(),
    }).reindex(sessions.index)
    d["decision"] = sessions["close"] - pd.Timedelta(minutes=decision_minutes)
    d["snap_age_min"] = (d["decision"] - d["snap_ts"]) / pd.Timedelta(minutes=1) - 1  # 0 = the bar just ended
    return d, counts


def adjust_dividends(daily, dividends):
    """Back-adjust prices for cash dividends: every price before an ex-date session E is multiplied by
    1 - cash / (close of the session before E). Returns (adjusted, applied ex-dates)."""
    d = daily.copy()
    days = d.index
    factors = []
    for ex, cash in zip(pd.to_datetime(dividends["ex_date"]), dividends["cash_amount"]):
        i = days.searchsorted(ex)
        if 0 < i < len(days) and days[i] == ex and not np.isnan(d["close"].iloc[i - 1]):
            factors.append((ex, 1 - cash / d["close"].iloc[i - 1]))
    for ex, f in factors:
        d.loc[d.index < ex, PRICES] *= f
    return d, pd.DataFrame(factors, columns=["ex_date", "factor"])


# ---------------------------------------------------------------- features

def ticker_features(d, full=True):
    """Features at each session's decision time, from that session's snapshot and earlier sessions'
    full bars (rows without a snapshot are skipped). full=False computes only what context tickers need."""
    d = d.dropna(subset=["snap", "close"])
    C, H, L = (d[k].to_numpy(float) for k in ("close", "high", "low"))
    S, SH, SL, O = (d[k].to_numpy(float) for k in ("snap", "snap_high", "snap_low", "open"))
    V = d["snap_volume"].to_numpy(float)
    nan = np.nan
    rows = []
    for t in range(len(d)):
        c, h, lo = np.r_[C[:t], S[t]], np.r_[H[:t], SH[t]], np.r_[L[:t], SL[t]]
        f = {f"ret_{n}d": (c[-1] / c[-1 - n] - 1) * 100 if t >= n else nan for n in (1, 2, 3, 5, 10, 20)}
        f["dist_sma10"] = (c[-1] / c[-10:].mean() - 1) * 100 if t >= 9 else nan
        if not full:
            rows.append(f)
            continue
        lr = np.diff(np.log(c))  # lr[-1] is today's move so far; earlier ones are full sessions
        rv20 = lr[-21:-1].std(ddof=1) * 100 if t >= 21 else nan  # volatility coming into today
        f.update({
            "retz_1d": f["ret_1d"] / rv20, "retz_3d": f["ret_3d"] / (rv20 * math.sqrt(3)),
            "retz_5d": f["ret_5d"] / (rv20 * math.sqrt(5)),
            "rv5": lr[-6:-1].std(ddof=1) * 100 if t >= 6 else nan, "rv20": rv20,
        })
        f["rv_ratio"] = f["rv5"] / rv20
        for n in (2, 3, 5, 14):
            f[f"rsi_{n}"] = talib.RSI(c, n)[-1] if t > n else nan
        for n in (5, 20, 50, 200):
            f[f"dist_sma{n}"] = (c[-1] / c[-n:].mean() - 1) * 100 if t >= n - 1 else nan
        f["bb_z20"] = (c[-1] - c[-20:].mean()) / c[-20:].std() if t >= 19 else nan
        atr = talib.ATR(h, lo, c, 14)[-1] if t > 14 else nan
        f["atr_pct"] = atr / c[-1] * 100
        f["range_atr"] = (SH[t] - SL[t]) / atr
        f["stoch_10"] = (c[-1] - lo[-10:].min()) / (h[-10:].max() - lo[-10:].min()) if t >= 9 else nan
        f["drawdown_20"] = (c[-1] / h[-20:].max() - 1) * 100 if t >= 19 else nan
        f["drawdown_60"] = (c[-1] / h[-60:].max() - 1) * 100 if t >= 59 else nan
        f["runup_20"] = (c[-1] / lo[-20:].min() - 1) * 100 if t >= 19 else nan
        f["low_10"] = float(c[-1] < C[t - 9:t].min()) if t >= 9 else nan
        f["high_10"] = float(c[-1] > C[t - 9:t].max()) if t >= 9 else nan
        moves = np.sign(np.diff(c[-21:])) if t >= 1 else np.array([])
        run = 0
        for m in moves[::-1]:
            if m == 0 or (run and np.sign(run) != m):
                break
            run += int(m)
        f["streak"] = float(run)
        f["ibs"] = (S[t] - SL[t]) / (SH[t] - SL[t]) if SH[t] > SL[t] else 0.5
        f["ret_intraday"] = (S[t] / O[t] - 1) * 100
        f["gap"] = (O[t] / C[t - 1] - 1) * 100 if t >= 1 else nan
        f["vol_ratio"] = V[t] / V[t - 20:t].mean() if t >= 20 else nan
        f["above_sma200"] = float(f["dist_sma200"] > 0) if t >= 199 else nan
        f["sma50_above_200"] = float(c[-50:].mean() > c[-200:].mean()) if t >= 199 else nan
        rows.append(f)
    return pd.DataFrame(rows, index=d.index)


def calendar_features(sessions):
    days = sessions.index
    month = days.to_period("M")
    pos = pd.Series(range(len(days)), index=days).groupby(month).rank(method="first").to_numpy() - 1
    size = pd.Series(1, index=days).groupby(month).transform("size").to_numpy()
    left = size - 1 - pos
    return pd.DataFrame({"dow": days.dayofweek.astype(float), "sessions_to_month_end": left.astype(float),
                         "turn_of_month": ((left == 0) | (pos <= 2)).astype(float)}, index=days)


def build_features(daily, sessions):
    """SPY features, context-ticker features and calendar features, one row per session."""
    spy = ticker_features(daily[TICKER])
    out = spy.copy()
    ctx = {t: ticker_features(daily[t], full=False).reindex(sessions.index) for t in CONTEXT if t in daily}
    out = out.reindex(sessions.index)
    get = lambda t, c: ctx[t][c] if t in ctx else np.nan  # noqa: E731
    out["qqq_ret_1d"] = get("QQQ", "ret_1d")
    out["iwm_rel_1d"] = get("IWM", "ret_1d") - out["ret_1d"]
    out["iwm_rel_5d"] = get("IWM", "ret_5d") - out["ret_5d"]
    for t in ("TLT", "HYG"):
        out[f"{t.lower()}_ret_1d"], out[f"{t.lower()}_ret_5d"] = get(t, "ret_1d"), get(t, "ret_5d")
    out["gld_ret_5d"], out["uup_ret_5d"] = get("GLD", "ret_5d"), get("UUP", "ret_5d")
    out["vixy_ret_1d"], out["vixy_ret_5d"] = get("VIXY", "ret_1d"), get("VIXY", "ret_5d")
    out["vixy_dist_sma10"] = get("VIXY", "dist_sma10")
    out = out.join(calendar_features(sessions))
    return out[ML_FEATURES]


# ---------------------------------------------------------------- outcomes

def forward_outcomes(d, last_day, horizons=HORIZONS):
    """Per session t: total return (%) from t's snapshot to the snapshot h sessions later, and the
    lowest low / highest high in between (after t's decision through t+h's decision), relative to
    t's snapshot. NaN unless every session in the window has data and t+h is on or before last_day."""
    S, SL, SH = (d[k].to_numpy(float) for k in ("snap", "snap_low", "snap_high"))
    L, H = d["low"].to_numpy(float), d["high"].to_numpy(float)
    PL, PH = d["post_low"].to_numpy(float), d["post_high"].to_numpy(float)
    days = d.index
    n = len(d)
    out = {}
    for h in horizons:
        fwd, lo, hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for t in range(n - h):
            if days[t + h] > last_day:
                break
            path_low = np.r_[PL[t], L[t + 1:t + h], SL[t + h]]
            path_high = np.r_[PH[t], H[t + 1:t + h], SH[t + h]]
            if np.isnan(S[t]) or np.isnan(S[t + h]) or np.isnan(path_low).any():
                continue
            fwd[t] = (S[t + h] / S[t] - 1) * 100
            lo[t], hi[t] = (path_low.min() / S[t] - 1) * 100, (path_high.max() / S[t] - 1) * 100
        out[f"fwd_{h}"], out[f"low_{h}"], out[f"high_{h}"] = fwd, lo, hi
    return pd.DataFrame(out, index=days)


# ---------------------------------------------------------------- statistics

def block_indices(n, reps, seed, block=BLOCK):
    """(reps, n) row indices from a circular moving-block bootstrap over consecutive sessions."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (reps, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def bh_qvalues(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def ic_table(feats, outs, rows, reps, seed):
    """Rank correlation of each feature with each forward return over `rows` (complete features and
    outcomes), with block-bootstrap intervals, two-sided p-values and BH q-values; plus the top-minus-
    bottom quintile difference in mean forward return."""
    records = []
    idx = block_indices(int(rows.sum()), reps, seed)
    for h in HORIZONS:
        y = outs.loc[rows, f"fwd_{h}"].to_numpy(float)
        ry = pd.Series(y).rank().to_numpy()
        for c in ML_FEATURES:
            x = feats.loc[rows, c].to_numpy(float)
            if np.nanstd(x) == 0:
                continue
            rx = pd.Series(x).rank().to_numpy()
            ic = np.corrcoef(rx, ry)[0, 1]
            a, b = rx[idx], ry[idx]
            a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):  # a rare yes/no indicator can be constant
                sims = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
            sims = sims[~np.isnan(sims)]
            p = 2 * min((sims <= 0).mean(), (sims >= 0).mean())
            if len(np.unique(x)) <= 5:  # yes/no and small-integer indicators: highest value vs lowest
                top, bottom = x == x.max(), x == x.min()
            else:
                q = pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False).to_numpy()
                top, bottom = q == 4, q == 0
            records.append({"feature": c, "group": LABELS[c], "horizon": h, "n": len(y), "ic": ic,
                            "ci_low": np.percentile(sims, 2.5), "ci_high": np.percentile(sims, 97.5),
                            "p": max(p, 1 / reps), "q5_minus_q1": y[top].mean() - y[bottom].mean()})
    t = pd.DataFrame(records)
    t["q"] = bh_qvalues(t["p"])
    return t


def event_stats(mask, fwd, low, high, side, idx):
    """One event over one period's rows (fwd valid): counts, mean/median/hit rate against the period's
    baseline, the excess with a block-bootstrap interval, and credit-spread odds (side +1: the put
    side, -1: the call side)."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    pos = np.flatnonzero(mask)
    clusters = int(1 + (np.diff(pos) > PRIMARY_HORIZON).sum()) if n else 0
    out = {"n": n, "clusters": clusters, "base_mean": fwd.mean(), "base_hit": (fwd > 0).mean() * 100}
    if n == 0:
        return out
    e = fwd[mask]
    sm, sf = mask[idx], fwd[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = (sf * sm).sum(1) / sm.sum(1) - sf.mean(1)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = 2 * min(np.nanmean(diffs <= 0), np.nanmean(diffs >= 0))
    wl, wh = alerts.wilson(int((e > 0).sum()), n)
    out.update(mean=e.mean(), median=np.median(e), hit=(e > 0).mean() * 100, hit_lo=wl, hit_hi=wh,
               excess=e.mean() - fwd.mean(), excess_lo=lo, excess_hi=hi, p=max(p, 1 / len(idx)))
    for k in CREDIT_K:
        if side > 0:
            out[f"close_ok_{k:g}"] = (e >= -k).mean() * 100
            out[f"close_ok_{k:g}_base"] = (fwd >= -k).mean() * 100
            out[f"path_ok_{k:g}"] = (low[mask] > -k).mean() * 100
            out[f"path_ok_{k:g}_base"] = (low > -k).mean() * 100
        else:
            out[f"close_ok_{k:g}"] = (e <= k).mean() * 100
            out[f"close_ok_{k:g}_base"] = (fwd <= k).mean() * 100
            out[f"path_ok_{k:g}"] = (high[mask] < k).mean() * 100
            out[f"path_ok_{k:g}_base"] = (high < k).mean() * 100
    return out


def event_table(feats, outs, period_rows, reps, seed, regimes=None):
    """Every pre-registered dip and rally over one period, per horizon (and per regime, if given)."""
    records = []
    for h in HORIZONS:
        valid = period_rows & outs[f"fwd_{h}"].notna()
        f, o = feats[valid], outs[valid]
        fwd, low, high = (o[f"{k}_{h}"].to_numpy(float) for k in ("fwd", "low", "high"))
        idx = block_indices(len(fwd), reps, seed)
        for side, events in ((1, DIPS), (-1, RALLIES)):
            for name, rule in events.items():
                mask = rule(f).fillna(False).to_numpy(bool)
                for regime, keep in (regimes(f) if regimes else {"all": np.ones(len(f), bool)}).items():
                    m = mask & keep
                    k_idx = idx if regime == "all" else block_indices(int(keep.sum()), reps, seed)
                    stats = (event_stats(m, fwd, low, high, side, idx) if regime == "all" else
                             event_stats(m[keep], fwd[keep], low[keep], high[keep], side, k_idx))
                    records.append({"event": name, "side": "dip" if side > 0 else "rally", "horizon": h,
                                    "regime": regime, **stats})
    return pd.DataFrame(records)


def run_rule(f, snap, entry, exit_when, max_hold, side, last_day):
    """One position at a time: enter at a signal day's snapshot, exit at the first later snapshot where
    the exit condition holds or after max_hold sessions. Trades that would end after last_day are dropped."""
    days = f.index
    entry = entry(f).fillna(False).to_numpy(bool)
    exit_now = exit_when(f).fillna(False).to_numpy(bool)
    S = snap.reindex(days).to_numpy(float)
    trades, i, n = [], 0, len(days)
    while i < n:
        if not entry[i] or np.isnan(S[i]):
            i += 1
            continue
        j = i + 1
        while j < n and j - i < max_hold and not exit_now[j]:
            j += 1
        if j >= n or days[j] > last_day or np.isnan(S[j]):
            break
        trades.append({"entry": days[i], "exit": days[j], "sessions": j - i, "ret": side * (S[j] / S[i] - 1) * 100})
        i = j
    return pd.DataFrame(trades, columns=["entry", "exit", "sessions", "ret"])


def rule_table(feats, snap, first_day, last_day):
    """Each classic rule over one period: trade statistics against holding SPY over the same period."""
    f = feats[(feats.index >= first_day) & (feats.index <= last_day)]
    s = snap.reindex(f.index)
    daily = (s / s.shift(1) - 1).dropna() * 100
    records = []
    for name, side, entry, exit_when, max_hold in RULES:
        t = run_rule(f, snap, entry, exit_when, max_hold, side, last_day)
        r = t["ret"].to_numpy(float)
        n = len(r)
        held = t["sessions"].sum()
        wl, wh = alerts.wilson(int((r > 0).sum()), n)
        records.append({"rule": name, "side": "long" if side > 0 else "short", "trades": n,
                        "win_rate": (r > 0).mean() * 100 if n else np.nan, "win_lo": wl, "win_hi": wh,
                        "mean": r.mean() if n else np.nan, "total": r.sum(),
                        "avg_sessions": held / n if n else np.nan, "exposure": held / len(daily) * 100,
                        "per_session": r.sum() / held if held else np.nan,
                        "spy_per_session": side * daily.mean(),
                        "consistency": r.mean() / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std() else np.nan,
                        "max_drawdown": float((np.maximum.accumulate(np.r_[0, np.cumsum(r)])[1:] - np.cumsum(r)).max())
                        if n else np.nan})
    return pd.DataFrame(records)


def composite_score(feats, design_rows):
    """The pre-registered oversold score: mean of design-standardized, sign-flipped stretch measures."""
    parts = []
    for c in SCORE_PARTS:
        x = feats[c]
        mu, sd = x[design_rows].mean(), x[design_rows].std()
        parts.append(-(x - mu) / sd)
    return pd.concat(parts, axis=1).mean(axis=1)


# ---------------------------------------------------------------- machine learning

def walk_forward(X, y, fwd_end, sessions_idx, start_pos, stop_pos, model_names, horizon):
    """Expanding walk-forward over session positions [start_pos, stop_pos): each block of TEST_SESSIONS
    is predicted by models fit only on rows whose label window (t + horizon) ends before the block's
    first session. Returns out-of-sample probabilities per model and the block of each row."""
    proba = {m: np.full(len(y), np.nan) for m in model_names}
    block = np.full(len(y), -1)
    fitted, records = [], []
    for k, first in enumerate(range(start_pos, stop_pos, TEST_SESSIONS)):
        last = min(first + TEST_SESSIONS, stop_pos)
        test = (sessions_idx >= first) & (sessions_idx < last) & ~np.isnan(y)
        train = (fwd_end < first) & ~np.isnan(y)
        if train.sum() < 200 or test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        block[test] = k
        rec = {"block": k, "n_train": int(train.sum()), "n_test": int(test.sum())}
        for m in model_names:
            with threadpool_limits(THREADS):
                model = MODELS[m]().fit(X[train], y[train].astype(int))
                p = model.predict_proba(X[test])[:, 1]
            proba[m][test] = p
            rec[f"auc_{m}"] = roc_auc_score(y[test], p) if len(np.unique(y[test])) == 2 else np.nan
            fitted.append({"model": m, "estimator": model, "test": test})
        records.append(rec)
    return proba, block, pd.DataFrame(records), fitted


def ml_study(feats, outs_design, outs_all, design_rows, holdout_rows, final_test, reps, seed):
    """Walk-forward logistic regression and boosted trees on every feature, for each ML horizon:
    out-of-sample AUC, rank IC, quintile returns and permutation importance (design blocks).

    Positions are session positions, so a label ending h sessions later is purged correctly even
    if some session lacks a complete row. Exploration runs train and score on design outcomes only;
    the final test also predicts the holdout, training each block only on earlier closed windows."""
    complete = feats[ML_FEATURES].notna().all(axis=1)
    rows = complete & (design_rows | holdout_rows) if final_test else complete & design_rows
    f = feats[rows]
    X = f[ML_FEATURES].to_numpy(float)
    spos = np.searchsorted(feats.index, f.index)
    in_design = np.asarray(f.index <= DESIGN_END)
    start = spos[MIN_TRAIN_SESSIONS]
    stop = len(feats) if final_test else int(np.searchsorted(feats.index, DESIGN_END, side="right"))
    score_all = composite_score(feats, design_rows).loc[f.index].to_numpy(float)
    results, importances, blocks = [], [], []
    for h in ML_HORIZONS:
        fwd_design = outs_design.loc[f.index, f"fwd_{h}"].to_numpy(float)
        fwd_train = outs_all.loc[f.index, f"fwd_{h}"].to_numpy(float) if final_test else fwd_design
        y = np.where(np.isnan(fwd_train), np.nan, (fwd_train > 0).astype(float))
        proba, block, rec, fitted = walk_forward(X, y, spos + h, spos, start, stop, list(MODELS), h)
        rec["horizon"] = h
        blocks.append(rec)
        mr_cols = [ML_FEATURES.index(c) for c in MR_FEATURES]
        proba[MR_MODEL] = walk_forward(X[:, mr_cols], y, spos + h, spos, start, stop, ["logistic"], h)[0]["logistic"]
        for period, keep, fwd in (("design", in_design, fwd_design), ("holdout", ~in_design, fwd_train)):
            yy = (fwd > 0).astype(float)
            for m in [*MODELS, MR_MODEL]:
                ok = keep & ~np.isnan(proba[m]) & ~np.isnan(fwd)
                if ok.sum() < 20:
                    continue
                results.append({"horizon": h, "model": m, "period": period,
                                **score_predictions(proba[m][ok], fwd[ok], yy[ok], reps, seed)})
            ok = keep & (block >= 0) & ~np.isnan(fwd)
            if ok.sum() >= 20:
                results.append({"horizon": h, "model": "oversold score (no ML)", "period": period,
                                **score_predictions(score_all[ok], fwd[ok], yy[ok], reps, seed)})
        if h == PRIMARY_HORIZON:
            for m in MODELS:
                drops = []
                for fit in fitted:
                    test = fit["test"] & in_design
                    if fit["model"] == m and test.sum() > 20 and len(np.unique(y[test])) == 2:
                        with threadpool_limits(THREADS):
                            r = permutation_importance(fit["estimator"], X[test], y[test].astype(int),
                                                       scoring="roc_auc", n_repeats=5, random_state=0)
                        drops.append(r.importances_mean)
                if drops:
                    oos = in_design & ~np.isnan(proba[m])
                    for i, c in enumerate(ML_FEATURES):
                        x = pd.Series(X[oos, i])
                        direction = (x.rank().corr(pd.Series(proba[m][oos]).rank()) if x.nunique() > 1
                                     else np.nan)  # constant over these blocks (e.g. always above the 200-day)
                        importances.append({"model": m, "feature": c, "auc_drop": np.mean([d[i] for d in drops]),
                                            "blocks_with_drop": int(sum(d[i] > 0 for d in drops)),
                                            "blocks": len(drops), "direction": direction})
    return pd.DataFrame(results), pd.DataFrame(importances), pd.concat(blocks, ignore_index=True)


def score_predictions(p, fwd, y, reps, seed):
    """AUC, rank IC with a block-bootstrap interval, and mean forward return by prediction quintile."""
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    rp, rf = pd.Series(p).rank().to_numpy(), pd.Series(fwd).rank().to_numpy()
    ic = np.corrcoef(rp, rf)[0, 1]
    idx = block_indices(len(p), reps, seed)
    a, b = rp[idx] - rp[idx].mean(1, keepdims=True), rf[idx] - rf[idx].mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    sims = sims[~np.isnan(sims)]
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False).to_numpy()
    out = {"n": len(p), "auc": auc, "ic": ic, "ic_low": np.percentile(sims, 2.5), "ic_high": np.percentile(sims, 97.5),
           "base_hit": y.mean() * 100}
    for k in range(5):
        out[f"q{k + 1}_mean"] = fwd[q == k].mean()
        out[f"q{k + 1}_hit"] = (fwd[q == k] > 0).mean() * 100
    out["q5_minus_q1"] = out["q5_mean"] - out["q1_mean"]
    return out


# ---------------------------------------------------------------- the study

def select_primary(design_events, ml_results):
    """The pre-registered rule for the holdout's primary hypotheses: per side, the event with the
    highest lower bound of its design 5-session excess (in the event's own direction) among events
    with at least MIN_EVENTS; and the ML model with the higher design out-of-sample AUC at 5 sessions."""
    e = design_events[(design_events["horizon"] == PRIMARY_HORIZON) & (design_events["regime"] == "all")
                      & (design_events["n"] >= MIN_EVENTS)].copy()
    picks = {}
    for side, sign in (("dip", 1), ("rally", -1)):
        s = e[e["side"] == side].copy()
        s["bound"] = s["excess_lo"] if sign > 0 else -s["excess_hi"]
        if len(s):
            picks[side] = s.sort_values("bound", ascending=False).iloc[0]["event"]
    m = ml_results[(ml_results["horizon"] == PRIMARY_HORIZON) & (ml_results["period"] == "design")
                   & ml_results["model"].isin(list(MODELS))]
    if len(m):
        picks["ml"] = m.sort_values("auc", ascending=False).iloc[0]["model"]
    return picks


def run_study(daily, sessions, *, final_test=False, reps=2000, seed=0, timings=None):
    """Features, outcomes and every analysis. No file I/O. daily: per-ticker dividend-adjusted daily tables."""
    timings = {} if timings is None else timings
    with timed(timings, "features"):
        feats = build_features(daily, sessions)
        design_mask = feats.index <= DESIGN_END
        feats["oversold_score"] = composite_score(feats, pd.Series(design_mask, index=feats.index))
        q10, q90 = feats.loc[design_mask, "oversold_score"].quantile([0.1, 0.9])  # design-period cut-offs
        feats["oversold_top10"] = (feats["oversold_score"] >= q90).astype(float).where(feats["oversold_score"].notna())
        feats["oversold_bottom10"] = (feats["oversold_score"] <= q10).astype(float).where(
            feats["oversold_score"].notna())
    spy = daily[TICKER]
    last_day = sessions.index[-1]
    with timed(timings, "outcomes"):
        outs_design = forward_outcomes(spy, DESIGN_END)
        outs_all = forward_outcomes(spy, last_day)
    design_rows = pd.Series(feats.index <= DESIGN_END, index=feats.index)
    holdout_rows = pd.Series(feats.index >= HOLDOUT_START, index=feats.index)
    rv_median = feats.loc[design_rows, "rv20"].median()
    regimes = lambda f: {"above 200-day": (f["above_sma200"] == 1).to_numpy(),  # noqa: E731
                         "below 200-day": (f["above_sma200"] == 0).to_numpy(),
                         "calm (20-day vol below design median)": (f["rv20"] < rv_median).to_numpy(),
                         "volatile (20-day vol above design median)": (f["rv20"] >= rv_median).to_numpy()}
    complete = feats[ML_FEATURES].notna().all(axis=1)
    res = {"features": feats, "rv_median": rv_median, "final_test": final_test, "reps": reps}
    with timed(timings, "indicators"):
        res["ic_design"] = ic_table(feats, outs_design, design_rows & complete & outs_design["fwd_5"].notna()
                                    & outs_design["fwd_1"].notna(), reps, seed)
    with timed(timings, "events"):
        res["events_design"] = event_table(feats, outs_design, design_rows, reps, seed)
        res["regimes_design"] = event_table(feats, outs_design, design_rows, reps, seed, regimes)
    with timed(timings, "rules"):
        res["rules_design"] = rule_table(feats, spy["snap"], feats.index[0], DESIGN_END)
    with timed(timings, "ml"):
        res["ml"], res["importance"], res["ml_blocks"] = ml_study(feats, outs_design, outs_all, design_rows,
                                                                  holdout_rows, final_test, reps, seed)
    res["primary"] = select_primary(res["events_design"], res["ml"])
    res["paths_design"] = event_paths(feats, spy, design_rows, DESIGN_END, res["primary"])
    if final_test:
        with timed(timings, "holdout"):
            res["ic_holdout"] = ic_table(feats, outs_all, holdout_rows & complete & outs_all["fwd_5"].notna()
                                         & outs_all["fwd_1"].notna(), reps, seed)
            res["events_holdout"] = event_table(feats, outs_all, holdout_rows, reps, seed)
            res["regimes_holdout"] = event_table(feats, outs_all, holdout_rows, reps, seed, regimes)
            res["rules_holdout"] = rule_table(feats, spy["snap"], HOLDOUT_START, last_day)
            res["paths_holdout"] = event_paths(feats, spy, holdout_rows, last_day, res["primary"])
    res["periods"] = {"design": (feats.index[0], DESIGN_END, int(design_rows.sum())),
                      "holdout": (HOLDOUT_START, last_day, int(holdout_rows.sum()))}
    return res


def event_paths(feats, spy, rows, last_day, primary=None, days_after=10):
    """Average cumulative excess return over the 10 sessions after selected events (excess over the
    period's average path), for the event-study chart."""
    S = spy["snap"].reindex(feats.index).to_numpy(float)
    idx = np.flatnonzero(rows.to_numpy())
    ok = idx[idx + days_after < len(S)]
    ok = ok[feats.index[ok + days_after] <= last_day]
    paths = np.array([(S[i:i + days_after + 1] / S[i] - 1) * 100 for i in ok])
    base = np.nanmean(paths, axis=0)
    out = {}
    picks = [p for p in ((primary or {}).get("dip"), (primary or {}).get("rally")) if p]
    for name in dict.fromkeys([*picks, "RSI(2) < 10", "1-day drop ≥ 1.5%", "Below lower Bollinger band",
                               "RSI(2) > 90", "1-day rise ≥ 1.5%"]):
        if len(out) == 6:
            break
        rule = {**DIPS, **RALLIES}[name]
        m = rule(feats.iloc[ok]).fillna(False).to_numpy(bool)
        if m.sum():
            out[name] = (np.nanmean(paths[m], axis=0) - base, int(m.sum()))
    return out


# ---------------------------------------------------------------- report

def pct(v, digits=2, sign=True):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def md_table(header, rows):
    return alerts.md_table(header, rows)


def verdict(lo, hi):
    return "✓" if lo > 0 else "✗ reversed" if hi < 0 else "–"


def render_report(res):
    lines = []
    w = lines.append
    final = res["final_test"]
    d0, d1, dn = res["periods"]["design"]
    h0, h1, hn = res["periods"]["holdout"]
    w("# Does SPY mean-revert? A 1–5 day signal study\n")
    w(f"Design period **{d0.date()} to {d1.date()}** ({dn} sessions). Holdout **{h0.date()} to {h1.date()}** "
      f"({hn} sessions): " + ("**opened once for this report** (final test)." if final else
                              "**not opened** (exploration run)."))
    w(f"\nSignals use prices up to {DECISION_MINUTES} minutes before the close (15:50 ET) plus earlier full days, so "
      "they could be acted on the same afternoon. Returns run from that 15:50 price to the 15:50 price 1, 2, 3 or 5 "
      "sessions later and include dividends. This is a study of SPY itself, not of option P&L.\n")
    w("## How to read this\n")
    w("- **Excess return:** an event's average forward return minus the average over *all* days in the same "
      "period. SPY drifted up most of the time, so a positive raw return after a dip means little on its own.")
    w("- **95% interval:** computed by resampling whole 10-session blocks, because multi-day windows overlap and "
      "signals cluster (a selloff fires the same signal several days running). **Clusters** counts the separate "
      "episodes, which is closer to the true sample size than the raw count.")
    w("- **IC (rank correlation):** −1 to +1. Negative for a return-type indicator means mean reversion (down "
      "days are followed by up days); positive means momentum. Values of 0.05–0.10 are large for daily returns.")
    w(f"- **q-value:** a p-value corrected for testing {len(ML_FEATURES)} indicators × {len(HORIZONS)} horizons at "
      f"once; q < {FDR:.2f} means fewer than {FDR:.0%} of such findings are expected to be false.")
    w("- **✓ / ✗ / –:** the 95% interval is entirely above zero / entirely below zero / includes zero.\n")

    primary = res["primary"]
    if final:
        w("## Primary results (holdout)\n")
        w(primary_section(res))
        w(scorecard_section(res))
    w(odds_by_expiry_section(res))
    w("**Options follow-up, fixed before the holdout was opened:** the next step prices "
      + "; ".join(OPTIONS_FOLLOW_UP) + f" with real option prices from 2024-10 on (primary picks: dip = "
      f"*{primary.get('dip')}*, rally = *{primary.get('rally')}*), whatever the holdout shows, so that choice "
      "cannot be steered by holdout results.\n")

    w("## 1. Which indicators carry information? (design period)\n")
    w(ic_section(res))
    w("## 2. After sharp drops (dips)\n")
    w(events_section(res, "dip"))
    w("## 3. After sharp rises (rallies)\n")
    w(events_section(res, "rally"))
    w("## 4. Does the market regime matter?\n")
    w(regime_section(res))
    w("## 5. Classic mean-reversion rules\n")
    w(rules_section(res))
    w("## 6. Machine learning\n")
    w(ml_section(res))
    w("## 7. Caveats\n")
    w("- **Five years, one data source.** About 800 design and 440 holdout sessions; a few large selloffs drive "
      "most dip results. Effects of a few hundredths of a percent a day are hard to separate from noise.")
    w("- **Signal study, not trading P&L.** Entries and exits are at the 15:50 price without costs; SPY's own "
      "spread is about 0.01%, but options add their own costs and payoff shape.")
    w("- **Many comparisons.** Dozens of indicators, events and rules were tested. Only the primary hypotheses, "
      "chosen by a fixed rule from the design period, are confirmatory; treat the rest as leads.")
    w("- **Holdout spent.** " + ("The holdout has now been looked at; it is no longer clean for daily SPY "
                                 "mean-reversion ideas. New ideas need new data (e.g. alerts going forward)."
                                 if final else "Not yet opened."))
    return "\n".join(lines) + "\n"


def primary_section(res):
    ev = res["events_holdout"]
    ed = res["events_design"]
    primary = res["primary"]
    rows = []
    for side in ("dip", "rally"):
        name = primary.get(side)
        if not name:
            continue
        for label, t in (("design", ed), ("holdout", ev)):
            r = t[(t["event"] == name) & (t["horizon"] == PRIMARY_HORIZON) & (t["regime"] == "all")].iloc[0]
            sign = 1 if side == "dip" else -1
            lo, hi = (r["excess_lo"], r["excess_hi"]) if sign > 0 else (-r["excess_hi"], -r["excess_lo"])
            rows.append([f"{side}: {name}", label, f"{int(r['n'])} ({int(r['clusters'])})", pct(r["mean"]),
                         pct(r["base_mean"]), f"{pct(r['excess'])} ({pct(r['excess_lo'])} to {pct(r['excess_hi'])})",
                         verdict(lo, hi)])
    out = (f"Chosen by the pre-registered rule from the design period: the dip and the rally with the strongest "
           f"lower bound on their {PRIMARY_HORIZON}-session excess return (at least {MIN_EVENTS} events), and the ML "
           "model with the better design AUC. ✓ means the effect held in the predicted direction.\n\n")
    out += md_table(["Hypothesis", "Period", "Events (clusters)", f"Mean {PRIMARY_HORIZON}-session return",
                     "All days", "Excess (95% interval)", "Held?"], rows)
    m = res["ml"]
    model = primary.get("ml")
    if model:
        rows = []
        for period in ("design", "holdout"):
            r = m[(m["horizon"] == PRIMARY_HORIZON) & (m["model"] == model) & (m["period"] == period)]
            if len(r):
                r = r.iloc[0]
                rows.append([f"ML: {model}", period, int(r["n"]), f"{r['auc']:.3f}",
                             f"{r['ic']:+.3f} ({r['ic_low']:+.3f} to {r['ic_high']:+.3f})",
                             pct(r["q5_minus_q1"]), verdict(r["ic_low"], r["ic_high"])])
        out += "\n\n" + md_table(["Hypothesis", "Period", "Predictions", "AUC (0.5 = coin flip)",
                                  "Rank IC (95% interval)", "Top minus bottom fifth, 5-session return", "Held?"], rows)
    return out + "\n"


def scorecard_section(res):
    """Every dip and rally at the primary horizon: did the design-period direction hold in the holdout?"""
    ed, eh = (res[k] for k in ("events_design", "events_holdout"))
    rows, tally = [], {"held": 0, "same direction": 0, "reversed": 0, "too few": 0}
    for side, events in (("dip", DIPS), ("rally", RALLIES)):
        sign = 1 if side == "dip" else -1
        for name in events:
            d = ed[(ed["event"] == name) & (ed["horizon"] == PRIMARY_HORIZON) & (ed["regime"] == "all")].iloc[0]
            h = eh[(eh["event"] == name) & (eh["horizon"] == PRIMARY_HORIZON) & (eh["regime"] == "all")].iloc[0]
            if not h["n"] or h["n"] < 5 or not d["n"]:
                tally["too few"] += 1
                continue
            lo, hi = (h["excess_lo"], h["excess_hi"]) if sign > 0 else (-h["excess_hi"], -h["excess_lo"])
            key = "held" if lo > 0 else "reversed" if sign * h["excess"] < 0 else "same direction"
            tally[key] += 1
            rows.append([f"{side}: {name}", f"{pct(d['excess'])} (n={int(d['n'])})",
                         f"{pct(h['excess'])} (n={int(h['n'])})",
                         {"held": "✓ held", "same direction": "right way, not significant",
                          "reversed": "✗ wrong way"}[key]])
    out = (f"\n**Scorecard: every dip and rally, {PRIMARY_HORIZON}-session excess return.** "
           f"{tally['held']} held significantly, {tally['same direction']} pointed the predicted way without "
           f"significance, {tally['reversed']} went the wrong way, {tally['too few']} had fewer than 5 holdout "
           "events. With no real effect, about half would point each way by chance.\n\n")
    return out + md_table(["Signal", "Design excess", "Holdout excess", "Holdout"], rows) + "\n"


def odds_by_expiry_section(res):
    """Credit-spread odds by expiry (1-5 sessions) for the two primary signals."""
    out = ["**Credit-spread odds by expiry** for the primary signals: share of trades where a short strike 1% "
           "beyond the 15:50 entry would have expired worthless, signal vs all days.\n"]
    rows = []
    for side in ("dip", "rally"):
        name = res["primary"].get(side)
        if not name:
            continue
        for label in ("design", "holdout"):
            t = res.get(f"events_{label}")
            if t is None:
                continue
            cells = []
            for h in HORIZONS:
                r = t[(t["event"] == name) & (t["horizon"] == h) & (t["regime"] == "all")].iloc[0]
                cells.append(f"{r['close_ok_1']:.0f}% vs {r['close_ok_1_base']:.0f}%" if r["n"] else "n/a")
            rows.append([f"{side}: {name}", label, *cells])
    out.append(md_table(["Signal", "Period", *(f"{h} session{'s' if h > 1 else ''}" for h in HORIZONS)], rows))
    return "\n".join(out) + "\n"


def ic_section(res):
    t = res["ic_design"]
    hold = res.get("ic_holdout")
    out = []
    piv = t.pivot(index="feature", columns="horizon", values="ic")
    qv = t.pivot(index="feature", columns="horizon", values="q")
    order = piv[[1, PRIMARY_HORIZON]].abs().max(axis=1).sort_values(ascending=False).index
    n = int(t["n"].iloc[0])
    sig = t[t["q"] < FDR]
    out.append(f"Rank correlation between each indicator at 15:50 and SPY's return over the next 1, 2, 3 and 5 "
               f"sessions, over {n} design sessions with every indicator available. **{len(sig)} of {len(t)}** "
               f"indicator-horizon pairs pass the false-discovery screen (q < {FDR:.2f}). Strongest 20 by |IC| "
               "at 1 or 5 sessions:\n")
    rows = []
    for c in order[:20]:
        r1 = t[(t["feature"] == c) & (t["horizon"] == 1)].iloc[0]
        r5 = t[(t["feature"] == c) & (t["horizon"] == PRIMARY_HORIZON)].iloc[0]
        row = [f"`{c}`", LABELS[c]] + [f"{piv.loc[c, h]:+.3f}" + (" *" if qv.loc[c, h] < FDR else "")
                                        for h in HORIZONS]
        row += [pct(r5["q5_minus_q1"])]
        if hold is not None:
            hr = hold[(hold["feature"] == c) & (hold["horizon"] == PRIMARY_HORIZON)]
            row += [f"{hr['ic'].iloc[0]:+.3f}" if len(hr) else "n/a"]
        rows.append(row)
    head = ["Indicator", "Group", *(f"IC {h}d" for h in HORIZONS), f"Top − bottom fifth, {PRIMARY_HORIZON}d return"]
    out.append(md_table(head + (["Holdout IC 5d"] if hold is not None else []), rows))
    out.append("\n\\* q < %.2f. Full table: `indicators_design.csv`%s.\n" % (
        FDR, " and `indicators_holdout.csv`" if hold is not None else ""))
    out.append("![Indicators](indicators.png)\n")
    return "\n".join(out)


def events_section(res, side):
    t = res["events_design"]
    hold = res.get("events_holdout")
    t = t[(t["side"] == side) & (t["regime"] == "all")]
    sign = 1 if side == "dip" else -1
    out = []
    word = ("SPY tends to bounce" if side == "dip" else "SPY tends to fall back")
    out.append(f"If mean reversion is real, {word}: the excess return should be "
               f"{'positive' if side == 'dip' else 'negative'}. Events at 15:50; returns to 15:50 later.\n")
    rows = []
    for name in (DIPS if side == "dip" else RALLIES):
        r = {h: t[(t["event"] == name) & (t["horizon"] == h)].iloc[0] for h in (1, 3, PRIMARY_HORIZON)}
        r5 = r[PRIMARY_HORIZON]
        if not r5["n"]:
            rows.append([name, 0, "", "", "", "", "", ""])
            continue
        lo, hi = (r5["excess_lo"], r5["excess_hi"]) if sign > 0 else (-r5["excess_hi"], -r5["excess_lo"])
        row = [name, f"{int(r5['n'])} ({int(r5['clusters'])})", pct(r[1]["excess"]), pct(r[3]["excess"]),
               f"**{pct(r5['excess'])}** ({pct(r5['excess_lo'])} to {pct(r5['excess_hi'])})", verdict(lo, hi),
               f"{r5['hit']:.0f}% vs {r5['base_hit']:.0f}%"]
        if hold is not None:
            hr = hold[(hold["event"] == name) & (hold["horizon"] == PRIMARY_HORIZON) & (hold["side"] == side)
                      & (hold["regime"] == "all")].iloc[0]
            if hr["n"]:
                hlo, hhi = (hr["excess_lo"], hr["excess_hi"]) if sign > 0 else (-hr["excess_hi"], -hr["excess_lo"])
                row.append(f"{int(hr['n'])}: {pct(hr['excess'])} {verdict(hlo, hhi)}")
            else:
                row.append("0 events")
        rows.append(row)
    head = ["Event", "Events (clusters)", "Excess 1d", "Excess 3d", f"Excess {PRIMARY_HORIZON}d (95% interval)",
            "", f"Up after {PRIMARY_HORIZON}d: event vs all days"]
    out.append("**Design period**" + (" with the holdout result in the last column" if hold is not None else "") +
               ":\n")
    out.append(md_table(head + (["Holdout: events, excess 5d"] if hold is not None else []), rows))
    # credit-spread odds
    k_rows = []
    for name in (DIPS if side == "dip" else RALLIES):
        r = t[(t["event"] == name) & (t["horizon"] == PRIMARY_HORIZON)].iloc[0]
        if r["n"] < 10:
            continue
        cells = []
        for k in CREDIT_K:
            cells.append(f"{r[f'close_ok_{k:g}']:.0f}% vs {r[f'close_ok_{k:g}_base']:.0f}%")
        cells.append(f"{r['path_ok_1']:.0f}% vs {r['path_ok_1_base']:.0f}%")
        k_rows.append([name, int(r["n"]), *cells])
    word = "below" if side == "dip" else "above"
    out.append(f"\n**Credit-spread odds** ({PRIMARY_HORIZON} sessions): how often SPY finished no more than k "
               f"{word} the 15:50 entry price, i.e. a short {'put' if side == 'dip' else 'call'} k% "
               f"{'below' if side == 'dip' else 'above'} the entry would have expired worthless, event vs all days. "
               f"Last column: SPY never traded more than 1% {word} the entry at any point.\n")
    out.append(md_table(["Event", "Events", *(f"Within {k:g}%" for k in CREDIT_K), "Never 1% against"], k_rows))
    out.append("")
    return "\n".join(out) + ("\n![Event paths](event_paths.png)\n" if side == "dip" else "\n")


def regime_section(res):
    t = res["regimes_design"]
    hold = res.get("regimes_holdout")
    out = ["Classic advice is to buy dips only in uptrends (above the 200-day average) and that mean reversion is "
           f"stronger when volatility is high. Excess {PRIMARY_HORIZON}-session return by regime, for the main dip "
           "and rally signals (volatility split at the design-period median of 20-day volatility, "
           f"{res['rv_median']:.2f}% a day):\n"]
    names = ["RSI(2) < 10", "1-day drop ≥ 1.5%", "Below lower Bollinger band", "3+ down days in a row",
             "RSI(2) > 90", "1-day rise ≥ 1.5%", "Above upper Bollinger band"]
    regs = [r for r in t["regime"].unique() if r != "all"]
    rows = []
    for name in names:
        row = [name]
        for reg in regs:
            r = t[(t["event"] == name) & (t["horizon"] == PRIMARY_HORIZON) & (t["regime"] == reg)].iloc[0]
            if r["n"] < 5:
                row.append(f"{int(r['n'])} events")
                continue
            sign = 1 if name in DIPS else -1
            lo, hi = (r["excess_lo"], r["excess_hi"]) if sign > 0 else (-r["excess_hi"], -r["excess_lo"])
            cell = f"{pct(r['excess'])} (n={int(r['n'])}) {verdict(lo, hi)}"
            if hold is not None:
                hr = hold[(hold["event"] == name) & (hold["horizon"] == PRIMARY_HORIZON) & (hold["regime"] == reg)]
                if len(hr) and hr.iloc[0]["n"] >= 5:
                    cell += f"; holdout {pct(hr.iloc[0]['excess'])} (n={int(hr.iloc[0]['n'])})"
            row.append(cell)
        rows.append(row)
    out.append(md_table(["Event", *regs], rows))
    return "\n".join(out) + "\n"


def rules_section(res):
    out = ["Each rule exactly as usually published, one position at a time, entering and exiting at the 15:50 "
           "price, no costs. *Per session in trade* is the average return per session held, next to SPY's average "
           "per session over the period (for shorts, minus SPY's): a rule only adds value if it beats that.\n"]
    for label, key in (("Design period", "rules_design"), ("Holdout", "rules_holdout")):
        t = res.get(key)
        if t is None:
            continue
        rows = [[r.rule, r.trades, f"{r.win_rate:.0f}%" if r.trades else "", pct(r.mean), pct(r.total),
                 f"{r.avg_sessions:.1f}" if r.trades else "", f"{r.exposure:.0f}%",
                 f"{pct(r.per_session, 3)} vs {pct(r.spy_per_session, 3)}",
                 f"{r.consistency:.2f}" if pd.notna(r.consistency) else "", pct(-r.max_drawdown, 1)]
                for r in t.itertuples()]
        out.append(f"**{label}:**\n")
        out.append(md_table(["Rule", "Trades", "Winners", "Avg trade", "Sum of trades", "Avg sessions",
                             "Time in market", "Per session in trade vs SPY", "Consistency", "Worst drawdown"], rows))
        out.append("")
    return "\n".join(out) + "\n"


def ml_section(res):
    m = res["ml"]
    out = [f"Logistic regression and boosted trees on all {len(ML_FEATURES)} indicators, predicting whether SPY is "
           f"higher 1 or {PRIMARY_HORIZON} sessions later. Settings fixed in advance (the same as the ML swing "
           f"study). Expanding walk-forward: each {TEST_SESSIONS}-session block is predicted by models trained only "
           f"on earlier sessions whose outcome window had closed; the first {MIN_TRAIN_SESSIONS} sessions train only. "
           "The *oversold score* is the pre-registered simple average of five stretch measures, as a no-ML baseline. "
           "AUC 0.5 = coin flip.\n"]
    rows = []
    for r in m.sort_values(["period", "horizon", "model"]).itertuples():
        rows.append([r.period, f"{r.horizon}d", r.model, r.n, f"{r.auc:.3f}",
                     f"{r.ic:+.3f} ({r.ic_low:+.3f} to {r.ic_high:+.3f})", verdict(r.ic_low, r.ic_high),
                     " / ".join(pct(getattr(r, f"q{k}_mean")) for k in range(1, 6)), pct(r.q5_minus_q1)])
    out.append(md_table(["Period", "Horizon", "Model", "Predictions", "AUC", "Rank IC (95% interval)", "",
                         "Mean return by prediction fifth (lowest → highest)", "Top − bottom"], rows))
    imp = res["importance"]
    if len(imp):
        out.append(f"\n**What the models used** (design blocks, {PRIMARY_HORIZON}-session target): drop in AUC "
                   "when an indicator is shuffled, and its direction (+ = higher value → model more bullish):\n")
        rows = []
        for model in MODELS:
            t = imp[imp["model"] == model].sort_values("auc_drop", ascending=False).head(8)
            sign = lambda v: "constant" if pd.isna(v) else "+" if v > 0 else "−"  # noqa: E731
            rows.append([model, "; ".join(f"`{x.feature}` {x.auc_drop:+.3f} ({sign(x.direction)})"
                                          for x in t.itertuples())])
        out.append(md_table(["Model", "Top indicators (AUC drop, direction)"], rows))
    out.append("\n![ML](ml.png)\n")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- charts

def _axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=7.5)
    return ax


def plot_event_paths(res, path):
    periods = [("Design", res["paths_design"])] + ([("Holdout", res["paths_holdout"])] if res["final_test"] else [])
    fig = plt.figure(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    colors = [SERIES[0], SERIES[1], TARGET, "#eda100", "#8a63d2", "#e87ba4"]
    width = 0.86 / len(periods)
    for k, (label, paths) in enumerate(periods):
        ax = _axes(fig, [0.09 + k * (width + 0.02), 0.22, width - 0.04, 0.55])
        ax.axhline(0, color=BASELINE, lw=1)
        for (name, (p, n)), color in zip(paths.items(), colors):
            ax.plot(range(len(p)), p, color=color, lw=1.8, label=f"{name} (n={n})" if k == 0 else None)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(label, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Sessions after the 15:50 signal", color=INK_2, fontsize=7.5)
        if k == 0:
            ax.set_ylabel("Average excess return (%)", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, "What SPY did after each signal, beyond its usual drift", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Average cumulative return after the signal minus the average path from all days in the "
             "same period.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_indicators(res, path):
    t = res["ic_design"]
    piv = t.pivot(index="feature", columns="horizon", values="ic")
    order = piv[[1, PRIMARY_HORIZON]].abs().max(axis=1).sort_values(ascending=True).index[-18:]
    fig = plt.figure(figsize=(8, 5.2), dpi=150, facecolor=SURFACE)
    ax = _axes(fig, [0.33, 0.14, 0.63, 0.7])
    y = np.arange(len(order))
    for j, (h, color) in enumerate(((1, SERIES[0]), (PRIMARY_HORIZON, SERIES[1]))):
        ax.barh(y + (j - 0.5) * 0.38, piv.loc[order, h], 0.36, color=color, label=f"{h}-session return", zorder=3)
    if res.get("ic_holdout") is not None:
        hp = res["ic_holdout"].pivot(index="feature", columns="horizon", values="ic")
        ax.scatter(hp.loc[order, PRIMARY_HORIZON], y + 0.19, marker="|", s=80, color=INK, zorder=4,
                   label=f"holdout, {PRIMARY_HORIZON}-session")
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y, [f"{c} ({LABELS[c].lower()})" for c in order], fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Rank correlation with SPY's forward return (negative = mean reversion for return-type "
                  "indicators)", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, "Which 15:50 indicators line up with SPY's next few days?", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.925, "Design period. Strongest 18 indicators by absolute correlation at 1 or 5 sessions.",
             color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_ml(res, path):
    m = res["ml"]
    m = m[m["horizon"] == PRIMARY_HORIZON]
    periods = [p for p in ("design", "holdout") if (m["period"] == p).any()]
    fig = plt.figure(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    names = list(MODELS) + ["oversold score (no ML)"]
    colors = [SERIES[0], SERIES[1], TARGET]
    width = 0.86 / len(periods)
    for k, period in enumerate(periods):
        ax = _axes(fig, [0.09 + k * (width + 0.02), 0.2, width - 0.04, 0.58])
        for j, (name, color) in enumerate(zip(names, colors)):
            r = m[(m["period"] == period) & (m["model"] == name)]
            if not len(r):
                continue
            vals = [r.iloc[0][f"q{q}_mean"] for q in range(1, 6)]
            ax.bar(np.arange(5) + (j - 1) * 0.27, vals, 0.25, color=color, zorder=3,
                   label=name if k == 0 else None)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_xticks(range(5), ["lowest", "2", "3", "4", "highest"])
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_title(f"{period.capitalize()} (out of sample)", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Fifth of predictions", color=INK_2, fontsize=7.5)
        if k == 0:
            ax.set_ylabel(f"Mean {PRIMARY_HORIZON}-session return (%)", color=INK_2, fontsize=7.5)
    fig.text(0.02, 0.97, "Do the models sort good weeks from bad ones?", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "SPY's average return over the next 5 sessions, by how bullish each model was. A working "
             "model rises left to right.", color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def load_daily(sessions, cache_dir, refresh, tickers):
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    daily, notes = {}, {}
    for t in tickers:
        try:
            minutes, source = download.load_minute_bars(t, first, last, cache_dir=cache_dir, refresh=refresh,
                                                        final_close=sessions["close"].iloc[-1])
        except RuntimeError as e:
            if t == TICKER:
                raise
            notes[t] = f"skipped: {e}"
            continue
        d, counts = daily_table(minutes, sessions)
        divs = load_dividends(t, first, last, cache_dir, refresh)
        daily[t], applied = adjust_dividends(d, divs)
        notes[t] = (f"{source}; {int(d['snap'].notna().sum())} sessions with a 15:50 price; "
                    f"{len(applied)} dividends applied")
    return daily, notes


def write_outputs(out_dir, res, notes):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["features"].to_parquet(out / "features.parquet")
    res["ic_design"].to_csv(out / "indicators_design.csv", index=False)
    res["events_design"].to_csv(out / "events_design.csv", index=False)
    res["regimes_design"].to_csv(out / "regimes_design.csv", index=False)
    res["rules_design"].to_csv(out / "rules_design.csv", index=False)
    res["ml"].to_csv(out / "ml.csv", index=False)
    res["importance"].to_csv(out / "ml_importance.csv", index=False)
    if res["final_test"]:
        for key in ("ic_holdout", "events_holdout", "regimes_holdout", "rules_holdout"):
            name = key.replace("ic_", "indicators_")
            res[key].to_csv(out / f"{name}.csv", index=False)
    plot_indicators(res, out / "indicators.png")
    plot_event_paths(res, out / "event_paths.png")
    plot_ml(res, out / "ml.png")
    text = render_report(res)
    text += "\n## Data\n\n" + "\n".join(f"- **{t}:** {n}" for t, n in notes.items()) + "\n"
    (out / "report.md").write_text(text)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPY 1-5 day mean-reversion signal study (design period; "
                                            "--final-test also evaluates the holdout once).")
    p.add_argument("--out", default="output/meanrev")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--final-test", action="store_true",
                   help="also evaluate the pre-registered candidates on the holdout (2025-01-01 on), once")
    p.add_argument("--reps", type=int, default=2000, help="bootstrap resamples (default 2,000)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(args.start, args.end, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    with timed(timings, "data"):
        daily, notes = load_daily(sessions, args.cache_dir, args.refresh, (TICKER, *CONTEXT))
    res = run_study(daily, sessions, final_test=args.final_test, reps=args.reps, seed=args.seed, timings=timings)
    with timed(timings, "outputs"):
        out = write_outputs(args.out, res, notes)
    print(f"Mean-reversion study: design {res['periods']['design'][0].date()}..{DESIGN_END.date()}"
          + (f", holdout {HOLDOUT_START.date()}..{res['periods']['holdout'][1].date()} (final test)"
             if args.final_test else " (holdout not opened)"))
    print("primary picks:", res["primary"])
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
