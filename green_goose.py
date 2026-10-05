"""Green Goose (versions 1 and 2): buy an at-the-money call or put near the close, exit the next morning. Tested
first on stock prices (five years) and then with SPY options (the data plan's two years). A signal study on stocks;
options per contract at traded prices plus slippage, no commissions.

  uv run --env-file .env python green_goose.py --out output/green_goose
  uv run --env-file .env python green_goose.py --skip-options --out output/green_goose   # stocks only

The rules as given, with the points the text leaves open fixed before any result was seen:
- Decision at 15:50 ET on daily bars, today's bar being the session so far (its 15:50 price as the close).
  - Base: Wilder RSI(2) > 85 -> puts, < 15 -> calls; otherwise trade with the day's candle (15:50 price above the
    open -> calls, below -> puts; "direction of candle" read as going with it).
  - Overrides, in priority order: (1) ADX(n) moves into the zone between +DI(n) and -DI(n) today, having been outside
    it yesterday -> puts if -DI is on top, calls if +DI is; (2) RSI(2) "stabs" the zone: above both DI lines yesterday
    and at or below the upper one today (into the zone or all the way below both) -> calls; below both yesterday and
    at or above the lower one today (into it or all the way above both) -> puts. No trade when ADX(n) > 60. n = 5 for
    version 1, 6 for version 2
    (its chart setup). Version 2's TRIX override is for EEM only and is not used; its Bollinger/EMA/SMA lines are
    "for guidance" and set no rule.
- Stocks: the move from the 15:50 price to the next session's open, 9:35 (version 2 exits in the first 5 minutes),
  9:40, 10:00 and 11:00, in the signal's direction, in bps; an ex-dividend morning's payout comes off the 15:50
  price. SPY, QQQ, IWM, AAPL, META, 2021-10 to 2026-09. Baselines: the same days with calls and puts in the same mix
  but at random (expected move = (share of calls - share of puts) x the average move), and always calls.
- Options (SPY, every session the option plan covers): forward F = K + C - P from the call and put at the strike
  nearest the 15:50 price; implied volatility, delta and theta (per calendar day) by Black-76 at 15:50; the strike
  bracketing F whose delta is within 0.47-0.53 (closest to 0.50) with theta below -0.12, else no trade. Expiry: the
  first session at least 5 calendar days out (primary) or the next session ("1-day", secondary). Bought at the open
  of the option's 15:50 minute bar. Version 2 exit: the 9:35 price. Version 1 exit, from the option's opening price O
  against the price paid: up 100%+ -> sell 55% at O and trail the rest; up less, or down less than 10% -> trail
  everything (stop 10% below the highest price since the open); down 10-40% -> sell at O; down more than 40% ->
  hold to the close; anything still open at 11:00 is sold then. Slippage $0.01, $0.02 (primary) and $0.05 a share
  per fill. Baselines: the same contracts and exits with always calls and always puts.
- Tests: SPY stocks, signal minus the random-mix baseline at 9:35 and 9:40 > 0; SPY options, mean P&L per contract
  > 0 for each version at $0.02 a fill (primary expiry); one-sided, BH q < 0.10 across the four.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import talib  # noqa: E402
from scipy.stats import norm  # noqa: E402

import alert_spreads as asp  # noqa: E402
import alerts  # noqa: E402
import call_spreads as cs  # noqa: E402
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from outcomes import _ns  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
TICKERS = ("SPY", "QQQ", "IWM", "AAPL", "META")
VERSIONS = {"version 1": 5, "version 2": 6}          # ADX/DMI period
HORIZONS = {"open": 0, "9:35": 5, "9:40": 10, "10:00": 30, "11:00": 90}  # minutes after the next open
DECISION = pd.Timedelta(minutes=10)                  # before the close: 15:50
WINDOW = 200                                         # earlier daily bars fed to the indicators each day
RSI_HI, RSI_LO, ADX_MAX = 85, 15, 60
DELTA_BAND, THETA_MAX = (0.47, 0.53), -0.12
EXPIRIES = {"5+ days": 5, "1-day": 1}                # calendar days to the first allowed expiry
PRIMARY_EXPIRY = "5+ days"
SLIPS = (0.01, 0.02, 0.05)
PRIMARY_SLIP = 0.02
OPTIONS_FROM = pd.Timestamp("2024-10-01")
TRAIL, BIG_WIN, SELL_PART, CUT_LO, CUT_HI = 0.10, 1.00, 0.55, -0.10, -0.40
FDR = 0.10
MIN = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------- signals

def daily_signals(daily, n):
    """One row per session with a 15:50 price: direction (+1 calls, -1 puts, 0 no trade), the rule that set it, and
    RSI(2), ADX(n), +DI(n), -DI(n) at 15:50 (and yesterday's, suffixed _y). Each day's indicators come from the
    earlier full bars plus today's bar so far, so nothing after 15:50 is used; yesterday's values are those of the
    full bars."""
    d = daily.dropna(subset=["snap", "close", "open"])
    O, H, L, C = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    SH, SL, S = (d[k].to_numpy(float) for k in ("snap_high", "snap_low", "snap"))
    rows = []
    for t in range(len(d)):
        a = max(0, t - WINDOW)
        c, h, l = np.r_[C[a:t], S[t]], np.r_[H[a:t], SH[t]], np.r_[L[a:t], SL[t]]
        if len(c) < 3 * n:
            rows.append({"direction": 0, "rule": "warm-up"})
            continue
        rsi, adx = talib.RSI(c, 2), talib.ADX(h, l, c, n)
        pdi, mdi = talib.PLUS_DI(h, l, c, n), talib.MINUS_DI(h, l, c, n)
        direction, rule = decide((rsi[-1], adx[-1], pdi[-1], mdi[-1]), (rsi[-2], adx[-2], pdi[-2], mdi[-2]),
                                 S[t] - O[t])
        rows.append({"direction": direction, "rule": rule, "rsi2": rsi[-1], "adx": adx[-1], "pdi": pdi[-1],
                     "mdi": mdi[-1], "rsi2_y": rsi[-2], "adx_y": adx[-2], "pdi_y": pdi[-2], "mdi_y": mdi[-2]})
    return pd.DataFrame(rows, index=d.index)


def decide(today, yesterday, candle):
    """The rules for one day: today/yesterday = (RSI(2), ADX, +DI, -DI); candle = 15:50 price minus the open.
    Returns (+1 calls / -1 puts / 0 no trade, the rule that decided)."""
    r, x, p, m = today
    ry, xy, py, my = yesterday
    lo, hi, lo_y, hi_y = min(p, m), max(p, m), min(py, my), max(py, my)
    if r > RSI_HI:
        direction, rule = -1, "RSI(2) above 85"
    elif r < RSI_LO:
        direction, rule = 1, "RSI(2) below 15"
    else:
        direction = int(np.sign(candle))
        rule = "candle up" if direction > 0 else "candle down" if direction < 0 else "flat candle"
    if lo < x < hi and not (lo_y < xy < hi_y):
        direction, rule = (-1 if m > p else 1), "ADX entered the DI zone"
    elif ry > hi_y and r <= hi:  # from above both lines into the zone or through it to below both
        direction, rule = 1, "RSI(2) stabbed the zone from above"
    elif ry < lo_y and r >= lo:  # from below both lines into the zone or through it to above both
        direction, rule = -1, "RSI(2) stabbed the zone from below"
    if x > ADX_MAX:
        direction, rule = 0, "ADX above 60"
    return direction, rule


# ---------------------------------------------------------------- stocks

def morning_moves(rth, sessions, dividends):
    """Per session t: the 15:50 price (less a dividend paid the next morning) and the next session's prices at each
    horizon (the open, then the last minute close before open + h), indexed by t."""
    days = sessions.index
    g = rth.groupby("session")
    first_open = g["open"].first()
    t_ns, close = _ns(rth["ts"]), rth["close"].to_numpy(float)
    paid = dividends.groupby("ex_date")["cash_amount"].sum() if len(dividends) else pd.Series(dtype=float)
    out = []
    for i in range(len(days) - 1):
        t, nxt = days[i], days[i + 1]
        dec = sessions.loc[t, "close"] - DECISION
        k = int(np.searchsorted(t_ns, dec.value)) - 1
        if k < 0 or rth["session"].iloc[k] != t:
            continue
        row = {"session": t, "entry": close[k] - float(paid.get(nxt, 0.0))}
        o = sessions.loc[nxt, "open"]
        for name, h in HORIZONS.items():
            if h == 0:
                row[name] = first_open.get(nxt, np.nan)
                continue
            j = int(np.searchsorted(t_ns, (o + pd.Timedelta(minutes=h)).value)) - 1
            row[name] = close[j] if j >= 0 and rth["session"].iloc[j] == nxt else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("session")


def stock_stats(sig, moves):
    """Per horizon: trades, share of calls, hit rate, mean move in the signal's direction (bps) with its 95%
    interval, the random-mix baseline, the difference and its one-sided p, and always-calls."""
    j = sig.join(moves, how="inner")
    j = j[j["direction"] != 0]
    rows = []
    for name in HORIZONS:
        ret = (j[name] / j["entry"] - 1) * 1e4
        ok = ret.notna()
        x, dirn = (ret * j["direction"])[ok].to_numpy(), j["direction"][ok].to_numpy()
        n = len(x)
        if n < 10:
            continue
        mean, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
        mix = (dirn > 0).mean() - (dirn < 0).mean()
        base = mix * ret[ok].mean()
        z = (mean - base) / se if se else np.nan
        lo, hi = alerts.wilson(int((x > 0).sum()), n)
        rows.append({"horizon": name, "trades": n, "calls": (dirn > 0).mean() * 100, "hit": (x > 0).mean() * 100,
                     "hit_lo": lo, "hit_hi": hi, "mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
                     "baseline": base, "excess": mean - base, "p_up": norm.sf(z) if np.isfinite(z) else np.nan,
                     "always_calls": ret[ok].mean()})
    return pd.DataFrame(rows)


def rule_stats(sig, moves, horizon="9:40"):
    """The move in the signal's direction by the rule that set it (for reading)."""
    j = sig.join(moves, how="inner")
    j = j[j["direction"] != 0]
    x = (j[horizon] / j["entry"] - 1) * 1e4 * j["direction"]
    g = pd.DataFrame({"rule": j["rule"], "x": x}).dropna().groupby("rule")["x"]
    return pd.DataFrame({"trades": g.size(), "hit": g.apply(lambda v: (v > 0).mean() * 100), "mean": g.mean()})


# ---------------------------------------------------------------- options

def b76(F, K, sigma, tau):
    """(call delta, theta per calendar day) by Black-76 with no discounting."""
    s = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    return norm.cdf(d1), -F * norm.pdf(d1) * sigma / (2 * math.sqrt(tau)) / 365


def price_before(bars, at_ns, look=5):
    """Close of the last bar starting before `at_ns` (within `look` minutes); NaN otherwise."""
    if bars is None or bars.empty:
        return np.nan
    t = _ns(bars["ts"])
    k = int(np.searchsorted(t, at_ns)) - 1
    return float(bars["close"].iloc[k]) if k >= 0 and t[k] >= at_ns - look * MIN.value else np.nan


def price_from(bars, at_ns, look=2):
    """Open of the first bar starting at or after `at_ns` (within `look` minutes); NaN otherwise."""
    if bars is None or bars.empty:
        return np.nan
    t = _ns(bars["ts"])
    k = int(np.searchsorted(t, at_ns))
    return float(bars["open"].iloc[k]) if k < len(t) and t[k] <= at_ns + look * MIN.value else np.nan


def expiry_for(day, sessions, min_days):
    """The first session at least `min_days` calendar days after `day` (SPY lists an expiry every session)."""
    later = sessions.index[sessions.index >= day + pd.Timedelta(days=min_days)]
    return later[0] if len(later) else None


def choose_contract(day, right, spot, expiry, sessions, load):
    """(ticker, strike, delta, theta, price at 15:50) of the contract the rules pick, or (None, reason)."""
    dec = (sessions.loc[day, "close"] - DECISION).value
    k0 = math.floor(spot + 0.5)
    try:
        c0 = price_before(load(asp.option_ticker(expiry, "C", k0, "SPY"), f"{day:%Y-%m-%d}"), dec)
        p0 = price_before(load(asp.option_ticker(expiry, "P", k0, "SPY"), f"{day:%Y-%m-%d}"), dec)
    except asp.NotInPlan:
        return None, "outside the data plan"
    if np.isnan(c0) or np.isnan(p0):
        return None, "no price for the forward"
    F = k0 + c0 - p0
    tau = (sessions.loc[expiry, "close"].value - dec) / 1e9 / (365 * 86400)
    best = None
    for k in sorted({math.floor(F), math.ceil(F)}):
        tk = asp.option_ticker(expiry, right, k, "SPY")
        px = price_before(load(tk, f"{day:%Y-%m-%d}"), dec)
        if np.isnan(px):
            continue
        call_px = px if right == "C" else px + F - k
        iv = float(cs.implied_vol(call_px, F, k, tau))
        if not np.isfinite(iv):
            continue
        dcall, theta = b76(F, k, iv, tau)
        delta = dcall if right == "C" else dcall - 1
        if DELTA_BAND[0] <= abs(delta) <= DELTA_BAND[1] and theta < THETA_MAX:
            if best is None or abs(abs(delta) - 0.5) < abs(abs(best[2]) - 0.5):
                best = (tk, k, delta, theta, px)
    return (best, "ok") if best else (None, "no strike in the delta band")


def exit_v2(bars, open_ns):
    """Version 2: the price at 9:35 (the last close in the first five minutes, else the next bar's open)."""
    p = price_before(bars, open_ns + 5 * MIN.value, look=5)
    return p if np.isfinite(p) else price_from(bars, open_ns + 5 * MIN.value)


def exit_v1(bars, open_ns, close_ns, paid):
    """Version 1's exits from the option's next-day bars; returns (average exit price, how it ended)."""
    o = price_from(bars, open_ns, look=5)
    if not np.isfinite(o):
        return np.nan, "no opening price"
    r = o / paid - 1
    if CUT_HI <= r <= CUT_LO:
        return o, "down 10-40%: out at the open"
    if r < CUT_HI:
        return price_before(bars, close_ns, look=30), "down over 40%: held to the close"
    part = SELL_PART if r >= BIG_WIN else 0.0
    t = _ns(bars["ts"])
    hi, lo, op, cl = (bars[k].to_numpy(float) for k in ("high", "low", "open", "close"))
    stop_end = open_ns + 90 * MIN.value
    peak, rest = o, np.nan
    for i in np.flatnonzero((t >= open_ns) & (t < stop_end)):
        stop = (1 - TRAIL) * peak
        if lo[i] <= stop:
            rest = min(stop, op[i])
            break
        peak = max(peak, hi[i])
    how = "up 100%+: half at the open, trailed" if part else "trailed"
    if np.isnan(rest):
        rest = price_before(bars, stop_end, look=30)
        how += ", out at 11:00"
    return part * o + (1 - part) * rest, how


def _fetch(reqs, load, workers=8):
    def one(r):
        try:
            load(*r)
        except asp.NotInPlan:
            pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, sorted(reqs)))


def option_trades(days, spots, sessions, load, expiry_days, workers=8):
    """For each session: the call and the put the rules pick, bought at the 15:50 bar's open and exited by both
    versions the next morning (prices before slippage). Downloads the entry-day candidates first, then the next
    day only for the chosen contracts."""
    plan = []
    for day in days:
        i = sessions.index.get_loc(day)
        expiry = expiry_for(day, sessions, expiry_days)
        if i + 1 < len(sessions) and expiry is not None and day in spots.index:
            plan.append((day, sessions.index[i + 1], expiry, math.floor(spots.loc[day] + 0.5)))
    if hasattr(load, "state"):
        _fetch({(asp.option_ticker(e, r, k, "SPY"), f"{d:%Y-%m-%d}") for d, _, e, k0 in plan
                for r in ("C", "P") for k in (k0 - 1, k0, k0 + 1)}, load, workers)
    picks = []
    for day, nxt, expiry, _ in plan:
        for right in ("C", "P"):
            pick, why = choose_contract(day, right, spots.loc[day], expiry, sessions, load)
            picks.append((day, nxt, expiry, right, pick, why))
    if hasattr(load, "state"):
        _fetch({(p[0], f"{nxt:%Y-%m-%d}") for _, nxt, _, _, p, _ in picks if p}, load, workers)
    rows = []
    for day, nxt, expiry, right, pick, why in picks:
        row = {"session": day, "right": right, "expiry": expiry, "status": why}
        if pick:
            tk, k, delta, theta, px = pick
            dec = (sessions.loc[day, "close"] - DECISION).value
            try:
                today, tomorrow = load(tk, f"{day:%Y-%m-%d}"), load(tk, f"{nxt:%Y-%m-%d}")
            except asp.NotInPlan:
                rows.append({**row, "status": "outside the data plan"})
                continue
            paid = price_from(today, dec)
            o_ns, c_ns = sessions.loc[nxt, "open"].value, sessions.loc[nxt, "close"].value
            v2 = exit_v2(tomorrow, o_ns)
            v1, how = exit_v1(tomorrow, o_ns, c_ns, paid) if np.isfinite(paid) else (np.nan, "no entry price")
            row.update(contract=tk, strike=k, delta=delta, theta=theta, paid=paid, exit_v1=v1, v1_how=how,
                       exit_v2=v2, status="ok" if np.isfinite(paid) and np.isfinite(v2) else "missing prices")
        rows.append(row)
    return pd.DataFrame(rows)


def option_stats(t, exit_col, slip, years):
    x = 100 * (t[exit_col] - t["paid"] - 2 * slip).to_numpy(float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"trades": n}
    if n < 10:
        return out
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    z = m / se if se else np.nan
    prem = 100 * (t.loc[np.isfinite(t[exit_col]), "paid"].to_numpy(float) + slip)
    cum = np.cumsum(x)
    return {**out, "win": (x > 0).mean() * 100, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(z) if np.isfinite(z) else np.nan, "pct_premium": (x / prem).mean() * 100,
            "premium": prem.mean(), "per_year": x.sum() / years, "worst": x.min(),
            "drawdown": float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max())}


# ---------------------------------------------------------------- study

def run_study(data, sessions, load=None, *, primary="SPY", timings=None):
    """data: {ticker: (minute bars, dividends)}. No file I/O beyond the option loader's cache."""
    timings = {} if timings is None else timings
    stock_rows, rules, sigs, raw = [], [], {}, None
    for tk, (minutes, dividends) in data.items():
        with timed(timings, "daily bars and signals"):
            daily, _ = mr.daily_table(minutes, sessions)
            adj, _ = mr.adjust_dividends(daily, dividends)
            rth, _ = features.regular_session_minutes(minutes, sessions)
            moves = morning_moves(rth, sessions, dividends)
        for version, n in VERSIONS.items():
            with timed(timings, "daily bars and signals"):
                sig = daily_signals(adj, n)
            sigs[(tk, version)] = sig
            for period, part in (("2021-26", sig), ("2021-24", sig[sig.index <= "2024-12-31"]),
                                 ("2025-26", sig[sig.index > "2024-12-31"])):
                st = stock_stats(part, moves)
                stock_rows.append(st.assign(ticker=tk, version=version, period=period))
            if tk == primary:
                rules.append(rule_stats(sig, moves).assign(version=version))
        if tk == primary:
            raw = daily["snap"]  # the unadjusted 15:50 price picks the strikes
    stocks = pd.concat(stock_rows, ignore_index=True)
    out = {"stocks": stocks, "rules": pd.concat(rules), "signals": sigs, "primary": primary, "options": None}
    if load is not None:
        days = sessions.index[(sessions.index >= OPTIONS_FROM)]
        opt = {}
        for name, exp_days in EXPIRIES.items():
            with timed(timings, "options"):
                opt[name] = option_trades(days, raw.dropna(), sessions, load, exp_days)
        rows = []
        for name, t in opt.items():
            ok = t[t["status"] == "ok"]
            years = (ok["session"].max() - ok["session"].min()).days / 365.25 if len(ok) else np.nan
            for version, exit_col in (("version 1", "exit_v1"), ("version 2", "exit_v2")):
                sig_v = sigs[(primary, version)]["direction"]
                d = sig_v.reindex(ok["session"]).to_numpy()
                chosen = ok[((d > 0) & (ok["right"] == "C")) | ((d < 0) & (ok["right"] == "P"))]
                for slip in SLIPS:
                    for label, part in (("signal", chosen), ("always calls", ok[ok["right"] == "C"]),
                                        ("always puts", ok[ok["right"] == "P"])):
                        rows.append({"expiry": name, "version": version, "slip": slip, "trades_by": label,
                                     **option_stats(part, exit_col, slip, years)})
        ost = pd.DataFrame(rows)
        out["options"], out["option_trades"] = ost, opt
    tests = []
    s = stocks[(stocks["ticker"] == primary) & (stocks["period"] == "2021-26") & stocks["horizon"].isin(["9:35", "9:40"])]
    for r in s.itertuples():
        tests.append({"test": f"stocks {r.version} {r.horizon}: signal minus random mix", "value": r.excess,
                      "p": r.p_up})
    if out["options"] is not None:
        o = out["options"]
        for version in VERSIONS:
            r = o[(o["expiry"] == PRIMARY_EXPIRY) & (o["version"] == version) & (o["slip"] == PRIMARY_SLIP)
                  & (o["trades_by"] == "signal")]
            if len(r):
                tests.append({"test": f"options {version}: mean per contract", "value": r["mean"].iloc[0],
                              "p": r["p_up"].iloc[0]})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = mr.bh_qvalues(tests["p"].fillna(1).to_numpy())
    out["tests"] = tests
    return out


# ---------------------------------------------------------------- report

def fmt(v, digits=1, sign=True):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def money(v):
    return "n/a" if v is None or pd.isna(v) else f"{'-' if v < 0 else '+'}${abs(v):,.0f}"


def render_report(res):
    s, pk = res["stocks"], res["primary"]
    lines = []
    w = lines.append
    w("# Green Goose\n")
    w("Buy an at-the-money call or put near the close and sell the next morning. Direction at 15:50 ET: RSI(2) above "
      "85 -> puts, below 15 -> calls, otherwise with the day's candle; overridden by ADX moving into the zone between "
      "the DI lines (puts if -DI is on top), then by RSI(2) crossing into that zone (from above -> calls, from below "
      "-> puts); no trade when ADX is above 60. Version 1 uses ADX/DMI 5, version 2 uses 6.\n")
    if len(res["tests"]):
        w("**Tests.**\n")
        rows = [[r.test, fmt(r.value, 2), f"{r.p:.3f}", f"{r.q:.3f}"] for r in res["tests"].itertuples()]
        w(alerts.md_table(["Test", "Value (bps or $ per contract)", "p", "q"], rows) + "\n")
    w(f"## Stocks: the move from 15:50 to the next morning, in the signal's direction ({pk}, 2021-10 to 2026-09)\n")
    w("*Random mix* = calls and puts in the same proportions at random; *always calls* = the average move itself.\n")
    rows = []
    for version in VERSIONS:
        for r in s[(s["ticker"] == pk) & (s["version"] == version) & (s["period"] == "2021-26")].itertuples():
            rows.append([version, r.horizon, f"{r.trades:,}", f"{r.calls:.0f}%",
                         f"{r.hit:.0f}% ({r.hit_lo:.0f}-{r.hit_hi:.0f}%)", f"{fmt(r.mean)} ({fmt(r.lo)} to {fmt(r.hi)})",
                         fmt(r.baseline), fmt(r.excess), f"{r.p_up:.3f}", fmt(r.always_calls)])
    w(alerts.md_table(["Version", "Exit", "Trades", "Calls", "Right direction (95%)", "Move, bps (95%)",
                       "Random mix", "Signal minus random", "p", "Always calls"], rows) + "\n")
    w("## Stocks: every ticker, both periods (exit 9:40, version 1)\n")
    rows = []
    for r in s[(s["version"] == "version 1") & (s["horizon"] == "9:40")].itertuples():
        rows.append([r.ticker, r.period, f"{r.trades:,}", f"{r.hit:.0f}%", f"{fmt(r.mean)} ({fmt(r.lo)} to {fmt(r.hi)})",
                     fmt(r.baseline), fmt(r.excess)])
    w(alerts.md_table(["Ticker", "Period", "Trades", "Right direction", "Move, bps (95%)", "Random mix",
                       "Signal minus random"], rows) + "\n")
    w(f"## Stocks: by the rule that set the direction ({pk}, exit 9:40)\n")
    rows = [[r["version"], rule, f"{int(r['trades']):,}", f"{r['hit']:.0f}%", fmt(r["mean"])]
            for rule, r in res["rules"].iterrows()]
    w(alerts.md_table(["Version", "Rule", "Trades", "Right direction", "Move, bps"], rows) + "\n")
    o = res["options"]
    if o is not None:
        w("## SPY options\n")
        w("At-the-money (delta 0.47-0.53, theta below -0.12) bought at the open of the 15:50 minute bar. Version 2 sells "
          "at 9:35; version 1 sells at the open if down 10-40%, holds to the close if down more, else trails 10% below "
          "the highest price since the open (selling 55% at the open first if up 100%+), out by 11:00. Per contract, "
          "traded prices plus slippage on each fill.\n")
        rows = []
        for r in o[o["slip"] == PRIMARY_SLIP].itertuples():
            if pd.isna(getattr(r, "mean", np.nan)):
                continue
            rows.append([r.expiry, r.version, r.trades_by, f"{int(r.trades):,}", f"{r.win:.0f}%",
                         f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})", f"{r.pct_premium:+.1f}%",
                         money(r.premium), money(r.per_year), f"{money(r.worst)} / {money(-r.drawdown)}"])
        w(alerts.md_table(["Expiry", "Version", "Trades", "Count", "Wins", f"Per contract at ${PRIMARY_SLIP:.2f} (95%)",
                           "% of premium", "Avg premium", "A year (1 contract)", "Worst / worst run"], rows) + "\n")
        rows = []
        for r in o[(o["trades_by"] == "signal") & (o["expiry"] == PRIMARY_EXPIRY)].itertuples():
            if pd.isna(getattr(r, "mean", np.nan)):
                continue
            rows.append([r.version, f"${r.slip:.2f}", f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})",
                         money(r.per_year)])
        w(f"Slippage, signal trades, {PRIMARY_EXPIRY} expiry:\n")
        w(alerts.md_table(["Version", "Slippage per fill", "Per contract (95%)", "A year (1 contract)"], rows) + "\n")
        for name, t in res["option_trades"].items():
            w(f"- {name} expiry, contracts: " + ", ".join(f"{k} {v:,}" for k, v in t["status"].value_counts().items())
              + ".")
        w("")
    w("## Notes\n")
    w("- Stocks: a signal study from the 15:50 price; it ignores option pricing. Options: traded prices (the plan has "
      "no quotes), Black-76 greeks from those prices.")
    w("- The rules' open points (candle direction read as going with it, the zone crossings, ADX above 60 as a veto, "
      "the expiry) were fixed before any result was seen; see green_goose.py.")
    return "\n".join(lines) + "\n"


def plot(res, path):
    pk = res["primary"]
    fig = plt.figure(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.18, 0.86, 0.62])
    sig = res["signals"][(pk, "version 1")]
    moves = res.get("moves_primary")
    if moves is not None:
        j = sig.join(moves, how="inner")
        j = j[j["direction"] != 0]
        x = ((j["9:40"] / j["entry"] - 1) * 1e4 * j["direction"]).dropna()
        ax.plot(x.index, x.cumsum() / 100, color=SERIES[0], lw=1.2, label="signal, out at 9:40")
        a = ((j["9:40"] / j["entry"] - 1) * 1e4).dropna()
        ax.plot(a.index, a.cumsum() / 100, color=MUTED, lw=1.0, ls=(0, (3, 2)), label="always calls (long overnight)")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative % (stock, whole account)", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.02, 0.97, f"Green Goose on {pk}: 15:50 to 9:40 the next morning", color=INK, fontsize=11,
             fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Stock moves in the signal's direction (version 1), against simply holding overnight.",
             color=INK_2, fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Green Goose: at-the-money options bought near the close, sold next morning.")
    p.add_argument("--out", default="output/green_goose")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--skip-options", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timings = {}
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    sessions = sessions[sessions.index < today]
    first, last = str(sessions.index[0].date()), str(sessions.index[-1].date())
    with timed(timings, "data"):
        data = {tk: (gr.load_minutes(tk, sessions, args.cache_dir, args.refresh)[0],
                     mr.load_dividends(tk, first, last, args.cache_dir, args.refresh)) for tk in args.tickers}
    load = None if args.skip_options else asp.option_loader(args.cache_dir, args.refresh)
    res = run_study(data, sessions, load, primary=args.tickers[0], timings=timings)
    minutes, dividends = data[args.tickers[0]]
    rth, _ = features.regular_session_minutes(minutes, sessions)
    res["moves_primary"] = morning_moves(rth, sessions, dividends)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["stocks"].to_csv(out / "stocks.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    if res["options"] is not None:
        res["options"].to_csv(out / "options.csv", index=False)
        pd.concat([t.assign(expiry_rule=k) for k, t in res["option_trades"].items()]).to_parquet(
            out / "option_trades.parquet", index=False)
    plot(res, out / "green_goose.png")
    (out / "report.md").write_text(render_report(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
