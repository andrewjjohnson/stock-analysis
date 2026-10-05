"""Green Goose indicator search: which indicators known at 15:50 tell how SPY moves overnight, so Green Goose can keep
the pieces that help, drop the ones that don't and add new ones. Chosen on 2021-10 to 2024-09; checked on 2024-10 to
2026-09 as stock moves and as the same SPY options green_goose.py buys.

  uv run --env-file .env python goose_indicators.py --out output/goose_indicators

Fixed before any result was seen:
- Conditions (sets of sessions, known at 15:50 from earlier full daily bars plus today's bar so far): Green Goose's
  own pieces (RSI(2) below 15 / above 85, the candle up / down, ADX(5) entering the DI zone with +DI or -DI on top,
  RSI(2) stabbing the zone from above / below, ADX(5) above 60) and new ones: IBS below 0.2 / above 0.8, beyond the
  lower / upper Bollinger band (20, 2), 3+ down / up closes in a row, down / up 1%+ on the day, MFI(14) below 20 /
  above 80, above / below the 50-day average, the MACD(12, 26, 9) histogram above / below 0, the last half hour
  (15:20-15:50) up / down, an open above / below yesterday's close, VIXY up / down 5%+ on the day, volume 50%+ above
  its 20-day average, the next session at the turn of the month (a month's last session or first three) and a
  weekend or holiday before the next session.
- Measure: the move from the 15:50 price to the next session's open, in bps (a dividend paid that morning comes off
  the 15:50 price). A condition's edge = the average move on its sessions minus the average on every session.
- Screen (2021-10 to 2024-09, SPY): two-sided p < 0.05 against the same condition slid to other dates (every
  circular shift of at least 20 sessions that doesn't line it up with itself, such as Fridays slid onto Fridays;
  `slides`), at least 30 sessions, and an edge of the same sign on QQQ and IWM (each
  with its own indicators). A passing condition with a positive edge is a call signal; with a negative edge it is a
  put signal if SPY fell on average after it (puts made money) and a warning otherwise (calls did worse than usual,
  but puts would not have paid).
- Rule: calls when call signals outnumber put signals and warnings together; puts when put signals outnumber call
  signals; no trade otherwise. "Strong" (for reading only): a margin of two or more.
- Check (2024-10 to 2026-09, run once): the rule as SPY options with green_goose.py's contracts and version 1 exits
  at $0.02 a share per fill, next-session expiry (primary) and 5+ days; and the SPY stock move against the rule slid
  to other dates. Tests: mean per contract > 0 for each expiry and the stock move above its slid copies, one-sided,
  BH q < 0.10 across the three. Green Goose version 1 and always calls are shown next to it.
"""

import argparse
import math
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
import features  # noqa: E402
import gap_recovery as gr  # noqa: E402
import green_goose as gg  # noqa: E402
import meanrev as mr  # noqa: E402
import stock_dip_spreads as sds  # noqa: E402
from report import BASELINE, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from run import timed  # noqa: E402

NY = features.NY
PRIMARY, CONFIRM, VIX = "SPY", ("QQQ", "IWM"), "VIXY"
DESIGN_END = pd.Timestamp("2024-09-30")
CHECK_START = gg.OPTIONS_FROM                       # 2024-10-01, where the option data starts
HORIZON = "open"                                    # the next session's opening price
P_SCREEN, MIN_DAYS, MIN_SHIFT = 0.05, 30, 20
SLIP = gg.PRIMARY_SLIP
PRIMARY_EXPIRY = "1-day"
STRONG = 2
FDR = 0.10
BOOT = 5000
HALF_HOUR = 40                                      # minutes before the close: 15:20
INPUTS = ["rsi2", "adx", "pdi", "mdi", "rsi2_y", "adx_y", "pdi_y", "mdi_y", "candle", "ibs", "bb_z20", "streak",
          "ret_1d", "mfi", "dist_sma50", "macd_hist", "last_half", "gap", "vixy_1d", "vol_ratio"]
GOOSE = ["RSI(2) below 15", "RSI(2) above 85", "Candle up", "Candle down", "ADX(5) entered the DI zone, +DI on top",
         "ADX(5) entered the DI zone, -DI on top", "RSI(2) stabbed the zone from above",
         "RSI(2) stabbed the zone from below", "ADX(5) above 60"]


# ---------------------------------------------------------------- indicators and conditions

def mfi_macd(d):
    """MFI(14) and the MACD(12, 26, 9) histogram at 15:50, from up to 200 earlier full bars plus today's bar so far."""
    d = d.dropna(subset=["snap", "close"])
    C, H, L, V = (d[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    S, SH, SL, SV = (d[k].to_numpy(float) for k in ("snap", "snap_high", "snap_low", "snap_volume"))
    mfi, hist = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for t in range(len(d)):
        a = max(0, t - gg.WINDOW)
        c = np.r_[C[a:t], S[t]]
        if t >= 14:
            mfi[t] = talib.MFI(np.r_[H[a:t], SH[t]], np.r_[L[a:t], SL[t]], c, np.r_[V[a:t], SV[t]], 14)[-1]
        if t >= 33:
            hist[t] = talib.MACD(c, 12, 26, 9)[2][-1]
    return pd.DataFrame({"mfi": mfi, "macd_hist": hist}, index=d.index)


def indicator_frame(adj, raw, raw_half, vixy_ret, sessions):
    """Every input at 15:50, one row per session with a 15:50 price. adj = dividend-adjusted daily table, raw / raw_half
    = unadjusted tables with the snapshot at 15:50 / 15:20, vixy_ret = VIXY's 15:50 price against its last close (%)."""
    ind = gg.daily_signals(adj, gg.VERSIONS["version 1"])
    f = mr.ticker_features(adj)[["ibs", "bb_z20", "streak", "ret_1d", "dist_sma50", "gap", "vol_ratio"]]
    ind = ind.join(f).join(mfi_macd(adj))
    ind["candle"] = (adj["snap"] - adj["open"]).reindex(ind.index)
    ind["last_half"] = ((raw["snap"] / raw_half["snap"] - 1) * 100).reindex(ind.index)
    ind["vixy_1d"] = vixy_ret.reindex(ind.index)
    days = sessions.index.to_series()
    ind["tom_next"] = mr.calendar_features(sessions)["turn_of_month"].shift(-1).reindex(ind.index)
    ind["days_to_next"] = (days.shift(-1) - days).dt.days.reindex(ind.index)
    return ind


def conditions(ind):
    """One bool column per condition (a missing input makes it false) plus `usable`: every input is available."""
    r, x, p, m = (ind[k] for k in ("rsi2", "adx", "pdi", "mdi"))
    ry, xy, py, my = (ind[k] for k in ("rsi2_y", "adx_y", "pdi_y", "mdi_y"))
    lo, hi, lo_y, hi_y = np.fmin(p, m), np.fmax(p, m), np.fmin(py, my), np.fmax(py, my)
    entered = (lo < x) & (x < hi) & ~((lo_y < xy) & (xy < hi_y))
    c = {
        "RSI(2) below 15": r < gg.RSI_LO,
        "RSI(2) above 85": r > gg.RSI_HI,
        "Candle up": ind["candle"] > 0,
        "Candle down": ind["candle"] < 0,
        "ADX(5) entered the DI zone, +DI on top": entered & (p > m),
        "ADX(5) entered the DI zone, -DI on top": entered & (m > p),
        "RSI(2) stabbed the zone from above": (ry > hi_y) & (r <= hi),
        "RSI(2) stabbed the zone from below": (ry < lo_y) & (r >= lo),
        "ADX(5) above 60": x > gg.ADX_MAX,
        "IBS below 0.2": ind["ibs"] < 0.2,
        "IBS above 0.8": ind["ibs"] > 0.8,
        "Below the lower Bollinger band": ind["bb_z20"] <= -2,
        "Above the upper Bollinger band": ind["bb_z20"] >= 2,
        "3+ down closes in a row": ind["streak"] <= -3,
        "3+ up closes in a row": ind["streak"] >= 3,
        "Down 1%+ on the day": ind["ret_1d"] <= -1,
        "Up 1%+ on the day": ind["ret_1d"] >= 1,
        "MFI(14) below 20": ind["mfi"] < 20,
        "MFI(14) above 80": ind["mfi"] > 80,
        "Above the 50-day average": ind["dist_sma50"] > 0,
        "Below the 50-day average": ind["dist_sma50"] < 0,
        "MACD histogram above 0": ind["macd_hist"] > 0,
        "MACD histogram below 0": ind["macd_hist"] < 0,
        "Last half hour up": ind["last_half"] > 0,
        "Last half hour down": ind["last_half"] < 0,
        "Opened above yesterday's close": ind["gap"] > 0,
        "Opened below yesterday's close": ind["gap"] < 0,
        "VIXY up 5%+ on the day": ind["vixy_1d"] >= 5,
        "VIXY down 5%+ on the day": ind["vixy_1d"] <= -5,
        "Volume 50%+ above normal": ind["vol_ratio"] >= 1.5,
        "Turn of the month next session": ind["tom_next"] == 1,
        "Weekend or holiday before the next session": ind["days_to_next"] > 1,
    }
    out = pd.DataFrame({k: v.astype(bool) for k, v in c.items()}, index=ind.index)
    out["usable"] = ind[INPUTS].notna().all(axis=1)
    return out


def overnight(moves, horizon=HORIZON):
    """The move from the 15:50 price to `horizon` the next session, in bps."""
    return (moves[horizon] / moves["entry"] - 1) * 1e4


# ---------------------------------------------------------------- screen

def slides(series, min_shift=MIN_SHIFT):
    """Circular shifts of at least min_shift sessions that don't line any of the bool series up with itself: after the
    shift no more than halfway from the chance overlap to a full one of its sessions are still its sessions (this
    skips, e.g., Fridays slid onto Fridays)."""
    n = len(series[0])
    keep = []
    for s in range(min_shift, n - min_shift + 1):
        ok = True
        for c in series:
            k, share = c.sum(), c.mean()
            if k and (c & np.roll(c, s)).sum() / k > share + (1 - share) / 2:
                ok = False
                break
        if ok:
            keep.append(s)
    return np.array(keep, int)


def edge_test(r, c, min_shift=MIN_SHIFT):
    """(edge, two-sided p): the average of r on the condition's sessions minus the average of r, against the same
    condition slid to other dates (`slides`)."""
    r, c = np.asarray(r, float), np.asarray(c, bool)
    n = len(r)
    if not 0 < c.sum() < n:
        return np.nan, np.nan
    edge = r[c].mean() - r.mean()
    shifts = slides([c], min_shift)
    if not len(shifts):
        return edge, np.nan
    null = np.array([r[np.roll(c, s)].mean() for s in shifts]) - r.mean()
    centre = null.mean()
    p = (1 + np.sum(np.abs(null - centre) >= abs(edge - centre) - 1e-12)) / (1 + len(null))
    return edge, p


def screen(design):
    """design: {ticker: (r, conditions)} over the design sessions (usable, with a move). One row per condition with
    SPY's sessions, average move after it, edge, share of rises and p, the other tickers' edges and the role."""
    r, cond = design[PRIMARY]
    r = r.to_numpy(float)
    rows = []
    for name in cond.columns.drop("usable"):
        c = cond[name].to_numpy(bool)
        edge, p = edge_test(r, c)
        others = {}
        for tk in CONFIRM:
            if tk in design:
                rt, ct = design[tk]
                ct = ct[name].to_numpy(bool)
                others[tk] = rt.to_numpy(float)[ct].mean() - rt.mean() if ct.any() else np.nan
        mean = r[c].mean() if c.any() else np.nan
        passed = (c.sum() >= MIN_DAYS and p < P_SCREEN
                  and all(np.sign(e) == np.sign(edge) for e in others.values()))
        role = ("call" if edge > 0 else "put" if mean < 0 else "warning") if passed else "not used"
        rows.append({"condition": name, "group": "Green Goose" if name in GOOSE else "new", "sessions": int(c.sum()),
                     "mean": mean, "edge": edge, "up": (r[c] > 0).mean() * 100 if c.any() else np.nan, "p": p,
                     **{f"edge_{tk}": e for tk, e in others.items()}, "role": role})
    return pd.DataFrame(rows)


def rule_direction(cond, roles):
    """+1 calls / -1 puts / 0 no trade per session, and the winning margin. roles: {condition: call/put/warning}."""
    count = {k: cond[[n for n, r in roles.items() if r == k]].sum(axis=1) for k in ("call", "put", "warning")}
    calls, puts, warns = count["call"], count["put"], count["warning"]
    d = np.where(calls > puts + warns, 1, np.where(puts > calls, -1, 0))
    margin = np.where(d > 0, calls - puts - warns, np.where(d < 0, puts - calls, 0))
    return pd.DataFrame({"direction": d, "margin": margin}, index=cond.index)


# ---------------------------------------------------------------- checks

def stock_check(direction, r, min_shift=MIN_SHIFT):
    """Trades, share of calls, share right, mean move in the trade's direction (95%), and the same direction series slid
    to other dates (`slides` of its call and put sessions): their average and the one-sided p."""
    j = pd.concat([direction.rename("d"), r.rename("r")], axis=1, join="inner").dropna()
    d, x = j["d"].to_numpy(float), j["r"].to_numpy(float)
    on = d != 0
    n = int(on.sum())
    if n < 10:
        return {"trades": n}
    s = d[on] * x[on]
    mean, se = s.mean(), s.std(ddof=1) / math.sqrt(n)
    null = []
    for k in slides([d > 0, d < 0], min_shift):
        dk = np.roll(d, k)
        null.append((dk * x)[dk != 0].mean())
    null = np.array(null) if null else np.array([np.nan])
    return {"trades": n, "calls": (d[on] > 0).mean() * 100, "hit": (s > 0).mean() * 100, "mean": mean,
            "lo": mean - 1.96 * se, "hi": mean + 1.96 * se, "slid": null.mean(),
            "p_up": (1 + np.sum(null >= mean)) / (1 + len(null))}


def option_pnl(direction, trades, slip=SLIP):
    """Per contract P&L (version 1 exits, `slip` a share per fill) of the contract matching each session's direction,
    indexed by session; sessions without that contract are skipped."""
    ok = trades[trades["status"] == "ok"]
    d = direction.reindex(ok["session"]).to_numpy()
    pick = ok[((d > 0) & (ok["right"] == "C")) | ((d < 0) & (ok["right"] == "P"))]
    pnl = pd.Series((100 * (pick["exit_v1"] - pick["paid"] - 2 * slip)).to_numpy(float), index=pick["session"])
    return pnl.dropna().sort_index()


def trade_stats(pnl, years):
    x = pnl.to_numpy(float)
    n = len(x)
    out = {"trades": n}
    if n < 10:
        return out
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    win, loss = x[x > 0], x[x <= 0]
    cum = np.cumsum(x)
    return {**out, "win": len(win) / n * 100, "avg_win": win.mean() if len(win) else np.nan,
            "avg_loss": loss.mean() if len(loss) else np.nan, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "p_up": norm.sf(m / se) if se else np.nan, "per_year": x.sum() / years, "worst": x.min(),
            "drawdown": float((np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()),
            "profit_factor": win.sum() / -loss.sum() if loss.sum() < 0 else np.nan}


def diff_interval(a, b, reps=BOOT, seed=0):
    """95% interval of mean(a) - mean(b), resampling sessions (a, b: P&L indexed by session)."""
    days = a.index.union(b.index)
    A, B = a.reindex(days).to_numpy(float), b.reindex(days).to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(days), (reps, len(days)))
    with np.errstate(invalid="ignore"):
        d = np.nanmean(A[idx], axis=1) - np.nanmean(B[idx], axis=1)
    return np.nanpercentile(d, [2.5, 97.5])


# ---------------------------------------------------------------- study

def ticker_tables(minutes, dividends, sessions, vixy_ret):
    """(conditions, overnight moves in bps at the open and 9:40, Green Goose v1 direction, unadjusted 15:50 price)."""
    raw, _ = mr.daily_table(minutes, sessions)
    raw_half, _ = mr.daily_table(minutes, sessions, decision_minutes=HALF_HOUR)
    adj, _ = mr.adjust_dividends(raw, dividends)
    rth, _ = features.regular_session_minutes(minutes, sessions)
    moves = gg.morning_moves(rth, sessions, dividends)
    ind = indicator_frame(adj, raw, raw_half, vixy_ret, sessions)
    r = pd.DataFrame({"open": overnight(moves, "open"), "9:40": overnight(moves, "9:40")})
    return conditions(ind), r, ind["direction"], raw["snap"]


def run_study(data, sessions, load=None, *, timings=None):
    """data: {ticker: (minute bars, dividends)} with SPY, QQQ, IWM and VIXY. No file I/O beyond the option loader's
    cache."""
    timings = {} if timings is None else timings
    with timed(timings, "indicators"):
        vx_raw, _ = mr.daily_table(data[VIX][0], sessions)
        vx_adj, _ = mr.adjust_dividends(vx_raw, data[VIX][1])
        vixy_ret = mr.ticker_features(vx_adj, full=False)["ret_1d"]
        tables = {tk: ticker_tables(*data[tk], sessions, vixy_ret) for tk in (PRIMARY, *CONFIRM) if tk in data}
    design, check = {}, {}
    for tk, (cond, r, _, _) in tables.items():
        ok = cond["usable"] & r["open"].reindex(cond.index).notna()
        rows = cond.index[ok]
        design[tk] = (r.loc[rows[rows <= DESIGN_END], "open"], cond.loc[rows[rows <= DESIGN_END]])
        check[tk] = (r.loc[rows[rows >= CHECK_START], "open"], cond.loc[rows[rows >= CHECK_START]])
    with timed(timings, "screen"):
        scr = screen(design)
        check_r, check_c = check[PRIMARY]
        scr["check_sessions"] = [int(check_c[n].sum()) for n in scr["condition"]]
        scr["check_mean"] = [check_r[check_c[n]].mean() if check_c[n].any() else np.nan for n in scr["condition"]]
        scr["check_edge"] = scr["check_mean"] - check_r.mean()
    roles = {r.condition: r.role for r in scr.itertuples() if r.role != "not used"}
    rules, stocks = {}, []
    for tk, (cond, r, goose, _) in tables.items():
        rule = rule_direction(cond.drop(columns="usable"), roles)
        rules[tk] = {"new rule": rule["direction"],
                     "new rule, strong": pd.Series(np.where(rule["margin"] >= STRONG, rule["direction"], 0),
                                                   index=rule.index),
                     "Green Goose v1": goose, "always calls": pd.Series(1, index=cond.index)}
        for period, rows in (("2021-10 to 2024-09", design[tk][1].index), ("2024-10 to 2026-09", check[tk][1].index)):
            for label, d in rules[tk].items():
                for horizon in ("open", "9:40"):
                    st = stock_check(d.reindex(rows), r.loc[rows, horizon])
                    stocks.append({"ticker": tk, "period": period, "rule": label, "horizon": horizon, **st})
    out = {"screen": scr, "roles": roles, "stocks": pd.DataFrame(stocks), "rules": rules[PRIMARY], "options": None,
           "design_days": len(design[PRIMARY][0]), "check_days": len(check_r),
           "design_mean": design[PRIMARY][0].mean(), "check_mean": check_r.mean()}
    tests = []
    if load is not None:
        days = sessions.index[sessions.index >= CHECK_START]
        snap = tables[PRIMARY][3].dropna()
        opt, rows, pnls = {}, [], {}
        for name, exp_days in gg.EXPIRIES.items():
            with timed(timings, "options"):
                opt[name] = gg.option_trades(days, snap, sessions, load, exp_days)
            ok = opt[name][opt[name]["status"] == "ok"]
            years = (ok["session"].max() - ok["session"].min()).days / 365.25 if len(ok) else np.nan
            for label, d in [*rules[PRIMARY].items(), ("always puts", -rules[PRIMARY]["always calls"])]:
                pnl = option_pnl(d, opt[name])
                pnls[(name, label)] = pnl
                rows.append({"expiry": name, "rule": label, **trade_stats(pnl, years)})
        ost = pd.DataFrame(rows)
        ost["minus_goose_lo"], ost["minus_goose_hi"] = np.nan, np.nan
        for name in gg.EXPIRIES:
            a, b = pnls[(name, "new rule")], pnls[(name, "Green Goose v1")]
            lo, hi = diff_interval(a, b) if len(a) and len(b) else (np.nan, np.nan)
            ost.loc[(ost["expiry"] == name) & (ost["rule"] == "new rule"), ["minus_goose_lo", "minus_goose_hi"]] = lo, hi
        per = []
        ok1 = opt[PRIMARY_EXPIRY][opt[PRIMARY_EXPIRY]["status"] == "ok"]
        yrs = (ok1["session"].max() - ok1["session"].min()).days / 365.25 if len(ok1) else np.nan
        cond_p = tables[PRIMARY][0]
        for name in scr["condition"]:
            on = cond_p.loc[cond_p.index >= CHECK_START, name]
            for side, sign in (("calls", 1), ("puts", -1)):
                st = trade_stats(option_pnl(on.astype(int) * sign, opt[PRIMARY_EXPIRY]), yrs)
                per.append({"condition": name, "side": side, **st})
        out.update(options=ost, option_trades=opt, pnls=pnls, per_condition=pd.DataFrame(per))
        for name in gg.EXPIRIES:
            row = ost[(ost["expiry"] == name) & (ost["rule"] == "new rule")]
            if len(row) and "p_up" in row and pd.notna(row["p_up"].iloc[0]):
                tests.append({"test": f"options, {name} expiry: new rule's mean per contract",
                              "value": row["mean"].iloc[0], "p": row["p_up"].iloc[0]})
    s = out["stocks"]
    row = s[(s["ticker"] == PRIMARY) & (s["period"] == "2024-10 to 2026-09") & (s["rule"] == "new rule")
            & (s["horizon"] == HORIZON)]
    if len(row) and "p_up" in row and pd.notna(row["p_up"].iloc[0]):
        tests.append({"test": "SPY stock: new rule's move minus its slid copies (bps)",
                      "value": row["mean"].iloc[0] - row["slid"].iloc[0], "p": row["p_up"].iloc[0]})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = mr.bh_qvalues(tests["p"].to_numpy())
    out["tests"] = tests
    return out


# ---------------------------------------------------------------- report

fmt, money = gg.fmt, gg.money


def roles_text(roles):
    if not roles:
        return "none"
    return "; ".join(f"{n} ({r})" for n, r in roles.items())


def render_report(res):
    scr, roles = res["screen"], res["roles"]
    lines = []
    w = lines.append
    w("# Green Goose: which indicators help\n")
    w("Every condition below is known at 15:50: Green Goose's own pieces and new ones. Each was judged on the move "
      "from SPY's 15:50 price to the next open, over 2021-10 to 2024-09 only; the kept ones were combined into one "
      "rule, which was then checked once on 2024-10 to 2026-09 as stock moves and as the SPY options Green Goose buys "
      "(same contracts, version 1 exits, $0.02 a share per fill, per contract). *Edge* = the average move after the "
      "condition minus the average on every session (SPY: "
      f"{fmt(res['design_mean'])} bps in 2021-24, {fmt(res['check_mean'])} bps in 2024-26).\n")
    w(f"**Kept:** {roles_text(roles)}.\n")
    w("Rule: calls when call signals outnumber put signals and warnings together; puts when put signals outnumber "
      "call signals; otherwise no trade.\n")
    o = res["options"]
    if o is not None:
        w(f"## The check: SPY options, 2024-10 to 2026-09 ({PRIMARY_EXPIRY} expiry first)\n")
        rows = []
        for r in o.itertuples():
            if pd.isna(getattr(r, "mean", np.nan)):
                rows.append([r.expiry, r.rule, f"{int(r.trades):,}"] + ["n/a"] * 7)
                continue
            vs = (f"{money(r.minus_goose_lo)} to {money(r.minus_goose_hi)}"
                  if pd.notna(getattr(r, "minus_goose_lo", np.nan)) else "")
            rows.append([r.expiry, r.rule, f"{int(r.trades):,}", f"{r.win:.0f}%",
                         f"{money(r.avg_win)} / {money(r.avg_loss)}", f"{money(r.mean)} ({money(r.lo)} to {money(r.hi)})",
                         money(r.per_year), f"{money(r.worst)} / {money(-r.drawdown)}",
                         "n/a" if pd.isna(r.profit_factor) else f"{r.profit_factor:.2f}", vs])
        w(alerts.md_table(["Expiry", "Rule", "Trades", "Wins", "Avg win / loss", "Per contract (95%)",
                           "A year (1 contract)", "Worst / worst run", "Profit factor", "New rule minus Goose (95%)"],
                          rows) + "\n")
    if len(res["tests"]):
        w("**Tests (one-sided).**\n")
        rows = [[r.test, fmt(r.value, 2), f"{r.p:.3f}", f"{r.q:.3f}"] for r in res["tests"].itertuples()]
        w(alerts.md_table(["Test", "Value ($ per contract or bps)", "p", "q"], rows) + "\n")
    w("## The screen: every condition, SPY 2021-10 to 2024-09\n")
    w(f"p = two-sided, against the same condition slid to other dates; kept if p < {P_SCREEN}, at least {MIN_DAYS} "
      "sessions, and the same sign on QQQ and IWM. The last two columns are the 2024-26 check period, for reading.\n")
    rows = []
    for r in scr.itertuples():
        rows.append([r.condition, r.group, f"{r.sessions:,}", fmt(r.mean), f"{fmt(r.edge)}", f"{r.up:.0f}%",
                     f"{r.p:.3f}", fmt(r.edge_QQQ), fmt(r.edge_IWM), r.role, f"{r.check_sessions:,}",
                     fmt(r.check_edge)])
    w(alerts.md_table(["Condition", "From", "Sessions", "Avg move, bps", "Edge", "SPY up overnight", "p", "QQQ edge",
                       "IWM edge", "Role", "2024-26 sessions", "2024-26 edge"], rows) + "\n")
    s = res["stocks"]
    w("## Stock moves in the trade's direction (15:50 to the next open)\n")
    w("*Slid* = the same calls/puts sequence slid to other dates (average over every shift); p = share of slid copies "
      "doing at least as well. The 2021-24 rows for the new rule are in-sample (its conditions were chosen there).\n")
    rows = []
    for r in s[s["horizon"] == HORIZON].itertuples():
        if pd.isna(getattr(r, "mean", np.nan)):
            rows.append([r.ticker, r.period, r.rule, f"{int(r.trades):,}"] + ["n/a"] * 5)
            continue
        rows.append([r.ticker, r.period, r.rule, f"{int(r.trades):,}", f"{r.calls:.0f}%", f"{r.hit:.0f}%",
                     f"{fmt(r.mean)} ({fmt(r.lo)} to {fmt(r.hi)})", fmt(r.slid), f"{r.p_up:.3f}"])
    w(alerts.md_table(["Ticker", "Period", "Rule", "Trades", "Calls", "Right direction", "Move, bps (95%)", "Slid",
                       "p"], rows) + "\n")
    if o is not None:
        pc = res["per_condition"]
        w(f"## Each condition alone as SPY options, 2024-10 to 2026-09 ({PRIMARY_EXPIRY} expiry, for reading)\n")
        w("Calls (or puts) on every session the condition holds. Chosen on nothing: picking the best rows here would "
          "be fitting the check period.\n")
        rows = []
        for name in scr["condition"]:
            cells = [name]
            for side in ("calls", "puts"):
                r = pc[(pc["condition"] == name) & (pc["side"] == side)].iloc[0]
                cells += (["n/a"] * 2 if pd.isna(r.get("mean", np.nan)) else
                          [f"{int(r['trades'])}, {r['win']:.0f}% wins", money(r["mean"])])
            rows.append(cells)
        w(alerts.md_table(["Condition", "Calls: trades, wins", "Calls per contract", "Puts: trades, wins",
                           "Puts per contract"], rows) + "\n")
        for name, t in res["option_trades"].items():
            w(f"- {name} expiry, contracts: " + ", ".join(f"{k} {v:,}" for k, v in t["status"].value_counts().items())
              + ".")
        w("")
    w("## Notes\n")
    w(f"- {res['design_days']:,} SPY sessions in the screen and {res['check_days']:,} in the check (sessions with every "
      "indicator and a next-morning price).")
    w("- Conditions and their tests were fixed before any result was seen (see goose_indicators.py). The check "
      "period is not untouched: Green Goose itself, and earlier daily SPY studies, were already measured on it, and "
      "splitting each indicator into a call and a put side was prompted partly by Green Goose's calls doing better "
      "than its puts over 2021-26.")
    w("- Stocks: a signal study from the 15:50 price. Options: traded prices (no quotes on the plan), per contract, no "
      "commissions.")
    return "\n".join(lines) + "\n"


def plot_screen(res, path):
    scr = res["screen"]
    colors = {"call": SERIES[0], "put": SERIES[1], "warning": INK_2, "not used": GRID}
    fig = plt.figure(figsize=(8.6, 8.4), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.40, 0.07, 0.47, 0.83])
    y = np.arange(len(scr))[::-1]
    ax.barh(y, scr["edge"], color=[colors[r] for r in scr["role"]], height=0.7)
    ax.axvline(0, color=BASELINE, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(scr["condition"], fontsize=6.8, color=INK_2)
    for yy, r in zip(y, scr.itertuples()):
        ax.text(1.02, yy, f"p {r.p:.2f}  n {r.sessions}", transform=ax.get_yaxis_transform(), fontsize=5.8,
                color=MUTED, va="center", ha="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("Edge: average move to the next open after the condition, minus every session's (bps)",
                  color=INK_2, fontsize=7)
    fig.text(0.02, 0.975, "Which 15:50 conditions told SPY's overnight move? (2021-10 to 2024-09)", color=INK,
             fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.95, "Blue = kept as a call signal, orange = put signal, dark = warning, light = not kept.",
             color=INK_2, fontsize=7.5, va="top")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_options(res, path):
    fig = plt.figure(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax = sds._axes(fig, [0.09, 0.18, 0.86, 0.62])
    styles = {"new rule": (SERIES[0], "-", 1.4), "Green Goose v1": (SERIES[1], "-", 1.1),
              "always calls": (MUTED, (0, (3, 2)), 1.0)}
    for label, (color, ls, lw) in styles.items():
        pnl = res["pnls"].get((PRIMARY_EXPIRY, label))
        if pnl is not None and len(pnl):
            ax.plot(pnl.index, pnl.cumsum(), color=color, ls=ls, lw=lw, label=label)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("Cumulative $ per contract", color=INK_2, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(0.02, 0.97, f"SPY options, {PRIMARY_EXPIRY} expiry: the new rule against Green Goose (2024-10 to 2026-09)",
             color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"One contract a trade, version 1 exits, ${SLIP:.2f} a share per fill.", color=INK_2,
             fontsize=7.5, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Which 15:50 indicators help Green Goose, chosen on 2021-24, checked on "
                                            "2024-26 as SPY options.")
    p.add_argument("--out", default="output/goose_indicators")
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
                     mr.load_dividends(tk, first, last, args.cache_dir, args.refresh))
                for tk in (PRIMARY, *CONFIRM, VIX)}
    load = None if args.skip_options else asp.option_loader(args.cache_dir, args.refresh)
    res = run_study(data, sessions, load, timings=timings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res["screen"].to_csv(out / "screen.csv", index=False)
    res["stocks"].to_csv(out / "stocks.csv", index=False)
    res["tests"].to_csv(out / "tests.csv", index=False)
    plot_screen(res, out / "screen.png")
    if res["options"] is not None:
        res["options"].to_csv(out / "options.csv", index=False)
        res["per_condition"].to_csv(out / "per_condition.csv", index=False)
        plot_options(res, out / "options.png")
    (out / "report.md").write_text(render_report(res))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
