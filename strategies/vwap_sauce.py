"""VWAP + Sauce: a band-extension reversal on 2-minute bars, with a small trade simulator.

Built from one trader's description. Each rule the trader did not state is a parameter in
DEFAULTS with the brief's default, and IMPLEMENTATION lists the choices this code had to
make where the brief is silent. Nothing else filters or adds trades.

Indicators, all known when a bar completes (its close), never earlier:
- VWAP anchored at the open of the session `vwap_lookback_sessions - 1` sessions ago and
  re-anchored every session: typical price (H+L+C)/3 x volume over usable 2-minute bars.
  sigma is the volume-weighted standard deviation of typical price around that VWAP over
  the same window. Bands are VWAP +/- k sigma for k = 1, 1.5, 2 (U1, U1.5, U2 above and
  L1, L1.5, L2 below).
- FAST = EMA 8 and SLOW = EMA 48 of close (TA-Lib, continuous across sessions).

Setup A on the lower band (long; the upper band mirrors it as a short):
- ARM when FAST closes below L2. setup_extreme is the lowest low while armed.
- SLOW "goes parallel" when 0 <= SLOW - L2 <= parallel_distance_sigma x sigma and the
  least-squares slope of SLOW - L2 over the last parallel_lookback bars is >= 0 or smaller
  in size than slope_threshold x sigma per bar. SLOW closing below L2 invalidates the setup.
- Standard entry: the first close with FAST back above L2, if SLOW went parallel (when
  required), goes long at the next bar's open.
- Target: the moving L1.5 band (L1 with the extended target). Stops: FAST closing back
  below L2 (filled at the next open), price trading below setup_extreme, the session close.

A-continuation on the lower band (a short): while FAST < L2, a close above FAST and then a
later close below FAST go short at the next open. It exits when FAST closes back above L2.

B, VWAP to VWAP: FAST closing across a ladder level (L2, L1.5, L1, VWAP, U1, U1.5, U2) sets
a bias toward the next level. The first later bar that touches FAST and closes back in the
bias direction enters at the next open, with that next level as the target. FAST closing
back across the crossed level is a head fake: no trade, or an exit at the next open.

Fills: a decision made at a close fills at the next usable bar's open in the same session.
Targets and price stops fill inside a bar against the level known at the previous close:
at that level, or at the open when the bar opens beyond it. A bar that touches both is
ambiguous and counted as the stop. P&L is per share of the underlying and gross.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BAR_MINUTES = 2
FAST_PERIOD, SLOW_PERIOD = 8, 48
BAND_SIGMAS = (1.0, 1.5, 2.0)
EXTENDED_TARGET = 1.0  # L1 / U1
LADDER = np.array([-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0])  # sigma multiples, bottom to top
LADDER_NAMES = ("L2", "L1.5", "L1", "VWAP", "U1", "U1.5", "U2")
CENTER = 3  # index of VWAP in the ladder
SETUPS = ("A", "A_cont", "B")
PRIORITY = {s: n for n, s in enumerate(SETUPS)}  # same-bar tie-break: Setup A first
SIDE_NAMES = {1: "long", -1: "short"}
BAND_NAMES = {1: "lower", -1: "upper"}
NAT = np.iinfo(np.int64).min

# Every [ASSUMPTION] in the source brief, with the brief's default.
DEFAULTS = {
    "rth_only": True,  # regular trading hours only; extended hours are not supported
    "instrument": "underlying",  # the only instrument simulated; options mapping is left for later
    "vwap_lookback_sessions": 3,  # current session plus the 2 before it; test 3 and 4
    "parallel_distance_sigma": 0.25,
    "parallel_lookback": 5,
    "slope_threshold": 0.02,  # sigma per bar
    "require_slow_parallel": True,
    "use_trendline_confirm": False,
    "trendline_lookback": 10,
    "allow_fade_entry": False,  # when enabled, only while the session's realized P&L > 0
    "target_level": 1.5,
    "allow_extended_target": False,  # target the 1-sigma band instead
    "structure_stop": True,
    "price_stop": True,
    "time_stop": True,  # flat at the session's last bar; off = positions carry overnight
    "allow_reentry": False,  # off = at most one trade per setup instance
    "max_reentries": 2,
    "enable_continuation": False,
    "enable_vwap_to_vwap": False,
}

# Not from the brief: two of the choices below made switchable, and an experiment knob. The
# defaults reproduce the brief.
CHOICES = {
    "latch_slow_parallel": True,  # off: SLOW must be parallel on the entry bar itself
    "ambiguous_fill": "stop",  # a bar touching both the price stop and the target: "stop" or "target"
    "stop_buffer_sigma": 0.0,  # widen the price stop by this many sigma (sigma at the entry signal)
}

# Choices the brief leaves open, recorded in settings.json and the README.
IMPLEMENTATION = {
    "bars": "2-minute bars anchored to each session open, built from Massive one-minute bars; a bar is used when "
            ">= min_coverage of its minutes exist (default 0.5: Massive emits no bar for a minute without trades, "
            "so one traded minute is what a chart shows); dropped bars are counted, never filled",
    "vwap_window": "whole XNYS sessions; NaN when an earlier session in the window has no usable bars (never "
                   "substituted); sigma is the population, volume-weighted standard deviation",
    "intraday": "setups are intraday: every unfinished setup ends at the session's last usable bar, and a signal on "
                "that bar has no next open in the session, so it does not trade",
    "arming": "a side arms when FAST closes outside its 2-sigma band while that side has no live setup; after a setup "
              "ends, FAST must close back inside the band before that side arms again (one setup per excursion); "
              "each session starts fresh",
    "latched": "slow_parallel and the trendline confirmation stay true for the setup once seen while armed; "
               "slow_parallel_time is the first such bar (latch_slow_parallel=False: SLOW must be parallel on the "
               "entry signal bar itself)",
    "slope": "least-squares slope per bar over the last parallel_lookback bars, which must be contiguous and in one "
             "session",
    "trendline": "a line fitted to the last trendline_lookback SLOW values (contiguous, one session); confirmation = "
                 "the previous close on or below the line and this close above it (long; mirrored for short)",
    "standard_entry": "decided on the first close with FAST back inside the band: without slow_parallel (when "
                      "required) or the trendline confirmation (when enabled) by then, the setup ends without a trade",
    "fade_entry": "while armed with FAST still outside: slow_parallel seen, trendline confirmed if enabled, and the "
                  "realized P&L of trades closed earlier in the session > 0",
    "structure_stop": "FAST closes outside the band after closing inside at or after the entry signal (so a fade or "
                      "re-entry taken with FAST outside needs FAST to come back in first)",
    "price_stop": "low below setup_extreme (long) or high above it (short); setup_extreme stops updating at the "
                  "entry signal; stop_buffer_sigma moves the stop that many sigma further away, fixed at the first "
                  "entry of the setup",
    "fills": "close-based entries and exits fill at the next usable bar's open; targets and price stops fill inside "
             "the bar against the level known at the previous close, at that level or at the open when the bar "
             "opens beyond it; stop and target in one bar = ambiguous, filled as the stop (ambiguous_fill='target': "
             "as the target); time stop = last bar's close",
    "reentry": "only after a structure stop (a price stop, target or session end finishes the setup); while flat the "
               "setup ends if SLOW crosses the band, price trades beyond setup_extreme or the target is touched; "
               "signal = low <= FAST < close (long), filled at the next open",
    "continuation": "one setup per FAST excursion outside the band, independent of Setup A's state; the pullback "
                    "close may be on any bar of the excursion; Setup A's price stop (setup_extreme) lies on the "
                    "trade's favorable side, so its price stop is the extreme of the latest pullback bar instead (the "
                    "high of the bar that closed above FAST, for a short), plus stop_buffer_sigma; exits when FAST "
                    "closes back inside the band (logged as structure_stop), on that price stop or on the time stop",
    "vwap_to_vwap": "cross up = previous same-session FAST <= level and this FAST > level (mirrored down); several "
                    "levels crossed at once count as the outermost; a new cross replaces a bias still waiting for its "
                    "pullback; a cross beyond U2 or L2 has no next level and makes no setup; pullback = low <= FAST < "
                    "close (long) on a bar after the cross; only the first pullback may enter",
    "one_position": "one position per symbol; an open position blocks new entries (counted as blocked); on the same "
                    "bar Setup A beats A-continuation beats B; an exit and an entry can fill at the same open, which "
                    "is how the continuation hands off to the reversal",
    "pnl": "one share of the underlying: side x (exit - entry), gross of costs and slippage; sigma units = that / "
           "sigma at the entry signal bar; not options P&L",
}

UTC = "datetime64[ns, UTC]"
LOG_COLUMNS = {
    "setup": "str", "band": "str", "side": "str", "session": "datetime64[ns]", "instance": int, "arm_time": UTC,
    "setup_extreme": float, "slow_parallel_time": UTC, "trendline_confirm_time": UTC, "pullback_time": UTC,
    "level_crossed": "str", "center_cross": "boolean", "invalidation_time": UTC, "invalidation_reason": "str",
    "outcome": "str", "n_blocked": int, "n_trades": int, "trade_no": "Int64", "signal_time": UTC, "entry_time": UTC,
    "entry_price": float, "entry_type": "str", "target_level": "str", "target_price_at_entry": float,
    "stop_price": float, "sigma_at_entry": float, "exit_time": UTC, "exit_price": float, "exit_reason": "str", "ambiguous": "boolean",
    "pnl_usd": float, "pnl_sigma": float, "pnl_pct": float, "bars_held": "Int64",
}
TIME_FIELDS = ("arm", "slow_parallel", "trendline_confirm", "pullback", "invalidation")


def _ns(times):
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def band_name(side, k):
    """'L1.5' for the lower band (side +1), 'U1.5' for the upper band (side -1); 'VWAP' for k = 0."""
    return "VWAP" if k == 0 else f"{'L' if side > 0 else 'U'}{k:g}"


def anchored_vwap(bars, sessions, lookback):
    """VWAP and sigma at each bar's completion, over its session and the `lookback - 1` sessions before it.

    Sums run over the usable bars given. A value is NaN until `lookback - 1` earlier
    sessions exist in `sessions`, or when any of them has no usable bars.
    """
    tp = bars[["high", "low", "close"]].to_numpy(float).mean(axis=1)
    v = bars["volume"].to_numpy(float)
    sums = pd.DataFrame({"v": v, "pv": tp * v, "pv2": tp * tp * v})
    pos = sessions.index.get_indexer(bars["session"])
    per_session = sums.groupby(pos).sum().reindex(range(len(sessions)), fill_value=0.0)
    n = lookback - 1
    prior = per_session * 0.0
    if n:
        complete = (per_session["v"] > 0).astype(float).rolling(n).min().shift(1) == 1
        prior = per_session.rolling(n).sum().shift(1).where(complete)
    total = sums.groupby(pos).cumsum().to_numpy() + prior.to_numpy()[pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = total[:, 1] / total[:, 0]
        var = total[:, 2] / total[:, 0] - vwap * vwap
    return vwap, np.sqrt(np.clip(var, 0.0, None))


def add_indicators(bars, sessions, lookback):
    """The feature columns the simulator reads: fast, slow (from build_features' EMAs), vwap, sigma."""
    vwap, sigma = anchored_vwap(bars, sessions, lookback)
    return bars.rename(columns={f"ema_{FAST_PERIOD}": "fast", f"ema_{SLOW_PERIOD}": "slow"}).assign(
        vwap=vwap, sigma=sigma)


def contiguous_runs(bars):
    """Number of contiguous same-session bars ending at each row (1 after a session start or a dropped bar)."""
    start, end = _ns(bars["bar_start"]), _ns(bars["bar_end"])
    sess = bars["session"].to_numpy()
    linked = np.r_[False, (sess[1:] == sess[:-1]) & (start[1:] == end[:-1])]
    idx = np.arange(len(bars))
    return idx - np.maximum.accumulate(np.where(linked, 0, idx)) + 1


def rolling_line(y, n, runs):
    """Least-squares (slope per bar, fitted value at the last bar) over each row's last n values.

    NaN unless those n rows are one contiguous, same-session run.
    """
    slope, last = np.full(len(y), np.nan), np.full(len(y), np.nan)
    if len(y) >= n:
        w = sliding_window_view(np.asarray(y, float), n)
        x = np.arange(n) - (n - 1) / 2
        slope[n - 1:] = w @ x / (x @ x)
        last[n - 1:] = w.mean(axis=1) + slope[n - 1:] * (n - 1) / 2
    ok = runs >= n
    return np.where(ok, slope, np.nan), np.where(ok, last, np.nan)


def validate(p):
    unknown = set(p) - set(DEFAULTS) - set(CHOICES)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    if not p["rth_only"]:
        raise ValueError("rth_only=False is not supported: only regular-session bars are loaded")
    if p["instrument"] != "underlying":
        raise ValueError("only instrument='underlying' is simulated; no options mapping exists yet")
    if p["vwap_lookback_sessions"] < 1 or p["parallel_lookback"] < 2 or p["trendline_lookback"] < 2:
        raise ValueError("need vwap_lookback_sessions >= 1, parallel_lookback >= 2 and trendline_lookback >= 2")
    if p["ambiguous_fill"] not in ("stop", "target"):
        raise ValueError("ambiguous_fill must be 'stop' or 'target'")
    if not 0 <= p["target_level"] < 2 or p["max_reentries"] < 0 or p["stop_buffer_sigma"] < 0:
        raise ValueError("need 0 <= target_level < 2 (VWAP up to a band inside L2/U2), max_reentries >= 0 and "
                         "stop_buffer_sigma >= 0")


def simulate(bars, **params):
    """Run the enabled setups over the study bars with one position at a time; return the setup log.

    `bars`: usable bars in time order with session, bar_start, bar_end, open, high, low,
    close, fast, slow, vwap, sigma and in_study. Only in_study rows are simulated; a
    warm-up row only supplies the previous bar's levels. One log row per setup instance
    and trade: an instance without a trade has a single row with empty trade columns.
    `vwap_lookback_sessions` is informational here: it was applied when vwap/sigma were built.
    """
    p = {**DEFAULTS, **CHOICES, **params}
    validate(p)
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    fast, slow, vwap, sigma = (bars[k].to_numpy(float) for k in ("fast", "slow", "vwap", "sigma"))
    start, end = _ns(bars["bar_start"]), _ns(bars["bar_end"])
    sess = bars["session"].to_numpy()
    n = len(bars)
    runs = contiguous_runs(bars)
    k_target = EXTENDED_TARGET if p["allow_extended_target"] else p["target_level"]
    use_tl, need_parallel = p["use_trendline_confirm"], p["require_slow_parallel"]

    # Everything that is a pure function of one bar's features, per band side s: +1 = lower
    # band (Setup A long), -1 = upper band (short). NaN inputs make every comparison False.
    sides = (1, -1)
    with np.errstate(invalid="ignore"):
        band2 = {s: vwap - s * 2 * sigma for s in sides}
        outside = {s: s * (band2[s] - fast) > 0 for s in sides}  # FAST below L2 / above U2
        gap = {s: s * (slow - band2[s]) for s in sides}  # SLOW's distance inside the band
        crossed = {s: gap[s] < 0 for s in sides}
        parallel = {}
        for s in sides:
            slope = rolling_line(gap[s], p["parallel_lookback"], runs)[0]
            flat = (slope >= 0) | (np.abs(slope) < p["slope_threshold"] * sigma)
            parallel[s] = (gap[s] >= 0) & (gap[s] <= p["parallel_distance_sigma"] * sigma) & flat
        confirm = {s: np.zeros(n, dtype=bool) for s in sides}
        if use_tl:
            slope, now = rolling_line(slow, p["trendline_lookback"], runs)
            before = now - slope  # the same line one bar earlier
            prev_c = np.r_[np.nan, c[:-1]]
            confirm = {s: (s * (c - now) > 0) & (s * (prev_c - before) <= 0) for s in sides}
        target = {s: vwap - s * k_target * sigma for s in sides}
        ladder = vwap[:, None] + sigma[:, None] * LADDER
        same = np.r_[False, sess[1:] == sess[:-1]]
        f_prev = np.r_[np.nan, fast[:-1]][:, None]
        l_prev = np.vstack([np.full((1, len(LADDER)), np.nan), ladder[:-1]])
        up = same[:, None] & (f_prev <= l_prev) & (fast[:, None] > ladder)
        down = same[:, None] & (f_prev >= l_prev) & (fast[:, None] < ladder)

    instances = []
    A, C = {1: None, -1: None}, {1: None, -1: None}
    rearm = {1: True, -1: True}
    B = pos = pending_exit = pending_entry = None
    day_pnl = 0.0

    def new(setup, band, side, i, **extra):
        inst = {"setup": setup, "band": band, "side": side, "session": sess[i], "instance": len(instances),
                "arm": i, "state": "armed", "outcome": None, "n_blocked": 0, "trades": [], "extreme": np.nan,
                "stop": None,
                **{f: None for f in TIME_FIELDS[1:]}, "invalidation_reason": None, "level": None, **extra}
        instances.append(inst)
        return inst

    def finish(inst, outcome):
        nonlocal B
        inst["state"], inst["outcome"] = "done", outcome
        if inst["setup"] == "A" and A[inst["band"]] is inst:
            A[inst["band"]] = None
            rearm[inst["band"]] = False
        elif inst["setup"] == "A_cont" and C[inst["band"]] is inst:
            C[inst["band"]] = None
        elif B is inst:
            B = None

    def signal(setup, side, kind, inst, i):
        return {"setup": setup, "side": side, "kind": kind, "inst": inst, "idx": i}

    def target_at(position, k):
        if position["setup"] == "A":
            return target[position["band"]][k]
        return ladder[k, position["target_j"]] if position["setup"] == "B" else np.nan

    def open_position(sig, i):
        nonlocal pos
        inst, k, s = sig["inst"], sig["idx"], sig["side"]
        inst["state"] = "entered"
        if inst["stop"] is None:  # fixed at the setup's first entry; a re-entry keeps it
            base = (inst["extreme"] if inst["setup"] == "A" else
                    (l if s > 0 else h)[inst["pullback"]] if inst["setup"] == "A_cont" else np.nan)
            inst["stop"] = base - s * p["stop_buffer_sigma"] * sigma[k] if p["price_stop"] else np.nan
        pos = {**sig, "band": inst["band"], "entry": i, "price": o[i], "sigma": sigma[k], "stop": inst["stop"],
               "fast_inside": inst["setup"] == "A" and not outside[inst["band"]][k],
               "level": inst.get("level_j"), "target_j": inst.get("target_j")}
        pos["target_name"] = (band_name(inst["band"], k_target) if inst["setup"] == "A" else
                              LADDER_NAMES[pos["target_j"]] if inst["setup"] == "B" else None)
        pos["target_price"] = target_at(pos, k)

    def close_position(i, price, at_open, reason, ambiguous=False):
        nonlocal pos, day_pnl
        inst, s = pos["inst"], pos["side"]
        pnl = s * (price - pos["price"]) + 0.0  # + 0.0: a flat short logs 0.0, not -0.0
        day_pnl += pnl
        inst["trades"].append({
            "signal": pos["idx"], "entry": pos["entry"], "entry_price": pos["price"], "entry_type": pos["kind"],
            "target_level": pos["target_name"], "target_price_at_entry": pos["target_price"],
            "stop_price": pos["stop"], "sigma_at_entry": pos["sigma"], "exit_ns": start[i] if at_open else end[i], "exit_price": price,
            "exit_reason": reason, "ambiguous": ambiguous, "pnl_usd": pnl, "pnl_sigma": pnl / pos["sigma"],
            "pnl_pct": pnl / pos["price"] * 100, "bars_held": i - pos["entry"] + (0 if at_open else 1)})
        pos = None
        if (inst["setup"] == "A" and p["allow_reentry"] and reason == "structure_stop"
                and len(inst["trades"]) <= p["max_reentries"]):
            inst["state"] = "wait_reentry"
        else:
            finish(inst, "exited")

    def intrabar(i):
        """Price stop and target against levels known at the previous close; a tie goes to ambiguous_fill."""
        s, stop, t = pos["side"], pos["stop"], target_at(pos, i - 1)
        adverse, favorable = (l[i], h[i]) if s > 0 else (h[i], l[i])
        if s * (o[i] - stop) < 0:
            close_position(i, o[i], True, "price_stop")
        elif s * (o[i] - t) >= 0:
            close_position(i, o[i], True, "target")
        else:
            hit_stop, hit_target = s * (adverse - stop) < 0, s * (favorable - t) >= 0
            both = bool(hit_stop and hit_target)
            if hit_stop and not (both and p["ambiguous_fill"] == "target"):
                close_position(i, stop, False, "price_stop", ambiguous=both)
            elif hit_target:
                close_position(i, t, False, "target", ambiguous=both)

    def exit_signal(i):
        """Close-based exit for the open position, filled at the next open."""
        setup, band = pos["setup"], pos["band"]
        if setup == "A":
            if not outside[band][i]:
                pos["fast_inside"] = True
            elif p["structure_stop"] and pos["fast_inside"]:
                return "structure_stop"
        elif setup == "A_cont":
            if not outside[band][i]:
                return "structure_stop"
        elif pos["side"] * (fast[i] - ladder[i, pos["level"]]) < 0:  # B head fake
            return "structure_stop"
        return None

    def invalidate(inst, i, reason, outcome):
        inst["invalidation"], inst["invalidation_reason"] = i, reason
        finish(inst, outcome)

    def step_a(s, i, signals):
        inst = A[s]
        if inst is None and outside[s][i] and rearm[s]:
            inst = A[s] = new("A", s, s, i, extreme=l[i] if s > 0 else h[i])
            rearm[s] = False
        if inst is None:
            return
        adverse = l[i] if s > 0 else h[i]
        if inst["state"] == "armed":
            if s * (inst["extreme"] - adverse) > 0:
                inst["extreme"] = adverse
            if crossed[s][i]:
                return invalidate(inst, i, "slow_crossed_band", "invalidated")
            if parallel[s][i] and inst["slow_parallel"] is None:
                inst["slow_parallel"] = i
            if confirm[s][i] and inst["trendline_confirm"] is None:
                inst["trendline_confirm"] = i
            parallel_now = inst["slow_parallel"] is not None if p["latch_slow_parallel"] else parallel[s][i]
            ok_parallel = not need_parallel or parallel_now
            ok_line = not use_tl or inst["trendline_confirm"] is not None
            if not outside[s][i]:
                if ok_parallel and ok_line:
                    signals.append(signal("A", s, "standard", inst, i))
                else:
                    finish(inst, "no_parallel" if not ok_parallel else "no_trendline")
            elif p["allow_fade_entry"] and parallel_now and ok_line and day_pnl > 0:
                signals.append(signal("A", s, "fade", inst, i))
        elif inst["state"] == "wait_reentry":
            favorable = h[i] if s > 0 else l[i]
            if crossed[s][i]:
                invalidate(inst, i, "slow_crossed_band", "invalidated")
            elif s * (adverse - inst["stop"]) < 0:
                finish(inst, "extreme_broken")
            elif s * (favorable - target[s][i - 1]) >= 0:
                finish(inst, "target_hit_while_flat")
            elif s * (c[i] - fast[i]) > 0 and s * (adverse - fast[i]) <= 0:
                signals.append(signal("A", s, "reentry", inst, i))

    def step_cont(s, i, signals):
        inst = C[s]
        if inst is None:
            if not outside[s][i]:
                return
            inst = C[s] = new("A_cont", s, -s, i)
        if inst["state"] != "armed":
            return
        if not outside[s][i]:
            return finish(inst, "excursion_ended")
        with_move = -s * (c[i] - fast[i])  # > 0: closed on the move's side of FAST
        if inst["pullback"] is not None and with_move > 0:
            signals.append(signal("A_cont", -s, "standard", inst, i))
        elif with_move < 0:
            inst["pullback"] = i

    def step_b(i, signals):
        nonlocal B
        inst = B
        if inst is not None and inst["side"] * (fast[i] - ladder[i, inst["level_j"]]) < 0:
            invalidate(inst, i, "fast_back_across_level", "head_fake")
        ups, downs = np.flatnonzero(up[i]), np.flatnonzero(down[i])
        if ups.size or downs.size:
            j, d = (ups[-1], 1) if ups.size else (downs[0], -1)
            if B is not None:
                finish(B, "replaced")
            if 0 <= j + d < len(LADDER):
                B = new("B", 0, d, i, level_j=j, target_j=j + d, level=LADDER_NAMES[j])
        elif B is not None:  # never the cross bar itself: that bar took the branch above
            d = B["side"]
            adverse = l[i] if d > 0 else h[i]
            if d * (c[i] - fast[i]) > 0 and d * (adverse - fast[i]) <= 0:
                signals.append(signal("B", d, "standard", B, i))

    def blocked(sig):
        inst = sig["inst"]
        inst["n_blocked"] += 1
        if (sig["setup"] == "A" and sig["kind"] == "standard") or sig["setup"] == "B":
            finish(inst, "blocked")  # the cross back inside / the first pullback happens once
        elif sig["setup"] == "A_cont":
            inst["pullback"] = None

    def accept(sig):
        nonlocal B
        sig["inst"]["state"] = "pending"
        if sig["setup"] == "B":
            B = None  # the bias became a trade; a new cross may start another

    study = np.flatnonzero(bars["in_study"].to_numpy(bool))
    for i in study:
        last = i == n - 1 or sess[i + 1] != sess[i]
        if i == 0 or sess[i - 1] != sess[i]:
            day_pnl = 0.0
        # 1. Orders from the previous close fill at this open: the exit first, then the entry.
        if pending_exit is not None:
            close_position(i, o[i], True, pending_exit)
            pending_exit = None
        if pending_entry is not None:
            open_position(pending_entry, i)
            pending_entry = None
        # 2. Price stop and target inside this bar.
        if pos is not None:
            intrabar(i)
        # 3. Setup states at this close.
        signals = []
        for s in sides:
            step_a(s, i, signals)
        if p["enable_continuation"]:
            for s in sides:
                step_cont(s, i, signals)
        if p["enable_vwap_to_vwap"]:
            step_b(i, signals)
        # 4. Close-based exit, then at most one entry for the next open.
        if pos is not None:
            pending_exit = exit_signal(i)
        free = pos is None or pending_exit is not None
        for rank, sig in enumerate(sorted(signals, key=lambda x: PRIORITY[x["setup"]])):
            if rank == 0 and free and not last:
                accept(sig)
                pending_entry = sig
            elif not last:
                blocked(sig)
        for s in sides:
            if not outside[s][i]:
                rearm[s] = True
        # 5. Session end: the time stop, and every setup that is not holding the position ends.
        if last:
            if pos is not None and (p["time_stop"] or i == study[-1]):
                close_position(i, c[i], False, "time_stop" if p["time_stop"] else "end_of_data")
                pending_exit = None
            holding = pos["inst"] if pos is not None else None
            for inst in [x for x in (*A.values(), *C.values(), B) if x is not None and x is not holding]:
                finish(inst, "session_end")
            rearm[1] = rearm[-1] = True
    return setup_log(instances, bars)


def setup_log(instances, bars):
    """One row per instance and trade, with timestamps (UTC) in place of bar positions."""
    end = _ns(bars["bar_end"])
    start = _ns(bars["bar_start"])
    rows = []
    for inst in instances:
        base = {"setup": inst["setup"], "band": BAND_NAMES.get(inst["band"]), "side": SIDE_NAMES[inst["side"]],
                "session": inst["session"], "instance": inst["instance"],
                "setup_extreme": inst["extreme"] if inst["setup"] == "A" else np.nan,
                **{f"{f}_time": NAT if inst[f] is None else end[inst[f]] for f in TIME_FIELDS},
                "level_crossed": inst["level"], "center_cross": inst.get("level_j") == CENTER if inst["level"] else None,
                "invalidation_reason": inst["invalidation_reason"], "outcome": inst["outcome"],
                "n_blocked": inst["n_blocked"], "n_trades": len(inst["trades"]),
                "signal_time": NAT, "entry_time": NAT, "exit_time": NAT}  # int64 until converted below
        for no, t in enumerate(inst["trades"], 1):
            rows.append({**base, "trade_no": no, "signal_time": end[t["signal"]], "entry_time": start[t["entry"]],
                         **{k: v for k, v in t.items() if k not in ("signal", "entry", "exit_ns")},
                         "exit_time": t["exit_ns"]})
        if not inst["trades"]:
            rows.append(base)
    log = pd.DataFrame(rows, columns=list(LOG_COLUMNS))
    for k in (c for c, d in LOG_COLUMNS.items() if d == UTC):
        log[k] = pd.to_datetime(log[k].to_numpy(np.int64), utc=True).as_unit("ns")  # int64 min -> NaT
    return log.astype(LOG_COLUMNS)


def _max_drawdown(pnl):
    equity = np.r_[0.0, np.cumsum(pnl)]
    return float((np.maximum.accumulate(equity) - equity).max())


def trade_stats(trades):
    """Win rate, average win/loss, expectancy and max drawdown in $ and sigma (trades in entry order).

    A statistic with no trades behind it is NaN, never 0.
    """
    t = trades.sort_values("entry_time", kind="stable")
    usd, sig = t["pnl_usd"].to_numpy(float), t["pnl_sigma"].to_numpy(float)
    n = len(t)
    nan = np.nan
    out = {"trades": n, "wins": int((usd > 0).sum()), "losses": int((usd < 0).sum()),
           "win_rate": (usd > 0).mean() if n else nan}
    for unit, v in (("usd", usd), ("sigma", sig)):
        out |= {f"avg_win_{unit}": v[usd > 0].mean() if (usd > 0).any() else nan,
                f"avg_loss_{unit}": v[usd < 0].mean() if (usd < 0).any() else nan,
                f"expectancy_{unit}": v.mean() if n else nan, f"total_{unit}": v.sum() if n else nan,
                f"max_drawdown_{unit}": _max_drawdown(v) if n else nan}
    out |= {f"n_exit_{r}": int((t["exit_reason"] == r).sum())
            for r in ("target", "structure_stop", "price_stop", "time_stop", "end_of_data")}
    out |= {"n_ambiguous": int(t["ambiguous"].fillna(False).sum()),
            "n_exit_at_entry_open": int((t["bars_held"] == 0).sum()),
            **{f"n_entry_{k}": int((t["entry_type"] == k).sum()) for k in ("standard", "fade", "reentry")},
            "median_bars_held": t["bars_held"].median() if n else nan}
    return out


def summarize(log, months):
    """One row per setup and side (plus both sides; B also split into center and off-center crosses).

    `months` = calendar months covered by the study sessions, for the per-month rates.
    """
    rows = []
    is_b, center = log["setup"] == "B", log["center_cross"].fillna(False).astype(bool)
    groups = [("A", "all", log["setup"] == "A"), ("A_cont", "all", log["setup"] == "A_cont"), ("B", "all", is_b),
              ("B", "center", is_b & center), ("B", "off_center", is_b & ~center)]
    for setup, subset, in_group in groups:
        for side in ("both", "long", "short"):
            g = log[in_group & ((log["side"] == side) if side != "both" else True)]
            inst = g.drop_duplicates("instance")
            trades = g[g["trade_no"].notna()]
            before_entry = inst["n_trades"] == 0
            row = {"setup": setup, "subset": subset, "side": side, "setups": len(inst),
                   "setups_per_month": len(inst) / months, "trades_per_month": len(trades) / months,
                   **trade_stats(trades),
                   "n_invalidated_slow_cross": int((before_entry & (inst["invalidation_reason"] ==
                                                                    "slow_crossed_band")).sum()),
                   "n_head_fake": int((before_entry & (inst["outcome"] == "head_fake")).sum()),
                   **{f"n_{k}": int((inst["outcome"] == k).sum()) for k in ("no_parallel", "no_trendline",
                                                                             "excursion_ended", "blocked",
                                                                             "session_end")}}
            row["pct_invalidated_slow_cross"] = (row["n_invalidated_slow_cross"] / len(inst) * 100
                                                 if setup == "A" and len(inst) else np.nan)
            row["pct_head_fake"] = row["n_head_fake"] / len(inst) * 100 if setup == "B" and len(inst) else np.nan
            rows.append(row)
    return rows


def monthly(log):
    """Setups, trades, wins and P&L per setup and calendar month (months without setups are absent)."""
    if log.empty:
        return pd.DataFrame(columns=["setup", "month", "setups", "trades", "wins", "pnl_usd", "pnl_sigma"])
    g = log.assign(month=log["session"].dt.strftime("%Y-%m"), win=log["pnl_usd"] > 0,
                   traded=log["trade_no"].notna()).groupby(["setup", "month"], sort=True)
    return pd.DataFrame({"setups": g["instance"].nunique(), "trades": g["traded"].sum(), "wins": g["win"].sum(),
                         "pnl_usd": g["pnl_usd"].sum(min_count=1),
                         "pnl_sigma": g["pnl_sigma"].sum(min_count=1)}).reset_index()
