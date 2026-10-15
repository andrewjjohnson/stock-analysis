"""How a candidate strategy sits next to the existing ones: daily P&L of SPY dip buying and of the TSLA breakouts,
correlations with a candidate's daily P&L, and a one-position-at-a-time account view.

Shared by the edge studies; no CLI. Daily returns are fractions of the account on each session (0 on sessions
without a position).

- SPY dip buying (`spy_dip_daily`): meanrev's rule, as in trend_etfs.dip_positions: buy at 15:50 on the third
  down session in a row, sell at 15:50 on the first up session (5 at most), one position at a time, on
  dividend-adjusted 15:50 prices, 2 bps a round trip.
- TSLA breakouts (`tsla_breakout_daily`): the two keeper rules' share trades (long through yesterday's high, short
  through yesterday's low) from breakout_check.py's saved trades for the period, net of 1 bp per side, whole
  account per trade.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import features
import meanrev as mr
import trend_etfs as te

TSLA_RULES = ("yesterday's high", "yesterday's low (check)")


def spy_dip_daily(index, cache_dir="data/cache", refresh=False, cost_bps=2.0):
    """Daily return of SPY dip buying on `index` (session dates). Computed over meanrev's standard five years (the
    cached SPY file); positions are causal, so cutting to `index` afterwards changes nothing."""
    sessions = features.trading_sessions(mr.START, mr.END, warmup_sessions=0)
    daily, _ = mr.load_daily(sessions, cache_dir, refresh, ("SPY",))
    f = mr.ticker_features(daily["SPY"]).reindex(sessions.index)
    pos = te.dip_positions(f)
    ret = daily["SPY"]["snap"].reindex(sessions.index).pct_change()
    held = pos.shift(1).fillna(0)
    entries = (pos.diff() > 0).astype(float)
    return ((held * ret).fillna(0) - entries * cost_bps / 1e4).reindex(index).fillna(0)


def tsla_breakout_daily(index, out_root="output/breakout_check", cost_bps=1.0):
    """Daily return of the two TSLA keeper rules on `index` (session dates), from breakout_check's saved share
    trades (design and holdout runs); None if neither file exists."""
    frames = [pd.read_parquet(p) for p in (Path(out_root) / k / "shares.parquet" for k in ("design", "holdout"))
              if p.exists()]
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    t = t[~t["fake"] & (t["status"] == "ok") & t["rule"].isin(TSLA_RULES)]
    net = (t["gross"] - 2 * cost_bps) / 1e4
    return net.groupby(t["session"]).sum().reindex(index).fillna(0)


def correlations(candidate, others):
    """Pearson correlation of the candidate's daily returns with each series in `others` ({name: series}) over
    the candidate's sessions, plus how many sessions both had a position."""
    rows = []
    for name, s in others.items():
        if s is None:
            rows.append({"with": name, "corr": np.nan, "both_active": 0, "sessions": 0})
            continue
        s = s.reindex(candidate.index).fillna(0)
        ok = candidate.notna()
        c = np.corrcoef(candidate[ok], s[ok])[0, 1] if candidate[ok].std() > 0 and s[ok].std() > 0 else np.nan
        rows.append({"with": name, "corr": c, "both_active": int(((candidate != 0) & (s != 0)).sum()),
                     "sessions": int(ok.sum())})
    return pd.DataFrame(rows)


def one_at_a_time(trades, start="start", end="end"):
    """Trades taken one at a time: in order of `start`, a trade is skipped while an earlier taken one is still
    open (its `end` is after this trade's `start`). Returns the taken rows."""
    t = trades.sort_values(start, kind="stable")
    keep, busy_until = [], None
    for i, s, e in zip(t.index, t[start], t[end]):
        if busy_until is not None and s < busy_until:
            continue
        keep.append(i)
        busy_until = e
    return t.loc[keep]


def account(daily_returns):
    """Compounded account path of daily returns: total, per year, worst drawdown, and each calendar year."""
    r = daily_returns.fillna(0)
    eq = (1 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    out = {"total": eq.iloc[-1] - 1, "per_year": eq.iloc[-1] ** (1 / years) - 1,
           "max_dd": float((eq / eq.cummax() - 1).min())}
    for y, g in r.groupby(r.index.year):
        out[f"y{y}"] = (1 + g).prod() - 1
    return out
