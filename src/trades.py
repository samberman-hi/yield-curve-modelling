"""DV01-neutral 2s10s curve trades: P&L approximation, signals and backtest.

Position convention: +1 = steepener (long the 2y, short the 10y), -1 = flattener,
each leg sized to the same DV01 (``DV01_PER_LEG`` dollars per bp). Because the two
DV01s are equal, a parallel 1bp move makes ~0 P&L and a 1bp *steepening* of the
2s10s spread makes +DV01_PER_LEG for the steepener.

Daily P&L of one leg held from close t to close t+1 (yields in decimals):

    r_i = (y_i,t - f_t) * dt                 carry over the repo/funding rate
        + MD_i * (y_i(T) - y_i(T - dt))       roll-down along today's curve
        - MD_i * dy_i + 0.5 * C_i * dy_i^2    price change (duration + convexity)

with f_t proxied by the 3-month CMT yield, dt in years (calendar days / 365, so
weekends accrue carry), y_i(T - dt) linearly interpolated between neighbouring
CMT nodes, and MD_i, C_i the modified duration and convexity of a par bond at
y_i,t. Notional N_i = DV01_PER_LEG / (MD_i * 1e-4), re-sized daily so each leg
always carries the target DV01 (re-hedging costs are ignored; see README).

Signals use only data up to the close of day t and are applied to the P&L from
t to t+1, so there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bonds import par_duration_convexity

DV01_PER_LEG = 10_000.0  # $ per bp on each leg
COST_BP_PER_LEG = 0.5  # one-way transaction cost, bp of yield per leg (assumption)
TRADING_DAYS = 252

# Neighbouring CMT node used for the roll-down slope of each traded maturity.
LEGS = {"2y": ("DGS2", 2.0, "DGS1", 1.0), "10y": ("DGS10", 10.0, "DGS7", 7.0)}


def leg_returns(yields: pd.DataFrame, leg: str) -> pd.DataFrame:
    """Daily excess return per $1 notional of a constant-maturity par bond, split by source."""
    col, T, col_prev, T_prev = LEGS[leg]
    df = yields[[col, col_prev, "DGS3MO"]].dropna() / 100.0
    y, f = df[col], df["DGS3MO"]
    mod, conv = par_duration_convexity(y.to_numpy(), T)
    mod, conv = pd.Series(mod, y.index), pd.Series(conv, y.index)
    dt = y.index.to_series().diff().dt.days.shift(-1) / 365.0  # holding period t -> t+1
    dy = y.shift(-1) - y
    slope = (y - df[col_prev]) / (T - T_prev)  # per year of maturity, at close t
    out = pd.DataFrame({
        "carry": (y - f) * dt,
        "rolldown": mod * slope * dt,
        "duration": -mod * dy,
        "convexity": 0.5 * conv * dy**2,
    })
    out["total"] = out.sum(axis=1)
    out["mod_dur"] = mod
    # Index by the day the P&L is *realised* (t+1), so it lines up with positions taken at t.
    out = out.shift(1).dropna()
    return out


def unit_pnl(yields: pd.DataFrame) -> pd.DataFrame:
    """Daily $ P&L of +1 steepener unit (DV01_PER_LEG long 2y, short 10y), by component."""
    r2, r10 = leg_returns(yields, "2y"), leg_returns(yields, "10y")
    idx = r2.index.intersection(r10.index)
    r2, r10 = r2.loc[idx], r10.loc[idx]
    n2 = DV01_PER_LEG / (r2["mod_dur"] * 1e-4)  # notional for the target DV01
    n10 = DV01_PER_LEG / (r10["mod_dur"] * 1e-4)
    comps = ["carry", "rolldown", "duration", "convexity", "total"]
    return r2[comps].mul(n2, axis=0) - r10[comps].mul(n10, axis=0)


def carry_rolldown_bp(yields: pd.DataFrame, horizon: float = 0.25) -> pd.Series:
    """Expected carry + roll-down of the steepener over ``horizon`` years, in bp of DV01.

    For each leg, in yield-bp equivalents: (y - f) * h / MD + (y(T) - y(T - h)).
    Steepener = 2y leg minus 10y leg. Positive means the steepener earns money if
    the curve does not move.
    """
    out = {}
    for leg, (col, T, col_prev, T_prev) in LEGS.items():
        df = yields[[col, col_prev, "DGS3MO"]].dropna() / 100.0
        y, f = df[col], df["DGS3MO"]
        mod, _ = par_duration_convexity(y.to_numpy(), T)
        slope = (y - df[col_prev]) / (T - T_prev)
        out[leg] = ((y - f) * horizon / mod + slope * horizon) * 1e4
    both = pd.DataFrame(out).dropna()
    return (both["2y"] - both["10y"]).rename("carry_roll_bp")


def zscore_signal(spread: pd.Series, window: int, entry: float) -> pd.Series:
    """Mean-reversion rule on the 2s10s spread.

    z_t = (s_t - mean_{t-W+1..t}) / std_{t-W+1..t}. Go flattener (-1) when z > entry,
    steepener (+1) when z < -entry, and close the position when z crosses zero.
    """
    s = spread.dropna()
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    z = ((s - mu) / sd).to_numpy()
    pos = np.zeros(len(s))
    cur = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            cur = 0.0
        elif zi > entry:
            cur = -1.0
        elif zi < -entry:
            cur = 1.0
        elif (cur == -1.0 and zi <= 0) or (cur == 1.0 and zi >= 0):
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=s.index, name="position")


def carry_signal(carry_bp: pd.Series, threshold: float) -> pd.Series:
    """Hold the side of the trade with positive expected carry + roll-down.

    +1 if carry > threshold, -1 if carry < -threshold, else flat.
    """
    pos = np.where(carry_bp > threshold, 1.0, np.where(carry_bp < -threshold, -1.0, 0.0))
    return pd.Series(pos, index=carry_bp.index, name="position")


def backtest(position: pd.Series, pnl_unit: pd.DataFrame,
             cost_bp: float = COST_BP_PER_LEG) -> pd.DataFrame:
    """Apply positions decided at close t to the P&L realised over (t, t+1].

    Transaction cost: each unit change in position trades both legs, costing
    cost_bp * DV01_PER_LEG per leg, charged on the day the trade is done.
    """
    pos = position.reindex(pnl_unit.index.union(position.index)).ffill().fillna(0.0)
    held = pos.shift(1).reindex(pnl_unit.index).fillna(0.0)  # position held into day t
    gross = held * pnl_unit["total"]
    trades = pos.diff().abs().fillna(pos.abs())
    trade_cost = trades * 2 * cost_bp * DV01_PER_LEG
    # A trade at close t is booked with the next P&L day after t (so trades made
    # before the first P&L day are charged on that day rather than dropped).
    nxt = np.searchsorted(pnl_unit.index.to_numpy(), trade_cost.index.to_numpy(), side="right")
    keep = nxt < len(pnl_unit.index)
    cost = (trade_cost[keep].groupby(pnl_unit.index[nxt[keep]]).sum()
            .reindex(pnl_unit.index).fillna(0.0))
    return pd.DataFrame({"position": held, "gross": gross, "cost": cost, "net": gross - cost})


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """t-statistic and two-sided p-value for mean(x) = 0 with Newey–West standard errors."""
    from scipy import stats

    x = np.asarray(x, float)
    T = len(x)
    lags = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    u = x - x.mean()
    lrv = u @ u / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    t = x.mean() / np.sqrt(lrv / T)
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=T - 1)))


@dataclass
class Stats:
    ann_pnl: float
    ann_vol: float
    sharpe: float
    t_stat: float
    p_value: float
    max_dd: float
    hit_rate: float
    time_in_market: float
    trades_per_year: float


def performance(bt: pd.DataFrame, col: str = "net") -> Stats:
    """Annualised statistics of a daily $ P&L series."""
    x = bt[col]
    ann = x.mean() * TRADING_DAYS
    vol = x.std() * np.sqrt(TRADING_DAYS)
    t, p = newey_west_tstat(x.to_numpy())
    cum = x.cumsum()
    dd = (cum - cum.cummax()).min()
    active = bt["position"] != 0
    years = len(x) / TRADING_DAYS
    n_trades = bt["position"].diff().abs().sum()
    return Stats(ann, vol, ann / vol if vol > 0 else np.nan, t, p, dd,
                 float((x[active] > 0).mean()) if active.any() else np.nan,
                 float(active.mean()), float(n_trades / years))


def select_params(score_fn, grid: list[tuple]) -> tuple[tuple, pd.DataFrame]:
    """Pick the parameter tuple with the highest training-period Sharpe."""
    rows = [(*g, score_fn(*g)) for g in grid]
    table = pd.DataFrame(rows)
    best = table.iloc[table.iloc[:, -1].idxmax()]
    return tuple(best.iloc[:-1]), table
