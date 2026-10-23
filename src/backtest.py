"""Walk-forward pairs-trading backtest.

Timeline for each window (indices are trading days):

    [s - formation_days, s)   formation: test, select pairs, estimate hedge
    [s, s + trading_days)     trading: z-score rules, positions, P&L

Windows roll forward by ``trading_days``, so trading periods do not overlap and
each day of the backtest belongs to exactly one portfolio.

Execution: the signal is computed from the close on day t and the trade is
filled at the close of day t + ``execution_lag``. Positions are held as fixed
share quantities between trades (no daily rebalancing). All open positions are
closed at the close of the last day of the trading window.

Returns are on *committed capital*: each selected pair gets an equal share of
capital, whether or not it ever trades (Gatev, Goetzmann & Rouwenhorst, 2006).
A pair position is $1 long and $1 short per $1 of the pair's capital, so the
returns are excess returns of a self-financing long-short book (interest on
collateral and the short rebate are assumed to offset the risk-free rate).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .cointegration import benjamini_hochberg, select_pairs, test_all_pairs
from .config import BENCHMARK, StrategyConfig
from .kalman import init_from_formation, kalman_filter

# Reasons a trade can close
EXIT, STOP, END = "exit", "stop", "window_end"


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def signal_positions(z: np.ndarray, entry: float, exit_: float, stop: float
                     ) -> tuple[np.ndarray, list[str]]:
    """Map a z-score path to desired spread positions (+1, 0, -1).

    +1 = long the spread (long y, short x): entered when z <= -entry.
    -1 = short the spread: entered when z >= +entry.
    A long is closed when z >= -exit_ (exit_=0 -> spread back at its mean) and
    stopped out when z <= -stop; symmetrically for a short. New entries need
    |z| < stop. After a stop-loss the pair is disabled for the rest of the
    window, on the view that the relationship has broken.

    Returns the position decided at each close and, for each day, the reason
    a position was closed that day ("" if none).
    """
    pos = np.zeros(len(z), dtype=int)
    reason = [""] * len(z)
    current, disabled = 0, False
    for t, zt in enumerate(z):
        if np.isnan(zt):
            pos[t] = current
            continue
        if current == 0 and not disabled:
            if entry <= zt < stop:
                current = -1
            elif -stop < zt <= -entry:
                current = 1
        elif current == 1:
            if zt <= -stop:
                current, disabled, reason[t] = 0, True, STOP
            elif zt >= -exit_:
                current, reason[t] = 0, EXIT
        elif current == -1:
            if zt >= stop:
                current, disabled, reason[t] = 0, True, STOP
            elif zt <= exit_:
                current, reason[t] = 0, EXIT
        pos[t] = current
    return pos, reason


# --------------------------------------------------------------------------- #
# Single pair in one trading window
# --------------------------------------------------------------------------- #
@dataclass
class PairPath:
    """Everything needed to plot / audit one pair in one trading window."""

    dates: pd.DatetimeIndex
    z: np.ndarray
    beta: np.ndarray  # hedge ratio in force on each day
    held: np.ndarray  # executed position held after each day's close
    pnl: np.ndarray  # daily net P&L per $1 of pair capital
    trades: list[dict] = field(default_factory=list)


def pair_zscore(pair: pd.Series, form_logp: pd.DataFrame, trade_logp: pd.DataFrame,
                cfg: StrategyConfig) -> tuple[np.ndarray, np.ndarray]:
    """z-score and hedge-ratio paths over the trading window."""
    y, x = trade_logp[pair.y].to_numpy(), trade_logp[pair.x].to_numpy()
    if cfg.hedge == "ols":
        spread = y - pair.alpha - pair.beta * x
        z = (spread - pair.spread_mean) / pair.spread_std
        return z, np.full(len(y), pair.beta)
    if cfg.hedge == "kalman":
        th0, P0, R, W = init_from_formation(form_logp[pair.y].to_numpy(),
                                            form_logp[pair.x].to_numpy(), cfg.kalman_delta)
        kf = kalman_filter(y, x, th0, P0, R, W)
        return kf.z, kf.beta
    raise ValueError(f"unknown hedge {cfg.hedge!r}")


def simulate_pair(py: np.ndarray, px: np.ndarray, z: np.ndarray, beta: np.ndarray,
                  cfg: StrategyConfig, dates: pd.DatetimeIndex | None = None) -> PairPath:
    """Trade one pair over one window. Prices are levels (not logs)."""
    n = len(z)
    dates = dates if dates is not None else pd.RangeIndex(n)
    desired, reason = signal_positions(z, cfg.entry_z, cfg.exit_z, cfg.stop_z)
    # Execution lag: what is held after the close of day t was decided on t-lag.
    held = np.zeros(n, dtype=int)
    held[cfg.execution_lag:] = desired[: n - cfg.execution_lag]
    held[-1] = 0  # forced close at the end of the window
    close_reason = [""] * n
    for t in range(cfg.execution_lag, n):
        close_reason[t] = reason[t - cfg.execution_lag]

    cost = cfg.cost_bps / 1e4
    borrow_daily = cfg.borrow_bps_pa / 1e4 / 252
    pnl = np.zeros(n)
    trades: list[dict] = []
    ny = nx = 0.0  # shares held (positive numbers; direction carried by `d`)
    d = 0
    trade: dict = {}
    for t in range(n):
        # 1) mark-to-market of the position held over (t-1, t]
        if t > 0 and d != 0:
            gross = d * (ny * (py[t] - py[t - 1]) - nx * (px[t] - px[t - 1]))
            short_value = nx * px[t - 1] if d == 1 else ny * py[t - 1]
            fee = borrow_daily * short_value
            pnl[t] += gross - fee
            trade["gross"] += gross
            trade["borrow"] += fee
        # 2) trades at the close of day t
        target = held[t]
        if target != d and d != 0:  # close existing position
            exit_notional = ny * py[t] + nx * px[t]
            c = cost * exit_notional
            pnl[t] -= c
            trade["costs"] += c
            # exit_z = z on the decision day (comparable with the rule thresholds);
            # exit_fill_z = z at the fill one day later.
            trade.update(exit_date=dates[t], exit_z=z[t - cfg.execution_lag],
                         exit_fill_z=z[t], days=t - trade["_t0"],
                         exit_notional=exit_notional,
                         reason=close_reason[t] or (END if t == n - 1 else EXIT))
            trade["net"] = trade["gross"] - trade["costs"] - trade["borrow"]
            trade.pop("_t0")
            trades.append(trade)
            d, ny, nx = 0, 0.0, 0.0
        if target != 0 and d == 0:  # open new position
            if cfg.weighting == "dollar_neutral":
                wy, wx = 1.0, 1.0
            elif cfg.weighting == "hedge_ratio":
                b = abs(beta[t - cfg.execution_lag])  # beta known at decision time
                wy, wx = 2.0 / (1 + b), 2.0 * b / (1 + b)
            else:
                raise ValueError(cfg.weighting)
            d, ny, nx = target, wy / py[t], wx / px[t]
            c = cost * (wy + wx)
            pnl[t] -= c
            trade = dict(entry_date=dates[t], entry_z=z[t - cfg.execution_lag],
                         entry_fill_z=z[t], direction=d, gross=0.0, costs=c, borrow=0.0,
                         notional=wy + wx, _t0=t)
    return PairPath(pd.DatetimeIndex(dates) if not isinstance(dates, pd.RangeIndex) else dates,
                    z, beta, held, pnl, trades)


def spread_adf_pvalue(pair: pd.Series, logp: pd.DataFrame) -> float:
    """ADF p-value of the spread built with the *formation* alpha and beta.

    Because the coefficients are fixed (not re-estimated on this data) the
    ordinary ADF/MacKinnon distribution applies. Applied to the trading window
    it is an out-of-sample diagnostic only and never feeds back into trading.
    """
    s = logp[pair.y].to_numpy() - pair.alpha - pair.beta * logp[pair.x].to_numpy()
    return float(adfuller(s, regression="c", autolag="aic", result_object=False)[1])


# --------------------------------------------------------------------------- #
# Walk-forward driver
# --------------------------------------------------------------------------- #
def window_starts(n_days: int, cfg: StrategyConfig) -> list[int]:
    return list(range(cfg.formation_days, n_days - 1, cfg.trading_days))


def precompute_tests(prices: pd.DataFrame, cfg: StrategyConfig,
                     workers: int | None = None) -> dict[int, pd.DataFrame]:
    """Cointegration tests for every formation window (independent of the
    trading rules, so it is computed once and reused across the grid).
    Each window only ever receives its own formation slice of prices."""
    stocks = prices.drop(columns=[BENCHMARK])
    starts = window_starts(len(prices), cfg)
    slices = [stocks.iloc[s - cfg.formation_days: s] for s in starts]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(test_all_pairs, slices))
    return dict(zip(starts, results))


@dataclass
class BacktestResult:
    cfg: StrategyConfig
    returns: pd.Series  # daily portfolio returns on committed capital
    trades: pd.DataFrame
    pairs: pd.DataFrame  # selected pair-windows with OOS diagnostics and P&L
    windows: pd.DataFrame  # one row per window: dates, #tested, #selected
    paths: dict[tuple[int, str, str], PairPath]
    n_open: pd.Series  # number of open pair positions after each close


def run_backtest(prices: pd.DataFrame, cfg: StrategyConfig,
                 tests: dict[int, pd.DataFrame] | None = None) -> BacktestResult:
    tests = tests if tests is not None else precompute_tests(prices, cfg)
    logp = np.log(prices)
    dates = prices.index
    port = pd.Series(0.0, index=dates[cfg.formation_days:])
    n_open = pd.Series(0, index=port.index)
    trade_rows, pair_rows, win_rows, paths = [], [], [], {}

    for s, tested in tests.items():
        e = min(s + cfg.trading_days, len(dates))
        tdates = dates[s:e]
        sel = select_pairs(tested, cfg)
        win_rows.append(dict(window_start=s, trade_start=tdates[0], trade_end=tdates[-1],
                             formation_start=dates[s - cfg.formation_days],
                             n_tested=len(tested),
                             n_raw_sig=int((tested["eg_pvalue"] < 0.05).sum()) if len(tested) else 0,
                             n_bh_sig=int(benjamini_hochberg(tested["eg_pvalue"].to_numpy(),
                                                             cfg.fdr_q).sum()) if len(tested) else 0,
                             n_selected=len(sel)))
        if sel.empty:
            continue
        form_logp = logp.iloc[s - cfg.formation_days: s]
        trade_logp = logp.iloc[s:e]
        window_pnl = np.zeros(e - s)
        for _, pair in sel.iterrows():
            z, beta = pair_zscore(pair, form_logp, trade_logp, cfg)
            path = simulate_pair(prices[pair.y].to_numpy()[s:e], prices[pair.x].to_numpy()[s:e],
                                 z, beta, cfg, tdates)
            paths[(s, pair.y, pair.x)] = path
            window_pnl += path.pnl
            n_open.loc[tdates] += (path.held != 0).astype(int)
            for tr in path.trades:
                trade_rows.append(dict(window_start=s, group=pair.group, y=pair.y, x=pair.x, **tr))
            n_stops = sum(tr["reason"] == STOP for tr in path.trades)
            # Out-of-sample calibration of the *static* spread, in formation units:
            # if the formation relationship held, this z would have sd close to 1.
            z_static = ((trade_logp[pair.y] - pair.alpha - pair.beta * trade_logp[pair.x]).to_numpy()
                        - pair.spread_mean) / pair.spread_std
            pair_rows.append(dict(
                window_start=s, trade_start=tdates[0], group=pair.group, y=pair.y, x=pair.x,
                beta=pair.beta, eg_pvalue=pair.eg_pvalue, half_life=pair.half_life,
                oos_adf_p=spread_adf_pvalue(pair, trade_logp),
                oos_z_sd=float(np.std(z_static, ddof=1)),
                oos_frac_beyond_stop=float(np.mean(np.abs(z_static) >= cfg.stop_z)),
                signal_z_ac1=float(pd.Series(z).autocorr(1)),
                n_trades=len(path.trades),
                n_stops=n_stops, net_pnl=float(path.pnl.sum())))
        port.loc[tdates] = window_pnl / len(sel)

    trades = pd.DataFrame(trade_rows)
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        # ADF rejects a unit root in the OOS spread. Failing to reject is NOT proof
        # of a breakdown: 126 days give the test low power, and selection on the
        # formation window biases the formation fit towards looking stationary.
        pairs["oos_adf_reject"] = pairs["oos_adf_p"] < 0.05
    return BacktestResult(cfg, port, trades, pairs, pd.DataFrame(win_rows), paths, n_open)
