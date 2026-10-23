"""Performance statistics and hypothesis tests.

Inference on daily returns uses Newey-West (HAC) standard errors, because
pairs-trading returns are autocorrelated (positions last several days) and
heteroskedastic. Lag length: floor(4 (n/100)^(2/9)) (Newey & West, 1994).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252


def nw_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def hac_mean_test(r: pd.Series) -> tuple[float, float]:
    """t-stat and two-sided p-value for H0: E[r] = 0, Newey-West SEs."""
    r = pd.Series(r).dropna()
    res = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(r))})
    return float(res.tvalues[0]), float(res.pvalues[0])


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe ratio of (excess) daily returns."""
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    # Treat floating-point noise (a constant series has sd ~ 1e-18) as zero.
    if not sd > 1e-12 * max(1.0, abs(r.mean())):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * r.mean() / sd)


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the compounded wealth index (negative)."""
    w = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def ann_return(r: pd.Series) -> float:
    """Geometric annualised return."""
    r = pd.Series(r).dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def market_exposure(r: pd.Series, mkt: pd.Series) -> dict[str, float]:
    """Regress r_t = a + b * mkt_t + e_t (HAC). Returns beta, its t-stat and
    p-value, annualised alpha and its p-value, and the correlation."""
    df = pd.concat([r, mkt], axis=1, keys=["r", "m"]).dropna()
    X = sm.add_constant(df["m"].to_numpy())
    res = sm.OLS(df["r"].to_numpy(), X).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": nw_lags(len(df))})
    return dict(beta=float(res.params[1]), beta_t=float(res.tvalues[1]),
                beta_p=float(res.pvalues[1]), alpha_ann=float(res.params[0] * TRADING_DAYS),
                alpha_p=float(res.pvalues[0]), corr=float(df["r"].corr(df["m"])))


def turnover(trades: pd.DataFrame, windows: pd.DataFrame, n_years: float) -> float:
    """Annual traded notional as a multiple of committed capital.

    In each window the capital is the number of selected pairs (each pair gets
    $1); traded notional is the entry plus exit notional of every trade.
    """
    if trades.empty:
        return 0.0
    traded = (trades["notional"] + trades["exit_notional"]).groupby(trades["window_start"]).sum()
    cap = windows.set_index("window_start")["n_selected"]
    return float((traded / cap.loc[traded.index]).sum() / n_years)


def summary(r: pd.Series, mkt: pd.Series, trades: pd.DataFrame,
            windows: pd.DataFrame) -> dict[str, float]:
    r = pd.Series(r).dropna()
    t, p = hac_mean_test(r)
    n_years = len(r) / TRADING_DAYS
    out = dict(
        ann_return=ann_return(r),
        ann_vol=float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        sharpe=sharpe(r),
        t_stat=t,
        p_value=p,
        max_drawdown=max_drawdown(r),
        turnover=turnover(trades, windows, n_years),
        n_trades=int(len(trades)),
        hit_rate=float((trades["net"] > 0).mean()) if len(trades) else float("nan"),
        # per $1 of pair capital (i.e. per $1 long + $1 short)
        avg_trade_net_bps=float(trades["net"].mean() * 1e4) if len(trades) else float("nan"),
        avg_holding_days=float(trades["days"].mean()) if len(trades) else float("nan"),
    )
    out.update(market_exposure(r, mkt))
    return out


def trade_level_tests(trades: pd.DataFrame) -> dict[str, float]:
    """Inference that does not lean on thousands of zero-return days.

    t-test and Wilcoxon signed-rank test on per-trade net P&L, a t-test on
    per-window P&L (windows with at least one trade), and the share of total
    net P&L contributed by the two best trades (concentration).
    """
    from scipy import stats

    if len(trades) < 3:
        return dict(trade_t_p=float("nan"), trade_wilcoxon_p=float("nan"),
                    window_t_p=float("nan"), top2_share=float("nan"))
    net = trades["net"].to_numpy()
    by_window = trades.groupby("window_start")["net"].sum().to_numpy()
    total = net.sum()
    return dict(
        trade_t_p=float(stats.ttest_1samp(net, 0.0).pvalue),
        trade_wilcoxon_p=float(stats.wilcoxon(net).pvalue),
        window_t_p=float(stats.ttest_1samp(by_window, 0.0).pvalue) if len(by_window) > 2 else float("nan"),
        n_windows_traded=int(len(by_window)),
        top2_share=float(np.sort(net)[-2:].sum() / total) if total != 0 else float("nan"),
    )


def difference_test(r1: pd.Series, r2: pd.Series) -> tuple[float, float, float]:
    """Mean of r1 - r2 (annualised), HAC t-stat and p-value."""
    d = (pd.Series(r1) - pd.Series(r2)).dropna()
    t, p = hac_mean_test(d)
    return float(d.mean() * TRADING_DAYS), t, p
