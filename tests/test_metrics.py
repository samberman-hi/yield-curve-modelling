"""Tests for src/metrics.py against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (ann_return, hac_mean_test, market_exposure, max_drawdown,
                         nw_lags, sharpe, turnover)


def test_sharpe_hand_example():
    # r = [0.02, 0.00]: mean = 0.01, sample sd (ddof=1) = sqrt(2 * 0.01^2 / 1)
    # = 0.01 * sqrt(2). Sharpe = sqrt(252) * 0.01 / (0.01 * sqrt(2)) = sqrt(126).
    assert sharpe(pd.Series([0.02, 0.0])) == pytest.approx(np.sqrt(126))


def test_sharpe_all_zero_returns_is_nan():
    # A strategy that never trades has sd exactly 0 -> Sharpe undefined.
    assert np.isnan(sharpe(pd.Series([0.0] * 10)))


def test_sharpe_constant_nonzero_series_is_nan():
    # Regression test: the sample sd of a constant series is ~1e-18 from
    # floating-point rounding, not exactly 0; it must still give nan, not ~1e17.
    assert np.isnan(sharpe(pd.Series([0.01] * 10)))


def test_max_drawdown_hand_example():
    # Wealth: 1.1, 0.55, 0.66. Peak 1.1, trough 0.55 -> 0.55/1.1 - 1 = -0.5.
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    # Monotone gains never draw down.
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_ann_return_geometric():
    # 504 days (2 years) compounding to 1.21 overall -> 10% a year.
    r = pd.Series(np.full(504, 1.21 ** (1 / 504) - 1))
    assert ann_return(r) == pytest.approx(0.10)


def test_nw_lags_values():
    # floor(4 * (1000/100)^(2/9)) = floor(4 * 1.668) = floor(6.67) = 6
    assert nw_lags(1000) == 6
    assert nw_lags(100) == 4  # (1)^(2/9) = 1


def test_hac_t_close_to_naive_t_for_iid():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, 2000))
    t_hac, p_hac = hac_mean_test(r)
    t_naive = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    # No autocorrelation -> the HAC correction should barely matter.
    assert t_hac == pytest.approx(t_naive, rel=0.15)
    assert 0 <= p_hac <= 1


def test_market_exposure_recovers_beta():
    rng = np.random.default_rng(8)
    m = pd.Series(rng.normal(0, 0.01, 1000))
    r = 0.5 * m + pd.Series(rng.normal(0, 0.001, 1000))
    out = market_exposure(r, m)
    assert out["beta"] == pytest.approx(0.5, abs=0.01)
    assert out["corr"] > 0.95
    assert out["beta_p"] < 1e-6


def test_turnover_hand_example():
    # Window 0: 2 pairs selected (capital 2). Trades: (2 + 2) and (2 + 1.9) = 7.9
    #           -> 7.9 / 2 = 3.95
    # Window 126: 4 pairs selected. One trade 2 + 2.1 = 4.1 -> 4.1 / 4 = 1.025
    # Window 252: 3 pairs, no trades -> contributes 0.
    # Over 2 years: (3.95 + 1.025) / 2 = 2.4875 x capital per year.
    trades = pd.DataFrame(dict(window_start=[0, 0, 126], notional=[2.0, 2.0, 2.0],
                               exit_notional=[2.0, 1.9, 2.1]))
    windows = pd.DataFrame(dict(window_start=[0, 126, 252], n_selected=[2, 4, 3]))
    assert turnover(trades, windows, n_years=2.0) == pytest.approx(2.4875)
    assert turnover(trades.iloc[:0], windows, 2.0) == 0.0
