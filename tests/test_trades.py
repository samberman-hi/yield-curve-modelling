"""DV01-neutral 2s10s steepener: P&L decomposition, signals, backtest timing and costs."""

import numpy as np
import pandas as pd
import pytest

from src.bonds import par_duration_convexity
from src.trades import (DV01_PER_LEG, backtest, carry_signal, newey_west_tstat, unit_pnl,
                        zscore_signal)

COLS = ["DGS3MO", "DGS1", "DGS2", "DGS7", "DGS10"]


def _flat(level=4.0, n=10):
    return pd.DataFrame(level, index=pd.bdate_range("2024-01-01", periods=n), columns=COLS)


def test_flat_static_curve_earns_nothing():
    """Flat 4% curve that never moves, funding = yield: carry, roll-down and price P&L are all 0."""
    pnl = unit_pnl(_flat())
    assert len(pnl) == 9
    np.testing.assert_allclose(pnl.to_numpy(), 0.0, atol=1e-9)


def test_parallel_shift_is_duration_neutral():
    """Parallel +1bp: duration P&L 0; convexity P&L = 0.5*DV01*1e4*(C2/MD2 - C10/MD10)*dy^2 ~ -$3.6."""
    y = _flat()
    y.iloc[5:] += 0.01  # +1bp on day 5, all maturities
    pnl = unit_pnl(y)
    day = y.index[5]
    assert pnl.loc[day, "duration"] == pytest.approx(0.0, abs=1e-6)
    md2, c2 = par_duration_convexity(0.04, 2.0)
    md10, c10 = par_duration_convexity(0.04, 10.0)
    expected = 0.5 * DV01_PER_LEG * 1e4 * (c2 / md2 - c10 / md10) * 1e-4**2
    assert pnl.loc[day, "convexity"] == pytest.approx(expected, rel=1e-9)
    assert -5.0 < expected < 0.0  # the steepener is short convexity, but only by a few dollars


def test_one_bp_steepening_pays_dv01():
    """10y +1bp with the 2y unchanged = 2s10s +1bp, so +1 steepener earns +$10,000 duration P&L."""
    y = _flat()
    y.iloc[5:, COLS.index("DGS10")] += 0.01
    pnl = unit_pnl(y)
    assert pnl.loc[y.index[5], "duration"] == pytest.approx(DV01_PER_LEG, rel=1e-9)
    assert pnl.loc[y.index[4], "duration"] == pytest.approx(0.0, abs=1e-9)  # nothing the day before


def test_backtest_applies_position_next_day_and_charges_costs():
    """Position set at close t earns P&L of t+1; each unit traded costs 2 legs x 0.5bp x $10k, booked at t+1."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    pnl = pd.DataFrame({"total": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=idx)
    pos = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0], index=idx)
    bt = backtest(pos, pnl, cost_bp=0.5)
    unit_cost = 2 * 0.5 * DV01_PER_LEG  # $10,000 per unit of position change
    np.testing.assert_allclose(bt["position"], [0, 0, 1, 1, -1])
    np.testing.assert_allclose(bt["gross"], [0, 0, 300, 400, -500])
    np.testing.assert_allclose(bt["cost"], [0, 0, unit_cost, 0, 2 * unit_cost])
    np.testing.assert_allclose(bt["net"], bt["gross"] - bt["cost"])


def test_backtest_charges_entry_cost_before_first_pnl_day():
    """A position entered on the day before the first P&L date still pays one unit of entry cost."""
    idx = pd.bdate_range("2024-01-01", periods=4)
    pnl = pd.DataFrame({"total": [10.0, 20.0, 30.0]}, index=idx[1:])  # P&L starts on day 1
    pos = pd.Series(1.0, index=idx)  # long from close of day 0
    bt = backtest(pos, pnl, cost_bp=0.5)
    assert bt["gross"].sum() == pytest.approx(60.0)  # the position does earn day 1's P&L
    assert bt["cost"].sum() == pytest.approx(2 * 0.5 * DV01_PER_LEG)


def test_zscore_signal_state_machine():
    """Window 4, entry 1: hand-computed z-scores drive enter / hold / close-at-zero / flip."""
    s = pd.Series([1.0, 1, 1, 1, 4, 3, 1, 0, -3, -2, 0, 2],
                  index=pd.bdate_range("2024-01-01", periods=12))
    # z from day 4: 1.5, 0.5, -0.83, -1.10, -1.30, -0.55, 0.83, 1.24 (days 0-3: NaN or 0/0)
    expected = [0, 0, 0, 0, -1, -1, 0, 1, 1, 1, 0, -1]
    np.testing.assert_array_equal(zscore_signal(s, window=4, entry=1.0).to_numpy(), expected)


def test_carry_signal_thresholds():
    """Threshold 1bp: carry 2 -> +1, 0.5 -> 0, -0.5 -> 0, -2 -> -1, exactly 1 -> 0 (strict)."""
    c = pd.Series([2.0, 0.5, -0.5, -2.0, 1.0, -1.0])
    np.testing.assert_array_equal(carry_signal(c, 1.0).to_numpy(), [1, 0, 0, -1, 0, 0])


def test_newey_west_lag0_is_ordinary_tstat():
    """With 0 lags, NW t = mean / (std / sqrt(T)) using the population (ddof=0) std."""
    x = np.random.default_rng(0).normal(0.2, 1.0, 250)
    t, p = newey_west_tstat(x, lags=0)
    assert t == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-12)
    assert 0 < p < 1


def test_newey_west_lag1_hand_computed():
    """x = (1, 2, 3, 4), 1 lag: lrv = (5 + 2*0.5*1.25)/4 = 1.5625, t = 2.5 / sqrt(1.5625/4) = 4."""
    t, _ = newey_west_tstat(np.array([1.0, 2.0, 3.0, 4.0]), lags=1)
    assert t == pytest.approx(4.0)
