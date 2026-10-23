"""Tests for src/backtest.py: the signal state machine and the single-pair
P&L engine, checked against hand-computed numbers."""

import numpy as np
import pytest

from src.backtest import END, EXIT, STOP, signal_positions, simulate_pair
from src.config import BASELINE

nan = np.nan


def sig(z):
    """Baseline rules: entry |z| >= 2, exit at 0, stop |z| >= 4."""
    pos, reason = signal_positions(np.array(z, dtype=float), 2.0, 0.0, 4.0)
    return pos.tolist(), reason


# --------------------------------------------------------------------------- #
# 7. signal_positions
# --------------------------------------------------------------------------- #
def test_long_entry_at_minus_two_and_exit_at_zero():
    pos, reason = sig([-1.0, -2.0, -1.5, -0.1, 0.0, 0.5])
    # -2.0 <= -entry -> long; holds until z >= 0 on day 4.
    assert pos == [0, 1, 1, 1, 0, 0]
    assert reason == ["", "", "", "", EXIT, ""]


def test_short_entry_and_exit_when_crossing_zero():
    pos, reason = sig([1.9, 2.5, 1.0, -0.3, -2.5])
    # Short on day 1; exit day 3 (z <= 0, overshoots to -0.3);
    # day 4 is a fresh long entry the day after the exit.
    assert pos == [0, -1, -1, 0, 1]
    assert reason == ["", "", "", EXIT, ""]


def test_stop_loss_disables_pair_for_rest_of_window():
    pos, reason = sig([-2.5, -3.0, -4.0, -2.5, 2.5, 0.0])
    # Long on day 0, stopped at z = -4 on day 2; later signals (-2.5, +2.5)
    # would normally be entries but the pair is disabled.
    assert pos == [1, 1, 0, 0, 0, 0]
    assert reason == ["", "", STOP, "", "", ""]


def test_short_stop_loss():
    pos, reason = sig([2.0, 3.9, 4.2, -2.1])
    assert pos == [-1, -1, 0, 0]
    assert reason == ["", "", STOP, ""]


def test_no_entry_when_z_already_beyond_stop():
    # |z| >= 4 on the entry day: no trade (entry requires |z| < stop), and
    # since nothing was stopped the pair can still enter later.
    pos, reason = sig([4.5, -4.0, 2.2])
    assert pos == [0, 0, -1]
    assert reason == ["", "", ""]


def test_nan_keeps_current_position():
    pos, reason = sig([nan, -2.2, nan, nan, 0.1, nan])
    assert pos == [0, 1, 1, 1, 0, 0]
    assert reason == ["", "", "", "", EXIT, ""]


# --------------------------------------------------------------------------- #
# 8. simulate_pair
# --------------------------------------------------------------------------- #
# Round-number costs: one-way cost 10 bps per leg = 0.001, borrow 252 bps p.a.
# = 1 bp per day = 0.0001 on short notional.
CFG = BASELINE.with_(commission_bps=0.0, slippage_bps=10.0, borrow_bps_pa=252.0,
                     execution_lag=1)


def test_simulate_long_spread_forced_close_hand_computed():
    z = np.array([-2.5, -1.0, -0.5, -0.2, -0.1])  # long signal on day 0, never exits
    py = np.array([10.0, 10.0, 11.0, 10.5, 12.0])
    px = np.array([20.0, 20.0, 19.0, 21.0, 20.0])
    path = simulate_pair(py, px, z, np.ones(5), CFG)

    # Lag 1: signal day 0 -> fill day 1; forced flat on the last day.
    assert path.held.tolist() == [0, 1, 1, 1, 0]

    # Day 1 open: $1 each leg -> ny = 1/10 = 0.1, nx = 1/20 = 0.05.
    #   entry cost = 0.001 * (1 + 1) = 0.002
    # Day 2: gross = 0.1*(11-10) - 0.05*(19-20) = 0.15
    #        borrow = 0.0001 * nx*px[1] = 0.0001 * 1.00 = 0.000100
    # Day 3: gross = 0.1*(10.5-11) - 0.05*(21-19) = -0.15
    #        borrow = 0.0001 * 0.05*19 = 0.000095
    # Day 4: gross = 0.1*(12-10.5) - 0.05*(20-21) = 0.20
    #        borrow = 0.0001 * 0.05*21 = 0.000105
    #        exit notional = 0.1*12 + 0.05*20 = 2.2, exit cost = 0.0022
    expected = [0.0, -0.002, 0.15 - 0.0001, -0.15 - 0.000095, 0.20 - 0.000105 - 0.0022]
    np.testing.assert_allclose(path.pnl, expected, atol=1e-12)

    (tr,) = path.trades
    assert tr["direction"] == 1 and tr["reason"] == END and tr["days"] == 3
    assert tr["entry_z"] == -2.5               # signal z (decision day 0)
    assert tr["notional"] == pytest.approx(2.0)
    assert tr["exit_notional"] == pytest.approx(2.2)
    assert tr["gross"] == pytest.approx(0.20)  # 0.15 - 0.15 + 0.20
    assert tr["borrow"] == pytest.approx(0.0003)
    assert tr["costs"] == pytest.approx(0.0042)  # 0.002 + 0.0022
    # net = 0.20 - 0.0042 - 0.0003 = 0.1955, and must equal the daily P&L sum.
    assert tr["net"] == pytest.approx(0.1955)
    assert tr["net"] == pytest.approx(path.pnl.sum())


def test_simulate_short_spread_signal_exit_hand_computed():
    z = np.array([2.5, 1.0, -0.1, 0.0, 0.0])  # short day 0, exit signal day 2
    py = np.array([10.0, 10.0, 9.0, 9.5, 9.0])
    px = np.array([20.0, 20.0, 20.0, 22.0, 22.0])
    path = simulate_pair(py, px, z, np.ones(5), CFG)
    assert path.held.tolist() == [0, -1, -1, 0, 0]  # exit filled on day 3

    # Short the spread = short y, long x; d = -1, ny = 0.1, nx = 0.05.
    # Day 2: gross = -(0.1*(9-10) - 0.05*0) = 0.10; borrow on y: 0.0001*0.1*10 = 0.0001
    # Day 3: gross = -(0.1*0.5 - 0.05*2) = 0.05; borrow 0.0001*0.1*9 = 0.00009
    #        exit notional = 0.1*9.5 + 0.05*22 = 2.05, cost 0.00205
    expected = [0.0, -0.002, 0.10 - 0.0001, 0.05 - 0.00009 - 0.00205, 0.0]
    np.testing.assert_allclose(path.pnl, expected, atol=1e-12)
    (tr,) = path.trades
    assert tr["direction"] == -1 and tr["reason"] == EXIT and tr["days"] == 2
    assert tr["net"] == pytest.approx(0.15 - 0.00019 - 0.002 - 0.00205)
    assert tr["net"] == pytest.approx(path.pnl.sum())


def test_simulate_stop_reason_carried_through_lag():
    z = np.array([-2.5, -4.5, 0.0, -2.5, 0.0])  # stop decided day 1, filled day 2
    p = np.full(5, 10.0)
    path = simulate_pair(p, p, z, np.ones(5), CFG)
    assert path.held.tolist() == [0, 1, 0, 0, 0]  # disabled after stop
    assert [tr["reason"] for tr in path.trades] == [STOP]


def test_simulate_zero_costs_flat_prices_zero_pnl():
    cfg = CFG.with_(slippage_bps=0.0, borrow_bps_pa=0.0)
    p = np.full(6, 50.0)
    path = simulate_pair(p, p, np.array([-3.0, -1, 0, 3, 1, 1]), np.ones(6), cfg)
    assert np.all(path.pnl == 0.0)
    assert len(path.trades) == 2  # one long (exit), one short (window end)


def test_hedge_ratio_weighting():
    # beta = 0.5 known on decision day 0 -> wy = 2/1.5 = 4/3, wx = 2*0.5/1.5 = 2/3.
    # Gross notional stays 2 (same as dollar-neutral), so entry cost = 0.002.
    cfg = CFG.with_(weighting="hedge_ratio")
    z = np.array([-2.5, -1.0, -1.0])
    py, px = np.array([10.0, 8.0, 8.0]), np.array([20.0, 25.0, 25.0])
    beta = np.array([0.5, 9.9, 9.9])  # day-1 value must NOT be used
    path = simulate_pair(py, px, z, beta, cfg)
    (tr,) = path.trades
    assert tr["notional"] == pytest.approx(2.0)
    # Shares bought at day-1 prices: ny = (4/3)/8, nx = (2/3)/25; exit notional
    # at unchanged prices on day 2 is again 4/3 + 2/3 = 2.
    assert tr["exit_notional"] == pytest.approx(2.0)
    assert path.pnl[1] == pytest.approx(-0.002)
    # Day 2 borrow on the x leg: 0.0001 * (2/3)/25 * 25 = 0.0001 * 2/3.
    assert tr["borrow"] == pytest.approx(0.0001 * 2 / 3)
