"""Recession probit pipeline: target alignment, no look-ahead, and forecast-evaluation formulas."""

import numpy as np
import pandas as pd
import pytest

from src.recession import (HORIZON, MIN_TRAIN_OBS, PUB_LAG, brier, diebold_mariano,
                           expanding_probit, fit_probit, make_target, monthly_spreads)


def _synthetic_monthly(n=360, seed=0):
    """Monthly spread and a USREC series where recessions follow inverted spreads 12m later."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-01", periods=n, freq="MS")
    spread = pd.Series(1.0 + 1.5 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 0.3, n), idx)
    latent = -0.8 - 1.2 * spread.shift(HORIZON).bfill() + rng.normal(0, 1.0, n)
    usrec = pd.Series((latent > 0).astype(int), idx)
    return spread, usrec


def test_make_target_shifts_forward_12_months():
    """y_t = USREC_{t+12}: the value at month 0 is the indicator 12 months later; the tail is NaN."""
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    usrec = pd.Series(np.arange(30), idx)
    y = make_target(usrec)
    assert y.iloc[0] == 12 and y.iloc[17] == 29
    assert y.iloc[-HORIZON:].isna().all()


def test_brier_hand_computed():
    """Brier of p=(0.9, 0.2, 0.5) vs y=(1, 0, 1) = (0.01 + 0.04 + 0.25) / 3 = 0.1."""
    assert brier(np.array([0.9, 0.2, 0.5]), np.array([1, 0, 1])) == pytest.approx(0.1)


def test_diebold_mariano_zero_mean_differential():
    """A loss differential with mean exactly zero gives DM stat 0 and p-value 1."""
    loss_b = np.random.default_rng(0).uniform(size=100)
    d = np.tile([0.1, -0.1], 50)
    stat, p = diebold_mariano(loss_b + d, loss_b, lags=4)
    assert stat == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_diebold_mariano_sign():
    """If model b's loss is uniformly smaller, DM stat is large and positive (and flips if swapped)."""
    rng = np.random.default_rng(1)
    loss_b = rng.uniform(size=300)
    loss_a = loss_b + 0.1 + rng.normal(0, 0.05, 300)
    stat, p = diebold_mariano(loss_a, loss_b)
    assert stat > 5 and p < 1e-6
    assert diebold_mariano(loss_b, loss_a)[0] == pytest.approx(-stat)


def test_fit_probit_recovers_coefficients():
    """Probit on data from P(y=1) = Phi(-1 - 1.5x) recovers a ~ -1 and b ~ -1.5."""
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=5000))
    y = pd.Series((-1.0 - 1.5 * x + rng.normal(size=5000) > 0).astype(float))
    res = fit_probit(x, y)
    assert res.params["const"] == pytest.approx(-1.0, abs=0.1)
    assert res.params["spread"] == pytest.approx(-1.5, abs=0.1)


def test_expanding_probit_has_no_look_ahead():
    """Rewriting every USREC value from month D on leaves forecasts at origins t < D + pub_lag unchanged."""
    spread, usrec = _synthetic_monthly()
    base = expanding_probit(spread, usrec)
    D = pd.Timestamp("1985-01-01")
    corrupted = usrec.copy()
    corrupted[corrupted.index >= D] = 1 - corrupted[corrupted.index >= D]
    alt = expanding_probit(spread, corrupted)

    # origin t may only use outcomes dated <= t - pub_lag, so any t < D + pub_lag is safe
    safe = base.prob.index[base.prob.index < D + pd.DateOffset(months=PUB_LAG)]
    assert len(safe) > 50
    pd.testing.assert_series_equal(base.prob.loc[safe], alt.prob.loc[safe])
    # ...and the very next origin does see the change (so the test has power)
    first_exposed = D + pd.DateOffset(months=PUB_LAG)
    assert base.prob.loc[first_exposed] != alt.prob.loc[first_exposed]


def test_expanding_probit_first_origin_respects_min_obs():
    """First origin is the first month whose training set (s <= t - 24m) has MIN_TRAIN_OBS points."""
    spread, usrec = _synthetic_monthly()
    assert make_target(usrec).iloc[:MIN_TRAIN_OBS].sum() > 0  # a recession is in the first window
    fc = expanding_probit(spread, usrec)
    lag = HORIZON + PUB_LAG
    # training months 0..k-lag at origin position k, so k = MIN_TRAIN_OBS + lag - 1
    assert fc.prob.index[0] == spread.index[MIN_TRAIN_OBS + lag - 1]
    assert fc.base_rate.iloc[0] == pytest.approx(make_target(usrec).iloc[:MIN_TRAIN_OBS].mean())


def test_monthly_spreads_drops_incomplete_final_month():
    """Daily data ending mid-March keeps Jan and Feb only; ending on the last business day keeps March."""
    def daily(end):
        idx = pd.bdate_range("2024-01-01", end)
        return pd.DataFrame({"DGS10": 4.0, "DGS3MO": 5.0, "DGS2": 4.5}, index=idx)

    mid = monthly_spreads(daily("2024-03-15"))
    assert list(mid.index) == list(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    full = monthly_spreads(daily("2024-03-29"))  # Friday 29 March is the last business day
    assert full.index[-1] == pd.Timestamp("2024-03-01")
    np.testing.assert_allclose(full["10y-3m"], -1.0)
    np.testing.assert_allclose(full["10y-2y"], -0.5)


def test_diebold_mariano_identical_losses():
    """Identical loss series give DM stat 0 and p-value 1 (not NaN)."""
    loss = np.random.default_rng(3).uniform(size=50)
    assert diebold_mariano(loss, loss) == (0.0, 1.0)
