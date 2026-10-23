"""Tests for src/kalman.py.

Key fact used below: with no state noise (W = 0) the state is constant, and
starting from the OLS prior theta0 = (X'X)^{-1} X'y, P0 = R (X'X)^{-1} the
Kalman filter is exactly recursive least squares. So the prior on trading day t
must equal OLS on the pooled sample (formation + trading days 0..t-1).
"""

import numpy as np
import pytest

from src.kalman import init_from_formation, kalman_filter


def synthetic_logprices(n, seed):
    rng = np.random.default_rng(seed)
    x = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    y = 0.3 + 1.1 * x + rng.normal(0, 0.005, n)
    return y, x


def ols(y, x):
    X = np.column_stack([np.ones_like(x), x])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def test_init_from_formation_matches_ols_and_W_formula():
    y, x = synthetic_logprices(252, seed=3)
    delta = 1e-3
    theta0, P0, R, W = init_from_formation(y, x, delta)
    X = np.column_stack([np.ones_like(x), x])
    n = len(y)
    resid = y - X @ ols(y, x)
    np.testing.assert_allclose(theta0, ols(y, x), rtol=1e-10)
    assert R == pytest.approx(resid @ resid / (n - 2))           # dof n - 2
    np.testing.assert_allclose(P0, R * np.linalg.inv(X.T @ X))   # OLS covariance
    np.testing.assert_allclose(W, delta * R * np.linalg.inv(X.T @ X / n))
    # delta = 0 switches the state noise off completely.
    assert np.all(init_from_formation(y, x, 0.0)[3] == 0.0)


def test_kalman_with_zero_state_noise_is_pooled_ols():
    y, x = synthetic_logprices(252 + 60, seed=4)
    yf, xf, yt, xt = y[:252], x[:252], y[252:], x[252:]
    theta0, P0, R, W = init_from_formation(yf, xf, delta=0.0)
    kf = kalman_filter(yt, xt, theta0, P0, R, W)
    for t in [0, 1, 5, 30, 59]:
        # Prior on day t = OLS on formation + trading days 0..t-1.
        expected = ols(np.r_[yf, yt[:t]], np.r_[xf, xt[:t]])
        np.testing.assert_allclose([kf.alpha[t], kf.beta[t]], expected, rtol=1e-6)


def test_first_prior_is_theta0_and_z_is_standardised_error():
    y, x = synthetic_logprices(300, seed=5)
    theta0, P0, R, W = init_from_formation(y[:250], x[:250], delta=1e-3)
    kf = kalman_filter(y[250:], x[250:], theta0, P0, R, W)
    # Day 0 uses only the formation prior.
    assert (kf.alpha[0], kf.beta[0]) == (theta0[0], theta0[1])
    # e_0 = y_0 - h_0' theta0 and F_0 = h_0' (P0 + W) h_0 + R.
    h = np.array([1.0, x[250]])
    assert kf.error[0] == pytest.approx(y[250] - h @ theta0)
    assert kf.error_var[0] == pytest.approx(h @ (P0 + W) @ h + R)
    np.testing.assert_allclose(kf.z, kf.error / np.sqrt(kf.error_var))
    # The predicted variance always exceeds the observation noise R.
    assert np.all(kf.error_var > R)


def test_kalman_does_not_mutate_inputs():
    y, x = synthetic_logprices(100, seed=6)
    theta0, P0, R, W = init_from_formation(y[:80], x[:80], 1e-3)
    th_copy, P_copy = theta0.copy(), P0.copy()
    kalman_filter(y[80:], x[80:], theta0, P0, R, W)
    np.testing.assert_array_equal(theta0, th_copy)
    np.testing.assert_array_equal(P0, P_copy)
