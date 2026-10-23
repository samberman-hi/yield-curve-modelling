"""Kalman filter for a time-varying hedge ratio.

State-space model (log prices):

    observation:  y_t = alpha_t + beta_t * x_t + e_t,      e_t ~ N(0, R)
    state:        theta_t = theta_{t-1} + w_t,             w_t ~ N(0, W)

with theta_t = (alpha_t, beta_t)'. The filter is initialised and tuned on the
formation window only:

* theta_0 and P_0 are the OLS estimate and its covariance on the formation window;
* R is the OLS residual variance on the formation window;
* W = delta * R * (X'X / n)^{-1}. This scaling makes delta unit-free: the
  state noise adds roughly 2 * delta * R of variance per day to the fitted
  value, so delta = 1e-3 lets the equilibrium drift by about
  sqrt(126 * 2e-3) ~ 0.5 residual standard deviations over a 6-month window.
  delta is fixed a priori (not fitted).

The trading signal uses the *one-step-ahead* prediction error
e_t = y_t - x_t' theta_{t|t-1}, standardised by its predicted variance
F_t = x_t' P_{t|t-1} x_t + R. The z-score on day t uses only information up to
day t, and theta_{t|t-1} only information up to t-1.

Caveat: the model assumes i.i.d. observation noise, but a cointegrating spread
is an AR(1) process. The model is therefore misspecified: prediction errors are
strongly autocorrelated (lag-1 autocorrelation about 0.9 here, versus ~0 for a
well-specified filter), and the drifting intercept partly absorbs the very
deviations the strategy trades. z is a *scaled* prediction error, not an
N(0, 1) innovation. It is still a valid, look-ahead-free signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    alpha: np.ndarray  # prior (one-step-ahead) intercept used on each day
    beta: np.ndarray  # prior hedge ratio used on each day
    error: np.ndarray  # prediction error e_t
    error_var: np.ndarray  # F_t
    z: np.ndarray  # e_t / sqrt(F_t)


def init_from_formation(y: np.ndarray, x: np.ndarray, delta: float
                        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """OLS on the formation window -> (theta0, P0, R, W)."""
    X = np.column_stack([np.ones_like(x), x])
    n = len(y)
    XtX = X.T @ X
    theta0 = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ theta0
    R = float(resid @ resid / (n - 2))
    P0 = R * np.linalg.inv(XtX)
    W = delta * R * np.linalg.inv(XtX / n)
    return theta0, P0, R, W


def kalman_filter(y: np.ndarray, x: np.ndarray, theta0: np.ndarray, P0: np.ndarray,
                  R: float, W: np.ndarray) -> KalmanResult:
    """Run the filter over (y, x) starting from the prior (theta0, P0)."""
    n = len(y)
    theta, P = theta0.astype(float).copy(), P0.astype(float).copy()
    out = {k: np.empty(n) for k in ("alpha", "beta", "error", "error_var")}
    for t in range(n):
        # Predict: random-walk state
        P = P + W
        h = np.array([1.0, x[t]])
        e = y[t] - h @ theta
        F = float(h @ P @ h + R)
        out["alpha"][t], out["beta"][t] = theta
        out["error"][t], out["error_var"][t] = e, F
        # Update
        K = P @ h / F
        theta = theta + K * e
        P = P - np.outer(K, h) @ P
    return KalmanResult(out["alpha"], out["beta"], out["error"], out["error_var"],
                        out["error"] / np.sqrt(out["error_var"]))
