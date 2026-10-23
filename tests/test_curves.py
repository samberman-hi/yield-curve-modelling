"""Nelson-Siegel(-Svensson) loadings and fitting checked on noiseless synthetic curves."""

import numpy as np
import pandas as pd
import pytest

from src.curves import (DIEBOLD_LI_TAU, TAU_GRID, curve, design_matrix, empirical_factors,
                        fit_curves, loading_curv, loading_slope)

MATS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
COLS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7",
        "DGS10", "DGS20", "DGS30"]


def _ns_panel(params, tau):
    """Days x maturities DataFrame of exact NS yields (percent)."""
    rows = [curve(MATS, np.array(b), tau) for b in params]
    return pd.DataFrame(rows, index=pd.bdate_range("2020-01-01", periods=len(params)),
                        columns=COLS)


def test_loading_limits():
    """L1(0) = 1, L2(0) = 0, and both loadings go to 0 as maturity -> infinity."""
    tau = 1.5
    assert loading_slope(0.0, tau) == pytest.approx(1.0)
    assert loading_curv(0.0, tau) == pytest.approx(0.0)
    assert loading_slope(1e4, tau) == pytest.approx(0.0, abs=1e-3)
    assert loading_curv(1e4, tau) == pytest.approx(0.0, abs=1e-3)
    # value at m = tau: L1 = 1 - e^-1 = 0.632121, L2 = 1 - 2e^-1 = 0.264241
    assert loading_slope(tau, tau) == pytest.approx(1 - np.exp(-1))
    assert loading_curv(tau, tau) == pytest.approx(1 - 2 * np.exp(-1))


def test_diebold_li_tau_puts_hump_at_30_months():
    """With lambda = 0.0609/month the curvature loading peaks at ~30 months (Diebold & Li 2006)."""
    m = np.linspace(0.01, 10, 100_001)
    peak_months = 12 * m[np.argmax(loading_curv(m, DIEBOLD_LI_TAU))]
    assert peak_months == pytest.approx(30, abs=1.0)  # exact argmax is 1.7933*tau = 29.4 months


def test_ns_fit_recovers_exact_parameters():
    """A noiseless NS curve with tau on TAU_GRID is fitted with RMSE ~0 and the true betas and tau."""
    tau = TAU_GRID[15]
    params = [(4.5, -2.0, 1.5), (3.0, 1.0, -2.0), (6.0, -4.0, 0.5)]
    fit = fit_curves(_ns_panel(params, tau), MATS, model="NS")
    np.testing.assert_allclose(fit.params[["b0", "b1", "b2"]].to_numpy(), params, atol=1e-8)
    np.testing.assert_allclose(fit.params["tau1"], tau)
    assert fit.rmse_bp.max() < 1e-6


def test_ns_fit_handles_missing_maturities():
    """Days with a NaN 30y (a different maturity pattern) are fitted separately and still exact."""
    tau = TAU_GRID[20]
    params = [(4.5, -2.0, 1.5), (5.0, -1.0, 2.0), (3.5, 0.5, -1.0)]
    y = _ns_panel(params, tau)
    y.iloc[1, -1] = np.nan  # 30y missing on day 2 (the 2002-06 gap)
    y.iloc[2, 0] = np.nan  # 1m missing on day 3
    fit = fit_curves(y, MATS, model="NS")
    np.testing.assert_allclose(fit.params[["b0", "b1", "b2"]].to_numpy(), params, atol=1e-8)
    # the fitted curve is filled in at the missing maturity
    assert fit.fitted.iloc[1, -1] == pytest.approx(curve(MATS, params[1], tau)[-1])
    assert np.isnan(fit.residuals.iloc[1, -1])


def test_fixed_tau_betas_equal_ols():
    """With tau fixed, the betas equal OLS of yields on [1, L1, L2] (numpy lstsq)."""
    rng = np.random.default_rng(0)
    y = _ns_panel([(4.5, -2.0, 1.5), (3.0, 1.0, -2.0)], 2.0)
    y = y + rng.normal(0, 0.05, y.shape)  # noisy, so the model is no longer exact
    fit = fit_curves(y, MATS, model="NS", tau=DIEBOLD_LI_TAU)
    X = design_matrix(MATS, DIEBOLD_LI_TAU)
    for i in range(len(y)):
        beta, *_ = np.linalg.lstsq(X, y.iloc[i].to_numpy(), rcond=None)
        np.testing.assert_allclose(fit.params.iloc[i][["b0", "b1", "b2"]], beta, atol=1e-10)
    assert fit.model == "NS (fixed tau)"


def test_nss_fits_at_least_as_well_as_ns():
    """NSS nests NS, so its in-sample RMSE is never larger on the same (noisy) data."""
    rng = np.random.default_rng(1)
    y = _ns_panel([(4.5, -2.0, 1.5), (3.0, 1.0, -2.0), (5.0, -3.0, 3.0)], 1.0)
    y = y + rng.normal(0, 0.03, y.shape)
    ns = fit_curves(y, MATS, model="NS")
    nss = fit_curves(y, MATS, model="NSS")
    assert (nss.rmse_bp.to_numpy() <= ns.rmse_bp.to_numpy() + 1e-9).all()


def test_empirical_factors_hand_row():
    """3M=1, 2Y=2, 10Y=4 -> level 7/3, slope 3, curvature 2*2 - 1 - 4 = -1."""
    y = pd.DataFrame({"DGS3MO": [1.0], "DGS2": [2.0], "DGS10": [4.0]})
    f = empirical_factors(y).iloc[0]
    assert f["level"] == pytest.approx(7 / 3)
    assert f["slope"] == pytest.approx(3.0)
    assert f["curvature"] == pytest.approx(-1.0)


def test_leave_one_out_scores_only_interpolation():
    """LOO error is NaN on days where the held-out maturity is the longest one observed."""
    from src.curves import leave_one_out_errors
    mats = [0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    row = curve(np.array(mats), np.array([4.0, -1.0, 1.0]), TAU_GRID[10])
    y = pd.DataFrame([row, row], columns=[f"m{m}" for m in mats])
    y.iloc[1, -1] = np.nan  # 30Y missing on day 2 -> 20Y would be extrapolation
    err = leave_one_out_errors(y, mats, "NS")
    assert np.isfinite(err.loc[0, "m20.0"]) and abs(err.loc[0, "m20.0"]) < 1e-6
    assert np.isnan(err.loc[1, "m20.0"])
    assert np.isfinite(err.loc[1, "m10.0"])
