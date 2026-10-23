"""Nelson–Siegel (NS) and Nelson–Siegel–Svensson (NSS) yield-curve fitting.

Model (yields y in percent, maturity m in years, decay parameters tau in years):

    NS:  y(m) = b0 + b1 * L1(m/tau1) + b2 * L2(m/tau1)
    NSS: y(m) = NS + b3 * L2(m/tau2)

    L1(x) = (1 - exp(-x)) / x                 slope loading, 1 at m=0 -> 0 as m->inf
    L2(x) = (1 - exp(-x)) / x - exp(-x)       hump loading, 0 at both ends

Interpretation: b0 = y(inf) (level), b0 + b1 = y(0) so -b1 = long minus short
(slope), b2 = medium-term hump (curvature).

Estimation: given the decay parameters the model is linear in the betas, so we
use *profile least squares*: for every tau (NS) or (tau1, tau2) pair (NSS) on a
log-spaced grid, solve OLS for the betas, and keep the grid point with the lowest
sum of squared errors. This finds the global optimum on the grid and avoids the
local minima that plague direct non-linear NSS optimisation. Everything is
vectorised over days; days with different sets of available maturities (the
30-year gap) are fitted separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Diebold & Li (2006): lambda = 0.0609 per month, which puts the maximum of the
# curvature loading at 30 months. In years: tau = 1 / (0.0609 * 12) = 1.368.
DIEBOLD_LI_TAU = 1.0 / (0.0609 * 12)

# Decay-parameter grid in years (step ratio ~1.10). The curvature loading peaks at
# m ~= 1.8 * tau, so [0.25, 10] places the hump anywhere from ~5 months to ~18 years,
# i.e. inside the observed maturity range. A wider grid (0.1-30y) fits slightly
# better in sample but lets b0 drift to the unidentified far end of the curve;
# see the robustness table in the README.
TAU_GRID = np.geomspace(0.25, 10.0, 40)
WIDE_TAU_GRID = np.geomspace(0.1, 30.0, 45)
NSS_MIN_TAU_RATIO = 2.0  # require tau2 >= 2 * tau1 so the two humps are identifiable


def loading_slope(m: np.ndarray, tau: float) -> np.ndarray:
    """L1 = (1 - e^{-m/tau}) / (m/tau), with the m -> 0 limit equal to 1."""
    x = np.asarray(m, dtype=float) / tau
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(x > 1e-10, -np.expm1(-x) / x, 1.0)
    return out


def loading_curv(m: np.ndarray, tau: float) -> np.ndarray:
    """L2 = L1 - e^{-m/tau}; zero at m = 0 and as m -> infinity."""
    x = np.asarray(m, dtype=float) / tau
    return loading_slope(m, tau) - np.exp(-x)


def design_matrix(m: np.ndarray, tau1: float, tau2: float | None = None) -> np.ndarray:
    """Columns [1, L1(tau1), L2(tau1)] (+ [L2(tau2)] for NSS) evaluated at maturities m."""
    m = np.asarray(m, dtype=float)
    cols = [np.ones_like(m), loading_slope(m, tau1), loading_curv(m, tau1)]
    if tau2 is not None:
        cols.append(loading_curv(m, tau2))
    return np.column_stack(cols)


def curve(m: np.ndarray, betas: np.ndarray, tau1: float, tau2: float | None = None) -> np.ndarray:
    """Evaluate a fitted NS/NSS curve at maturities m."""
    return design_matrix(m, tau1, tau2) @ np.asarray(betas, dtype=float)


@dataclass
class CurveFit:
    """Daily fit results. ``params`` has columns b0..b2[, b3], tau1[, tau2]."""

    model: str
    params: pd.DataFrame
    fitted: pd.DataFrame  # fitted yields at the observed maturities (percent)
    residuals: pd.DataFrame  # observed minus fitted, in basis points
    rmse_bp: pd.Series

    def predict(self, maturities: np.ndarray) -> pd.DataFrame:
        """Fitted yields (percent) at arbitrary maturities for every day."""
        m = np.asarray(maturities, dtype=float)
        out = np.empty((len(self.params), len(m)))
        p = self.params
        nss = self.model == "NSS"
        for i, row in enumerate(p.itertuples(index=False)):
            betas = [row.b0, row.b1, row.b2] + ([row.b3] if nss else [])
            out[i] = curve(m, np.array(betas), row.tau1, row.tau2 if nss else None)
        return pd.DataFrame(out, index=p.index, columns=m)


def _tau_candidates(model: str, tau: float | None,
                    grid: np.ndarray) -> list[tuple[float, float | None]]:
    if model == "NS":
        return [(tau, None)] if tau is not None else [(t, None) for t in grid]
    if model == "NSS":
        return [(t1, t2) for t1 in grid for t2 in grid if t2 >= NSS_MIN_TAU_RATIO * t1]
    raise ValueError(f"Unknown model {model!r}")


def _fit_block(Y: np.ndarray, m: np.ndarray, cands: list[tuple[float, float | None]]):
    """Fit all rows of Y (days x maturities, no NaNs) at once; return betas, taus, fitted."""
    n_days = Y.shape[0]
    best_sse = np.full(n_days, np.inf)
    best_beta = None
    best_tau = np.zeros((n_days, 2))
    for tau1, tau2 in cands:
        X = design_matrix(m, tau1, tau2)
        B = Y @ np.linalg.pinv(X).T  # OLS betas for every day at once
        sse = ((Y - B @ X.T) ** 2).sum(axis=1)
        better = sse < best_sse
        if best_beta is None:
            best_beta = np.zeros((n_days, X.shape[1]))
        best_sse[better] = sse[better]
        best_beta[better] = B[better]
        best_tau[better] = (tau1, np.nan if tau2 is None else tau2)
    return best_beta, best_tau


def fit_curves(yields: pd.DataFrame, mats: list[float] | np.ndarray, model: str = "NS",
               tau: float | None = None, tau_grid: np.ndarray | None = None) -> CurveFit:
    """Fit NS or NSS to every row of ``yields`` (percent; NaN = maturity unavailable).

    Parameters
    ----------
    yields : DataFrame, days x maturities, in percent.
    mats : maturities in years, aligned with the columns.
    model : "NS" or "NSS".
    tau : for NS only, fix the decay parameter (e.g. ``DIEBOLD_LI_TAU``); if None it
        is chosen each day by grid search.
    tau_grid : decay-parameter grid (defaults to ``TAU_GRID``).
    """
    m_all = np.asarray(mats, dtype=float)
    Yall = yields.to_numpy(dtype=float)
    k = 4 if model == "NSS" else 3
    betas = np.full((len(yields), k), np.nan)
    taus = np.full((len(yields), 2), np.nan)
    fitted = np.full_like(Yall, np.nan)

    cands = _tau_candidates(model, tau, TAU_GRID if tau_grid is None else tau_grid)
    masks = ~np.isnan(Yall)
    patterns, inverse = np.unique(masks, axis=0, return_inverse=True)
    for p_idx, pattern in enumerate(patterns):
        rows = np.where(inverse.ravel() == p_idx)[0]
        if pattern.sum() <= k + 1:  # need more maturities than parameters
            continue
        m = m_all[pattern]
        B, T = _fit_block(Yall[np.ix_(rows, np.where(pattern)[0])], m, cands)
        betas[rows], taus[rows] = B, T
        for j, r in enumerate(rows):  # fitted values at all maturities, observed or not
            fitted[r] = curve(m_all, B[j], T[j, 0], None if np.isnan(T[j, 1]) else T[j, 1])

    names = ["b0", "b1", "b2", "b3"][:k]
    params = pd.DataFrame(betas, index=yields.index, columns=names)
    params["tau1"] = taus[:, 0]
    if model == "NSS":
        params["tau2"] = taus[:, 1]
    fitted_df = pd.DataFrame(fitted, index=yields.index, columns=yields.columns)
    resid = (yields - fitted_df) * 100.0  # bp; NaN where the maturity is unobserved
    rmse = np.sqrt((resid**2).mean(axis=1))
    label = model if (model == "NSS" or tau is None) else "NS (fixed tau)"
    return CurveFit(label, params, fitted_df, resid, rmse.rename(f"rmse_bp_{label}"))


def leave_one_out_errors(yields: pd.DataFrame, mats: list[float], model: str = "NS",
                         tau: float | None = None, holdouts: list[str] | None = None,
                         tau_grid: np.ndarray | None = None) -> pd.DataFrame:
    """Out-of-sample check for the curve fit: drop one maturity, refit, predict it.

    Returns a DataFrame (days x held-out maturities) of prediction errors in bp.
    Only interior maturities are held out by default, because extrapolating
    beyond the shortest or longest observed point is a different exercise.
    """
    holdouts = holdouts or list(yields.columns[1:-1])
    out = {}
    for col in holdouts:
        y_train = yields.copy()
        y_train[col] = np.nan
        fit = fit_curves(y_train, mats, model=model, tau=tau, tau_grid=tau_grid)
        out[col] = (yields[col] - fit.fitted[col]) * 100.0
    return pd.DataFrame(out)


def empirical_factors(yields: pd.DataFrame) -> pd.DataFrame:
    """Model-free level, slope and curvature proxies (percent), as in Diebold & Li (2006).

    level = (3M + 2Y + 10Y) / 3,  slope = 10Y - 3M,  curvature = 2 * 2Y - 3M - 10Y.
    """
    y3m, y2, y10 = yields["DGS3MO"], yields["DGS2"], yields["DGS10"]
    return pd.DataFrame({
        "level": (y3m + y2 + y10) / 3,
        "slope": y10 - y3m,
        "curvature": 2 * y2 - y3m - y10,
    })
