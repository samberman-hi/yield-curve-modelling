"""Formation-window statistics: hedge ratio, cointegration tests, half-life,
multiple-testing correction and pair selection.

All functions here take *only* formation-window data. Nothing in this module
ever sees the trading window, which is what rules out look-ahead bias in pair
selection.

Conventions: ``y`` and ``x`` are log prices. The spread is
``s_t = y_t - alpha - beta * x_t``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .config import UNIVERSE, StrategyConfig


def ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """OLS of y on [1, x]. Returns (alpha, beta, residual variance)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = float(resid @ resid / (len(y) - 2))
    return float(coef[0]), float(coef[1]), sigma2


def half_life(spread: np.ndarray) -> float:
    """Half-life (in periods) of mean reversion from an AR(1) fit.

    Fit  ds_t = a + b * s_{t-1} + e_t  by OLS. In the discrete-time
    Ornstein-Uhlenbeck model s_t = phi * s_{t-1} + ..., phi = 1 + b and the
    half-life is ln(2) / -ln(phi). Returns inf if the spread does not mean-revert
    (b >= 0) and nan if phi <= 0 (an explosive oscillation, not an OU process).
    """
    s = np.asarray(spread, dtype=float)
    lag, ds = s[:-1], np.diff(s)
    X = np.column_stack([np.ones_like(lag), lag])
    (_, b), *_ = np.linalg.lstsq(X, ds, rcond=None)
    phi = 1.0 + b
    if phi >= 1.0:
        return float("inf")
    if phi <= 0.0:
        return float("nan")
    return float(np.log(2.0) / -np.log(phi))


def engle_granger_pvalues(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Engle-Granger two-step test in both directions: (p for y on x, p for x on y).

    Step 1 regresses one log price on the other; step 2 runs an ADF test on the
    residual and reads the p-value from MacKinnon's cointegration response
    surfaces (which account for the estimated hedge ratio).
    """
    p_yx = coint(y, x, trend="c", autolag="aic")[1]
    p_xy = coint(x, y, trend="c", autolag="aic")[1]
    return float(p_yx), float(p_xy)


def engle_granger_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """Direction-free EG p-value: the larger of the two directional p-values.

    The EG test depends on which series is the dependent variable. To avoid
    choosing the more favourable ordering after the fact we take the larger
    (more conservative) p-value. Rejecting with the max means rejecting in both
    directions.
    """
    return max(engle_granger_pvalues(y, x))


def johansen_trace_rejects(y: np.ndarray, x: np.ndarray, level: str = "5%") -> bool:
    """Johansen trace test of H0: rank = 0 (no cointegration) vs rank >= 1.

    Constant in the cointegrating relation (det_order=0) and one lagged
    difference in the VECM. statsmodels reports critical values (90/95/99%)
    rather than p-values, so the Johansen test is used as a confirmation filter.
    """
    col = {"10%": 0, "5%": 1, "1%": 2}[level]
    with warnings.catch_warnings():
        # statsmodels' eigen-solver can return eigenvalues with ~1e-17 imaginary
        # parts; it discards them with a ComplexWarning, which is harmless.
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        res = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
    return bool(res.lr1[0] > res.cvt[0, col])


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected by Benjamini-Hochberg at FDR q.

    Sort p-values p_(1) <= ... <= p_(m); find the largest k with
    p_(k) <= k q / m; reject the k smallest.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])  # last index satisfying the bound
        reject[order[: k + 1]] = True
    return reject


@dataclass
class PairStats:
    group: str
    y: str  # dependent leg (the spread is long y / short x when z < 0)
    x: str
    alpha: float
    beta: float
    sigma2: float  # residual variance
    spread_mean: float
    spread_std: float
    eg_pvalue: float  # max of the two directional p-values
    eg_p_yx: float
    eg_p_xy: float
    johansen: bool
    half_life: float


def candidate_pairs(available: list[str]) -> list[tuple[str, str, str]]:
    """All within-industry pairs whose two tickers are available."""
    out = []
    for group, tickers in UNIVERSE.items():
        tk = [t for t in tickers if t in available]
        out += [(group, a, b) for a, b in combinations(tk, 2)]
    return out


def pair_stats(group: str, a: str, b: str, logp: pd.DataFrame) -> PairStats:
    ya, yb = logp[a].to_numpy(), logp[b].to_numpy()
    p_yx, p_xy = engle_granger_pvalues(ya, yb)
    joh = johansen_trace_rejects(ya, yb)
    alpha, beta, sigma2 = ols_hedge(ya, yb)
    spread = ya - alpha - beta * yb
    return PairStats(group, a, b, alpha, beta, sigma2, float(spread.mean()),
                     float(spread.std(ddof=1)), max(p_yx, p_xy), p_yx, p_xy, joh,
                     half_life(spread))


def test_all_pairs(formation_prices: pd.DataFrame) -> pd.DataFrame:
    """Run the tests on every within-industry pair with complete data."""
    complete = formation_prices.columns[formation_prices.notna().all()].tolist()
    logp = np.log(formation_prices[complete])
    rows = [pair_stats(g, a, b, logp).__dict__ for g, a, b in candidate_pairs(complete)]
    return pd.DataFrame(rows)


def select_pairs(tests: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Apply the selection rules to the output of ``test_all_pairs``.

    1. EG p-value significant after Benjamini-Hochberg across *all* pairs tested
       in the window (or raw p < q if ``use_fdr`` is False);
    2. Johansen trace test also rejects no-cointegration at 5%;
    3. half-life within [min_half_life, max_half_life] trading days;
    4. positive hedge ratio (otherwise it is not a long/short relative-value trade);
    5. keep at most ``max_pairs``, ranked by EG p-value.
    """
    if tests.empty:
        return tests
    t = tests.copy()
    if cfg.use_fdr:
        t["significant"] = benjamini_hochberg(t["eg_pvalue"].to_numpy(), cfg.fdr_q)
    else:
        t["significant"] = t["eg_pvalue"] < cfg.fdr_q
    keep = t["significant"]
    if cfg.require_johansen:
        keep &= t["johansen"]
    keep &= t["half_life"].between(cfg.min_half_life, cfg.max_half_life)
    keep &= t["beta"] > 0
    return t[keep].sort_values("eg_pvalue").head(cfg.max_pairs).reset_index(drop=True)
