"""Principal component analysis of daily Treasury yield changes.

PCA is run on the covariance matrix of daily yield changes in basis points (all
maturities share the same units, so no standardisation is needed and the
components keep a direct "bp move" interpretation). Eigenvector signs are
arbitrary, so we fix them for interpretation:

* PC1 (level):     loadings sum to a positive number (all yields rise).
* PC2 (slope):     long end loading > short end loading (steepening).
* PC3 (curvature): belly loading > average of the two ends (hump rises).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def yield_changes(yields: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Daily changes in bp over consecutive observations where all chosen maturities exist."""
    cols = columns or list(yields.columns)
    return yields[cols].dropna().diff().dropna() * 100.0


@dataclass
class PCAResult:
    loadings: pd.DataFrame  # maturities x components (unit-length eigenvectors)
    eigenvalues: pd.Series  # variance of each component (bp^2 per day)
    explained: pd.Series  # share of total variance
    scores: pd.DataFrame  # days x components (bp)


def _orient(vecs: np.ndarray) -> np.ndarray:
    """Apply the sign conventions in the module docstring to the first 3 eigenvectors."""
    v = vecs.copy()
    n = v.shape[0]
    if v[:, 0].sum() < 0:
        v[:, 0] *= -1
    if v.shape[1] > 1 and v[-1, 1] < v[0, 1]:
        v[:, 1] *= -1
    if v.shape[1] > 2:
        belly = v[n // 3: 2 * n // 3 + 1, 2].mean()
        ends = 0.5 * (v[0, 2] + v[-1, 2])
        if belly < ends:
            v[:, 2] *= -1
    return v


def fit_pca(changes: pd.DataFrame, n_components: int | None = None) -> PCAResult:
    """Eigen-decomposition of the sample covariance matrix of yield changes."""
    X = changes.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)  # ascending order for symmetric matrices
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], _orient(vecs[:, order])
    k = n_components or len(vals)
    names = [f"PC{i + 1}" for i in range(k)]
    loadings = pd.DataFrame(vecs[:, :k], index=changes.columns, columns=names)
    eig = pd.Series(vals[:k], index=names)
    explained = pd.Series(vals[:k] / vals.sum(), index=names)
    scores = pd.DataFrame(X @ vecs[:, :k], index=changes.index, columns=names)
    return PCAResult(loadings, eig, explained, scores)


def congruence(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """Absolute cosine similarity (Tucker's congruence) between two loading vectors.

    1 = identical shape; values above ~0.95 are conventionally read as 'the same factor'.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def subperiod_stability(changes: pd.DataFrame, periods: dict[str, tuple[str, str]],
                        k: int = 3) -> tuple[pd.DataFrame, dict[str, PCAResult]]:
    """Fit PCA in each sub-period; compare variance shares and loadings with the full sample."""
    full = fit_pca(changes, k)
    rows, fits = [], {}
    for name, (start, end) in periods.items():
        sub = changes.loc[start:end]
        res = fit_pca(sub, k)
        fits[name] = res
        row = {"period": name, "start": sub.index[0].date(), "end": sub.index[-1].date(),
               "days": len(sub)}
        for i in range(k):
            pc = f"PC{i + 1}"
            row[f"{pc} var %"] = 100 * res.explained[pc]
            row[f"{pc} congruence"] = congruence(res.loadings[pc], full.loadings[pc])
        row["PC1-3 var %"] = 100 * res.explained.iloc[:k].sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("period"), fits


def out_of_sample_explained(train: pd.DataFrame, test: pd.DataFrame, k: int = 3) -> pd.Series:
    """Share of test-period variance captured by eigenvectors estimated on the training period.

    explained_k = trace(V_k' S_test V_k) / trace(S_test), with V_k the first k
    training eigenvectors and S_test the test-period covariance matrix. If the
    factor structure were unstable this would fall well below the in-sample share.
    """
    V = fit_pca(train, k).loadings.to_numpy()
    S = np.cov(test.to_numpy(dtype=float), rowvar=False)
    total = np.trace(S)
    out = {f"PC1-{j}": float(np.trace(V[:, :j].T @ S @ V[:, :j]) / total) for j in range(1, k + 1)}
    return pd.Series(out)
