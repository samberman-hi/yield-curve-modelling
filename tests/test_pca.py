"""PCA of yield changes checked on synthetic data with a known 3-factor structure."""

import numpy as np
import pandas as pd
import pytest

from src.pca import congruence, fit_pca, out_of_sample_explained, yield_changes

MATS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
SDS = np.array([10.0, 3.0, 1.0])  # bp/day of level, slope, curvature


def _factors():
    """Orthonormal level / slope / curvature shapes (Gram-Schmidt of 1, log m, (log m)^2)."""
    x = np.log(MATS)
    q, _ = np.linalg.qr(np.column_stack([np.ones_like(x), x, x**2]))
    if q[:, 0].sum() < 0:  # level loadings positive
        q[:, 0] *= -1
    if q[-1, 1] < q[0, 1]:  # slope rises with maturity
        q[:, 1] *= -1
    return q


def _changes(T=2000, seed=0):
    """Changes = scores @ factors', with scores exactly uncorrelated with variances SDS^2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, 3))
    z -= z.mean(axis=0)
    qz, _ = np.linalg.qr(z)  # orthonormal, mean-zero columns
    scores = qz * np.sqrt(T - 1) * SDS
    X = scores @ _factors().T
    return pd.DataFrame(X, index=pd.bdate_range("2010-01-01", periods=T), columns=MATS)


def test_three_factor_variance_shares():
    """Explained shares equal 100/110, 9/110, 1/110 and the first three sum to 1."""
    res = fit_pca(_changes())
    np.testing.assert_allclose(res.explained.iloc[:3], SDS**2 / (SDS**2).sum(), atol=1e-10)
    assert res.explained.iloc[:3].sum() == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(res.eigenvalues.iloc[:3], SDS**2, rtol=1e-10)


def test_loadings_recover_true_factors():
    """Each estimated loading vector has congruence 1 with the factor that generated it."""
    res = fit_pca(_changes(), 3)
    for i, f in enumerate(_factors().T):
        assert congruence(res.loadings.iloc[:, i], f) == pytest.approx(1.0, abs=1e-8)


def test_sign_conventions():
    """PC1 loadings sum > 0, PC2 long end > short end, PC3 belly > ends (even if data are negated)."""
    for X in (_changes(), -_changes()):
        L = fit_pca(X, 3).loadings
        assert L["PC1"].sum() > 0
        assert L["PC2"].iloc[-1] > L["PC2"].iloc[0]
        n = len(L)
        assert L["PC3"].iloc[n // 3: 2 * n // 3 + 1].mean() > 0.5 * (L["PC3"].iloc[0] + L["PC3"].iloc[-1])


def test_congruence_known_values():
    """Congruence of v with v or -v is 1, with an orthogonal vector 0, and (1,0)~(1,1) is 1/sqrt(2)."""
    v = np.array([1.0, 2.0, 3.0])
    assert congruence(v, v) == pytest.approx(1.0)
    assert congruence(v, -v) == pytest.approx(1.0)
    assert congruence(v, np.array([3.0, 0.0, -1.0])) == pytest.approx(0.0)
    assert congruence([1, 0], [1, 1]) == pytest.approx(1 / np.sqrt(2))


def test_out_of_sample_equals_in_sample_when_train_is_test():
    """With train == test, trace(V'SV)/trace(S) equals the cumulative in-sample explained share."""
    rng = np.random.default_rng(3)
    X = _changes() + rng.normal(0, 0.5, (2000, len(MATS)))
    oos = out_of_sample_explained(X, X, k=3)
    ins = fit_pca(X).explained.iloc[:3].cumsum()
    np.testing.assert_allclose(oos.to_numpy(), ins.to_numpy(), rtol=1e-10)


def test_yield_changes_in_bp():
    """Percent levels 4.00 -> 4.10 -> 4.05 become changes of +10bp and -5bp; NaN rows are dropped."""
    y = pd.DataFrame({"a": [4.00, 4.10, np.nan, 4.05], "b": [1.0, 1.0, 1.0, 1.2]},
                     index=pd.bdate_range("2024-01-01", periods=4))
    ch = yield_changes(y)
    np.testing.assert_allclose(ch["a"], [10.0, -5.0])
    np.testing.assert_allclose(ch["b"], [0.0, 20.0])
