"""Shared synthetic data for the unit tests.

Everything is generated from fixed seeds, so every run sees identical numbers.
No test touches the network or data/raw/.
"""

import numpy as np
import pytest


@pytest.fixture
def cointegrated_pair():
    """x: random walk in log prices; y = 0.5 + 1.2 x + stationary AR(1) noise (phi=0.8)."""
    rng = np.random.default_rng(0)
    n = 500
    x = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    e = rng.normal(0, 0.01, n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + e[t]
    y = 0.5 + 1.2 * x + u
    return y, x


@pytest.fixture
def independent_walks():
    """Two unrelated random walks (seed 1): the null of no cointegration is true."""
    rng = np.random.default_rng(1)
    n = 500
    a = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    b = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    return a, b
