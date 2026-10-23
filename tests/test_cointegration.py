"""Tests for src/cointegration.py: BH correction, half-life, OLS hedge,
Engle-Granger / Johansen sanity checks and the pair-selection filters."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.cointegration import (benjamini_hochberg, engle_granger_pvalue,
                               engle_granger_pvalues, half_life,
                               johansen_trace_rejects, ols_hedge, select_pairs)
from src.config import BASELINE


# --------------------------------------------------------------------------- #
# 1. Benjamini-Hochberg
# --------------------------------------------------------------------------- #
# m = 10, q = 0.05, so the threshold for the k-th smallest p is k * 0.005:
#   k : 1      2      3      4      5      6      7      8      9      10
#   p : 0.001  0.008  0.039  0.041  0.042  0.060  0.074  0.205  0.212  0.216
# thr : 0.005  0.010  0.015  0.020  0.025  0.030  0.035  0.040  0.045  0.050
#  ok : yes    yes    no     no     no     no     no     no     no     no
# Largest k with p_(k) <= thr is k = 2, so exactly the two smallest are rejected.
BH_P = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]


def test_bh_hand_worked_example_rejects_two_smallest():
    rej = benjamini_hochberg(np.array(BH_P), q=0.05)
    assert rej.tolist() == [True, True] + [False] * 8


def test_bh_is_order_invariant():
    # Shuffling the inputs must shuffle the mask the same way.
    perm = np.random.default_rng(0).permutation(10)
    rej = benjamini_hochberg(np.array(BH_P)[perm], q=0.05)
    assert rej.tolist() == (np.arange(10)[perm] < 2).tolist()


def test_bh_step_up_rejects_p_above_its_own_threshold():
    # m = 3, q = 0.05: thresholds 0.0167, 0.0333, 0.05.
    # p_(1)=0.02 > 0.0167 fails on its own, but p_(3)=0.04 <= 0.05 passes,
    # so the step-up rule rejects all k <= 3, including p_(1).
    rej = benjamini_hochberg(np.array([0.04, 0.02, 0.035]), q=0.05)
    assert rej.all()


def test_bh_empty_input():
    rej = benjamini_hochberg(np.array([]), q=0.05)
    assert rej.shape == (0,) and rej.dtype == bool


def test_bh_all_null_rejects_nothing():
    # Smallest p 0.3 > 1 * 0.05 / 4 and every later one is above its threshold.
    assert not benjamini_hochberg(np.array([0.3, 0.5, 0.9, 0.99]), q=0.05).any()


def test_bh_matches_statsmodels_fdr_bh():
    rng = np.random.default_rng(42)
    # Mix of true signals (tiny p) and nulls (uniform p).
    p = np.concatenate([rng.uniform(0, 0.01, 15), rng.uniform(0, 1, 85)])
    expected = multipletests(p, alpha=0.05, method="fdr_bh")[0]
    np.testing.assert_array_equal(benjamini_hochberg(p, 0.05), expected)
    np.testing.assert_array_equal(benjamini_hochberg(BH_P, 0.05),
                                  multipletests(BH_P, alpha=0.05, method="fdr_bh")[0])


# --------------------------------------------------------------------------- #
# 2. Half-life
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phi", [0.5, 0.8, 0.95])
def test_half_life_exact_ar1_decay(phi):
    # s_t = phi^t * s0  =>  ds_t = (phi - 1) s_{t-1} exactly, so OLS gives
    # a = 0, b = phi - 1 with zero residuals and the half-life is ln2 / -ln(phi).
    s = 3.0 * phi ** np.arange(40)
    assert half_life(s) == pytest.approx(np.log(2) / -np.log(phi), rel=1e-6)


def test_half_life_phi_half_is_one_period():
    # Halving every period: the half-life is exactly 1.
    assert half_life(0.5 ** np.arange(30)) == pytest.approx(1.0, rel=1e-9)


def test_half_life_no_mean_reversion_is_inf():
    # Explosive growth: b = 0.1 > 0 -> phi > 1 -> inf.
    assert half_life(1.1 ** np.arange(50)) == np.inf
    # Linear trend: ds is constant, b = 0 -> phi = 1 -> inf.
    assert half_life(np.arange(50.0)) == np.inf


def test_half_life_negative_or_zero_phi_is_nan():
    # phi = -0.5: oscillating sign, not an OU process.
    assert np.isnan(half_life(3.0 * (-0.5) ** np.arange(30)))
    # phi = 0: s = [1, 0, 0, ...] gives ds_t = -s_{t-1} exactly, b = -1.
    assert np.isnan(half_life(np.r_[1.0, np.zeros(10)]))


# --------------------------------------------------------------------------- #
# 3. OLS hedge
# --------------------------------------------------------------------------- #
def test_ols_hedge_exact_line():
    x = np.linspace(3.0, 5.0, 50)
    alpha, beta, sigma2 = ols_hedge(0.7 + 1.3 * x, x)
    assert alpha == pytest.approx(0.7)
    assert beta == pytest.approx(1.3)
    assert sigma2 == pytest.approx(0.0, abs=1e-20)


def test_ols_hedge_known_residuals_and_dof():
    # Residuals e = [1, -1, -1, 1] are orthogonal to [1, x] for x = [0, 1, 2, 3]
    # (sum e = 0, sum x*e = 0 - 1 - 2 + 3 = 0), so OLS recovers alpha=2, beta=3
    # exactly and leaves e as the residuals. sigma2 = sum(e^2) / (n-2) = 4/2 = 2.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    e = np.array([1.0, -1.0, -1.0, 1.0])
    alpha, beta, sigma2 = ols_hedge(2.0 + 3.0 * x + e, x)
    assert (alpha, beta) == (pytest.approx(2.0), pytest.approx(3.0))
    assert sigma2 == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 4. Engle-Granger and Johansen sanity checks
# --------------------------------------------------------------------------- #
def test_cointegrated_pair_is_detected(cointegrated_pair):
    y, x = cointegrated_pair
    assert engle_granger_pvalue(y, x) < 0.01
    assert johansen_trace_rejects(y, x)
    # The hedge regression recovers the true relation approximately.
    _, beta, _ = ols_hedge(y, x)
    assert beta == pytest.approx(1.2, abs=0.1)


def test_independent_random_walks_not_cointegrated(independent_walks):
    a, b = independent_walks
    # With this seed the p-value is far from any significance level.
    assert engle_granger_pvalue(a, b) > 0.3
    assert not johansen_trace_rejects(a, b)


def test_eg_pvalue_is_max_of_both_directions(cointegrated_pair, independent_walks):
    for y, x in (cointegrated_pair, independent_walks):
        assert engle_granger_pvalue(y, x) == max(engle_granger_pvalues(y, x))


# --------------------------------------------------------------------------- #
# 5. select_pairs filters
# --------------------------------------------------------------------------- #
def make_tests_frame():
    """Hand-made output of test_all_pairs (m = 6 pairs tested).

    BH at q = 0.05, thresholds k * 0.05 / 6 for the sorted p-values:
      k  pair  p      thr     p <= thr?
      1  A     0.001  0.0083  yes
      2  B     0.002  0.0167  yes
      3  C     0.003  0.0250  yes
      4  D     0.004  0.0333  yes
      5  E     0.030  0.0417  yes
      6  F     0.200  0.0500  no
    So BH rejects A-E; each of B-E then fails exactly one other filter.
    """
    rows = [
        # name eg_p   johansen half_life beta
        ("A", 0.001, True, 10.0, 1.0),   # passes everything
        ("B", 0.002, False, 10.0, 1.0),  # fails Johansen
        ("C", 0.003, True, 1.0, 1.0),    # half-life too short (< 2)
        ("D", 0.004, True, 100.0, 1.0),  # half-life too long (> 63)
        ("E", 0.030, True, 20.0, -0.5),  # negative beta
        ("F", 0.200, True, 20.0, 1.0),   # not significant
    ]
    return pd.DataFrame([dict(group="g", y=n, x="X", alpha=0.0, beta=b, sigma2=1e-4,
                              spread_mean=0.0, spread_std=0.01, eg_pvalue=p,
                              eg_p_yx=p, eg_p_xy=p, johansen=j, half_life=hl)
                         for n, p, j, hl, b in rows])


def test_select_pairs_baseline_keeps_only_pair_passing_all_filters():
    sel = select_pairs(make_tests_frame(), BASELINE)
    assert sel["y"].tolist() == ["A"]


def test_select_pairs_without_johansen_requirement():
    sel = select_pairs(make_tests_frame(), BASELINE.with_(require_johansen=False))
    assert sel["y"].tolist() == ["A", "B"]


def test_select_pairs_half_life_bounds_are_inclusive():
    t = make_tests_frame()
    t.loc[t.y == "C", "half_life"] = 2.0   # exactly min -> kept
    t.loc[t.y == "D", "half_life"] = 63.0  # exactly max -> kept
    assert select_pairs(t, BASELINE)["y"].tolist() == ["A", "C", "D"]
    wide = BASELINE.with_(min_half_life=0.5, max_half_life=200.0)
    assert select_pairs(make_tests_frame(), wide)["y"].tolist() == ["A", "C", "D"]


def test_select_pairs_rejects_negative_beta():
    t = make_tests_frame()
    t.loc[t.y == "E", "beta"] = 0.5  # flip E to positive beta -> now selected
    assert select_pairs(t, BASELINE)["y"].tolist() == ["A", "E"]


def test_select_pairs_fdr_vs_raw_pvalues():
    # m = 4 pairs, all passing the other filters, p = 0.001, 0.02, 0.04, 0.06.
    # BH thresholds k * 0.05 / 4 = 0.0125, 0.025, 0.0375, 0.05:
    #   0.001 ok, 0.02 ok, 0.04 > 0.0375 no, 0.06 > 0.05 no -> BH keeps 2.
    # Raw rule p < 0.05 keeps 3 (0.001, 0.02, 0.04).
    t = make_tests_frame().iloc[:4].copy()
    t["johansen"], t["half_life"], t["beta"] = True, 10.0, 1.0
    t["eg_pvalue"] = [0.001, 0.02, 0.04, 0.06]
    assert select_pairs(t, BASELINE)["y"].tolist() == ["A", "B"]
    assert select_pairs(t, BASELINE.with_(use_fdr=False))["y"].tolist() == ["A", "B", "C"]


def test_select_pairs_sorts_by_pvalue_and_caps_count():
    t = make_tests_frame()
    t["johansen"], t["half_life"], t["beta"] = True, 10.0, 1.0
    t["eg_pvalue"] = [0.004, 0.001, 0.003, 0.002, 0.0005, 0.9]  # shuffled order
    sel = select_pairs(t, BASELINE.with_(max_pairs=3))
    assert sel["y"].tolist() == ["E", "B", "D"]  # three smallest p, ascending
    assert sel.index.tolist() == [0, 1, 2]       # index reset


def test_select_pairs_empty_input():
    assert select_pairs(pd.DataFrame(), BASELINE).empty
