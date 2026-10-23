"""Par-bond analytics checked against hand-computed and textbook values."""

import numpy as np
import pytest

from src.bonds import (approx_return, bond_price, dv01, dv01_neutral_ratio,
                       par_duration_convexity)


def test_par_bond_prices_at_par():
    """A bond priced at a yield equal to its coupon is worth exactly 1.0 per 1 of face."""
    for y, mat in [(0.01, 2), (0.045, 10), (0.08, 30)]:
        assert bond_price(y, y, mat) == pytest.approx(1.0, abs=1e-12)


def test_two_year_5pct_bond_at_6pct_hand_computed():
    """2y 5% semi-annual bond at 6%: cash flows 2.5, 2.5, 2.5, 102.5 discounted at 3% = 98.1415."""
    expected = (2.5 / 1.03 + 2.5 / 1.03**2 + 2.5 / 1.03**3 + 102.5 / 1.03**4) / 100
    assert bond_price(0.05, 0.06, 2.0) == pytest.approx(expected, abs=1e-12)
    assert bond_price(0.05, 0.06, 2.0) == pytest.approx(0.981415, abs=1e-6)


def test_zero_coupon_price():
    """With a zero coupon, price = (1 + y/2)^(-2T): 10y at 4% -> 1.02^-20 = 0.672971."""
    assert bond_price(0.0, 0.04, 10.0) == pytest.approx(1.02**-20, abs=1e-12)
    assert bond_price(0.0, 0.04, 10.0) == pytest.approx(0.672971, abs=1e-6)


def test_par_modified_duration_closed_form():
    """Par-bond modified duration equals (1 - (1 + y/2)^(-n)) / y, e.g. 8.1757 for 10y at 4%."""
    for y, mat in [(0.02, 2), (0.04, 10), (0.05, 30)]:
        mod, _ = par_duration_convexity(y, mat)
        n = 2 * mat
        assert mod == pytest.approx((1 - (1 + y / 2) ** (-n)) / y, rel=1e-12)
    assert par_duration_convexity(0.04, 10)[0] == pytest.approx(8.1757, abs=1e-4)


def test_duration_convexity_match_finite_differences():
    """MD = -P'/P and C = P''/P, checked by central differences of the exact price."""
    y, mat, h = 0.04, 10.0, 1e-5
    mod, conv = par_duration_convexity(y, mat)
    p_up, p_0, p_dn = (bond_price(y, y + d, mat) for d in (h, 0.0, -h))
    assert mod == pytest.approx(-(p_up - p_dn) / (2 * h) / p_0, rel=1e-6)
    assert conv == pytest.approx((p_up - 2 * p_0 + p_dn) / h**2 / p_0, rel=1e-4)


@pytest.mark.parametrize("dy", [0.001, -0.001, 0.01, -0.01])
def test_second_order_approximation_beats_first_order(dy):
    """For a 10y 4% par bond, duration+convexity error is far smaller than duration-only error."""
    y, mat = 0.04, 10.0
    mod, conv = par_duration_convexity(y, mat)
    exact = bond_price(y, y + dy, mat) - 1.0
    first = -mod * dy
    second = approx_return(dy, mod, conv)
    assert abs(second - exact) < 0.1 * abs(first - exact)
    assert abs(second - exact) < 5e-4  # < 5 cents per $100 even for a 100bp move


def test_dv01_10y_par_bond():
    """DV01 of $10mm 10y par at 4% = 1e7 * 8.1757e-4 = $8,175.7 (i.e. ~$818 per $1mm)."""
    md = (1 - 1.02**-20) / 0.04
    d = dv01(0.04, 10.0, notional=10_000_000)
    assert d == pytest.approx(10_000_000 * md * 1e-4, rel=1e-12)
    assert 8_100 < d < 8_200
    assert 810 < dv01(0.04, 10.0, notional=1_000_000) < 820


def test_dv01_neutral_ratio_equalises_dv01():
    """N10 = N2 * DV01_2 / DV01_10 makes the two legs' dollar DV01s equal."""
    d2, d10 = dv01(0.04, 2.0), dv01(0.045, 10.0)
    n2 = 50_000_000
    n10 = n2 * dv01_neutral_ratio(d2, d10)
    assert n10 * d10 == pytest.approx(n2 * d2, rel=1e-12)
    assert n10 < n2  # the longer bond needs less notional
