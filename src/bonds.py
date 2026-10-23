"""Par-bond analytics: price, modified duration, convexity, DV01 and approximate P&L.

Constant-maturity Treasury (CMT) yields are par yields on a semi-annual
bond-equivalent basis, so we treat each maturity as a par bond paying a
semi-annual coupon equal to its yield. All yields here are decimals (0.045 = 4.5%).
"""

from __future__ import annotations

import numpy as np

FREQ = 2  # semi-annual coupons, as for US Treasury notes and bonds


def bond_price(coupon: float, ytm: float | np.ndarray, maturity: float, freq: int = FREQ):
    """Price per 1 of face of a bullet bond with an integer number of coupon periods."""
    n = int(round(maturity * freq))
    c = coupon / freq
    y = np.asarray(ytm, dtype=float) / freq
    t = np.arange(1, n + 1)
    disc = (1 + y[..., None]) ** (-t)
    return c * disc.sum(axis=-1) + disc[..., -1]


def par_duration_convexity(ytm, maturity: float, freq: int = FREQ):
    """Modified duration and convexity (in years, years^2) of a par bond.

    Computed directly from the discounted cash flows (coupon y/f each period,
    principal at period n), with t measured in coupon periods:
        Macaulay D = sum(t * PV_t) / (P * f),   MD = D / (1 + y/f)
        C = sum(t (t+1) PV_t) / (P * f^2 * (1 + y/f)^2)
    For a par bond MD has the closed form (1 - (1 + y/f)^(-n)) / y, which the
    tests use as a check. Works on scalars or numpy arrays of yields.
    """
    y = np.maximum(np.asarray(ytm, dtype=float), 1e-6)  # guard the y -> 0 limit
    n = int(round(maturity * freq))
    t = np.arange(1, n + 1)  # coupon periods
    per = 1 + y[..., None] / freq
    cf = y[..., None] / freq + (t == n)  # coupon each period, plus principal at maturity
    pv = cf * per ** (-t)
    price = pv.sum(axis=-1)
    mac = (pv * t).sum(axis=-1) / price / freq  # Macaulay duration in years
    mod = mac / (1 + y / freq)
    conv = (pv * t * (t + 1)).sum(axis=-1) / price / (freq**2 * (1 + y / freq) ** 2)
    return mod, conv


def dv01(ytm, maturity: float, notional: float = 1.0, freq: int = FREQ):
    """Dollar value of a 1bp yield fall for a par bond: notional * MD * 1e-4."""
    mod, _ = par_duration_convexity(ytm, maturity, freq)
    return notional * mod * 1e-4


def approx_return(dy, mod_dur, convexity):
    """Second-order price return: dP/P ~= -MD * dy + 0.5 * C * dy^2 (dy in decimals)."""
    dy = np.asarray(dy, dtype=float)
    return -mod_dur * dy + 0.5 * convexity * dy**2


def dv01_neutral_ratio(dv01_short_leg: float, dv01_long_leg: float) -> float:
    """Notional of the 10y leg per 1 notional of the 2y leg so that DV01s offset.

    N_10 * DV01_10 = N_2 * DV01_2  =>  N_10 / N_2 = DV01_2 / DV01_10.
    """
    return dv01_short_leg / dv01_long_leg
