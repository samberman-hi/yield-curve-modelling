"""Estrella–Mishkin style probit models of NBER recessions 12 months ahead.

    P(USREC_{t+h} = 1 | spread_t) = Phi(a + b * spread_t),   h = 12 months

``spread_t`` is the monthly average of the daily 10y-3m or 10y-2y spread (percentage
points). Out-of-sample probabilities come from an expanding window: at each
forecast origin t the model is re-estimated using only (spread_s, USREC_{s+h})
pairs whose outcome had been *published* by t. NBER announces turning points with
a lag (4 to 21 months historically), so we assume an outcome month is known
``pub_lag`` = 12 months after it happens:

    training set at origin t:  s + h <= t - pub_lag

This is stricter than simply requiring s + h <= t, and removes the look-ahead
that comes from using recession dates nobody knew at the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

HORIZON = 12  # months ahead
PUB_LAG = 12  # months until NBER outcome is assumed known
MIN_TRAIN_OBS = 60  # at least 5 years of monthly data before the first forecast


def monthly_spreads(yields: pd.DataFrame) -> pd.DataFrame:
    """Monthly averages of the 10y-3m and 10y-2y spreads (pp), complete months only.

    Each spread uses its own full history (it only needs two maturities).
    The current, incomplete month is dropped.
    """
    daily = pd.DataFrame({
        "10y-3m": yields["DGS10"] - yields["DGS3MO"],
        "10y-2y": yields["DGS10"] - yields["DGS2"],
    })
    monthly = daily.resample("MS").mean()
    last_day = yields.index.max()
    next_trading_day = last_day + pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    if next_trading_day.month == last_day.month:  # month not finished
        monthly = monthly.iloc[:-1]
    return monthly


def make_target(usrec: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """y_t = USREC_{t+h}: was the economy in recession h months after month t?"""
    return usrec.shift(-horizon).rename(f"USREC_t+{horizon}")


def fit_probit(x: pd.Series, y: pd.Series, hac_lags: int | None = HORIZON):
    """Probit of y on a constant and x. HAC (Newey–West) standard errors by default,
    because overlapping 12-month-ahead targets make the residuals serially correlated."""
    X = sm.add_constant(x.rename("spread"))
    model = sm.Probit(y, X)
    if hac_lags:
        return model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model.fit(disp=0)


@dataclass
class OOSForecast:
    prob: pd.Series  # out-of-sample P(recession in t+h), indexed by origin month t
    base_rate: pd.Series  # expanding unconditional frequency (the naive benchmark)
    target: pd.Series  # realised USREC_{t+h} (NaN where not yet observed)
    coef: pd.DataFrame  # expanding-window estimates (a, b) at each origin


def expanding_probit(spread: pd.Series, usrec: pd.Series, horizon: int = HORIZON,
                     pub_lag: int = PUB_LAG, min_obs: int = MIN_TRAIN_OBS,
                     start: str | None = None) -> OOSForecast:
    """Genuinely out-of-sample recession probabilities from expanding-window probits.

    The first origin is the earliest month at which the training set has at least
    ``min_obs`` observations and contains at least one recession month (a probit
    cannot be estimated otherwise), or ``start`` if that is later.
    """
    spread = spread.dropna()
    target = make_target(usrec, horizon).reindex(spread.index)
    probs, base, coefs = {}, {}, {}
    months = spread.index
    for t in months:
        if start is not None and t < pd.Timestamp(start):
            continue
        cutoff = t - pd.DateOffset(months=horizon + pub_lag)  # last usable s
        train_idx = months[months <= cutoff]
        y_tr = target.loc[train_idx].dropna()
        if len(y_tr) < min_obs or y_tr.sum() == 0:
            continue
        res = fit_probit(spread.loc[y_tr.index], y_tr, hac_lags=None)
        a, b = res.params["const"], res.params["spread"]
        probs[t] = float(stats.norm.cdf(a + b * spread.loc[t]))
        base[t] = float(y_tr.mean())
        coefs[t] = (a, b)
    idx = pd.DatetimeIndex(list(probs))
    return OOSForecast(
        prob=pd.Series(probs, name="prob").reindex(idx),
        base_rate=pd.Series(base, name="base_rate").reindex(idx),
        target=target.reindex(idx),
        coef=pd.DataFrame.from_dict(coefs, orient="index", columns=["a", "b"]),
    )


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of probability forecasts (0 = perfect)."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = HORIZON) -> tuple[float, float]:
    """Diebold–Mariano test of equal expected loss, Newey–West variance with ``lags``.

    d_t = loss_a - loss_b; returns (DM statistic, two-sided p-value). A positive
    statistic means model b has the lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    dbar = d.mean()
    u = d - dbar
    lrv = u @ u / T
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)  # Bartlett kernel
        lrv += 2 * w * (u[k:] @ u[:-k]) / T
    if lrv <= 0:  # identical losses: no evidence either way
        return 0.0, 1.0
    stat = dbar / np.sqrt(lrv / T)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def block_bootstrap_auc(p: np.ndarray, y: np.ndarray, block: int = 24, n_boot: int = 2000,
                        seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for AUC from a moving-block bootstrap.

    Recessions are rare and persistent, so resampling single months would
    overstate precision; blocks of 24 months keep episodes intact.
    """
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p), np.asarray(y)
    T = len(p)
    n_blocks = int(np.ceil(T / block))
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        if 0 < y[idx].sum() < T:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(fc: OOSForecast) -> dict[str, float]:
    """AUC, Brier score, Brier skill score vs the expanding base rate and a DM test."""
    mask = fc.target.notna()
    p, b, y = fc.prob[mask].to_numpy(), fc.base_rate[mask].to_numpy(), fc.target[mask].to_numpy()
    bs_model, bs_base = brier(p, y), brier(b, y)
    dm, dm_p = diebold_mariano((b - y) ** 2, (p - y) ** 2)
    lo, hi = block_bootstrap_auc(p, y)
    return {
        "first origin": fc.prob.index[0].strftime("%Y-%m"),
        "last evaluated origin": fc.prob[mask].index[-1].strftime("%Y-%m"),
        "months": int(mask.sum()),
        "recession months": int(y.sum()),
        "AUC": roc_auc_score(y, p),
        "AUC 95% CI low": lo,
        "AUC 95% CI high": hi,
        "Brier": bs_model,
        "Brier (base rate)": bs_base,
        "Brier skill score": 1 - bs_model / bs_base,
        "DM stat": dm,
        "DM p-value": dm_p,
    }
