"""End-to-end pipeline: data -> curve fits -> PCA -> recession probits -> curve trades.

Run from the repo root with ``python -m src.pipeline``. Writes figures to figures/,
tables to results/*.csv and the headline numbers to results/key_numbers.json.
Each ``section_*`` function can also be called on its own (the notebook does this).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from sklearn.metrics import roc_curve

from . import curves, data, pca, recession, trades
from .bonds import bond_price, dv01, par_duration_convexity
from .plots import C, INK2, save, setup, shade_recessions

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"

# Key dates for the small-multiple episode chart (nearest prior trading day is used).
EPISODES: dict[str, list[str]] = {
    "2008 financial crisis": ["2007-06-29", "2008-09-12", "2008-12-31"],
    "2013 taper tantrum": ["2013-05-01", "2013-06-25", "2013-09-05"],
    "2020 Covid shock": ["2020-02-14", "2020-03-09", "2020-08-04"],
    "2022 hiking cycle": ["2021-12-31", "2022-06-15", "2022-10-21"],
    "2023 deepest inversion": ["2023-03-08", "2023-10-19", "2024-08-30"],
}
PCA_COLS = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20"]
PCA_PERIODS = {  # split at Fed policy regime changes
    "2001-08 pre-ZLB": ("2001-07-31", "2008-12-15"),
    "2008-15 zero lower bound": ("2008-12-16", "2015-12-15"),
    "2015-22 normalisation/Covid": ("2015-12-16", "2022-03-16"),
    "2022-26 hiking and after": ("2022-03-17", "2099-12-31"),
}
TRADE_TRAIN = ("1981-01-01", "2001-07-30")
TRADE_TEST = ("2001-07-31", "2099-12-31")
COST_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]

K: dict[str, object] = {}  # headline numbers collected as we go


def _r(x, n=2):
    return float(np.round(x, n))


def _save_table(df: pd.DataFrame, name: str, digits: int = 3) -> pd.DataFrame:
    RES_DIR.mkdir(exist_ok=True)
    df.round(digits).to_csv(RES_DIR / f"{name}.csv")
    return df


# --------------------------------------------------------------------------- data
def section_data():
    yields = data.load_yields()  # 2001-07-31 onward, all 11 maturities, 30y gap masked
    yields_full = data.load_yields(full_history=True)  # each series' own history
    usrec = data.load_recessions()
    K["sample_start"] = str(yields.index[0].date())
    K["sample_end"] = str(yields.index[-1].date())
    K["n_days"] = int(len(yields))
    K["dgs30_masked_days"] = int(yields["DGS30"].isna().sum())
    K["sources"] = dict(data.SOURCES_USED)

    setup()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    shade_recessions(ax, data.recession_periods(usrec.loc["2001":]))
    for i, col in enumerate(["DGS3MO", "DGS2", "DGS10", "DGS30"]):
        ax.plot(yields.index, yields[col], color=C[i], lw=1.1, label=data.LABELS[col])
    ax.set(title="US Treasury constant-maturity yields, 2001–2026",
           ylabel="Yield (%)", xlabel="Date")
    ax.annotate("30Y masked 2002–06\n(extrapolated by Treasury)", xy=(pd.Timestamp("2004-01-01"), 5.6),
                fontsize=8, color=INK2, ha="center")
    ax.legend(ncol=5, loc="upper right")
    save(fig, "fig01_yields_history")
    return yields, yields_full, usrec


# ------------------------------------------------------------------------- curves
def section_curves(yields: pd.DataFrame, usrec: pd.Series):
    mats = data.maturities(yields.columns)
    fits = {
        "NS (fixed tau)": curves.fit_curves(yields, mats, "NS", tau=curves.DIEBOLD_LI_TAU),
        "NS": curves.fit_curves(yields, mats, "NS"),
        "NSS": curves.fit_curves(yields, mats, "NSS"),
        "NSS (wide tau grid)": curves.fit_curves(yields, mats, "NSS", tau_grid=curves.WIDE_TAU_GRID),
    }
    loo = {
        "NS (fixed tau)": curves.leave_one_out_errors(yields, mats, "NS", tau=curves.DIEBOLD_LI_TAU),
        "NS": curves.leave_one_out_errors(yields, mats, "NS"),
        "NSS": curves.leave_one_out_errors(yields, mats, "NSS"),
        "NSS (wide tau grid)": curves.leave_one_out_errors(yields, mats, "NSS",
                                                           tau_grid=curves.WIDE_TAU_GRID),
    }

    rows = []
    for name, f in fits.items():
        p = f.params
        betas = p[[c for c in p if c.startswith("b")]]
        grid = curves.WIDE_TAU_GRID if "wide" in name else curves.TAU_GRID
        # Edges of what the candidate list actually allows (for NSS, tau2 >= 2 tau1
        # caps tau1 below the top of the grid).
        t1_max = grid[grid * curves.NSS_MIN_TAU_RATIO <= grid[-1] * 1.001].max() if "NSS" in name else grid[-1]
        at_bound = ((p["tau1"] <= grid[0] * 1.001) | (p["tau1"] >= t1_max * 0.999)).mean()
        t2_bound = ((p["tau2"] >= grid[-1] * 0.999).mean()) if "NSS" in name else np.nan
        rows.append({
            "model": name,
            "params": betas.shape[1] + (0 if "fixed" in name else (2 if "NSS" in name else 1)),
            "mean RMSE (bp)": f.rmse_bp.mean(),
            "median RMSE (bp)": f.rmse_bp.median(),
            "95th pct RMSE (bp)": f.rmse_bp.quantile(0.95),
            "max RMSE (bp)": f.rmse_bp.max(),
            "LOO MAE (bp)": loo[name].abs().mean().mean(),
            "median |daily chg b0| (bp)": 100 * betas["b0"].diff().abs().median(),
            "median |daily chg b2| (bp)": 100 * betas["b2"].diff().abs().median(),
            "% days |beta| > 20pp": 100 * (betas.abs() > 20).any(axis=1).mean(),
            "% days tau1 at grid edge": 100 * at_bound if "fixed" not in name else np.nan,
            "% days tau2 at grid edge": 100 * t2_bound,
        })
    summary = _save_table(pd.DataFrame(rows).set_index("model"), "curve_fit_summary")
    loo_table = _save_table(pd.DataFrame({k: v.abs().mean() for k, v in loo.items()})
                            .rename(index=data.LABELS), "curve_loo_mae_by_maturity")
    for name in ["NS (fixed tau)", "NS", "NSS"]:
        K[f"rmse_mean_bp[{name}]"] = _r(summary.loc[name, "mean RMSE (bp)"])
        K[f"loo_mae_bp[{name}]"] = _r(summary.loc[name, "LOO MAE (bp)"])
    K["rmse_mean_bp[NSS wide]"] = _r(summary.loc["NSS (wide tau grid)", "mean RMSE (bp)"])
    K["loo_mae_bp[NSS wide]"] = _r(summary.loc["NSS (wide tau grid)", "LOO MAE (bp)"])
    K["blowup_pct[NSS wide]"] = _r(summary.loc["NSS (wide tau grid)", "% days |beta| > 20pp"])
    K["tau_edge_pct"] = {m: {"tau1": _r(summary.loc[m, "% days tau1 at grid edge"], 1),
                             "tau2": _r(summary.loc[m, "% days tau2 at grid edge"], 1)}
                         for m in ["NS", "NSS", "NSS (wide tau grid)"]}
    K["loo_mae_bp_by_maturity[NSS]"] = loo_table["NSS"].round(2).to_dict()
    K["loo_mae_bp_by_maturity[NS]"] = loo_table["NS"].round(2).to_dict()
    K["rmse_by_year_NSS"] = fits["NSS"].rmse_bp.groupby(fits["NSS"].rmse_bp.index.year).mean().round(2).to_dict()

    # Factor mapping: NS betas vs model-free level/slope/curvature proxies.
    emp = curves.empirical_factors(yields)
    # Extra proxies: b0 is the yield at infinite maturity, so compare it with the
    # longest gap-free yield (20Y); b0 + b1 is the instantaneous short rate (vs 3M).
    emp["level (20Y)"] = yields["DGS20"]
    emp["short rate (3M)"] = yields["DGS3MO"]
    map_rows = []
    for name in ["NS (fixed tau)", "NS"]:
        p = fits[name].params
        ns_f = pd.DataFrame({"level": p["b0"], "level (20Y)": p["b0"], "slope": -p["b1"],
                             "curvature": p["b2"], "short rate (3M)": p["b0"] + p["b1"]})
        labels = {"level": "b0", "level (20Y)": "b0", "slope": "-b1", "curvature": "b2",
                  "short rate (3M)": "b0+b1"}
        for f in ns_f:
            map_rows.append({"model": name, "factor": f, "NS parameter": labels[f],
                             "corr (levels)": ns_f[f].corr(emp[f]),
                             "corr (daily changes)": ns_f[f].diff().corr(emp[f].diff())})
    mapping = _save_table(pd.DataFrame(map_rows).set_index(["model", "factor"]), "factor_mapping")
    for f in ["level", "level (20Y)", "slope", "curvature", "short rate (3M)"]:
        K[f"corr_levels[{f}, NS fixed tau]"] = _r(mapping.loc[("NS (fixed tau)", f), "corr (levels)"], 3)
        K[f"corr_levels[{f}, NS]"] = _r(mapping.loc[("NS", f), "corr (levels)"], 3)

    _plot_episodes(yields, fits["NSS"], mats)
    _plot_rmse(fits, loo_table)
    _plot_factors(fits["NS (fixed tau)"].params, emp, usrec)
    _plot_param_stability(fits)
    _animate(yields, fits["NSS"], mats)
    return fits, summary, loo_table, mapping


def _plot_episodes(yields, fit, mats):
    grid = np.linspace(1 / 12, 30, 200)
    panels = list(EPISODES.items()) + [("Now vs one and two years ago", [
        str((yields.index[-1] - pd.DateOffset(years=2)).date()),
        str((yields.index[-1] - pd.DateOffset(years=1)).date()),
        str(yields.index[-1].date())])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, (title, dates) in zip(axes.ravel(), panels):
        for i, d in enumerate(dates):
            t = yields.index[yields.index.get_indexer([pd.Timestamp(d)], method="pad")[0]]
            row = fit.params.loc[t]
            curve = curves.curve(grid, row[["b0", "b1", "b2", "b3"]].to_numpy(), row.tau1, row.tau2)
            ax.plot(grid, curve, color=C[i], lw=1.8, label=f"{t.date()}")
            obs = yields.loc[t]
            ax.plot(np.array(mats)[obs.notna().to_numpy()], obs.dropna(), "o", color=C[i],
                    ms=5, mec="white", mew=0.8)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
    fig.suptitle("Treasury curve through key episodes: observed CMT yields (dots) and daily "
                 "Nelson–Siegel–Svensson fit (lines)", fontsize=12, fontweight="bold")
    save(fig, "fig02_curve_episodes")


def _plot_rmse(fits, loo_table):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, name in enumerate(["NS (fixed tau)", "NS", "NSS"]):
        r = fits[name].rmse_bp.rolling(63).mean()
        a1.plot(r.index, r, color=C[i], lw=1.3, label=name)
    a1.set(title="In-sample fit error (63-day rolling mean of daily RMSE)",
           ylabel="RMSE across maturities (bp)", xlabel="Date")
    a1.legend()
    t = loo_table[["NS (fixed tau)", "NS", "NSS"]]
    x = np.arange(len(t))
    for i, c in enumerate(t.columns):
        a2.bar(x + (i - 1) * 0.27, t[c], width=0.25, color=C[i], label=c)
    a2.set_xticks(x, t.index)
    a2.set(title="Out-of-sample: leave-one-maturity-out error",
           ylabel="Mean absolute error (bp)", xlabel="Held-out maturity")
    a2.legend(fontsize=8)
    save(fig, "fig03_fit_errors")


def _plot_factors(p, emp, usrec):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    pairs = [("level (20Y)", p["b0"], "b0 (long-run level)", "20Y yield"),
             ("slope", -p["b1"], "−b1", "10Y − 3M"),
             ("curvature", p["b2"], "b2", "2×2Y − 3M − 10Y")]
    periods = data.recession_periods(usrec.loc["2001":])
    for ax, (f, ns, lab_ns, lab_emp) in zip(axes, pairs):
        shade_recessions(ax, periods, label=(f == "level (20Y)"))
        ax.plot(ns.index, ns, color=C[0], lw=1.1, label=f"NS {lab_ns}")
        ax.plot(emp.index, emp[f], color=C[1], lw=1.1, label=f"Empirical: {lab_emp}")
        name = f.split(" ")[0].capitalize()
        ax.set_ylabel(f"{name} (%, pp)")
        ax.set_title(f"{name}: corr = {ns.corr(emp[f]):.2f}")
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Nelson–Siegel factors (Diebold–Li fixed decay) vs model-free proxies",
                 fontsize=12, fontweight="bold")
    save(fig, "fig04_ns_factors")


def _plot_param_stability(fits):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ns, nss, wide = fits["NS"].params, fits["NSS"].params, fits["NSS (wide tau grid)"].params
    axes[0].plot(ns.index, ns["tau1"], ".", ms=1.5, color=C[0], label="NS tau")
    axes[0].plot(nss.index, nss["tau1"], ".", ms=1.5, color=C[1], label="NSS tau1")
    axes[0].axhline(curves.DIEBOLD_LI_TAU, color=INK2, ls="--", lw=1, label="Diebold–Li fixed")
    axes[0].set(yscale="log", title="Daily decay parameters", ylabel="tau (years, log scale)",
                xlabel="Date")
    axes[0].legend(fontsize=8, markerscale=6, loc="lower left")
    for ax, (p, ttl) in zip(axes[1:], [(nss, "NSS betas, tau grid 0.25–10y"),
                                       (wide, "NSS betas, wide tau grid 0.1–30y")]):
        for i, b in enumerate(["b0", "b1", "b2", "b3"]):
            ax.plot(p.index, p[b].clip(-30, 30), lw=0.7, color=C[i], label=b)
        ax.set(title=ttl, ylabel="Parameter value (pp, clipped at ±30)", xlabel="Date")
        ax.legend(fontsize=8, ncol=4)
    fig.suptitle("Parameter stability: profiled decay parameters jump, and a wide grid lets NSS "
                 "betas explode in offsetting pairs", fontsize=11, fontweight="bold")
    save(fig, "fig05_parameter_stability")


def _animate(yields, fit, mats):
    """Month-end NSS curves as a GIF (figures/curve_animation.gif)."""
    month_ends = yields.groupby(yields.index.to_period("M")).tail(1).index
    grid = np.linspace(1 / 12, 30, 150)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    line, = ax.plot([], [], color=C[0], lw=2)
    pts, = ax.plot([], [], "o", color=C[0], ms=5, mec="white")
    ttl = ax.set_title("")
    ax.set(xlim=(0, 30.5), ylim=(-0.2, 6.5), xlabel="Maturity (years)", ylabel="Yield (%)")
    fig.text(0.01, 0.01, "Source: FRED CMT yields; NSS fit. Sam Berman", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    def draw(t):
        r = fit.params.loc[t]
        line.set_data(grid, curves.curve(grid, r[["b0", "b1", "b2", "b3"]].to_numpy(), r.tau1, r.tau2))
        obs = yields.loc[t]
        pts.set_data(np.array(mats)[obs.notna().to_numpy()], obs.dropna())
        ttl.set_text(f"US Treasury curve {t:%b %Y}   10Y−3M = {obs.DGS10 - obs.DGS3MO:+.2f} pp")
        return line, pts, ttl

    anim = animation.FuncAnimation(fig, draw, frames=list(month_ends), blit=False)
    anim.save(ROOT / "figures" / "curve_animation.gif", writer=animation.PillowWriter(fps=6), dpi=70)
    plt.close(fig)


# ---------------------------------------------------------------------------- PCA
def section_pca(yields: pd.DataFrame, ns_params: pd.DataFrame):
    ch = pca.yield_changes(yields, PCA_COLS)
    full = pca.fit_pca(ch, 5)
    expl = pd.DataFrame({"variance share %": 100 * full.explained,
                         "cumulative %": 100 * full.explained.cumsum(),
                         "std dev (bp/day)": np.sqrt(full.eigenvalues)})
    _save_table(expl, "pca_explained")
    _save_table(full.loadings.rename(index=data.LABELS), "pca_loadings")
    K["pca_var_pct"] = {k: _r(100 * v, 1) for k, v in full.explained.items()}
    K["pca_cum3_pct"] = _r(100 * full.explained.iloc[:3].sum(), 1)
    K["pca_days"] = int(len(ch))

    # Robustness: include the 1M bill; and all 11 maturities once the 30Y is back.
    rob = {
        "Main: 3M–20Y, 2001–2026": full.explained,
        "Add 1M: 1M–20Y, 2001–2026": pca.fit_pca(pca.yield_changes(yields, ["DGS1MO"] + PCA_COLS), 5).explained,
        "All 11: 1M–30Y, 2006–2026": pca.fit_pca(pca.yield_changes(yields.loc["2006-02-09":]), 5).explained,
    }
    robust = _save_table(100 * pd.DataFrame(rob).T.iloc[:, :3].assign(
        **{"PC1-3": lambda d: d.sum(axis=1)}), "pca_robustness")

    stab, sub_fits = pca.subperiod_stability(ch, PCA_PERIODS, 3)
    _save_table(stab, "pca_subperiods")
    K["pca_subperiod_cum3_pct"] = {k: _r(v, 1) for k, v in stab["PC1-3 var %"].items()}
    K["pca_subperiod_min_congruence"] = {f"PC{i}": _r(stab[f"PC{i} congruence"].min(), 3) for i in (1, 2, 3)}

    oos = pca.out_of_sample_explained(ch.loc[:"2013-12-31"], ch.loc["2014-01-01":], 3)
    ins = pca.fit_pca(ch.loc["2014-01-01":], 3).explained.cumsum()
    oos_tab = _save_table(pd.DataFrame({"train 2001–13 loadings on 2014–26 data %": 100 * oos.values,
                                        "2014–26 own loadings (in-sample) %": 100 * ins.values},
                                       index=oos.index), "pca_out_of_sample")
    K["pca_oos_cum3_pct"] = _r(oos_tab.iloc[2, 0], 1)
    K["pca_is_cum3_pct_2014"] = _r(oos_tab.iloc[2, 1], 1)

    # Link to Nelson–Siegel: correlate PC scores with daily changes in NS factors.
    dns = pd.DataFrame({"level (b0)": ns_params["b0"], "slope (-b1)": -ns_params["b1"],
                        "curvature (b2)": ns_params["b2"]}).diff().loc[full.scores.index]
    link = pd.DataFrame({pc: [full.scores[pc].corr(dns[c]) for c in dns]
                         for pc in ["PC1", "PC2", "PC3"]}, index=dns.columns)
    _save_table(link, "pca_vs_ns_factors")
    K["corr_PC_vs_dNS"] = {f"PC{i + 1} vs {c}": _r(link.iloc[i, i], 3) for i, c in enumerate(dns.columns)}

    _plot_pca(full, sub_fits)
    return full, expl, robust, stab, oos_tab, link


def _plot_pca(full, sub_fits):
    labels = [data.LABELS[c] for c in PCA_COLS]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    names = ["PC1 level", "PC2 slope", "PC3 curvature"]
    for i in range(3):
        pc = f"PC{i + 1}"
        a1.plot(x, full.loadings[pc], "o-", color=C[i], ms=5,
                label=f"{names[i]} ({100 * full.explained[pc]:.1f}%)")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xticks(x, labels)
    a1.set(title="PCA loadings on daily yield changes, 2001–2026", xlabel="Maturity",
           ylabel="Loading (unit-length eigenvector)")
    a1.legend()
    e = 100 * full.explained
    a2.bar(e.index, e.values, color=C[0], width=0.6, label="Individual")
    a2.plot(e.index, e.cumsum().values, "o-", color=C[1], label="Cumulative")
    for i, v in enumerate(e.cumsum().values[:3]):
        a2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8)
    a2.set(title="Share of variance explained", ylabel="% of total variance", ylim=(0, 105),
           xlabel="Component")
    a2.legend(loc="center right")
    save(fig, "fig06_pca_loadings")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, ax in enumerate(axes):
        pc = f"PC{i + 1}"
        for j, (period, res) in enumerate(sub_fits.items()):
            ax.plot(x, res.loadings[pc], "o-", ms=4, lw=1.4, color=C[j], label=period)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(title=f"{names[i]} loadings by sub-period", xlabel="Maturity")
    axes[0].set_ylabel("Loading")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stability of the factor structure across Fed policy regimes "
                 "(short-end loadings collapse at the zero lower bound)", fontsize=11, fontweight="bold")
    save(fig, "fig07_pca_stability")


# --------------------------------------------------------------------- recession
def section_recession(yields_full: pd.DataFrame, usrec: pd.Series):
    ms = recession.monthly_spreads(yields_full)
    target = recession.make_target(usrec).reindex(ms.index)
    # Full-sample (in-sample) probits with HAC standard errors, for inference on the slope.
    ins_rows = []
    for c in ms:
        d = pd.concat([ms[c], target], axis=1).dropna()
        res = recession.fit_probit(d[c], d.iloc[:, 1])
        ins_rows.append({"spread": c, "sample": f"{d.index[0]:%Y-%m}–{d.index[-1]:%Y-%m}",
                         "months": len(d), "a": res.params["const"], "b": res.params["spread"],
                         "z (HAC)": res.tvalues["spread"], "p-value": res.pvalues["spread"],
                         "McFadden R2": res.prsquared,
                         "spread at P=50%": -res.params["const"] / res.params["spread"]})
    ins = _save_table(pd.DataFrame(ins_rows).set_index("spread"), "probit_in_sample", 4)

    fcs = {c: recession.expanding_probit(ms[c], usrec) for c in ms}
    common_start = max(f.prob.index[0] for f in fcs.values())
    windows = {  # the last three are sensitivity checks; the 2021-12 cut is post-hoc
        "common window": (common_start, None),
        "own full OOS window": (None, None),
        "origins from 1990-05": (pd.Timestamp("1990-05-01"), None),
        "post-hoc: origins to 2021-12": (common_start, pd.Timestamp("2021-12-01")),
    }
    rows = []
    for c, fc in fcs.items():
        for wname, (s, e) in windows.items():
            sl = slice(s, e)
            sub = recession.OOSForecast(fc.prob.loc[sl], fc.base_rate.loc[sl], fc.target.loc[sl],
                                        fc.coef.loc[sl])
            rows.append({"spread": c, "window": wname, **recession.evaluate(sub)})
    oos = _save_table(pd.DataFrame(rows).set_index(["spread", "window"]), "recession_oos", 4)

    probs = pd.DataFrame({c: f.prob for c, f in fcs.items()})
    _save_table(probs, "recession_probabilities_oos", 4)
    for c in ms:
        K[f"probit_b[{c}]"] = _r(ins.loc[c, "b"], 3)
        K[f"probit_p[{c}]"] = float(f"{ins.loc[c, 'p-value']:.2g}")
        r = oos.loc[(c, "common window")]
        K[f"auc[{c}]"] = _r(r["AUC"], 3)
        K[f"auc_ci[{c}]"] = [_r(r["AUC 95% CI low"], 3), _r(r["AUC 95% CI high"], 3)]
        K[f"brier[{c}]"] = _r(r["Brier"], 4)
        K[f"brier_base[{c}]"] = _r(r["Brier (base rate)"], 4)
        K[f"bss[{c}]"] = _r(r["Brier skill score"], 3)
        K[f"dm_p[{c}]"] = _r(r["DM p-value"], 3)
        K[f"cw_p[{c}]"] = _r(r["CW p-value (1-sided)"], 3)
        K[f"bss_from_1990_05[{c}]"] = _r(oos.loc[(c, "origins from 1990-05"), "Brier skill score"], 3)
        K[f"auc_from_1990_05[{c}]"] = _r(oos.loc[(c, "origins from 1990-05"), "AUC"], 3)
        # Sensitivity: 21-month publication lag (the longest NBER trough announcement).
        fc21 = recession.expanding_probit(ms[c], usrec, pub_lag=21, start=f"{common_start:%Y-%m}")
        e21 = recession.evaluate(fc21)
        K[f"pub_lag21[{c}]"] = {"AUC": _r(e21["AUC"], 3), "BSS": _r(e21["Brier skill score"], 3),
                                "first origin": e21["first origin"]}
        r2 = oos.loc[(c, "post-hoc: origins to 2021-12")]
        K[f"auc_pre2022[{c}]"] = _r(r2["AUC"], 3)
        K[f"bss_pre2022[{c}]"] = _r(r2["Brier skill score"], 3)
        p22 = fcs[c].prob.loc["2022-01":"2024-12"]
        K[f"peak_prob_2022_24[{c}]"] = _r(p22.max(), 3)
        K[f"peak_prob_origin[{c}]"] = f"{p22.idxmax():%Y-%m}"
        K[f"months_prob_gt50_2022_24[{c}]"] = int((p22 > 0.5).sum())
        K[f"latest_prob[{c}]"] = _r(fcs[c].prob.iloc[-1], 3)
        K[f"oos_slope_b_at_peak[{c}]"] = _r(fcs[c].coef.loc[p22.idxmax(), "b"], 3)
        K[f"oos_slope_b_latest[{c}]"] = _r(fcs[c].coef["b"].iloc[-1], 3)
        hit = fcs[c].target.notna()
        K[f"signals_p_gt50[{c}]"] = {
            "false alarms (months)": int(((fcs[c].prob > 0.5) & (fcs[c].target == 0) & hit).sum()),
            "hits (months)": int(((fcs[c].prob > 0.5) & (fcs[c].target == 1)).sum()),
            "recession months": int((fcs[c].target == 1).sum())}
        K[f"latest_origin"] = f"{fcs[c].prob.index[-1]:%Y-%m}"
    inv = ms["10y-3m"].loc["2022":"2025"]
    K["inversion_10y3m_months_2022_25"] = int((inv < 0).sum())
    neg = (inv < 0).astype(int)
    runs = neg.groupby((neg.diff() != 0).cumsum())
    longest = max((g for _, g in runs if g.iloc[0] == 1), key=len)
    K["inversion_10y3m_longest_run"] = {"months": int(len(longest)),
                                        "first": f"{longest.index[0]:%Y-%m}",
                                        "last": f"{longest.index[-1]:%Y-%m}"}
    K["inversion_10y3m_min_pp"] = _r(inv.min(), 2)
    K["share_days_20y_above_30y"] = _r((yields_full["DGS20"] > yields_full["DGS30"])
                                       .loc["2006-02-09":].mean(), 3)
    K["oos_common_start"] = f"{common_start:%Y-%m}"
    K["usrec_last"] = f"{usrec.index[-1]:%Y-%m}"

    _plot_recession(ms, fcs, usrec)
    return ms, ins, fcs, oos


def _plot_recession(ms, fcs, usrec):
    periods = data.recession_periods(usrec.loc["1981":])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.3]})
    shade_recessions(a1, periods)
    for i, c in enumerate(ms):
        a1.plot(ms.index, ms[c], color=C[i], lw=1.1, label=c)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set(title="Monthly average Treasury spreads", ylabel="Spread (pp)", ylim=(-2.5, 5.5))
    a1.legend(ncol=3, loc="upper right")
    shade_recessions(a2, periods, label=False)
    for i, (c, fc) in enumerate(fcs.items()):
        # Plot each probability at the month it is forecasting (t + 12) so it lines up with shading.
        a2.plot(fc.prob.index + pd.DateOffset(months=recession.HORIZON), fc.prob, color=C[i],
                lw=1.3, label=f"{c} probit")
    a2.axhline(0.5, color=INK2, ls="--", lw=0.8)
    a2.set(title="Out-of-sample probability of recession, plotted at the target month "
                 "(expanding window, 12-month publication lag)",
           ylabel="P(recession in 12 months)", xlabel="Target month", ylim=(0, 1))
    a2.legend(loc="upper left")
    save(fig, "fig08_recession_probabilities")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    common = max(f.prob.index[0] for f in fcs.values())
    for i, (c, fc) in enumerate(fcs.items()):
        m = fc.target.notna() & (fc.prob.index >= common)
        fpr, tpr, _ = roc_curve(fc.target[m], fc.prob[m])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=C[i], lw=1.8,
                label=f"{c} (AUC {roc_auc_score(fc.target[m], fc.prob[m]):.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.8, label="No skill")
    ax.set(title=f"Out-of-sample ROC, origins {common:%Y-%m} onward", xlabel="False positive rate",
           ylabel="True positive rate")
    ax.legend(loc="lower right")
    save(fig, "fig09_recession_roc", source="Source: FRED (DGS10, DGS3MO, DGS2, USREC).")


# ------------------------------------------------------------------------- trades
def section_trades(yields_full: pd.DataFrame):
    pnl = trades.unit_pnl(yields_full)
    spread = (yields_full["DGS10"] - yields_full["DGS2"]).dropna()
    carry = trades.carry_rolldown_bp(yields_full)
    tr, te = slice(*TRADE_TRAIN), slice(*TRADE_TEST)

    def sharpe_train(pos):
        return trades.performance(trades.backtest(pos, pnl).loc[tr]).sharpe

    z_grid = [(w, k) for w in (63, 126, 252, 504) for k in (0.5, 1.0, 1.5, 2.0)]
    (w_best, k_best), z_tab = trades.select_params(
        lambda w, k: sharpe_train(trades.zscore_signal(spread, int(w), k)), z_grid)
    c_grid = [(k,) for k in (0.0, 2.0, 5.0, 10.0)]
    (c_best,), c_tab = trades.select_params(lambda k: sharpe_train(trades.carry_signal(carry, k)),
                                            c_grid)
    _save_table(z_tab.set_axis(["window", "entry z", "train Sharpe"], axis=1), "trades_train_grid_zscore")
    _save_table(c_tab.set_axis(["threshold bp", "train Sharpe"], axis=1), "trades_train_grid_carry")

    positions = {
        f"Slope mean reversion (W={int(w_best)}, z={k_best})": trades.zscore_signal(spread, int(w_best), k_best),
        f"Carry & roll-down (threshold {c_best:g}bp)": trades.carry_signal(carry, c_best),
        "Always steepener (benchmark)": pd.Series(1.0, index=pnl.index),
    }
    rows, cost_rows, curves_ = [], [], {}
    for name, pos in positions.items():
        for per_name, per in [("train", tr), ("test", te)]:
            bt = trades.backtest(pos, pnl).loc[per]
            for col in ["gross", "net"]:
                s = trades.performance(bt, col)
                rows.append({"strategy": name, "period": per_name, "P&L": col,
                             "ann. P&L ($k)": s.ann_pnl / 1e3, "ann. vol ($k)": s.ann_vol / 1e3,
                             "Sharpe": s.sharpe, "t-stat (NW)": s.t_stat, "p-value": s.p_value,
                             "max drawdown ($k)": s.max_dd / 1e3, "hit rate": s.hit_rate,
                             "time in market": s.time_in_market, "trades / yr": s.trades_per_year})
            if per_name == "test":
                curves_[name] = bt["net"].cumsum()
        for cst in COST_GRID:
            s = trades.performance(trades.backtest(pos, pnl, cost_bp=cst).loc[te])
            cost_rows.append({"strategy": name, "cost (bp/leg)": cst, "Sharpe": s.sharpe,
                              "ann. P&L ($k)": s.ann_pnl / 1e3})
    perf = _save_table(pd.DataFrame(rows).set_index(["strategy", "period", "P&L"]), "trades_performance")
    costs = _save_table(pd.DataFrame(cost_rows).pivot(index="cost (bp/leg)", columns="strategy",
                                                      values="Sharpe"), "trades_cost_sensitivity")

    # Robustness: execute one day after the signal instead of at the signal close.
    lag_rows = {name: trades.performance(trades.backtest(pos.shift(1).fillna(0.0), pnl).loc[te]).sharpe
                for name, pos in positions.items()}
    K["test_sharpe_1day_exec_lag"] = {k.split(" (")[0]: _r(v) for k, v in lag_rows.items()}

    # POST-HOC (formed after seeing that mean reversion fails): trend following,
    # i.e. the mirror image of the z-score rule, selected on the same training
    # grid. The test period has already been looked at, so this is a hypothesis
    # for future data, not a validated out-of-sample result.
    (w_tr, k_tr), t_tab = trades.select_params(
        lambda w, k: sharpe_train(-trades.zscore_signal(spread, int(w), k)), z_grid)
    t_tab = t_tab.set_axis(["window", "entry z", "train Sharpe"], axis=1)
    t_tab["test Sharpe"] = [trades.performance(trades.backtest(
        -trades.zscore_signal(spread, int(w), k), pnl).loc[te]).sharpe for w, k in z_grid]
    _save_table(t_tab, "trades_posthoc_trend_grid")
    trend = trades.performance(trades.backtest(-trades.zscore_signal(spread, int(w_tr), k_tr), pnl).loc[te])
    K["posthoc_trend"] = {"window": int(w_tr), "entry_z": k_tr,
                          "train_sharpe": _r(t_tab["train Sharpe"].max()),
                          "test_sharpe": _r(trend.sharpe), "test_p": _r(trend.p_value, 3),
                          "test_ann_pnl_k": _r(trend.ann_pnl / 1e3, 0),
                          "test_sharpe_range_all_16": [_r(t_tab["test Sharpe"].min()),
                                                       _r(t_tab["test Sharpe"].max())]}
    K["carry_train_time_long_steepener"] = _r((positions[list(positions)[1]].loc[tr] > 0).mean(), 3)

    attrib = _save_table((pnl.loc[te].mean() * trades.TRADING_DAYS / 1e3).to_frame(
        "always-steepener ann. $k (test)").T, "trades_pnl_attribution")
    attrib_tr = (pnl.loc[tr].mean() * trades.TRADING_DAYS / 1e3)
    K["attrib_train_$k"] = {k: _r(v, 1) for k, v in attrib_tr.items()}
    K["attrib_test_$k"] = {k: _r(v, 1) for k, v in attrib.iloc[0].items()}
    for name in positions:
        key = name.split(" (")[0]
        for per in ["train", "test"]:
            r = perf.loc[(name, per, "net")]
            K[f"trade_sharpe[{key}, {per}]"] = _r(r["Sharpe"])
            K[f"trade_p[{key}, {per}]"] = _r(r["p-value"], 3)
            K[f"trade_annpnl_k[{key}, {per}]"] = _r(r["ann. P&L ($k)"], 0)
    K["trade_selected"] = {"zscore_window": int(w_best), "zscore_entry": k_best, "carry_threshold_bp": c_best}
    K["trade_test_start"] = str(pnl.loc[te].index[0].date())
    K["trade_train_start"] = str(pnl.loc[tr].index[0].date())

    # DV01 example at the latest close, and accuracy of the duration+convexity approximation.
    last = yields_full[["DGS2", "DGS10"]].dropna().iloc[-1] / 100
    d2, d10 = dv01(last.DGS2, 2.0, 1e6), dv01(last.DGS10, 10.0, 1e6)
    K["dv01_per_mm"] = {"2y": _r(d2, 1), "10y": _r(d10, 1), "hedge_ratio_10y_per_2y": _r(d2 / d10, 3),
                        "date": str(yields_full.index[-1].date())}
    y10 = yields_full["DGS10"].dropna() / 100
    y0, y1 = y10.to_numpy()[:-1], y10.to_numpy()[1:]
    exact = np.array([bond_price(c, y, 10.0) for c, y in zip(y0, y1)]) - 1.0
    md, cv = par_duration_convexity(y0, 10.0)
    approx_err = (exact - (-md * (y1 - y0) + 0.5 * cv * (y1 - y0) ** 2)) * 1e8 / md  # $ per $10k DV01
    dur_only_err = (exact - (-md * (y1 - y0))) * 1e8 / md
    K["approx_err_$_per_10k_dv01"] = {
        "max duration+convexity": _r(np.abs(approx_err).max(), 1),
        "max duration only": _r(np.abs(dur_only_err).max(), 1),
        "p99 duration+convexity": _r(np.quantile(np.abs(approx_err), 0.99), 2),
        "p99 duration only": _r(np.quantile(np.abs(dur_only_err), 0.99), 1),
        "worst day": str(y10.index[1:][np.abs(approx_err).argmax()].date()),
    }

    _plot_trades(curves_, spread, carry, te)
    return pnl, positions, perf, costs, attrib


def _plot_trades(curves_, spread, carry, te):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1.6, 1, 1]})
    for i, (name, cum) in enumerate(curves_.items()):
        a1.plot(cum.index, cum / 1e6, color=C[i], lw=1.3, label=name)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set(title="Out-of-sample cumulative net P&L of DV01-neutral 2s10s trades "
                 "($10k DV01 per leg, 0.5bp/leg costs)", ylabel="Cumulative P&L ($ million)")
    a1.set_ylim(bottom=min(a1.get_ylim()[0], -7.5))
    a1.legend(fontsize=8, loc="lower left")
    s = spread.loc[te]
    a2.plot(s.index, s, color=C[0], lw=1.0)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set(title="Mean-reversion signal input: 2s10s spread (10Y − 2Y)", ylabel="Spread (pp)")
    c = carry.loc[te]
    a3.plot(c.index, c, color=C[1], lw=1.0)
    a3.axhline(0, color=INK2, lw=0.8)
    a3.set(title="Carry signal input: expected 3-month carry + roll-down of the steepener",
           ylabel="bp of DV01", xlabel="Date")
    save(fig, "fig10_curve_trades")


# --------------------------------------------------------------------------- main
def main() -> dict:
    setup()
    yields, yields_full, usrec = section_data()
    print(f"Data: {K['sample_start']} -> {K['sample_end']} ({K['n_days']} days)")
    fits, *_ = section_curves(yields, usrec)
    print("Curves done")
    section_pca(yields, fits["NS (fixed tau)"].params)
    print("PCA done")
    section_recession(yields_full, usrec)
    print("Recession done")
    section_trades(yields_full)
    print("Trades done")
    RES_DIR.mkdir(exist_ok=True)
    (RES_DIR / "key_numbers.json").write_text(json.dumps(K, indent=2, default=str))
    return K


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))
