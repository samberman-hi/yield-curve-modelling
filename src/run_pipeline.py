"""End-to-end pipeline: data -> pair selection -> backtests -> tables -> figures.

Usage:  python -m src.run_pipeline

Writes CSV/JSON tables to results/ and PNGs to figures/. Two selection
specifications are run with both hedge-ratio methods:

* "BH (pre-specified)": Benjamini-Hochberg at q = 5% (the baseline);
* "Uncorrected": raw Engle-Granger p < 5%, everything else identical. This
  was planned from the start as a comparison, to show what skipping the
  multiple-testing correction does.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import plots
from .backtest import STOP, BacktestResult, precompute_tests, run_backtest
from .config import (BASELINE, BENCHMARK, END, FIGURES_DIR, GRID_COST_BPS, GRID_ENTRY_Z,
                     RESULTS_DIR, START, StrategyConfig)
from .data import data_sources, large_moves, load_prices
from .metrics import TRADING_DAYS, difference_test, hac_mean_test, summary

log = logging.getLogger(__name__)

SPECS: dict[str, StrategyConfig] = {
    "BH (pre-specified)": BASELINE,
    "Uncorrected": BASELINE.with_(use_fdr=False),
}
HEDGES = ("ols", "kalman")

# Worked example: a BH-selected pair whose window shows all three outcomes
# (a convergence exit, a large win and a stop-loss). Chosen for teaching value.
WORKED_EXAMPLE = ("C", "PNC", "2020-01-08")


def label(spec: str, hedge: str) -> str:
    return f"{spec} / {hedge.upper()}"


def split_cost(cfg: StrategyConfig, total_bps: float) -> StrategyConfig:
    """Set the one-way cost per leg keeping the baseline 1:4 commission/slippage split."""
    return cfg.with_(commission_bps=total_bps / 5, slippage_bps=total_bps * 4 / 5)


def selection_table(tests: dict[int, pd.DataFrame], runs: dict[str, BacktestResult]
                    ) -> pd.DataFrame:
    w = runs[label("BH (pre-specified)", "ols")].windows.copy()
    w["n_selected_uncorrected"] = runs[label("Uncorrected", "ols")].windows["n_selected"].to_numpy()
    w = w.rename(columns={"n_selected": "n_selected_bh"})
    w["n_dir_sig_yx"] = [int((t["eg_p_yx"] < 0.05).sum()) for t in tests.values()]
    w["n_johansen"] = [int(t["johansen"].sum()) for t in tests.values()]
    return w


def performance_table(runs: dict[str, BacktestResult], mkt: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, r in runs.items():
        s = summary(r.returns, mkt, r.trades, r.windows)
        s["n_pair_windows"] = len(r.pairs)
        s["time_in_market"] = float((r.n_open > 0).mean())
        rows[name] = s
    m = mkt.loc[next(iter(runs.values())).returns.index]
    t, p = hac_mean_test(m)
    rows["S&P 500 (SPY, total return)"] = dict(
        ann_return=float((1 + m).prod() ** (TRADING_DAYS / len(m)) - 1),
        ann_vol=float(m.std() * np.sqrt(TRADING_DAYS)),
        sharpe=float(np.sqrt(TRADING_DAYS) * m.mean() / m.std()), t_stat=t, p_value=p,
        max_drawdown=float(((1 + m).cumprod() / (1 + m).cumprod().cummax() - 1).min()))
    return pd.DataFrame(rows).T


def kalman_vs_ols(runs: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        d, t, p = difference_test(runs[label(spec, "kalman")].returns,
                                  runs[label(spec, "ols")].returns)
        rows.append(dict(spec=spec, ann_mean_diff=d, t_stat=t, p_value=p))
    return pd.DataFrame(rows)


def breakdown_table(runs: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        for hedge in HEDGES:
            r = runs[label(spec, hedge)]
            pr, tr = r.pairs, r.trades
            if pr.empty:
                continue
            stops = tr[tr["reason"] == STOP] if not tr.empty else tr
            rows.append(dict(
                spec=spec, hedge=hedge, pair_windows=len(pr),
                pct_broken_oos=pr["oos_broken"].mean(),
                pct_broken_is_benchmark=pr["is_broken"].mean(),
                mean_pnl_held=pr.loc[~pr["oos_broken"], "net_pnl"].mean(),
                mean_pnl_broken=pr.loc[pr["oos_broken"], "net_pnl"].mean(),
                n_held=int((~pr["oos_broken"]).sum()), n_broken=int(pr["oos_broken"].sum()),
                n_trades=len(tr), n_stops=len(stops),
                stop_rate=len(stops) / len(tr) if len(tr) else np.nan,
                stop_pnl_total=stops["net"].sum() if len(stops) else 0.0,
                nonstop_pnl_total=tr.loc[tr["reason"] != STOP, "net"].sum() if len(tr) else 0.0,
                window_end_pnl_total=tr.loc[tr["reason"] == "window_end", "net"].sum() if len(tr) else 0.0,
            ))
    return pd.DataFrame(rows)


def run_grid(prices: pd.DataFrame, tests: dict[int, pd.DataFrame], mkt: pd.Series) -> pd.DataFrame:
    rows = []
    for spec, cfg in SPECS.items():
        for hedge in HEDGES:
            for ez in GRID_ENTRY_Z:
                for cost in GRID_COST_BPS:
                    c = split_cost(cfg.with_(hedge=hedge, entry_z=ez), cost)
                    r = run_backtest(prices, c, tests)
                    s = summary(r.returns, mkt, r.trades, r.windows)
                    rows.append(dict(spec=spec, hedge=hedge, entry_z=ez, cost_bps=cost,
                                     sharpe=s["sharpe"], ann_return=s["ann_return"],
                                     p_value=s["p_value"], n_trades=s["n_trades"],
                                     hit_rate=s["hit_rate"]))
    return pd.DataFrame(rows)


def run_robustness(prices: pd.DataFrame, tests: dict[int, pd.DataFrame], mkt: pd.Series
                   ) -> pd.DataFrame:
    variants = {
        "baseline": {},
        "hedge-ratio weighting (not dollar-neutral)": dict(weighting="hedge_ratio"),
        "no Johansen confirmation": dict(require_johansen=False),
        "Kalman delta = 1e-4": dict(kalman_delta=1e-4),
        "Kalman delta = 1e-2": dict(kalman_delta=1e-2),
        "no execution lag (same-close fill, optimistic)": dict(execution_lag=0),
    }
    rows = []
    for spec, cfg in SPECS.items():
        for hedge in HEDGES:
            for name, kw in variants.items():
                if "Kalman" in name and hedge != "kalman":
                    continue
                c = cfg.with_(hedge=hedge, **kw)
                if c.execution_lag == 0:
                    # with no lag the position is filled on the signal close itself
                    c = c.with_(execution_lag=0)
                r = run_backtest(prices, c, tests)
                s = summary(r.returns, mkt, r.trades, r.windows)
                rows.append(dict(spec=spec, hedge=hedge, variant=name, sharpe=s["sharpe"],
                                 ann_return=s["ann_return"], p_value=s["p_value"],
                                 n_trades=s["n_trades"], n_pair_windows=len(r.pairs)))
    return pd.DataFrame(rows)


def worked_example(prices: pd.DataFrame, tests: dict[int, pd.DataFrame],
                   runs: dict[str, BacktestResult]) -> dict:
    y, x, start = WORKED_EXAMPLE
    ols, kal = runs[label("BH (pre-specified)", "ols")], runs[label("BH (pre-specified)", "kalman")]
    s = int(prices.index.get_loc(pd.Timestamp(start)))
    key = (s, y, x)
    if key not in ols.paths:  # fall back to the BH pair-window with most trades
        key = max(ols.paths, key=lambda k: len(ols.paths[k].trades))
        y, x, s = key[1], key[2], key[0]
    po, pk = ols.paths[key], kal.paths[key]
    pair = ols.pairs.set_index(["window_start", "y", "x"]).loc[key]
    form_dates = prices.index[s - BASELINE.formation_days: s]
    plots.worked_example(prices, y, x, form_dates, po, pk, BASELINE.entry_z, BASELINE.stop_z,
                         f"Worked example: {y} vs {x} (banks), formation "
                         f"{form_dates[0]:%b %Y}-{form_dates[-1]:%b %Y}, trading "
                         f"{po.dates[0]:%b}-{po.dates[-1]:%b %Y}",
                         FIGURES_DIR / "fig6_worked_example.png")
    tested = tests[s].set_index(["y", "x"]).loc[(y, x)]
    trades = pd.DataFrame([dict(hedge=h, **t) for h, p in (("ols", po), ("kalman", pk))
                           for t in p.trades])
    trades.to_csv(RESULTS_DIR / "worked_example_trades.csv", index=False)
    return dict(y=y, x=x, formation_start=str(form_dates[0].date()),
                formation_end=str(form_dates[-1].date()),
                trade_start=str(po.dates[0].date()), trade_end=str(po.dates[-1].date()),
                alpha=float(tested["alpha"]), beta=float(tested["beta"]),
                spread_std=float(tested["spread_std"]),
                eg_p_yx=float(tested["eg_p_yx"]), eg_p_xy=float(tested["eg_p_xy"]),
                johansen=bool(tested["johansen"]), half_life=float(tested["half_life"]),
                oos_adf_p=float(pair["oos_adf_p"]), is_adf_p=float(pair["is_adf_p"]),
                net_pnl_ols=float(po.pnl.sum()), net_pnl_kalman=float(pk.pnl.sum()),
                n_pairs_in_window=int(ols.windows.set_index("window_start").loc[s, "n_selected"]))


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    prices = load_prices()
    log.info("prices %s, %s to %s", prices.shape, prices.index[0].date(), prices.index[-1].date())
    large_moves(prices).to_csv(RESULTS_DIR / "data_quality_large_moves.csv", index=False)
    mkt = prices[BENCHMARK].pct_change()

    tests = precompute_tests(prices, BASELINE)
    all_tests = pd.concat([t.assign(window_start=s) for s, t in tests.items()], ignore_index=True)
    all_tests.to_csv(RESULTS_DIR / "formation_tests.csv", index=False)
    log.info("cointegration tests done (%d pair-windows) in %.0fs", len(all_tests), time.time() - t0)

    runs = {label(spec, h): run_backtest(prices, cfg.with_(hedge=h), tests)
            for spec, cfg in SPECS.items() for h in HEDGES}
    for name, r in runs.items():
        slug = name.split(" ")[0].lower() + "_" + r.cfg.hedge
        r.trades.to_csv(RESULTS_DIR / f"trades_{slug}.csv", index=False)
        r.pairs.to_csv(RESULTS_DIR / f"pairs_{slug}.csv", index=False)
    daily = pd.DataFrame({n: r.returns for n, r in runs.items()})
    daily[BENCHMARK] = mkt.loc[daily.index]
    daily.to_csv(RESULTS_DIR / "daily_returns.csv")

    sel = selection_table(tests, runs)
    perf = performance_table(runs, mkt)
    kvo = kalman_vs_ols(runs)
    brk = breakdown_table(runs)
    sel.to_csv(RESULTS_DIR / "selection_by_window.csv", index=False)
    perf.to_csv(RESULTS_DIR / "performance.csv")
    kvo.to_csv(RESULTS_DIR / "kalman_vs_ols.csv", index=False)
    brk.to_csv(RESULTS_DIR / "breakdown.csv", index=False)
    log.info("main backtests done in %.0fs", time.time() - t0)

    grid = run_grid(prices, tests, mkt)
    grid.to_csv(RESULTS_DIR / "sensitivity_grid.csv", index=False)
    rob = run_robustness(prices, tests, mkt)
    rob.to_csv(RESULTS_DIR / "robustness.csv", index=False)
    log.info("grid + robustness done in %.0fs", time.time() - t0)

    # Figures
    plots.cumulative_returns({n: r.returns for n, r in runs.items()},
                             FIGURES_DIR / "fig1_cumulative_returns.png")
    plots.pvalue_histogram(all_tests, FIGURES_DIR / "fig2_pvalue_histogram.png")
    plots.selection_per_window(sel, FIGURES_DIR / "fig3_pairs_per_window.png")
    plots.sensitivity_grid(grid, FIGURES_DIR / "fig4_sensitivity_grid.png")
    unc = runs[label("Uncorrected", "ols")].pairs
    plots.breakdown(unc, FIGURES_DIR / "fig5_breakdown.png",
                    f"Uncorrected selection, static OLS hedge: {len(unc)} pair-windows")
    example = worked_example(prices, tests, runs)

    # Headline numbers for the write-up
    dir_p = np.concatenate([all_tests["eg_p_yx"], all_tests["eg_p_xy"]])
    best = grid.loc[grid["sharpe"].idxmax()]
    key = dict(
        sample=dict(start=str(prices.index[0].date()), end=str(prices.index[-1].date()),
                    first_trade_day=str(daily.index[0].date()), n_days=len(daily),
                    n_years=len(daily) / TRADING_DAYS, n_tickers=prices.shape[1] - 1,
                    requested_start=START, requested_end_exclusive=END,
                    sources=sorted(set(data_sources().values()))),
        selection=dict(
            n_windows=len(tests), n_pair_tests=len(all_tests),
            directional_reject_rate_5pct=float((dir_p < 0.05).mean()),
            # Storey (2002) estimate of the share of true nulls, lambda = 0.5:
            # p-values above 0.5 come (almost) only from nulls, which are uniform.
            storey_pi0=float(min(1.0, (dir_p > 0.5).mean() / 0.5)),
            max_p_reject_rate_5pct=float((all_tests["eg_pvalue"] < 0.05).mean()),
            johansen_reject_rate_5pct=float(all_tests["johansen"].mean()),
            n_bh_significant=int(sel["n_bh_sig"].sum()),
            n_selected_bh=int(sel["n_selected_bh"].sum()),
            n_selected_uncorrected=int(sel["n_selected_uncorrected"].sum()),
            windows_with_no_bh_pair=int((sel["n_selected_bh"] == 0).sum())),
        performance=perf.replace({np.nan: None}).to_dict(orient="index"),
        kalman_vs_ols=kvo.to_dict(orient="records"),
        breakdown=brk.to_dict(orient="records"),
        grid_best=dict(best.items()) | dict(bonferroni_p=min(1.0, float(best["p_value"]) * len(grid)),
                                            n_cells=len(grid)),
        worked_example=example,
        baseline_config=asdict(BASELINE),
        runtime_seconds=time.time() - t0,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(key, indent=2, default=float))
    log.info("pipeline finished in %.0fs", time.time() - t0)
    return key


if __name__ == "__main__":
    main()
