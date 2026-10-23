"""Builds notebooks/analysis.ipynb from the cells below (run once; the notebook is
then executed with `jupyter nbconvert --execute`). Keeping the source here makes
the notebook diff-able."""

from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str]] = [
    ("md", """# Cointegration pairs trading on US large caps, 2011-2026

**Sam Berman, University College Dublin**

This notebook runs the whole pipeline (`src/run_pipeline.py`) from cached raw
data and walks through the results. Every number in the README comes from the
tables produced here. Runtime: about 2-3 minutes (the cointegration tests on
5,455 pair-windows dominate).

Structure:
1. Data and data-quality screen
2. Pair selection: how many pairs are really cointegrated?
3. Portfolio results (BH vs uncorrected selection; OLS vs Kalman hedge)
4. Out-of-sample breakdown of cointegration
5. Sensitivity grid
6. Worked example, recomputed by hand"""),
    ("code", """import json, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import Image, display

from src import run_pipeline
from src.config import BASELINE, UNIVERSE, RESULTS_DIR, FIGURES_DIR
from src.data import load_prices, data_sources, large_moves

pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)"""),
    ("md", "## 1. Data\nDaily adjusted closes (splits and dividends) from Yahoo Finance via `yfinance`, cached in `data/raw/`."),
    ("code", """prices = load_prices()
print(prices.shape, prices.index[0].date(), "to", prices.index[-1].date())
print("sources used:", sorted(set(data_sources().values())))
print("first valid date of late listings:")
print(prices.apply(lambda s: s.first_valid_index().date()).sort_values().tail(3))
pd.DataFrame({g: [", ".join(t)] for g, t in UNIVERSE.items()}, index=["tickers"]).T"""),
    ("md", """Data-quality screen: daily moves larger than 25% in log terms. All of them
match known events rather than data errors: the oil and COVID crash of
9-18 March 2020, the Pfizer vaccine rally (VLO, 9 Nov 2020), earnings moves
(AMD 2016/2017, NVDA Nov 2016, PayPal Feb 2022, Intel Aug 2024), Constellation's
Modelo deal (STZ, Feb 2013) and Coca-Cola's stake in Monster (Aug 2014). The
COP/PSX spin-off (May 2012) is adjusted in Yahoo's series and does not appear
as a jump."""),
    ("code", "large_moves(prices)"),
    ("md", "## Run the pipeline\nThis regenerates every table in `results/` and every figure in `figures/`."),
    ("code", """key = run_pipeline.main()
print(json.dumps(key["sample"], indent=1))"""),
    ("md", """## 2. Pair selection

Each 252-day formation window tests every within-industry pair (153-174 pairs
per window, 5,455 pair-windows in total) with Engle-Granger in both directions
and with Johansen's trace test. If no pair were cointegrated, directional
EG p-values would be uniform, so about 5% would fall below 0.05 by chance."""),
    ("code", """sel = key["selection"]
pd.Series(sel)"""),
    ("code", 'display(Image(FIGURES_DIR / "fig2_pvalue_histogram.png"))'),
    ("code", 'display(Image(FIGURES_DIR / "fig3_pairs_per_window.png"))\npd.read_csv(RESULTS_DIR / "selection_by_window.csv")[["trade_start", "n_tested", "n_raw_sig", "n_bh_sig", "n_selected_bh", "n_selected_uncorrected"]]'),
    ("md", """**Reading.** 8.2% of directional tests reject at 5%, against 5% expected
under the null. That is some evidence of cointegration in aggregate, but weak:
Storey's estimate of the share of true nulls is about 0.89. After the
Benjamini-Hochberg correction only 7 pair-windows survive in 32 windows, and 26
windows have no tradable pair at all. Johansen rejects 12% of the time, more
than EG; the Johansen trace test is known to over-reject in samples this short,
which is why it is used only as a confirmation filter."""),
    ("md", "## 3. Portfolio results (after costs, committed capital)"),
    ("code", """perf = pd.read_csv(RESULTS_DIR / "performance.csv", index_col=0)
cols = ["ann_return", "ann_vol", "sharpe", "t_stat", "p_value", "max_drawdown", "turnover",
        "n_trades", "hit_rate", "avg_holding_days", "time_in_market", "beta", "beta_p", "corr",
        "alpha_ann", "alpha_p"]
perf[cols + ["trade_t_p", "trade_wilcoxon_p", "window_t_p", "top2_share"]].T"""),
    ("code", 'display(Image(FIGURES_DIR / "fig1_cumulative_returns.png"))'),
    ("code", """# Kalman minus OLS: mean daily return difference, Newey-West t-test
pd.read_csv(RESULTS_DIR / "kalman_vs_ols.csv")"""),
    ("md", "## 4. Does cointegration survive out of sample, and what does a breakdown cost?"),
    ("code", 'pd.read_csv(RESULTS_DIR / "breakdown.csv").set_index(["spec", "hedge"]).T'),
    ("code", 'display(Image(FIGURES_DIR / "fig5_breakdown.png"))'),
    ("md", """Two diagnostics of the *formation* spread out of sample:

- **`pct_oos_adf_reject`** is the share of pair-windows where the ADF test still rejects a unit root. The test has low power on 126 days, and selection biases formation half-lives downward, so failing to reject is not proof of a break.
- **`median_oos_z_sd`** and **`mean_frac_days_beyond_stop`** measure how wide the spread became in formation units, where the sd is 1 by construction.

`median_signal_z_ac1` shows the Kalman prediction errors are strongly autocorrelated (about 0.9): the filter is misspecified for an AR(1) spread."""),
    ("code", """t = pd.read_csv(RESULTS_DIR / "trades_uncorrected_kalman.csv")
print("Worst Kalman trades (uncorrected selection): large losses closed by the mean-crossing exit, not the stop")
t.nsmallest(5, "net")[["group", "y", "x", "entry_date", "exit_date", "reason", "net", "days"]]"""),
    ("md", "## 5. Sensitivity grid (reported, not optimised)"),
    ("code", """grid = pd.read_csv(RESULTS_DIR / "sensitivity_grid.csv")
display(Image(FIGURES_DIR / "fig4_sensitivity_grid.png"))
print("best of", key["grid_best"]["n_cells"], "cells:", {k: key["grid_best"][k] for k in ["spec", "hedge", "entry_z", "cost_bps", "sharpe", "p_value", "n_trades", "bonferroni_p"]})"""),
    ("code", 'pd.read_csv(RESULTS_DIR / "robustness.csv")'),
    ("md", """## 6. Worked example: C vs PNC, trading window January-July 2020

The pair was selected by the pre-specified BH rule. Below, the first entry
signal is recomputed by hand from raw prices and the formation-window
estimates, to show there is nothing hidden in the code."""),
    ("code", """ex = key["worked_example"]
pd.Series(ex)"""),
    ("code", """y, x = ex["y"], ex["x"]
trades = pd.read_csv(RESULTS_DIR / "worked_example_trades.csv", parse_dates=["entry_date", "exit_date"])
first = trades[trades.hedge == "ols"].iloc[0]
# The fill happens one day after the signal: find the signal day.
i_fill = prices.index.get_loc(first.entry_date)
signal_day = prices.index[i_fill - 1]
ly, lx = np.log(prices.loc[signal_day, y]), np.log(prices.loc[signal_day, x])
form = np.log(prices.loc[ex["formation_start"]:ex["formation_end"], [y, x]])
b, a = np.polyfit(form[x], form[y], 1)          # OLS: log C = a + b log PNC
resid = form[y] - a - b * form[x]
z = (ly - a - b * lx - resid.mean()) / resid.std(ddof=1)
print(f"signal day {signal_day.date()}: log C = {ly:.4f}, log PNC = {lx:.4f}")
print(f"alpha = {a:.4f}, beta = {b:.4f}, spread = {ly - a - b * lx:+.4f}, sd = {resid.std(ddof=1):.4f}")
print(f"z = {z:+.2f}  (>= +2 -> short the spread: short C, long PNC, filled at the next close)")
print(f"pipeline recorded entry_z = {first.entry_z:+.2f}")
trades[["hedge", "entry_date", "exit_date", "direction", "entry_z", "exit_z", "reason", "gross", "costs", "borrow", "net"]]"""),
    ("code", 'display(Image(FIGURES_DIR / "fig6_worked_example.png"))'),
    ("md", """**Walk-through (OLS hedge).**
1. *Formation (Jan 2019 - Jan 2020):* log C = -0.19 + 0.90 log PNC; EG p < 0.0001 in both directions, Johansen rejects, half-life 3.8 days. It is the only pair to pass BH in this window.
2. *14 Jan 2020:* z = +2.07 (C rich vs PNC), so short the spread: short $1 of C, long $1 of PNC, filled at the 15 Jan close. The spread converges; the exit signal fills on 25 Feb: **+6.2%** net.
3. *6 Mar:* z = +2.65, so short again, filled Monday 9 Mar. The COVID sell-off then drives the spread straight through its mean to z = -6.6 by 12 Mar: **+11.5%**. This is luck, not the thesis: the spread overshot by six standard deviations.
4. *13 Mar:* z = -3.70 (inside the [-4, -2] band), so buy the spread. It fills on Monday 16 Mar at z = -6.7, already beyond the stop, and the stop fills on 17 Mar at z = -10.9: **-9.2%**.
5. The Kalman hedge ratio drifts from 0.90 to 1.5 during the crisis: the "equilibrium" was not stable. Net for the window: **+8.5%** of pair capital (OLS) and **+7.5%** (Kalman), most of it from one overshoot. Out of sample the spread had an sd of 4.6 formation units (ADF p = 0.53), yet the pair made money. This example was chosen after seeing the results, as an illustration."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
               for kind, src in cells]
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("analysis.ipynb")
nbf.write(nb, out)
print("wrote", out)
