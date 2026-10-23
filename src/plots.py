"""Figures for the README and notebook (static PNGs, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SOURCE = "Source: Yahoo Finance adjusted daily closes (via yfinance); author's calculations."

# Palette: fixed categorical order, recessive chrome.
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#eda100", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")
DIVERGING = LinearSegmentedColormap.from_list("rdgybu", [RED, "#f0efec", BLUE])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
    "font.family": "DejaVu Sans",
})


def _finish(fig: plt.Figure, path: Path, source: str = SOURCE) -> None:
    fig.text(0.01, 0.005, source, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cumulative_returns(returns: dict[str, pd.Series], path: Path) -> None:
    """Wealth index and drawdown of each strategy (one y-axis per panel)."""
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for (name, r), c in zip(returns.items(), colors):
        w = (1 + r.fillna(0)).cumprod()
        ax1.plot(w.index, w.to_numpy(), color=c, label=name)
        ax2.plot(w.index, (w / w.cummax() - 1).to_numpy() * 100, color=c, linewidth=1.2)
    ax1.axhline(1, color=AXIS, linewidth=0.8)
    ax1.set_ylabel("Growth of $1 (excess return)")
    ax1.set_title("Walk-forward pairs portfolios, after costs (2011-2026)")
    ax1.legend(loc="upper left", ncol=2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    _finish(fig, path)


def pvalue_histogram(tests: pd.DataFrame, path: Path) -> None:
    """Directional Engle-Granger p-values pooled over windows vs the uniform
    distribution expected if no pair were cointegrated."""
    p = np.concatenate([tests["eg_p_yx"].to_numpy(), tests["eg_p_xy"].to_numpy()])
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(p, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
            label=f"Observed ({len(p):,} directional tests)")
    ax.axhline(len(p) / 20, color=ORANGE, linestyle="--", linewidth=1.6,
               label="Expected under H0 for every pair (uniform)")
    ax.set_xlabel("Engle-Granger p-value (formation window)")
    ax.set_ylabel("Number of tests")
    share = (p < 0.05).mean()
    ax.set_title(f"Only a mild excess of small p-values: {share:.1%} reject at 5% "
                 "(5% expected if no pair is cointegrated)")
    ax.legend(loc="upper left")
    _finish(fig, path)


def selection_per_window(windows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = pd.to_datetime(windows["trade_start"])
    width = 60  # days
    ax.bar(x - pd.Timedelta(days=width / 2), windows["n_raw_sig"], width=width, color=BLUE,
           label="Raw EG p < 0.05 (both directions)")
    ax.bar(x + pd.Timedelta(days=width / 2), windows["n_bh_sig"], width=width, color=ORANGE,
           label="Significant after Benjamini-Hochberg (q = 5%)")
    ax.plot(x, 0.05 * windows["n_tested"], color=INK2, linestyle="--", linewidth=1.4,
            label="5% of pairs tested (upper bound on false positives if no pair is cointegrated)")
    ax.set_xlabel("Start of 6-month trading window")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Pairs passing the cointegration test in each formation window")
    ax.legend(loc="upper left")
    _finish(fig, path)


def sensitivity_grid(grid: pd.DataFrame, path: Path) -> None:
    """Sharpe ratio heatmaps: rows = entry z, cols = one-way cost per leg."""
    panels = list(grid.groupby(["spec", "hedge"], sort=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels) + 1, 4.2), squeeze=False,
                             layout="constrained")
    vmax = max(0.5, float(np.nanmax(np.abs(grid["sharpe"]))))
    for ax, ((spec, hedge), g) in zip(axes[0], panels):
        m = g.pivot(index="entry_z", columns="cost_bps", values="sharpe")
        im = ax.imshow(m.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(m.shape[1]), [f"{c:g}" for c in m.columns])
        ax.set_yticks(range(m.shape[0]), [f"{z:g}" for z in m.index])
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_title(f"{spec}\n{hedge.upper()} hedge", fontsize=9)
        base = (list(m.index).index(2.0), list(m.columns).index(5.0))
        ax.add_patch(plt.Rectangle((base[1] - 0.5, base[0] - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=2))
        ax.set_xlabel("One-way cost per leg (bp)")
        ax.set_ylabel("Entry |z|")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Sharpe ratio")
    fig.suptitle("Sensitivity of the Sharpe ratio to entry threshold and costs "
                 "(boxed cell = pre-specified baseline: |z| = 2, 5 bp)",
                 fontsize=11, fontweight="bold", color=INK)
    fig.get_layout_engine().set(h_pad=0.15)
    fig.supxlabel(SOURCE, fontsize=7, color=MUTED, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def breakdown(pairs: pd.DataFrame, path: Path, subtitle: str = "") -> None:
    """Out-of-sample vs in-sample ADF p-values of the formation spread."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 21)
    ax1.hist(pairs["is_adf_p"], bins=bins, color=BLUE, alpha=0.85, edgecolor=SURFACE,
             label="Last 126 days of formation window")
    ax1.hist(pairs["oos_adf_p"], bins=bins, color=ORANGE, alpha=0.65, edgecolor=SURFACE,
             label="126-day trading window (out of sample)")
    ax1.axvline(0.05, color=INK2, linestyle="--", linewidth=1.2)
    ax1.set_xlabel("ADF p-value of spread (formation hedge ratio)")
    ax1.set_ylabel("Pair-windows")
    ax1.set_title("Does the spread stay stationary?")
    ax1.legend(loc="upper right")

    g = pairs.assign(status=np.where(pairs["oos_broken"], "Broke OOS\n(ADF p >= 0.05)",
                                     "Held OOS\n(ADF p < 0.05)"))
    stats = g.groupby("status")["net_pnl"].agg(["mean", "count"])
    ax2.bar(stats.index, stats["mean"] * 100, color=[ORANGE, BLUE][: len(stats)], width=0.5)
    for i, (m, n) in enumerate(zip(stats["mean"], stats["count"])):
        ax2.text(i, m * 100, f"{m * 100:+.2f}%\n(n = {n})", ha="center",
                 va="bottom" if m >= 0 else "top", fontsize=8, color=INK)
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    ax2.set_ylabel("Mean net P&L per pair-window (% of pair capital)")
    ax2.set_title("What a breakdown costs")
    ax2.margins(y=0.25)
    if subtitle:
        fig.suptitle(subtitle, fontsize=10, color=INK2)
    _finish(fig, path)


def worked_example(prices: pd.DataFrame, y: str, x: str, form_dates: pd.DatetimeIndex,
                   ols_path, kal_path, entry: float, stop: float, title: str,
                   path: Path) -> None:
    """Four panels: prices, z-score with trades, hedge ratio, cumulative P&L."""
    td = ols_path.dates
    fig, axes = plt.subplots(4, 1, figsize=(9, 10.5), sharex=False,
                             gridspec_kw={"height_ratios": [1.2, 1.4, 0.8, 1]})
    # 1) normalised prices across formation + trading
    ax = axes[0]
    span = prices.loc[form_dates[0]: td[-1], [y, x]]
    norm = span / span.iloc[0] * 100
    ax.plot(norm.index, norm[y].to_numpy(), color=BLUE, label=y)
    ax.plot(norm.index, norm[x].to_numpy(), color=ORANGE, label=x)
    ax.axvspan(td[0], td[-1], color=GRID, alpha=0.6, linewidth=0)
    ax.text(td[0], ax.get_ylim()[1], "  trading window", va="top", fontsize=8, color=INK2)
    ax.set_ylabel("Price (start = 100)")
    ax.set_title(title)
    ax.legend(loc="lower left")

    # 2) z-score with thresholds and trades (OLS hedge)
    ax = axes[1]
    ax.plot(td, ols_path.z, color=BLUE, label="z-score, static OLS hedge")
    ax.plot(td, kal_path.z, color=VIOLET, linewidth=1.0, alpha=0.8,
            label="z-score, Kalman hedge")
    for lvl, ls in ((entry, "--"), (-entry, "--"), (stop, ":"), (-stop, ":")):
        ax.axhline(lvl, color=INK2, linestyle=ls, linewidth=1)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    zmap = pd.Series(ols_path.z, index=td)
    for tr in ols_path.trades:
        ax.scatter(tr["entry_date"], zmap[tr["entry_date"]], marker="^" if tr["direction"] == 1 else "v",
                   s=70, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.scatter(tr["exit_date"], zmap[tr["exit_date"]], marker="X" if tr["reason"] == "stop" else "o",
                   s=60, color=RED if tr["reason"] == "stop" else INK, edgecolor=SURFACE,
                   linewidth=1.5, zorder=5)
    ax.scatter([], [], marker="^", color=AQUA, label="enter long spread (fill, OLS)")
    ax.scatter([], [], marker="v", color=AQUA, label="enter short spread (fill, OLS)")
    ax.scatter([], [], marker="o", color=INK, label="exit (fill)")
    ax.scatter([], [], marker="X", color=RED, label="stop-loss (fill)")
    ax.set_ylabel("Spread z-score")
    ax.text(td[0], entry, f" entry ±{entry:g}", va="bottom", fontsize=7, color=INK2)
    ax.text(td[0], stop, f" stop ±{stop:g}", va="bottom", fontsize=7, color=INK2)
    # Reserve empty space below the data for the legend.
    lo = min(np.nanmin(ols_path.z), np.nanmin(kal_path.z), -stop)
    hi = max(np.nanmax(ols_path.z), np.nanmax(kal_path.z), stop)
    ax.set_ylim(lo - 0.45 * (hi - lo), hi + 1)
    ax.legend(loc="lower right", ncol=2, fontsize=7)

    # 3) hedge ratio
    ax = axes[2]
    ax.plot(td, ols_path.beta, color=BLUE, label="static OLS beta (formation)")
    ax.plot(td, kal_path.beta, color=VIOLET, label="Kalman beta (one-step-ahead)")
    ax.set_ylabel("Hedge ratio")
    ax.legend(loc="best")

    # 4) cumulative P&L
    ax = axes[3]
    ax.plot(td, np.cumsum(ols_path.pnl) * 100, color=BLUE, label="OLS hedge")
    ax.plot(td, np.cumsum(kal_path.pnl) * 100, color=VIOLET, label="Kalman hedge")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Cumulative net P&L\n(% of pair capital)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    _finish(fig, path)
