"""Shared matplotlib style for the figures (light surface, recessive grid, fixed colour order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Categorical colours, always assigned in this order (validated colour-blind-safe set).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RECESSION = "#d9d8d3"

SOURCE_FRED = "Source: FRED (Board of Governors H.15 constant-maturity yields; NBER via USREC)."


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
        "axes.prop_cycle": matplotlib.cycler(color=C),
    })


def shade_recessions(ax, periods, label: bool = True) -> None:
    for i, (s, e) in enumerate(periods):
        ax.axvspan(s, e, color=RECESSION, alpha=0.8, lw=0, zorder=0,
                   label="NBER recession" if (label and i == 0) else None)


def save(fig, name: str, source: str = SOURCE_FRED) -> Path:
    """Add the data-source footnote and write figures/<name>.png."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.text(0.01, 0.005, source + "  Author: Sam Berman.", fontsize=7.5, color=INK2,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
