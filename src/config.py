"""Project configuration: universe, sample dates and the pre-specified baseline.

Every parameter in ``BASELINE`` was fixed *before* any backtest was run. The
sensitivity grid in ``run_pipeline.py`` varies only the entry threshold and
the cost level around this baseline; it is reported for transparency, not used
to pick the headline result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Sample period. END is exclusive (yfinance convention) and pinned so that the
# results are reproducible; the last bar used is 2026-10-22.
START = "2010-01-01"
END = "2026-10-23"

# Industry groups. Pairs are only ever formed *within* a group. Tickers are
# current (2026) liquid US large caps, which introduces survivorship bias:
# see the README ("Survivorship bias").
UNIVERSE: dict[str, list[str]] = {
    "energy": ["XOM", "CVX", "COP", "EOG", "OXY", "SLB", "HAL", "VLO", "MPC", "PSX"],
    "banks": ["JPM", "BAC", "C", "WFC", "USB", "PNC", "GS", "MS"],
    "payments": ["V", "MA", "AXP", "COF", "PYPL"],
    "beverages": ["KO", "PEP", "MNST", "STZ", "TAP", "BF-B"],
    "semis": ["INTC", "TXN", "ADI", "QCOM", "NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC"],
    "utilities": ["DUK", "SO", "AEP", "D", "XEL", "ED", "NEE", "WEC"],
    # ETF pairs: each group contains exactly one candidate pair.
    "etf_commodity_fx": ["EWA", "EWC"],  # Australia vs Canada (commodity exporters)
    "etf_gold": ["GLD", "GDX"],  # gold vs gold miners
    "etf_europe": ["EWG", "EWQ"],  # Germany vs France
}

BENCHMARK = "SPY"  # S&P 500 total-return proxy for beta / correlation


@dataclass(frozen=True)
class StrategyConfig:
    """All knobs of the strategy. Defaults = the pre-specified baseline."""

    # Walk-forward windows (trading days)
    formation_days: int = 252  # ~12 months
    trading_days: int = 126  # ~6 months

    # Pair selection (formation window only)
    fdr_q: float = 0.05  # Benjamini-Hochberg false discovery rate
    use_fdr: bool = True  # False -> raw p < fdr_q (used only as a comparison)
    require_johansen: bool = True  # Johansen trace test must reject r=0 at 5%
    min_half_life: float = 2.0  # trading days
    max_half_life: float = 63.0  # trading days (half the trading window)
    max_pairs: int = 20

    # Hedge ratio: "ols" (static, formation window) or "kalman" (time-varying)
    hedge: str = "ols"
    kalman_delta: float = 1e-3  # state-noise scale, see kalman.py

    # Trading rules on the z-score of the spread
    entry_z: float = 2.0
    exit_z: float = 0.0  # exit when the spread crosses its mean
    stop_z: float = 4.0  # stop-loss; the pair is then disabled for the window
    execution_lag: int = 1  # signal at close t, trade at close t+1

    # Position sizing: "dollar_neutral" ($1 long / $1 short) or
    # "hedge_ratio" ($1/(1+|b|) long, $|b|/(1+|b|) short; robustness only)
    weighting: str = "dollar_neutral"

    # Costs
    commission_bps: float = 1.0  # per leg, per side, on traded notional
    slippage_bps: float = 4.0  # half bid-ask spread + impact, per leg per side
    borrow_bps_pa: float = 50.0  # short-borrow fee, annual, on short notional

    @property
    def cost_bps(self) -> float:
        """One-way transaction cost per leg (commission + slippage), bps."""
        return self.commission_bps + self.slippage_bps

    def with_(self, **kwargs: object) -> "StrategyConfig":
        return replace(self, **kwargs)  # type: ignore[arg-type]


BASELINE = StrategyConfig()

# Sensitivity grid (reported, not optimised). Exit/stop/windows stay fixed.
GRID_ENTRY_Z = [1.5, 2.0, 2.5, 3.0]
GRID_COST_BPS = [0.0, 5.0, 10.0, 20.0]  # total one-way cost per leg
