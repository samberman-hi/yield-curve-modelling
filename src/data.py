"""Download and cache US Treasury yield and NBER recession data.

Primary source: FRED's public CSV endpoint (no API key)
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID

Fallback for the yield series: Stooq daily CSVs (``<n>usy.b`` tickers). Stooq has
no recession indicator, so USREC is FRED-only. Nothing is ever simulated: if every
source fails, a ``DataUnavailableError`` is raised.

Raw files are cached in ``data/raw/`` (gitignored) so the pipeline only hits the
network once. Delete the folder (or pass ``refresh=True``) to re-download.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"

# FRED constant-maturity Treasury series -> maturity in years.
YIELD_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
LABELS: dict[str, str] = {
    "DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M", "DGS1": "1Y", "DGS2": "2Y",
    "DGS3": "3Y", "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y", "DGS20": "20Y",
    "DGS30": "30Y",
}
# Stooq equivalents (only used if FRED is unreachable).
STOOQ_TICKERS: dict[str, str] = {
    "DGS1MO": "1musy.b", "DGS3MO": "3musy.b", "DGS6MO": "6musy.b", "DGS1": "1usy.b",
    "DGS2": "2usy.b", "DGS3": "3usy.b", "DGS5": "5usy.b", "DGS7": "7usy.b",
    "DGS10": "10usy.b", "DGS20": "20usy.b", "DGS30": "30usy.b",
}
RECESSION_SERIES = "USREC"

# Window in which FRED's DGS30 is an extrapolated estimate, not a market yield.
DGS30_GAP = ("2002-02-19", "2006-02-08")

# Which source each series actually came from in this run (reported in the README).
SOURCES_USED: dict[str, str] = {}


class DataUnavailableError(RuntimeError):
    """Raised when a series cannot be obtained from any free source."""


def _get(url: str, timeout: int = 60, retries: int = 4) -> str:
    """HTTP GET with exponential back-off (FRED occasionally drops connections)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _parse_fred_csv(text: str, sid: str) -> pd.Series:
    """Parse a fredgraph CSV. Missing observations ('.' or blank) become NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2 or sid not in df.columns:
        raise ValueError(f"Unexpected FRED payload for {sid}: columns={list(df.columns)}")
    date_col = df.columns[0]  # 'observation_date' (older payloads used 'DATE')
    s = pd.Series(pd.to_numeric(df[sid], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name=sid)
    return s.sort_index()


def _parse_stooq_csv(text: str, sid: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if "Close" not in df.columns or "Date" not in df.columns:
        raise ValueError(f"Unexpected Stooq payload for {sid}")
    s = pd.Series(df["Close"].astype(float).values, index=pd.to_datetime(df["Date"]), name=sid)
    return s.sort_index()


def fetch_series(sid: str, refresh: bool = False) -> pd.Series:
    """Return one series, from the local cache if present, else FRED, else Stooq."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{sid}.csv"
    if cache.exists() and not refresh:
        SOURCES_USED.setdefault(sid, "FRED (cached)")
        return _parse_fred_csv(cache.read_text(), sid)

    errors: list[str] = []
    try:
        text = _get(FRED_URL.format(sid=sid))
        s = _parse_fred_csv(text, sid)  # validate before caching
        cache.write_text(text)
        SOURCES_USED[sid] = "FRED"
        return s
    except Exception as exc:  # network block, HTTP error or malformed payload
        errors.append(f"FRED: {exc}")
        logger.warning("FRED download failed for %s: %s", sid, exc)

    if sid in STOOQ_TICKERS:
        try:
            s = _parse_stooq_csv(_get(STOOQ_URL.format(ticker=STOOQ_TICKERS[sid])), sid)
            # Cache in FRED format so the reader above works unchanged.
            out = s.rename_axis("observation_date").reset_index()
            cache.write_text(out.to_csv(index=False))
            SOURCES_USED[sid] = "Stooq"
            return s
        except Exception as exc:
            errors.append(f"Stooq: {exc}")

    raise DataUnavailableError(f"Could not download {sid}: " + "; ".join(errors))


def load_yields(refresh: bool = False, full_history: bool = False,
                mask_30y_gap: bool = True) -> pd.DataFrame:
    """Daily constant-maturity Treasury yields in percent, one column per FRED series.

    Rows where *every* series is missing (US bond-market holidays) are dropped.
    Individual gaps are kept as NaN. In particular, Treasury stopped issuing the
    30-year bond in 2001; between 2002-02-19 and 2006-02-08 FRED's DGS30 is not a
    market yield but Treasury's extrapolated estimate (the 20-year CMT plus an
    adjustment factor). The series jumps by +17bp / -16bp on the first and last
    days of that window, so we set it to NaN there (``mask_30y_gap=True``).

    By default the sample starts at the earliest date on which all eleven series
    have started (2001-07-31, the first DGS1MO print). ``full_history=True`` keeps
    each series' full history (used by the recession and trade backtests, which
    only need a few maturities and benefit from more business cycles).
    """
    df = pd.concat([fetch_series(sid, refresh) for sid in YIELD_SERIES], axis=1, sort=True)
    df = df.dropna(how="all")
    if mask_30y_gap:
        df.loc[DGS30_GAP[0]:DGS30_GAP[1], "DGS30"] = float("nan")
    if not full_history:
        start = max(df[c].first_valid_index() for c in df.columns)
        df = df.loc[start:]
    return df


def load_recessions(refresh: bool = False) -> pd.Series:
    """Monthly NBER recession indicator (1 = recession month), indexed by month start."""
    s = fetch_series(RECESSION_SERIES, refresh).dropna().astype(int)
    return s


def recession_periods(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert the monthly 0/1 indicator into (start, end) date pairs for chart shading."""
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for date, flag in usrec.items():
        if flag == 1 and start is None:
            start = date
        elif flag == 0 and start is not None:
            periods.append((start, date))
            start = None
    if start is not None:
        periods.append((start, usrec.index[-1] + pd.offsets.MonthEnd(0)))
    return periods


def maturities(columns: list[str] | pd.Index) -> list[float]:
    """Maturities in years for a list of FRED series ids."""
    return [YIELD_SERIES[c] for c in columns]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    y = load_yields(refresh=True, full_history=True)
    r = load_recessions(refresh=True)
    print(y.describe().T[["count"]].assign(first=[y[c].first_valid_index().date() for c in y],
                                             last=[y[c].last_valid_index().date() for c in y]))
    print("USREC", r.index[0].date(), "->", r.index[-1].date())
    print(SOURCES_USED)
