"""Download and cache daily adjusted close prices.

Source order: Yahoo Finance (via ``yfinance``), then Stooq's public CSV
endpoint as a fallback. Each ticker is cached as ``data/raw/<TICKER>.csv``
(gitignored) together with the source it came from, so repeated runs do not hit
the network. No data is ever simulated: if every source fails for a ticker the
ticker is reported as missing and dropped from the universe.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import BENCHMARK, END, RAW_DIR, START, UNIVERSE

log = logging.getLogger(__name__)

SOURCES_FILE = "sources.json"


def _from_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    # auto_adjust=True -> prices adjusted for splits and dividends, so
    # pct_change() of the series is a total return.
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance >= 0.2 returns MultiIndex columns
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def _from_stooq(ticker: str, start: str, end: str) -> pd.Series:
    sym = ticker.lower().replace("-", ".") + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    if "Close" not in df or df.empty:
        raise ValueError(f"Stooq returned no data for {ticker}")
    s = df["Close"].loc[start:end].dropna()
    # END is exclusive, matching yfinance
    return s[s.index < pd.Timestamp(end)].rename(ticker)


def _load_sources(raw_dir: Path) -> dict[str, str]:
    f = raw_dir / SOURCES_FILE
    return json.loads(f.read_text()) if f.exists() else {}


def download_ticker(ticker: str, start: str = START, end: str = END,
                    raw_dir: Path = RAW_DIR, refresh: bool = False) -> pd.Series | None:
    """Return the cached price series, downloading it first if necessary."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{ticker}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(ticker)

    sources = _load_sources(raw_dir)
    for name, fn in (("yahoo", _from_yahoo), ("stooq", _from_stooq)):
        for attempt in range(3):
            try:
                s = fn(ticker, start, end)
                s.index = pd.DatetimeIndex(s.index).tz_localize(None)
                s.index.name = "Date"
                s.to_frame().to_csv(path)
                sources[ticker] = name
                (raw_dir / SOURCES_FILE).write_text(json.dumps(sources, indent=1, sort_keys=True))
                return s
            except Exception as exc:  # network / parsing errors -> retry, then next source
                log.warning("%s failed for %s (attempt %d): %s", name, ticker, attempt + 1, exc)
                time.sleep(2 ** attempt)
    log.error("All sources failed for %s; it is excluded (no data is simulated).", ticker)
    return None


def all_tickers() -> list[str]:
    tickers = [t for group in UNIVERSE.values() for t in group]
    return sorted(set(tickers + [BENCHMARK]))


def load_prices(refresh: bool = False, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Wide DataFrame of adjusted closes (dates x tickers), NaN before listing."""
    series = {}
    for t in all_tickers():
        s = download_ticker(t, raw_dir=raw_dir, refresh=refresh)
        if s is not None:
            series[t] = s
    prices = pd.DataFrame(series).sort_index()
    # Keep NYSE trading days only: the benchmark defines the calendar.
    prices = prices.loc[prices[BENCHMARK].notna()]
    return prices


def data_sources(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    return _load_sources(raw_dir)


def large_moves(prices: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """List daily |log returns| above ``threshold``: a data-quality screen for
    unadjusted corporate actions (e.g. spin-offs) that would corrupt spreads."""
    lr = np.log(prices).diff()
    hits = lr.stack()
    hits = hits[hits.abs() > threshold]
    return hits.rename("log_return").reset_index().rename(
        columns={"level_0": "date", "level_1": "ticker"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = load_prices()
    print(p.shape, p.index.min().date(), p.index.max().date())
    print(p.apply(lambda s: s.first_valid_index().date()).sort_values().tail())
    print(large_moves(p))
