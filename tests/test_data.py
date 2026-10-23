"""FRED CSV parsing and recession-period extraction on inline inputs (no network, no files)."""

import numpy as np
import pandas as pd
import pytest

from src.data import YIELD_SERIES, _parse_fred_csv, maturities, recession_periods


def test_parse_fred_csv_missing_values_and_order():
    """'.' and blank cells become NaN, dates become a sorted DatetimeIndex, values stay floats."""
    text = ("observation_date,DGS10\n"
            "2024-01-03,3.91\n"
            "2024-01-01,.\n"
            "2024-01-02,3.95\n"
            "2024-01-04,\n")
    s = _parse_fred_csv(text, "DGS10")
    assert list(s.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02",
                                                 "2024-01-03", "2024-01-04"]))
    np.testing.assert_allclose(s.to_numpy(), [np.nan, 3.95, 3.91, np.nan])
    assert s.name == "DGS10"


def test_parse_fred_csv_rejects_wrong_series():
    """A payload whose value column is not the requested series id raises ValueError."""
    with pytest.raises(ValueError):
        _parse_fred_csv("observation_date,DGS2\n2024-01-01,4.0\n", "DGS10")


def test_recession_periods_hand_series():
    """0,1,1,0,0,1,1 from Jan 2020 -> (Feb 1, Apr 1) and an open episode (Jun 1, Jul 31)."""
    idx = pd.date_range("2020-01-01", periods=7, freq="MS")
    usrec = pd.Series([0, 1, 1, 0, 0, 1, 1], idx)
    assert recession_periods(usrec) == [
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
        (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-07-31")),
    ]


def test_maturities_mapping():
    """FRED ids map to years: DGS3MO -> 0.25, DGS2 -> 2, DGS30 -> 30."""
    assert maturities(["DGS3MO", "DGS2", "DGS30"]) == [0.25, 2.0, 30.0]
    assert YIELD_SERIES["DGS1MO"] == pytest.approx(1 / 12)
