"""Tests for drawdown series, maximum drawdown, and drawdown duration."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.drawdown import (
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
)


@pytest.fixture
def known_close() -> pd.Series:
    # Index 0..5; peaks at 110 (idx 1) and 120 (idx 5)
    return pd.Series([100.0, 110.0, 90.0, 105.0, 95.0, 120.0])


def test_drawdown_series_values(known_close):
    dd = drawdown_series(known_close)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)          # new peak
    assert dd.iloc[2] == pytest.approx(90.0 / 110.0 - 1.0)
    assert dd.iloc[3] == pytest.approx(105.0 / 110.0 - 1.0)
    assert dd.iloc[5] == pytest.approx(0.0)          # recovery to new peak
    assert (dd <= 0).all()


def test_max_drawdown(known_close):
    assert max_drawdown(known_close) == pytest.approx(90.0 / 110.0 - 1.0)


def test_max_drawdown_duration():
    close = pd.Series([100.0, 110.0, 90.0, 85.0, 80.0, 120.0])
    # Peak at idx 1 (110); trough 80 at idx 4; recovery at idx 5 (120).
    # Sessions below the peak: idx 2, 3, 4 -> 3 sessions.
    assert max_drawdown_duration(close) == 3


def test_drawdown_duration_monotonic_decline():
    close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])
    # Only ever one peak (idx 0); below it from idx 1 onward.
    assert max_drawdown_duration(close) == 4
