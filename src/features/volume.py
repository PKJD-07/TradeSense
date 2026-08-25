"""
Volume-based features.

All features are CAUSAL: each value at timestamp t uses only data available
at or before close_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    """Compute ratio while preserving the original Series index."""
    if isinstance(numerator, pd.Series):
        index = numerator.index
    elif isinstance(denominator, pd.Series):
        index = denominator.index
    else:
        index = None

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full_like(numerator, fill_value=np.nan, dtype=float)

    valid = (
        (denominator != 0)
        & np.isfinite(denominator)
        & np.isfinite(numerator)
    )

    result[valid] = numerator[valid] / denominator[valid]

    return pd.Series(result, index=index)


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(
        window=window,
        min_periods=window,
    ).mean()


def volume_change_1d(df: pd.DataFrame) -> pd.Series:
    """1-session volume change."""
    vol = df["volume"].astype(float)
    vol_prev = vol.shift(1)

    return _safe_ratio(vol, vol_prev) - 1.0


def relative_volume_10d(df: pd.DataFrame) -> pd.Series:
    """Volume relative to its 10-session average."""
    vol = df["volume"].astype(float)
    sma_vol = _sma(vol, 10)

    return _safe_ratio(vol, sma_vol)


def volume_trend_5d(df: pd.DataFrame) -> pd.Series:
    """5-session volume average relative to 20-session average."""
    vol = df["volume"].astype(float)

    sma_5 = _sma(vol, 5)
    sma_20 = _sma(vol, 20)

    return _safe_ratio(sma_5, sma_20) - 1.0