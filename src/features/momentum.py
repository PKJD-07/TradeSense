"""
Momentum and trend features using moving averages.

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
    """Simple moving average with explicit window."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with explicit minimum period."""
    return series.ewm(
        span=span,
        min_periods=span,
        adjust=False,
    ).mean()


def price_sma_ratio_10(df: pd.DataFrame) -> pd.Series:
    """Price relative to 10-session SMA."""
    sma = _sma(df["close"], 10)
    return _safe_ratio(df["close"], sma)


def price_sma_ratio_20(df: pd.DataFrame) -> pd.Series:
    """Price relative to 20-session SMA."""
    sma = _sma(df["close"], 20)
    return _safe_ratio(df["close"], sma)


def price_ema_ratio_10(df: pd.DataFrame) -> pd.Series:
    """Price relative to 10-session EMA."""
    ema = _ema(df["close"], 10)
    return _safe_ratio(df["close"], ema)


def sma_cross_10_20(df: pd.DataFrame) -> pd.Series:
    """SMA(10) / SMA(20) - 1."""
    sma_10 = _sma(df["close"], 10)
    sma_20 = _sma(df["close"], 20)

    return _safe_ratio(sma_10, sma_20) - 1.0