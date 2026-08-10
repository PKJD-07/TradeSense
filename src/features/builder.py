"""
Feature builder / orchestrator.

Combines individual feature groups into a unified feature matrix with proper
per-symbol isolation and cross-asset alignment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.features.price import (
    return_1d,
    return_5d,
    return_10d,
    return_20d,
    log_return_1d,
    intraday_return,
)
from src.features.momentum import (
    price_sma_ratio_10,
    price_sma_ratio_20,
    price_ema_ratio_10,
    sma_cross_10_20,
)
from src.features.volatility import (
    volatility_10d,
    volatility_20d,
    high_low_range,
    atr_ratio_14,
    volatility_ratio,
)
from src.features.volume import (
    volume_change_1d,
    relative_volume_10d,
    volume_trend_5d,
)
from src.features.market_context import compute_market_context


# Feature function registry
STOCK_FEATURES: list[tuple[str, Callable]] = [
    ("return_1d", return_1d),
    ("return_5d", return_5d),
    ("return_10d", return_10d),
    ("return_20d", return_20d),
    ("log_return_1d", log_return_1d),
    ("intraday_return", intraday_return),
    ("price_sma_ratio_10", price_sma_ratio_10),
    ("price_sma_ratio_20", price_sma_ratio_20),
    ("price_ema_ratio_10", price_ema_ratio_10),
    ("sma_cross_10_20", sma_cross_10_20),
    ("volatility_10d", volatility_10d),
    ("volatility_20d", volatility_20d),
    ("high_low_range", high_low_range),
    ("atr_ratio_14", atr_ratio_14),
    ("volatility_ratio", volatility_ratio),
    ("volume_change_1d", volume_change_1d),
    ("relative_volume_10d", relative_volume_10d),
    ("volume_trend_5d", volume_trend_5d),
]

MARKET_CONTEXT_FEATURES = ["spy_return_1d", "relative_return_1d"]

ALL_FEATURE_NAMES = [name for name, _ in STOCK_FEATURES] + MARKET_CONTEXT_FEATURES


def build_features(
    df: pd.DataFrame,
    include_market_context: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix from OHLCV data.

    Per-symbol isolation: all rolling features are computed independently for
    each symbol using groupby. No rolling window ever crosses symbol boundaries.

    Cross-asset alignment: market-context features use LEFT JOIN from each
    stock's timestamps onto SPY. Stock observations are always preserved.
    Missing SPY data results in NaN market-context features.

    Args:
        df: Long-form DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
        include_market_context: Whether to include SPY-relative features

    Returns:
        DataFrame with columns [timestamp, symbol, feature_1, feature_2, ...]
        Feature columns contain NaN where lookback is insufficient.
    """
    if "symbol" not in df.columns:
        raise ValueError("Input DataFrame must have 'symbol' column")
    if "timestamp" not in df.columns:
        raise ValueError("Input DataFrame must have 'timestamp' column")

    # Ensure proper dtypes and sorting
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Compute stock-specific features per symbol
    feature_frames = []

    for symbol, group in df.groupby("symbol"):
        group = group.set_index("timestamp")

        # Compute each feature
        features = {}
        for name, func in STOCK_FEATURES:
            features[name] = func(group.reset_index().set_index("timestamp"))

        # Combine features for this symbol
        symbol_features = pd.DataFrame(features, index=group.index)
        symbol_features["symbol"] = symbol
        symbol_features = symbol_features.reset_index()
        feature_frames.append(symbol_features)

    result = pd.concat(feature_frames, ignore_index=True)

    # Add market-context features
    if include_market_context:
        market_ctx = compute_market_context(df)

        # Merge market context onto result (LEFT JOIN preserves all stock rows)
        result = result.merge(
            market_ctx,
            on=["timestamp", "symbol"],
            how="left",
        )

    # Ensure stable column ordering
    output_columns = ["timestamp", "symbol"] + get_feature_names(
        include_market_context=include_market_context
    )
    result = result[output_columns]

    return result


def get_feature_names(include_market_context: bool = True) -> list[str]:
    """Return the list of feature column names.

    Args:
        include_market_context: Whether to include market-context features

    Returns:
        List of feature names in output order
    """
    if include_market_context:
        return ALL_FEATURE_NAMES.copy()
    return [name for name, _ in STOCK_FEATURES]
