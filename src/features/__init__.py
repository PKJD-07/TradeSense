"""
TradeSense feature-engineering layer.

Constructs causal features for ML models. Every feature at timestamp t uses only
information available at or before close_t. No feature uses future data.

This layer does NOT:
- Train ML models
- Perform feature scaling (deferred to ML pipeline)
- Include target columns in the feature matrix
- Modify data in src/data/ or src/analysis/

Features are organized into groups:
- price: return and price-based features
- momentum: moving-average and trend features
- volatility: volatility and range features
- volume: volume-based features
- market_context: SPY-relative cross-asset features

All features are CANDIDATE features — predictive value must be established
through out-of-sample ML evaluation.
"""

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
from src.features.market_context import (
    spy_return_1d,
    relative_return_1d,
)
from src.features.builder import build_features, get_feature_names

__all__ = [
    # Price features
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    "intraday_return",
    # Momentum features
    "price_sma_ratio_10",
    "price_sma_ratio_20",
    "price_ema_ratio_10",
    "sma_cross_10_20",
    # Volatility features
    "volatility_10d",
    "volatility_20d",
    "high_low_range",
    "atr_ratio_14",
    "volatility_ratio",
    # Volume features
    "volume_change_1d",
    "relative_volume_10d",
    "volume_trend_5d",
    # Market context features
    "spy_return_1d",
    "relative_return_1d",
    # Builder
    "build_features",
    "get_feature_names",
]
