"""
TradeSense EDA and ML target-definition layer.

Analytical utilities for understanding the statistical properties of historical
OHLCV data and for constructing ML target labels under an explicit
temporal/execution convention.

Modules:
    convert     Candle/CandleCollection -> pandas DataFrame bridge
    quality     data quality assessment
    returns     simple / log / N-period returns
    volatility  rolling and realized volatility
    drawdown    drawdown series and statistics
    statistics  descriptive stats, autocorrelation, cross-asset correlation
    targets     ML target definitions (labels only, no models)
    split       chronological and walk-forward temporal splits
    plots       visualization helpers

NOTE: This layer does NOT implement ML models, feature engineering, trading
strategies, or backtesting.
"""

from src.analysis.convert import (
    candles_to_dataframe,
    collections_to_long_dataframe,
    pivot_close_prices,
    save_candles_csv,
    load_candles_csv,
)
from src.analysis.quality import assess_quality, DataQualityReport
from src.analysis.returns import (
    simple_returns,
    log_returns,
    n_period_returns,
    open_to_close_returns,
)
from src.analysis.volatility import (
    annualized_volatility,
    rolling_volatility,
    realized_volatility,
)
from src.analysis.drawdown import (
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
)
from src.analysis.statistics import (
    describe_returns,
    autocorrelation,
    cross_asset_correlation,
)
from src.analysis.targets import (
    next_session_direction,
    forward_return,
    future_realized_volatility,
    DEFAULT_EPSILON,
    DEFAULT_HORIZON,
)
from src.analysis.split import chronological_split, walk_forward_windows

__all__ = [
    "candles_to_dataframe",
    "collections_to_long_dataframe",
    "pivot_close_prices",
    "save_candles_csv",
    "load_candles_csv",
    "assess_quality",
    "DataQualityReport",
    "simple_returns",
    "log_returns",
    "n_period_returns",
    "open_to_close_returns",
    "annualized_volatility",
    "rolling_volatility",
    "realized_volatility",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "describe_returns",
    "autocorrelation",
    "cross_asset_correlation",
    "next_session_direction",
    "forward_return",
    "future_realized_volatility",
    "DEFAULT_EPSILON",
    "DEFAULT_HORIZON",
    "chronological_split",
    "walk_forward_windows",
]
