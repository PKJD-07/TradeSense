"""Deterministic synthetic fixtures for ML-layer tests.

Builds ML panels from synthetic OHLCV via ``build_ml_dataset`` (reusing the
feature-layer fixtures), plus raw feature/target arrays for model unit tests.
All data is generated with fixed random seeds; no live API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.dataset import build_ml_dataset, MLDataset
from tests.features.fixtures import make_long_ohlcv_df, make_long_ohlcv_with_gaps

DEFAULT_SYMBOLS = ("AAPL", "MSFT", "JPM", "XOM", "SPY")


def make_ml_dataset(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    n: int = 100,
    seed: int = 7,
    epsilon: float = 0.001,
    **kwargs,
) -> MLDataset:
    """Build an MLDataset from synthetic multi-symbol OHLCV.

    If SPY is not among ``symbols``, market-context features cannot be computed
    (they require SPY as the market proxy); they are disabled automatically so
    the dataset builds cleanly.
    """
    long_ohlcv = make_long_ohlcv_df(symbols=symbols, n=n, seed=seed)
    if "SPY" not in symbols and "include_market_context" not in kwargs:
        kwargs["include_market_context"] = False
    return build_ml_dataset(long_ohlcv, epsilon=epsilon, **kwargs)


def make_ml_dataset_with_gaps(
    symbols: tuple[str, ...] = ("AAPL", "SPY"),
    n: int = 50,
    seed: int = 11,
    drop_timestamps: dict[str, list[int]] | None = None,
    max_nan_fraction: float = 0.5,
) -> MLDataset:
    """Build an MLDataset from OHLCV with intentional SPY gaps (NaN features).

    The default NaN guard is raised so the intentional gaps (a few rows in a
    50-row symbol) are dropped as feature-NaN rows instead of failing the build.
    """
    if drop_timestamps is None:
        drop_timestamps = {"SPY": [25, 30]}
    long_ohlcv = make_long_ohlcv_with_gaps(
        symbols=symbols,
        n=n,
        seed=seed,
        drop_timestamps=drop_timestamps,
    )
    return build_ml_dataset(long_ohlcv, max_nan_fraction=max_nan_fraction)


def make_synthetic_xy(
    n: int = 600,
    n_features: int = 5,
    seed: int = 11,
    signal: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) for model unit tests, independent of the panel pipeline.

    y is a noisy, thresholded linear rule over X so a logistic model can
    recover signal; labels are the 3-class set {-1, 0, +1}.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    score = signal * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0.0, 0.5, size=n)
    y = np.where(score > 0.5, 1, np.where(score < -0.5, -1, 0)).astype(int)
    return X, y
