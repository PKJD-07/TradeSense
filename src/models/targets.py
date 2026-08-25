"""Target generation for ML training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_targets(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Create forward-return classification targets.

    Labels:
        1  = UP
        0  = NEUTRAL
       -1  = DOWN

    Both the original target columns and the ML-specific target columns
    are returned for compatibility with the existing pipeline.
    """

    required_columns = {"symbol", "close"}

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    result = df.copy()

    # Calculate future close separately for each symbol.
    result["future_close"] = (
        result.groupby("symbol")["close"]
        .shift(-horizon)
    )

    # Original target interface.
    result["forward_return"] = (
        result["future_close"] / result["close"] - 1.0
    )

    # ML-specific target return.
    result["target_return_1d"] = result["forward_return"]

    # Three-class target.
    result["target"] = np.select(
        [
            result["forward_return"] > threshold,
            result["forward_return"] < -threshold,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    # ML-specific target direction.
    result["target_direction_1d"] = result["target"]

    # Rows without enough future data do not have a valid target.
    invalid_rows = result["future_close"].isna()

    result.loc[invalid_rows, "forward_return"] = np.nan
    result.loc[invalid_rows, "target_return_1d"] = np.nan
    result.loc[invalid_rows, "target"] = np.nan
    result.loc[invalid_rows, "target_direction_1d"] = np.nan

    # future_close is an intermediate calculation and should not be exposed.
    result = result.drop(columns=["future_close"])

    return result