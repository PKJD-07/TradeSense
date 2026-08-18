"""Target generation for ML training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_targets(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Create forward-return classification targets.

    Labels:
        1  = UP
        0  = NEUTRAL
       -1  = DOWN

    The target uses future prices intentionally. Feature generation
    remains responsible for preventing feature leakage.
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

    result["future_close"] = (
        result.groupby("symbol")["close"]
        .shift(-horizon)
    )

    result["forward_return"] = (
        result["future_close"] / result["close"] - 1.0
    )

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

    # Rows without enough future data cannot have a valid target.
    result.loc[
        result["future_close"].isna(),
        "target",
    ] = np.nan

    return result.drop(columns=["future_close"])