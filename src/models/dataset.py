"""Dataset preparation for ML training."""

from __future__ import annotations

import pandas as pd


def build_ml_dataset(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine features and target labels into an ML-ready dataset."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    if not isinstance(targets, pd.DataFrame):
        raise TypeError("targets must be a pandas DataFrame")

    required_target_columns = {
        "target_direction_1d",
    }

    missing = required_target_columns - set(targets.columns)

    if missing:
        raise ValueError(
            f"Missing required target columns: {sorted(missing)}"
        )

    if len(features) != len(targets):
        raise ValueError(
            "features and targets must have the same number of rows"
        )

    X = features.copy()
    y = targets["target_direction_1d"].copy()

    # Target-related columns must never enter the ML feature matrix.
    target_columns = {
        "target_return_1d",
        "target_direction_1d",
        "future_close",
        "forward_return",
        "target",
    }

    X = X.drop(
        columns=[
            column
            for column in target_columns
            if column in X.columns
        ],
        errors="ignore",
    )

    # Metadata is not an ML feature.
    X = X.drop(
        columns=[
            column
            for column in ("timestamp", "symbol")
            if column in X.columns
        ],
        errors="ignore",
    )

    # Remove rows where the target is unavailable.
    valid_rows = y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    return X, y