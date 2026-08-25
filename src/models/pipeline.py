"""End-to-end ML training pipeline for TradeSense."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.builder import build_features
from src.models.classifier import TradingClassifier
from src.models.dataset import build_ml_dataset
from src.models.targets import create_targets


@dataclass
class MLDatasetSplit:
    """Chronological train/validation/test ML split."""

    X_train: pd.DataFrame
    y_train: pd.Series

    X_validation: pd.DataFrame
    y_validation: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class MLPipelineResult:
    """Result of training the TradeSense ML pipeline."""

    classifier: TradingClassifier
    split: MLDatasetSplit


def prepare_ml_dataset(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.001,
    include_market_context: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features and targets.

    Features are generated using information available at or before
    the current timestamp. Targets use future prices and are therefore
    generated separately.
    """

    features = build_features(
        df,
        include_market_context=include_market_context,
    )

    targets = create_targets(
        df,
        horizon=horizon,
        threshold=threshold,
    )

    # The current target implementation uses target_direction_1d /
    # target_return_1d. Normalize the generic target output here so the
    # dataset builder has one stable interface.
    targets = targets.copy()

    if "target" in targets.columns:
        targets["target_direction_1d"] = targets["target"]

    if "forward_return" in targets.columns:
        targets["target_return_1d"] = targets["forward_return"]

    # Align target metadata with feature rows.
    target_columns = [
        column
        for column in [
            "timestamp",
            "symbol",
            "target_return_1d",
            "target_direction_1d",
        ]
        if column in targets.columns
    ]

    targets = targets[target_columns]

    merged = features.merge(
        targets,
        on=["timestamp", "symbol"],
        how="left",
    )

    feature_columns = [
        column
        for column in features.columns
        if column not in {"timestamp", "symbol"}
    ]

    target_columns = [
        "target_return_1d",
        "target_direction_1d",
    ]

    merged = merged.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    # Remove rows where the target does not exist or feature warm-up
    # values are unavailable.
    required_columns = feature_columns + [
        column
        for column in target_columns
        if column in merged.columns
    ]

    merged = merged.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    X = merged[
        ["timestamp", "symbol"] + feature_columns
    ]

    targets_for_dataset = merged[
        ["timestamp", "symbol"]
        + [
            column
            for column in target_columns
            if column in merged.columns
        ]
    ]

    return build_ml_dataset(
        X,
        targets_for_dataset,
    )


def chronological_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> MLDatasetSplit:
    """Split data chronologically without shuffling."""

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same number of rows"
        )

    if X.empty:
        raise ValueError("X cannot be empty")

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1"
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1"
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1"
        )

    n = len(X)

    train_end = int(n * train_ratio)
    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    if train_end == 0:
        raise ValueError("Training split is empty")

    if validation_end <= train_end:
        raise ValueError("Validation split is empty")

    if validation_end >= n:
        raise ValueError("Test split is empty")

    return MLDatasetSplit(
        X_train=X.iloc[:train_end].reset_index(drop=True),
        y_train=y.iloc[:train_end].reset_index(drop=True),

        X_validation=X.iloc[
            train_end:validation_end
        ].reset_index(drop=True),
        y_validation=y.iloc[
            train_end:validation_end
        ].reset_index(drop=True),

        X_test=X.iloc[
            validation_end:
        ].reset_index(drop=True),
        y_test=y.iloc[
            validation_end:
        ].reset_index(drop=True),
    )


def train_ml_pipeline(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.001,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    include_market_context: bool = True,
) -> MLPipelineResult:
    """Prepare data, split chronologically, and train the classifier."""

    X, y = prepare_ml_dataset(
        df,
        horizon=horizon,
        threshold=threshold,
        include_market_context=include_market_context,
    )

    split = chronological_train_validation_test_split(
        X,
        y,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    classifier = TradingClassifier()

    classifier.fit(
        split.X_train,
        split.y_train,
    )

    return MLPipelineResult(
        classifier=classifier,
        split=split,
    )