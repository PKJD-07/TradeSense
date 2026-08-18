"""Tests for ML dataset preparation."""

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import build_ml_dataset


def make_features() -> pd.DataFrame:
    """Create a small synthetic feature DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
            "symbol": ["AAPL"] * 5,
            "return_1d": [0.01, 0.02, -0.01, 0.03, 0.01],
            "volatility_10d": [0.10, 0.11, 0.12, 0.13, 0.14],
            "volume_change_1d": [0.05, -0.02, 0.10, 0.03, -0.01],
        }
    )


def make_targets() -> pd.DataFrame:
    """Create matching synthetic target data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
            "symbol": ["AAPL"] * 5,
            "target_return_1d": [
                0.03,
                -0.01,
                0.04,
                0.00,
                np.nan,
            ],
            "target_direction_1d": [
                1,
                0,
                1,
                0,
                np.nan,
            ],
        }
    )


def test_build_ml_dataset_returns_features_and_target():
    """Test that dataset preparation returns X and y."""
    features = make_features()
    targets = make_targets()

    X, y = build_ml_dataset(features, targets)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_ml_dataset_removes_rows_without_target():
    """Rows without a valid future target must be removed."""
    features = make_features()
    targets = make_targets()

    X, y = build_ml_dataset(features, targets)

    assert len(X) == 4
    assert len(y) == 4


def test_target_columns_never_enter_feature_matrix():
    """Target columns must never become ML features."""
    features = make_features()

    # Deliberately add target columns to the feature DataFrame.
    features["target_return_1d"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    features["target_direction_1d"] = [1, 1, 0, -1, 0]

    targets = make_targets()

    X, y = build_ml_dataset(features, targets)

    assert "target_return_1d" not in X.columns
    assert "target_direction_1d" not in X.columns


def test_future_target_columns_never_enter_feature_matrix():
    """Other target-related columns must also be excluded."""
    features = make_features()

    features["future_close"] = [
        101,
        102,
        103,
        104,
        105,
    ]
    features["forward_return"] = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]
    features["target"] = [1, 1, 0, -1, 1]

    targets = make_targets()

    X, _ = build_ml_dataset(features, targets)

    assert "future_close" not in X.columns
    assert "forward_return" not in X.columns
    assert "target" not in X.columns


def test_metadata_columns_are_not_features():
    """Timestamp and symbol should not be passed to the ML model."""
    features = make_features()
    targets = make_targets()

    X, _ = build_ml_dataset(features, targets)

    assert "timestamp" not in X.columns
    assert "symbol" not in X.columns


def test_feature_values_are_preserved():
    """Valid feature values should remain unchanged."""
    features = make_features()
    targets = make_targets()

    X, _ = build_ml_dataset(features, targets)

    assert X.loc[0, "return_1d"] == 0.01
    assert X.loc[1, "volatility_10d"] == 0.11
    assert X.loc[2, "volume_change_1d"] == 0.10


def test_target_values_are_preserved():
    """Valid target labels should remain unchanged."""
    features = make_features()
    targets = make_targets()

    _, y = build_ml_dataset(features, targets)

    assert y.tolist() == [1, 0, 1, 0]


def test_features_and_targets_must_have_same_length():
    """Mismatched feature and target lengths should fail."""
    features = make_features()
    targets = make_targets().iloc[:4]

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        build_ml_dataset(features, targets)


def test_missing_target_column_raises_error():
    """Missing target direction should fail."""
    features = make_features()
    targets = make_targets().drop(
        columns=["target_direction_1d"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required target columns",
    ):
        build_ml_dataset(features, targets)


def test_features_must_be_dataframe():
    """Features must be provided as a DataFrame."""
    targets = make_targets()

    with pytest.raises(
        TypeError,
        match="features must be a pandas DataFrame",
    ):
        build_ml_dataset([], targets)


def test_targets_must_be_dataframe():
    """Targets must be provided as a DataFrame."""
    features = make_features()

    with pytest.raises(
        TypeError,
        match="targets must be a pandas DataFrame",
    ):
        build_ml_dataset(features, [])


def test_output_indices_are_reset():
    """Output X and y should have clean sequential indices."""
    features = make_features()
    targets = make_targets()

    X, y = build_ml_dataset(features, targets)

    assert X.index.tolist() == [0, 1, 2, 3]
    assert y.index.tolist() == [0, 1, 2, 3]