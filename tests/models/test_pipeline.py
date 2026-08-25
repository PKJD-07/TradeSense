import numpy as np
import pandas as pd
import pytest

from src.models.pipeline import (
    chronological_train_validation_test_split,
    prepare_ml_dataset,
    train_ml_pipeline,
)


def make_market_data(rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2021-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    rng = np.random.default_rng(42)

    returns = rng.normal(
        loc=0.0005,
        scale=0.02,
        size=rows,
    )

    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["AAPL"] * rows,
            "open": prices * (1 + rng.normal(0, 0.003, rows)),
            "high": prices * (1 + rng.uniform(0.005, 0.02, rows)),
            "low": prices * (1 - rng.uniform(0.005, 0.02, rows)),
            "close": prices,
            "volume": rng.integers(
                500_000,
                2_000_000,
                size=rows,
            ),
        }
    )


def test_prepare_ml_dataset_returns_features_and_targets():
    df = make_market_data()

    X, y = prepare_ml_dataset(
        df,
        horizon=1,
        threshold=0.001,
        include_market_context=False,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0


def test_prepare_ml_dataset_contains_no_metadata_features():
    df = make_market_data()

    X, y = prepare_ml_dataset(
        df,
        include_market_context=False,
    )

    assert "timestamp" not in X.columns
    assert "symbol" not in X.columns


def test_prepare_ml_dataset_contains_no_target_features():
    df = make_market_data()

    X, y = prepare_ml_dataset(
        df,
        include_market_context=False,
    )

    forbidden = {
        "target",
        "target_return_1d",
        "target_direction_1d",
        "future_close",
        "forward_return",
    }

    assert forbidden.isdisjoint(X.columns)


def test_chronological_split_preserves_order():
    df = make_market_data()

    X, y = prepare_ml_dataset(
        df,
        include_market_context=False,
    )

    split = chronological_train_validation_test_split(
        X,
        y,
        train_ratio=0.7,
        validation_ratio=0.15,
    )

    assert len(split.X_train) > 0
    assert len(split.X_validation) > 0
    assert len(split.X_test) > 0

    total = (
        len(split.X_train)
        + len(split.X_validation)
        + len(split.X_test)
    )

    assert total == len(X)


def test_chronological_split_does_not_shuffle():
    X = pd.DataFrame(
        {
            "feature": range(100),
        }
    )

    y = pd.Series(
        [0] * 100,
    )

    split = chronological_train_validation_test_split(
        X,
        y,
        train_ratio=0.7,
        validation_ratio=0.15,
    )

    assert split.X_train["feature"].tolist() == list(range(70))
    assert split.X_validation["feature"].tolist() == list(range(70, 85))
    assert split.X_test["feature"].tolist() == list(range(85, 100))


def test_split_rejects_mismatched_rows():
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series(range(9))

    with pytest.raises(ValueError):
        chronological_train_validation_test_split(X, y)


def test_train_ml_pipeline_returns_fitted_classifier():
    df = make_market_data(300)

    result = train_ml_pipeline(
        df,
        horizon=1,
        threshold=0.001,
        include_market_context=False,
    )

    assert result.classifier.is_fitted
    assert len(result.split.X_train) > 0
    assert len(result.split.X_validation) > 0
    assert len(result.split.X_test) > 0


def test_trained_pipeline_can_predict():
    df = make_market_data(300)

    result = train_ml_pipeline(
        df,
        horizon=1,
        threshold=0.001,
        include_market_context=False,
    )

    predictions = result.classifier.predict(
        result.split.X_test
    )

    assert len(predictions) == len(result.split.X_test)
    assert set(predictions).issubset({-1, 0, 1})


def test_trained_pipeline_returns_probabilities():
    df = make_market_data(300)

    result = train_ml_pipeline(
        df,
        horizon=1,
        threshold=0.001,
        include_market_context=False,
    )

    probabilities = result.classifier.predict_proba(
        result.split.X_test
    )

    assert probabilities.shape == (
        len(result.split.X_test),
        3,
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )