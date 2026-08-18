"""Tests for the ML classifier."""

import numpy as np
import pandas as pd
import pytest

from src.models.classifier import (
    TradingClassifier,
)


def make_training_data():
    """Create synthetic training data with three classes."""

    X = pd.DataFrame(
        {
            "feature_1": [
                -2.0, -1.8, -1.5,
                0.0, 0.1, -0.1,
                1.5, 1.8, 2.0,
            ],
            "feature_2": [
                -1.0, -1.2, -0.8,
                0.0, 0.1, -0.1,
                0.8, 1.2, 1.0,
            ],
        }
    )

    y = pd.Series(
        [
            -1, -1, -1,
             0,  0,  0,
             1,  1,  1,
        ],
        name="target_direction_1d",
    )

    return X, y


def test_classifier_can_be_created():
    """Test classifier initialization."""

    classifier = TradingClassifier()

    assert classifier is not None
    assert classifier.is_fitted is False


def test_classifier_can_fit():
    """Test that the classifier can learn from training data."""

    X, y = make_training_data()

    classifier = TradingClassifier()

    classifier.fit(X, y)

    assert classifier.is_fitted is True


def test_predict_requires_fitted_model():
    """Prediction before fitting should fail."""

    X, _ = make_training_data()

    classifier = TradingClassifier()

    with pytest.raises(
        RuntimeError,
        match="not fitted",
    ):
        classifier.predict(X)


def test_predict_returns_three_class_labels():
    """Predictions should use the DOWN/NEUTRAL/UP labels."""

    X, y = make_training_data()

    classifier = TradingClassifier()
    classifier.fit(X, y)

    predictions = classifier.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({-1, 0, 1})


def test_predict_proba_returns_three_probabilities():
    """Each prediction should contain three class probabilities."""

    X, y = make_training_data()

    classifier = TradingClassifier()
    classifier.fit(X, y)

    probabilities = classifier.predict_proba(X)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X), 3)


def test_probabilities_sum_to_one():
    """Class probabilities must sum to one."""

    X, y = make_training_data()

    classifier = TradingClassifier()
    classifier.fit(X, y)

    probabilities = classifier.predict_proba(X)

    sums = probabilities.sum(axis=1)

    assert np.allclose(sums, 1.0)


def test_classes_are_down_neutral_up():
    """Classifier must expose the three expected classes."""

    X, y = make_training_data()

    classifier = TradingClassifier()
    classifier.fit(X, y)

    assert set(classifier.classes_) == {-1, 0, 1}


def test_predict_proba_requires_fitted_model():
    """Probability prediction before fitting should fail."""

    X, _ = make_training_data()

    classifier = TradingClassifier()

    with pytest.raises(
        RuntimeError,
        match="not fitted",
    ):
        classifier.predict_proba(X)


def test_fit_rejects_empty_data():
    """Empty training data should be rejected."""

    classifier = TradingClassifier()

    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        classifier.fit(X, y)


def test_fit_rejects_mismatched_rows():
    """Feature and target row counts must match."""

    X, y = make_training_data()

    classifier = TradingClassifier()

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        classifier.fit(X, y.iloc[:-1])


def test_fit_rejects_missing_class():
    """Training data must contain all three classes."""

    X, _ = make_training_data()

    y = pd.Series(
        [-1, -1, -1, 0, 0, 0, 0, 0, 0],
        name="target_direction_1d",
    )

    classifier = TradingClassifier()

    with pytest.raises(
        ValueError,
        match="three classes",
    ):
        classifier.fit(X, y)


def test_predict_proba_requires_dataframe():
    """Prediction input must be a DataFrame."""

    X, y = make_training_data()

    classifier = TradingClassifier()
    classifier.fit(X, y)

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        classifier.predict_proba(X.values)


def test_classifier_uses_only_numeric_features():
    """The classifier should reject non-numeric feature values."""

    X, y = make_training_data()

    X["bad_feature"] = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
    ]

    classifier = TradingClassifier()

    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        classifier.fit(X, y)