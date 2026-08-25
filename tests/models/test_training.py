import pandas as pd
import pytest

from src.models.training import (
    EvaluationResult,
    evaluate_classifier,
    train_classifier,
)


def make_dataset():
    X = pd.DataFrame(
        {
            "feature_1": [
                -2.0, -1.5, -1.0,
                0.0, 0.1, -0.1,
                1.0, 1.5, 2.0,
            ],
            "feature_2": [
                -1.0, -0.5, -0.8,
                0.0, 0.1, -0.1,
                0.8, 0.5, 1.0,
            ],
        }
    )

    y = pd.Series(
        [
            -1, -1, -1,
             0,  0,  0,
             1,  1,  1,
        ]
    )

    return X, y


def test_train_classifier_returns_fitted_classifier():
    X, y = make_dataset()

    classifier = train_classifier(X, y)

    assert classifier.is_fitted
    assert classifier.classes == (-1, 0, 1)


def test_trained_classifier_can_predict():
    X, y = make_dataset()

    classifier = train_classifier(X, y)

    predictions = classifier.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({-1, 0, 1})


def test_trained_classifier_returns_probabilities():
    X, y = make_dataset()

    classifier = train_classifier(X, y)

    probabilities = classifier.predict_proba(X)

    assert probabilities.shape == (len(X), 3)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()

    row_sums = probabilities.sum(axis=1)

    assert row_sums.tolist() == pytest.approx(
        [1.0] * len(X)
    )


def test_evaluate_classifier():
    X, y = make_dataset()

    classifier = train_classifier(X, y)

    result = evaluate_classifier(
        classifier,
        X,
        y,
    )

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.confusion_matrix) == 3
    assert len(result.confusion_matrix[0]) == 3
    assert "DOWN" in result.classification_report
    assert "NEUTRAL" in result.classification_report
    assert "UP" in result.classification_report