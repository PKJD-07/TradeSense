"""Training and evaluation utilities for TradeSense ML models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.models.classifier import TradingClassifier


@dataclass
class EvaluationResult:
    """Results from evaluating a trained classifier."""

    accuracy: float
    confusion_matrix: list[list[int]]
    classification_report: dict


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> TradingClassifier:
    """Train and return a TradingClassifier."""

    classifier = TradingClassifier()
    classifier.fit(X_train, y_train)

    return classifier


def evaluate_classifier(
    classifier: TradingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Evaluate a trained classifier on unseen data."""

    predictions = classifier.predict(X_test)

    return EvaluationResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion_matrix=confusion_matrix(
            y_test,
            predictions,
            labels=list(classifier.classes),
        ).tolist(),
        classification_report=classification_report(
            y_test,
            predictions,
            labels=list(classifier.classes),
            target_names=["DOWN", "NEUTRAL", "UP"],
            output_dict=True,
            zero_division=0,
        ),
    )