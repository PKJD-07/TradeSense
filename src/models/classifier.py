"""ML trading classifier for TradeSense."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TradingClassifier:
    """Three-class ML classifier for DOWN, NEUTRAL, and UP predictions."""

    CLASSES = (-1, 0, 1)

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the classifier has been fitted."""
        return self._fitted

    @property
    def classes_(self) -> np.ndarray:
        """Return the classifier's learned classes."""
        if not self._fitted:
            return np.array([])

        return np.asarray(self.model.classes_)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate the feature matrix."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if X.empty:
            raise ValueError("X cannot be empty")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X.dtypes
        ):
            raise ValueError("All features must be numeric")

        if X.isna().any().any():
            raise ValueError("X cannot contain NaN values")

    def _validate_targets(self, y: pd.Series) -> None:
        """Validate target labels."""
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        if y.empty:
            raise ValueError("y cannot be empty")

        values = set(y.tolist())

        if not values.issubset(set(self.CLASSES)):
            raise ValueError(
                "Targets must contain only -1, 0, and 1"
            )

        if values != set(self.CLASSES):
            raise ValueError(
                "Training data must contain all three classes"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TradingClassifier:
        """Fit the classifier."""
        self._validate_features(X)

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of rows"
            )

        self._validate_targets(y)

        self.model.fit(X, y)
        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict DOWN (-1), NEUTRAL (0), or UP (1)."""
        if not self._fitted:
            raise RuntimeError(
                "Classifier is not fitted"
            )

        self._validate_features(X)

        return np.asarray(
            self.model.predict(X),
            dtype=int,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities in DOWN, NEUTRAL, UP order."""
        if not self._fitted:
            raise RuntimeError(
                "Classifier is not fitted"
            )

        self._validate_features(X)

        raw_probabilities = self.model.predict_proba(X)

        probabilities = np.zeros(
            (len(X), 3),
            dtype=float,
        )

        for index, class_value in enumerate(self.model.classes_):
            class_value = int(class_value)

            target_index = self.CLASSES.index(class_value)

            probabilities[:, target_index] = (
                raw_probabilities[:, index]
            )

        return probabilities

    @property
    def classes(self) -> tuple[int, int, int]:
        """Return canonical class ordering."""
        return self.CLASSES