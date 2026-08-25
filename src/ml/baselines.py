"""Baseline models for the ML pipeline.

Every baseline implements the sklearn estimator interface
``fit(X, y)`` / ``predict(X)`` / ``predict_proba(X)`` so that ``experiment.py``
can evaluate them with the same per-phase metric blocks as the candidate
models. Baselines are deliberately trivial: they establish the reference point
a candidate model must beat before any predictive signal can be claimed.

Baselines (docs/ml_pipeline.md, §6):
    - MajorityClass: predicts the most frequent class from TRAINING labels
      only. No future information.
    - Persistence: predicts ``y_hat_t = sign(OC_t)`` thresholded by the same
      epsilon as the target, where ``OC_t = close_t/open_t - 1`` is session
      t's own open-to-close move ("tomorrow continues today's direction").
      Causally valid: OC_t is known immediately after close_t. It consumes the
      existing ``intraday_return`` feature column from the panel; the baseline
      does NOT add a column to X.
    - PriorProbability: predicts training-label priors as probabilities. A
      calibration reference point.

All three are deterministic and learn nothing from validation or test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from src.ml.constants import DEFAULT_EPSILON

#: Feature column consumed by the persistence baseline (session t's OC move).
INTRADAY_RETURN = "intraday_return"

#: Canonical 3-class output domain of the persistence rule. Fixed (not derived
#: from training labels) so ``predict_proba`` stays aligned even when a
#: training phase happens to lack one of the classes.
PERSISTENCE_CLASSES = np.array([-1, 0, 1])

#: Registry of baseline names accepted by ``build_baseline``.
BASELINE_NAMES = ("majority_class", "persistence", "prior_probability")


def _label_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique classes and their counts in ``y``."""
    y = np.asarray(y)
    return np.unique(y, return_counts=True)


class MajorityClass(ClassifierMixin, BaseEstimator):
    """Predict the most frequent class of the training labels.

    Learns nothing from X: the prediction is a constant based only on the
    class distribution of ``y`` passed to ``fit``. Ties break to the lowest
    class value for determinism.
    """

    def fit(self, X, y):  # noqa: N803 (sklearn signature)
        classes, counts = _label_counts(y)
        self.classes_ = classes
        self.majority_class_ = classes[int(np.argmax(counts))]
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):  # noqa: N803
        proba = np.zeros((len(X), len(self.classes_)))
        col = int(np.flatnonzero(self.classes_ == self.majority_class_)[0])
        proba[:, col] = 1.0
        return proba


class PriorProbability(ClassifierMixin, BaseEstimator):
    """Predict training-label priors as probabilities.

    A calibration reference: this is the best possible constant model under
    log loss given only the training distribution. ``predict`` returns the
    most likely class (identical to MajorityClass); ``predict_proba`` returns
    the full training prior vector.
    """

    def fit(self, X, y):  # noqa: N803
        classes, counts = _label_counts(y)
        self.classes_ = classes
        self.priors_ = counts.astype(float) / counts.sum()
        self.majority_class_ = classes[int(np.argmax(counts))]
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):  # noqa: N803
        return np.tile(self.priors_, (len(X), 1))


class Persistence(ClassifierMixin, BaseEstimator):
    """Predict session t+1's direction from session t's own OC move.

    ``y_hat_t = sign(OC_t)`` thresholded by ``epsilon``, where
    ``OC_t = close_t/open_t - 1`` ("tomorrow continues today's direction").
    OC_t is the ``intraday_return`` feature already present in the panel; this
    baseline reads it from X and does not add any column.

    Causally valid: OC_t is known immediately after close_t. ``fit`` records
    no statistics from training labels — predictions depend only on X.

    NOTE: do not run persistence through a scaling pipeline. The epsilon
    threshold is defined on the raw OC move, so ``X`` must be the raw feature
    matrix.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        feature: str = INTRADAY_RETURN,
    ):
        self.epsilon = epsilon
        self.feature = feature

    def fit(self, X, y):  # noqa: N803
        if not isinstance(self.epsilon, (int, float)) or not self.epsilon >= 0:
            raise ValueError("epsilon must be a non-negative number")
        # Fixed output domain; independent of the training label distribution.
        self.classes_ = PERSISTENCE_CLASSES
        self._feature_col = self._resolve_feature(X)
        return self

    def _resolve_feature(self, X) -> int:
        """Column index of the OC-move feature in ``X``."""
        if isinstance(X, pd.DataFrame):
            if self.feature not in X.columns:
                raise ValueError(
                    f"Persistence requires the '{self.feature}' column in X; "
                    "pass the raw feature matrix, not a preprocessed one"
                )
            return X.columns.get_loc(self.feature)
        arr = np.asarray(X)
        if arr.ndim not in (1, 2) or (arr.ndim == 2 and arr.shape[1] != 1):
            raise ValueError(
                "Persistence with a numpy X expects the single "
                f"'{self.feature}' column (shape (n,) or (n, 1))"
            )
        return 0

    def _oc(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.feature].to_numpy(dtype=float)
        arr = np.asarray(X)
        return arr[:, self._feature_col].astype(float) if arr.ndim == 2 else arr.astype(float)

    def predict(self, X):  # noqa: N803
        if not hasattr(self, "classes_"):
            raise NotFittedError(
                "This Persistence instance is not fitted yet; call 'fit' first."
            )
        oc = self._oc(X)
        y_hat = np.zeros(len(oc), dtype=int)
        y_hat[oc > self.epsilon] = 1
        y_hat[oc < -self.epsilon] = -1
        return y_hat

    def predict_proba(self, X):  # noqa: N803
        preds = self.predict(X)
        proba = np.zeros((len(preds), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            proba[:, i] = preds == cls
        return proba


def build_baseline(name: str, epsilon: float = DEFAULT_EPSILON):
    """Instantiate a baseline by registry name.

    Args:
        name: One of ``BASELINE_NAMES``.
        epsilon: Target epsilon, threaded into Persistence so its threshold
            matches the target definition exactly.

    Returns:
        A new, unfitted baseline instance.

    Raises:
        ValueError: If ``name`` is not a known baseline.
    """
    if name == "majority_class":
        return MajorityClass()
    if name == "persistence":
        return Persistence(epsilon=epsilon)
    if name == "prior_probability":
        return PriorProbability()
    raise ValueError(f"Unknown baseline '{name}'; expected one of {BASELINE_NAMES}")
