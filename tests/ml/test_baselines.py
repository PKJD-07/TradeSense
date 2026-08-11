"""Tests for src.ml.baselines: deterministic reference models.

These verify the three baselines' contracts:
    - MajorityClass / PriorProbability learn from TRAINING labels only.
    - Persistence is a causal rule over session t's OC move, reads only the
      ``intraday_return`` column, and learns nothing from training labels.
    - All baselines are deterministic sklearn classifiers with
      fit / predict / predict_proba.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from src.ml.baselines import (
    BASELINE_NAMES,
    INTRADAY_RETURN,
    MajorityClass,
    Persistence,
    PriorProbability,
    build_baseline,
)
from src.ml.constants import DEFAULT_EPSILON
from tests.ml.fixtures import make_ml_dataset


def _X(n: int = 50, seed: int = 1, oc: np.ndarray | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
    df[INTRADAY_RETURN] = (
        oc if oc is not None else rng.normal(0.0, 0.01, size=n)
    )
    return df


def _y(n: int = 50, seed: int = 2, p: tuple = (0.4, 0.2, 0.4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=n, p=p)


class TestSklearnContract:
    """All baselines are deterministic sklearn classifiers."""

    @pytest.mark.parametrize("cls", [MajorityClass, Persistence, PriorProbability])
    def test_is_classifier(self, cls):
        assert is_classifier(cls())

    @pytest.mark.parametrize("cls", [MajorityClass, Persistence, PriorProbability])
    def test_predict_requires_fit(self, cls):
        with pytest.raises(AttributeError):
            cls().predict(_X(n=10))

    @pytest.mark.parametrize("cls", [MajorityClass, Persistence, PriorProbability])
    def test_outputs_deterministic(self, cls):
        X, y = _X(n=40), _y(n=40)
        a, b = cls(), cls()
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))

    @pytest.mark.parametrize("cls", [MajorityClass, Persistence, PriorProbability])
    def test_output_shapes(self, cls):
        X, y = _X(n=35), _y(n=35)
        est = cls().fit(X, y)
        n = len(X)
        assert est.predict(X).shape == (n,)
        assert est.predict_proba(X).shape == (n, len(est.classes_))
        assert set(est.predict(X)) <= set(est.classes_)


class TestMajorityClass:
    def test_predicts_most_frequent_class(self):
        y = np.array([0, 0, 0, 1, 1, -1, -1, -1, -1])
        est = MajorityClass().fit(_X(n=9), y)
        assert est.majority_class_ == -1
        np.testing.assert_array_equal(est.predict(_X(n=5)), [-1] * 5)

    def test_proba_is_one_hot(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        est = MajorityClass().fit(_X(n=6), y)
        proba = est.predict_proba(_X(n=3))
        # classes_ are the training labels seen: [0, 1]; majority=0 -> col 0
        np.testing.assert_array_equal(est.classes_, [0, 1])
        expected = np.zeros((3, 2))
        expected[:, 0] = 1.0
        np.testing.assert_allclose(proba, expected)

    def test_tie_breaks_to_lowest_class(self):
        y = np.array([-1, -1, 0, 0, 1, 1])  # all counts equal -> lowest = -1
        est = MajorityClass().fit(_X(n=6), y)
        assert est.majority_class_ == -1

    def test_training_labels_only(self):
        """Future/test labels must never affect the prediction."""
        train_y = np.zeros(10, dtype=int)  # all neutral in train
        est = MajorityClass().fit(_X(n=10), train_y)
        # test labels are all bullish; a leaked baseline would predict 1
        test_y = np.ones(5, dtype=int)
        est.fit(_X(n=10), train_y)
        np.testing.assert_array_equal(est.predict(_X(n=5)), [0] * 5)
        assert test_y.sum() > 0  # the discriminating signal exists


class TestPriorProbability:
    def test_priors_match_training_counts(self):
        y = np.array([-1] * 5 + [0] * 3 + [1] * 2)
        est = PriorProbability().fit(_X(n=10), y)
        np.testing.assert_allclose(est.priors_, [0.5, 0.3, 0.2])
        np.testing.assert_array_equal(est.classes_, [-1, 0, 1])

    def test_proba_rows_are_constant_priors(self):
        y = np.array([1] * 7 + [0] * 3)
        est = PriorProbability().fit(_X(n=10), y)
        proba = est.predict_proba(_X(n=4))
        np.testing.assert_allclose(proba, np.tile(est.priors_, (4, 1)))

    def test_predict_is_argmax_prior(self):
        y = np.array([-1] * 2 + [0] * 8)
        est = PriorProbability().fit(_X(n=10), y)
        np.testing.assert_array_equal(est.predict(_X(n=3)), [0] * 3)

    def test_training_labels_only(self):
        train_y = np.zeros(8, dtype=int)
        est = PriorProbability().fit(_X(n=8), train_y)
        assert est.majority_class_ == 0
        np.testing.assert_array_equal(est.classes_, [0])
        np.testing.assert_allclose(est.priors_, [1.0])


class TestPersistence:
    def test_threshold_matches_target_epsilon(self):
        eps = 0.001
        oc = np.array([0.02, -0.02, 0.0005, 0.0, 0.0015, -0.0015])
        est = Persistence(epsilon=eps).fit(_X(n=6, oc=oc), np.zeros(6, dtype=int))
        np.testing.assert_array_equal(est.predict(_X(n=6, oc=oc)), [1, -1, 0, 0, 1, -1])

    def test_epsilon_boundary_is_neutral(self):
        """OC exactly at +/-epsilon is inside the neutral band (strict <, >)."""
        eps = 0.001
        oc = np.array([eps, -eps])
        est = Persistence(epsilon=eps).fit(_X(n=2, oc=oc), np.zeros(2, dtype=int))
        np.testing.assert_array_equal(est.predict(_X(n=2, oc=oc)), [0, 0])

    def test_uses_only_intraday_column(self):
        """Perturbing every other feature must not change predictions."""
        oc = np.array([0.02, -0.02, 0.0004])
        X = _X(n=3, oc=oc)
        est = Persistence().fit(X, np.zeros(3, dtype=int))
        base = est.predict(X)
        X_noisy = X.copy()
        X_noisy[["f1", "f2", "f3", "f4"]] *= 1000.0
        np.testing.assert_array_equal(est.predict(X_noisy), base)

    def test_fit_ignores_training_labels(self):
        oc = np.array([0.02, -0.02, 0.0004])
        X = _X(n=3, oc=oc)
        p1 = Persistence().fit(X, np.array([1, 1, 1]))
        p2 = Persistence().fit(X, np.array([-1, -1, -1]))
        np.testing.assert_array_equal(p1.predict(X), p2.predict(X))

    def test_missing_column_raises(self):
        X = _X(n=5).drop(columns=[INTRADAY_RETURN])
        est = Persistence()
        with pytest.raises(ValueError, match=INTRADAY_RETURN):
            est.fit(X, np.zeros(5, dtype=int))

    def test_proba_is_one_hot(self):
        oc = np.array([0.02, -0.02, 0.0004])
        est = Persistence().fit(_X(n=3, oc=oc), np.zeros(3, dtype=int))
        proba = est.predict_proba(_X(n=3, oc=oc))
        # classes_ = [-1, 0, 1]; predictions [1, -1, 0]
        expected = np.zeros((3, 3))
        expected[0, 2] = expected[1, 0] = expected[2, 1] = 1.0
        np.testing.assert_allclose(proba, expected)

    def test_numpy_single_column_supported(self):
        oc = np.array([0.02, -0.02])
        est = Persistence().fit(np.asarray(oc)[:, None], np.zeros(2, dtype=int))
        np.testing.assert_array_equal(est.predict(np.asarray(oc)[:, None]), [1, -1])

    def test_default_epsilon_is_target_epsilon(self):
        assert Persistence().epsilon == DEFAULT_EPSILON


class TestRegistry:
    def test_all_names_build(self):
        for name in BASELINE_NAMES:
            est = build_baseline(name)
            assert est is not None

    def test_persistence_receives_epsilon(self):
        est = build_baseline("persistence", epsilon=0.005)
        assert isinstance(est, Persistence)
        assert est.epsilon == 0.005

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline("oracle")


class TestOnRealDataset:
    """End-to-end shape/value sanity on a real ML panel."""

    @pytest.mark.parametrize("name", BASELINE_NAMES)
    def test_fit_predict_on_panel(self, name):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "SPY"), n=60, seed=7)
        X, y = ds.X, ds.y.to_numpy()
        est = build_baseline(name, epsilon=ds.report.epsilon).fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(ds.df),)
        assert set(np.unique(preds)) <= {-1, 0, 1}
        assert est.predict_proba(X).shape == (len(ds.df), 3)

    def test_persistence_matches_intraday_rule_on_panel(self):
        """On the panel, persistence must equal sign(OC_t) from the feature."""
        ds = make_ml_dataset(symbols=("AAPL", "SPY"), n=60, seed=9)
        X, y = ds.X, ds.y.to_numpy()
        est = Persistence(epsilon=ds.report.epsilon).fit(X, y)
        oc = X[INTRADAY_RETURN].to_numpy()
        eps = ds.report.epsilon
        expected = np.where(oc > eps, 1, np.where(oc < -eps, -1, 0))
        np.testing.assert_array_equal(est.predict(X), expected)
