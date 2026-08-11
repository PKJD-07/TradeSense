"""Tests for src.ml.models: the fixed-config model registry.

Verifies that the three candidate models carry exactly the documented V1
config, that model family drives the preprocessing policy (scaler for linear,
none for trees), that ``make_model_pipeline`` composes train-fitted
preprocessing + model, and that everything is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.constants import SEED
from src.ml.models import (
    CLASS_WEIGHT_SUPPORTED,
    MODEL_CONFIGS,
    MODEL_FAMILY,
    MODEL_NAMES,
    build_model,
    make_model_pipeline,
    model_family,
)
from src.ml.preprocessing import requires_scaling
from tests.ml.fixtures import make_ml_dataset, make_synthetic_xy


class TestRegistry:
    def test_model_names(self):
        assert MODEL_NAMES == (
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        )

    def test_configs_match_locked_design(self):
        assert MODEL_CONFIGS["logistic_regression"]["C"] == 1.0
        rf = MODEL_CONFIGS["random_forest"]
        assert rf["max_depth"] == 6
        assert rf["min_samples_leaf"] == 20
        assert rf["n_estimators"] == 100
        gbm = MODEL_CONFIGS["gradient_boosting"]
        assert gbm["n_estimators"] == 200
        assert gbm["max_depth"] == 3
        assert gbm["learning_rate"] == 0.1
        assert gbm["n_iter_no_change"] is None  # no early stopping
        for name in MODEL_NAMES:
            assert MODEL_CONFIGS[name]["random_state"] == SEED

    def test_family_mapping(self):
        assert MODEL_FAMILY == {
            "logistic_regression": "linear",
            "random_forest": "tree",
            "gradient_boosting": "tree",
        }
        assert model_family("logistic_regression") == "linear"
        assert model_family("random_forest") == "tree"
        assert model_family("gradient_boosting") == "tree"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("knn")
        with pytest.raises(ValueError, match="Unknown model"):
            model_family("knn")

    def test_build_returns_correct_classes(self):
        assert isinstance(build_model("logistic_regression"), LogisticRegression)
        assert isinstance(build_model("random_forest"), RandomForestClassifier)
        assert isinstance(
            build_model("gradient_boosting"), GradientBoostingClassifier
        )

    def test_build_applies_fixed_config(self):
        rf = build_model("random_forest")
        assert rf.max_depth == 6
        assert rf.min_samples_leaf == 20
        assert rf.n_estimators == 100
        lr = build_model("logistic_regression")
        assert lr.C == 1.0
        gbm = build_model("gradient_boosting")
        assert gbm.n_iter_no_change is None
        assert gbm.learning_rate == 0.1

    def test_build_returns_fresh_instances(self):
        a, b = build_model("random_forest"), build_model("random_forest")
        assert a is not b

    def test_default_unweighted(self):
        assert build_model("logistic_regression").class_weight is None
        assert build_model("random_forest").class_weight is None

    def test_class_weight_override(self):
        assert build_model("random_forest", class_weight="balanced").class_weight == "balanced"
        assert (
            build_model("logistic_regression", class_weight="balanced").class_weight
            == "balanced"
        )

    def test_class_weight_rejected_for_gbm(self):
        with pytest.raises(ValueError, match="class_weight"):
            build_model("gradient_boosting", class_weight="balanced")

    def test_class_weight_supported_models(self):
        assert set(CLASS_WEIGHT_SUPPORTED) == {
            "logistic_regression",
            "random_forest",
        }

    def test_models_are_classifiers(self):
        for name in MODEL_NAMES:
            assert is_classifier(build_model(name))


class TestFamilyScaling:
    def test_linear_needs_scaling(self):
        assert requires_scaling(model_family("logistic_regression")) is True

    def test_tree_needs_no_scaling(self):
        assert requires_scaling(model_family("random_forest")) is False
        assert requires_scaling(model_family("gradient_boosting")) is False


class TestModelPipeline:
    def test_pipeline_ends_with_model(self):
        for name in MODEL_NAMES:
            pipe = make_model_pipeline(name)
            assert isinstance(pipe, Pipeline)
            assert pipe.steps[-1][0] == "model"
            assert pipe.steps[-1][1] is not None

    def test_linear_pipeline_has_scaler(self):
        pipe = make_model_pipeline("logistic_regression")
        assert any(
            isinstance(step, StandardScaler) for _, step in pipe.steps[:-1]
        )

    def test_tree_pipeline_has_no_scaler(self):
        for name in ("random_forest", "gradient_boosting"):
            pipe = make_model_pipeline(name)
            assert not any(
                isinstance(step, StandardScaler) for _, step in pipe.steps[:-1]
            )

    def test_fit_predict_on_synthetic(self):
        X, y = make_synthetic_xy(n=400, seed=11)
        for name in MODEL_NAMES:
            pipe = make_model_pipeline(name)
            pipe.fit(X, y)
            preds = pipe.predict(X)
            assert preds.shape == (len(y),)
            assert set(np.unique(preds)) <= {-1, 0, 1}
            proba = pipe.predict_proba(X)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_deterministic_across_runs(self):
        X, y = make_synthetic_xy(n=200, seed=13)
        p1 = make_model_pipeline("random_forest").fit(X, y)
        p2 = make_model_pipeline("random_forest").fit(X, y)
        np.testing.assert_array_equal(p1.predict(X), p2.predict(X))

    def test_fit_on_real_panel(self):
        ds = make_ml_dataset(symbols=("AAPL", "MSFT", "JPM", "XOM", "SPY"), n=120, seed=15)
        X, y = ds.X, ds.y.to_numpy()
        pipe = make_model_pipeline("logistic_regression").fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(ds.df),)
        assert set(np.unique(preds)) <= {-1, 0, 1}

    def test_imputer_opt_in_composes(self):
        pipe = make_model_pipeline("logistic_regression", impute=True)
        assert pipe.steps[0][0] == "imputer"
        assert isinstance(pipe.steps[-1][1], LogisticRegression)
