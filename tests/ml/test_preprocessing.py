"""Tests for src.ml.preprocessing: leakage-safe pipelines.

These verify structure (scaler for linear, none for trees), correctness of the
transforms, and that every fitted parameter comes from training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.ml.preprocessing import (
    build_preprocessing_pipeline,
    requires_scaling,
    preprocessing_policy,
    MODEL_FAMILIES,
    PREPROCESSING_NONE,
    PREPROCESSING_SCALER,
)


def _xy(seed: int = 1, n: int = 100, d: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    return X, X


class TestPipelineStructure:
    def test_linear_pipeline_has_scaler(self):
        pipe = build_preprocessing_pipeline("linear")
        assert isinstance(pipe.steps[-1][1], StandardScaler)

    def test_tree_pipeline_has_no_scaler(self):
        pipe = build_preprocessing_pipeline("tree")
        # one identity passthrough step so the pipeline is not empty
        assert [name for name, _ in pipe.steps] == ["passthrough"]
        assert not any(isinstance(t, StandardScaler) for _, t in pipe.steps)

    def test_policies(self):
        assert requires_scaling("linear") is True
        assert requires_scaling("tree") is False
        assert preprocessing_policy("linear") == PREPROCESSING_SCALER
        assert preprocessing_policy("tree") == PREPROCESSING_NONE

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            build_preprocessing_pipeline("mars")
        with pytest.raises(ValueError):
            requires_scaling("mars")

    def test_imputer_opt_in(self):
        pipe = build_preprocessing_pipeline("tree", impute=True)
        assert isinstance(pipe.steps[0][1], SimpleImputer)
        pipe_lin = build_preprocessing_pipeline("linear", impute=True)
        assert isinstance(pipe_lin.steps[0][1], SimpleImputer)
        assert isinstance(pipe_lin.steps[-1][1], StandardScaler)


class TestTransformCorrectness:
    def test_linear_scale_matches_manual_standardscaler(self):
        X_train, _ = _xy(seed=2, n=120)
        X_test, _ = _xy(seed=3, n=50)
        pipe = build_preprocessing_pipeline("linear")
        pipe.fit(X_train)
        got = pipe.transform(X_test)
        scaler = StandardScaler().fit(X_train)
        expected = scaler.transform(X_test)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_tree_pipeline_is_identity(self):
        X_train, X_other = _xy(seed=4, n=80)
        pipe = build_preprocessing_pipeline("tree")
        pipe.fit(X_train)
        out = pipe.transform(X_other)
        pd.testing.assert_frame_equal(out, X_other)

    def test_transform_does_not_refit(self):
        X_train, _ = _xy(seed=5, n=120)
        X_test, _ = _xy(seed=6, n=50)
        pipe = build_preprocessing_pipeline("linear")
        pipe.fit(X_train)
        scaler = pipe.named_steps["scaler"]
        before = (scaler.mean_.copy(), scaler.scale_.copy())
        _ = pipe.transform(X_test)  # must not change fitted params
        after = (scaler.mean_, scaler.scale_)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


class TestTrainOnlyFit:
    def test_imputer_statistics_from_train_only(self):
        """Imputer fitted on train must use ONLY training statistics."""
        X_train = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, 20.0, 30.0, 40.0]}
        )
        X_test = pd.DataFrame(
            {"a": [np.nan, 100.0], "b": [0.0, 0.0]}
        )
        pipe = build_preprocessing_pipeline("tree", impute=True)
        pipe.fit(X_train)
        imputer = pipe.named_steps["imputer"]
        # train column 'a' mean excludes NaN -> (1+2+3)/3 = 2.0
        assert np.isclose(imputer.statistics_[0], 2.0)
        assert np.isclose(imputer.statistics_[1], 25.0)  # (10+20+30+40)/4
        filled = pipe.transform(X_test)  # ndarray, column order preserved
        # test row 'a' NaN is filled with the TRAIN mean (2.0), never with 100
        assert np.isclose(filled[0, 0], 2.0)
        # test row 'a' = 100 is left untouched
        assert np.isclose(filled[1, 0], 100.0)

    def test_scaler_statistics_from_train_only(self):
        X_train = pd.DataFrame({"a": [0.0, 10.0, 20.0]})
        X_test = pd.DataFrame({"a": [1000.0]})
        pipe = build_preprocessing_pipeline("linear")
        pipe.fit(X_train)
        scaler = pipe.named_steps["scaler"]
        assert np.isclose(scaler.mean_[0], 10.0)
        # StandardScaler uses population std (ddof=0): sqrt((100+0+100)/3)
        assert np.isclose(scaler.scale_[0], np.sqrt(200.0 / 3.0))
        transformed = pipe.transform(X_test)
        # (1000 - 10) / scale_
        assert np.isclose(transformed[0, 0], (1000.0 - 10.0) / np.sqrt(200.0 / 3.0))
