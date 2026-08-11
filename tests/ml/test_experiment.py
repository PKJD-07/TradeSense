"""Tests for src.ml.experiment: the full V1 pipeline orchestrator.

Covers config defaults, the complete run (all baselines + models), per-phase
metrics and prediction frames, test-region discipline (test never enters
walk-forward), determinism, walk-forward geometry, and output serialization.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.constants import SEED, TARGET_COLUMN, WARMUP_ROWS
from src.ml.experiment import (
    WALK_FORWARD_PHASE,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    save_experiment,
)
from src.ml.baselines import BASELINE_NAMES
from src.ml.models import MODEL_NAMES
from tests.features.fixtures import make_long_ohlcv_df


def _ohlcv(n: int = 120, symbols: tuple = ("AAPL", "MSFT", "SPY"), seed: int = 3):
    return make_long_ohlcv_df(symbols=symbols, n=n, seed=seed)


class TestConfigDefaults:
    def test_frozen_v1_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.seed == SEED
        assert cfg.epsilon == 0.001
        assert cfg.include_market_context is True
        assert cfg.warmup_rows == WARMUP_ROWS
        assert cfg.train_fraction == 0.70
        assert cfg.val_fraction == 0.15
        assert cfg.purge_gap == 1
        assert cfg.test_block == 63
        assert cfg.step == 63
        assert cfg.min_train_rows == 504
        assert cfg.baselines == BASELINE_NAMES
        assert cfg.models == MODEL_NAMES
        assert cfg.compute_calibration is False


class TestRunStructure:
    def _run(self, **cfg_overrides):
        cfg = ExperimentConfig(**cfg_overrides)
        return run_experiment(_ohlcv(n=120), config=cfg)

    def test_result_metadata_recorded(self):
        res = self._run(run_id="run_meta")
        assert isinstance(res, ExperimentResult)
        assert res.config.run_id == "run_meta"
        assert res.dataset["feature_names"]
        assert res.dataset["target"]["epsilon"] == 0.001
        assert res.dataset["final_rows"] > 0
        assert set(res.split) >= {"train_end", "val_end", "n_train_rows", "n_test_rows"}
        assert set(res.library_versions) >= {
            "python",
            "numpy",
            "pandas",
            "scikit_learn",
        }
        assert res.library_versions["scikit_learn"]

    def test_all_estimators_present(self):
        res = self._run()
        names = set(res.estimator_results)
        assert names == set(BASELINE_NAMES) | set(MODEL_NAMES)
        for name in MODEL_NAMES:
            assert res.estimator_results[name]["kind"] == "model"
        for name in BASELINE_NAMES:
            assert res.estimator_results[name]["kind"] == "baseline"

    def test_phase_metrics_for_all_phases(self):
        res = self._run()
        for name in res.estimator_results:
            phases = set(res.estimator_results[name]["phase_metrics"])
            assert phases == {"train", "validation", "test"}
            assert res.estimator_results[name]["phase_metrics"]["test"]["n_samples"] > 0

    def test_prediction_frames_columns_and_alignment(self):
        res = self._run()
        split = res.split
        for name, phase_frames in res.predictions.items():
            assert phase_frames["train"]["phase"].eq("train").all()
            assert phase_frames["validation"]["phase"].eq("validation").all()
            test = phase_frames["test"]
            assert test["phase"].eq("test").all()
            for phase, frame in phase_frames.items():
                for col in ("symbol", "timestamp", TARGET_COLUMN, "y_pred",
                            "y_prob_-1", "y_prob_0", "y_prob_1", "realized_return"):
                    assert col in frame.columns
                # y_prob rows sum to 1
                prob_cols = ["y_prob_-1", "y_prob_0", "y_prob_1"]
                np.testing.assert_allclose(
                    frame[prob_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-8
                )
            assert len(test) == split["n_test_rows"]

    def test_financial_diagnostics_in_test(self):
        res = self._run()
        test_metrics = res.estimator_results["persistence"]["phase_metrics"]["test"]
        assert test_metrics["financial"] is not None
        assert "long_short_proxy_spread" in test_metrics["financial"]

    def test_split_boundaries_materialized(self):
        res = self._run()
        assert res.split["train_end"] < res.split["val_end"]
        assert res.split["n_train_rows"] + res.split["n_val_rows"] + res.split["n_test_rows"] == res.dataset["final_rows"]

    def test_test_region_never_in_walk_forward(self):
        """With the default min_train=504 there are no folds; more importantly
        a run configured for folds must keep the test region out."""
        cfg = ExperimentConfig(
            models=("logistic_regression",),
            baselines=("majority_class", "persistence"),
            min_train_rows=15,
            test_block=10,
            step=10,
            run_id="wf_test",
        )
        res = run_experiment(_ohlcv(n=100, symbols=("AAPL", "MSFT", "SPY")), config=cfg)
        val_end = pd.Timestamp(res.split["val_end"])
        for name in res.predictions:
            if WALK_FORWARD_PHASE in res.predictions[name]:
                oos = res.predictions[name][WALK_FORWARD_PHASE]
                assert (pd.to_datetime(oos["timestamp"], utc=True) <= val_end).all()
                assert res.estimator_results[name]["walk_forward"]["n_folds"] > 0
            # the test phase frame must not be confused with walk-forward OOS
            assert WALK_FORWARD_PHASE not in res.estimator_results[name]["phase_metrics"]

    def test_baselines_use_raw_features(self):
        """Persistence must reproduce sign(OC_t) from intraday_return, which
        only holds if it is NOT fed a scaled feature matrix."""
        from src.ml.dataset import build_ml_dataset

        res = self._run()
        test = res.predictions["persistence"]["test"]
        # Rebuild the same panel to recover intraday_return (session-t OC move).
        ds = build_ml_dataset(_ohlcv(n=120), epsilon=res.config.epsilon)
        merged = test.merge(
            ds.df[["symbol", "timestamp", "intraday_return"]],
            on=["symbol", "timestamp"],
            how="left",
        )
        assert merged["intraday_return"].notna().all()
        eps = res.config.epsilon
        expected = np.where(
            merged["intraday_return"] > eps,
            1,
            np.where(merged["intraday_return"] < -eps, -1, 0),
        )
        np.testing.assert_array_equal(test["y_pred"].to_numpy(), expected)


class TestDeterminism:
    def test_identical_metrics_across_runs(self):
        cfg = ExperimentConfig(run_id="det")
        r1 = run_experiment(_ohlcv(n=90), config=cfg)
        r2 = run_experiment(_ohlcv(n=90), config=cfg)
        for name in r1.estimator_results:
            m1 = r1.estimator_results[name]["phase_metrics"]
            m2 = r2.estimator_results[name]["phase_metrics"]
            for phase in ("train", "validation", "test"):
                assert m1[phase]["accuracy"] == m2[phase]["accuracy"]
                assert m1[phase]["macro_f1"] == m2[phase]["macro_f1"]

    def test_identical_predictions_across_runs(self):
        cfg = ExperimentConfig(run_id="det2")
        r1 = run_experiment(_ohlcv(n=90), config=cfg)
        r2 = run_experiment(_ohlcv(n=90), config=cfg)
        for name in r1.predictions:
            for phase in ("train", "validation", "test"):
                pd.testing.assert_frame_equal(
                    r1.predictions[name][phase], r2.predictions[name][phase]
                )


class TestWalkForward:
    def test_oos_frames_and_folds(self):
        cfg = ExperimentConfig(
            models=("logistic_regression", "random_forest"),
            baselines=("majority_class", "persistence"),
            min_train_rows=15,
            test_block=10,
            step=10,
            run_id="wf",
        )
        res = run_experiment(_ohlcv(n=100, symbols=("AAPL", "MSFT", "SPY")), config=cfg)
        for name in ("logistic_regression", "random_forest", "majority_class", "persistence"):
            wf = res.estimator_results[name]["walk_forward"]
            assert wf is not None
            assert wf["n_folds"] > 0
            assert wf["n_oos_rows"] == len(res.predictions[name][WALK_FORWARD_PHASE])
            assert "oos_metrics" in wf
            assert wf["oos_metrics"]["n_samples"] == wf["n_oos_rows"]
            oos = res.predictions[name][WALK_FORWARD_PHASE]
            # one fold label per block, contiguous fold ids
            assert set(oos["fold"].unique()) == set(range(wf["n_folds"]))

    def test_default_config_has_no_folds_on_short_panel(self):
        res = run_experiment(
            _ohlcv(n=120), config=ExperimentConfig(run_id="short")
        )
        for name in res.estimator_results:
            assert res.estimator_results[name]["walk_forward"] is None


class TestSave:
    def test_writes_json_and_csvs(self, tmp_path):
        cfg = ExperimentConfig(
            models=("logistic_regression",),
            baselines=("majority_class",),
            min_train_rows=15,
            test_block=10,
            step=10,
            run_id="saved_run",
            outputs_dir=tmp_path,
        )
        res = run_experiment(_ohlcv(n=100, symbols=("AAPL", "MSFT", "SPY")), config=cfg)
        run_dir = save_experiment(res, outputs_dir=tmp_path)
        assert run_dir == tmp_path / "saved_run"
        assert (run_dir / "experiment.json").exists()
        pred_dir = run_dir / "predictions"
        csvs = list(pred_dir.glob("*.csv"))
        assert csvs
        # each estimator x phase has a CSV
        for name in res.predictions:
            for phase in res.predictions[name]:
                assert (pred_dir / f"{name}_{phase}.csv").exists()
        # JSON round-trips and matches the in-memory summary
        payload = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
        assert payload["run_id"] == "saved_run"
        assert set(payload["estimators"]) == {"logistic_regression", "majority_class"}
        # a prediction CSV round-trips
        df = pd.read_csv(pred_dir / "logistic_regression_test.csv")
        assert TARGET_COLUMN in df.columns
        assert "y_prob_-1" in df.columns


class TestErrors:
    def test_unknown_estimator_raises(self):
        cfg = ExperimentConfig(
            baselines=("not_a_baseline",), models=(), run_id="err"
        )
        with pytest.raises(ValueError, match="Unknown baseline"):
            run_experiment(_ohlcv(n=60), config=cfg)
