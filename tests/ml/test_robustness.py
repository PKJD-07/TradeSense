"""Tests for src.ml.robustness: the 13-step V1 diagnostic suite.

Verifies that each analysis function runs without error, returns the expected
schema, and that the orchestrator produces a complete RobustnessReport.
All data is generated with fixed random seeds; no live API calls.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.robustness import (
    RobustnessReport,
    run_robustness_analysis,
    save_robustness_report,
)
from src.ml.experiment import ExperimentConfig, ExperimentResult
from tests.features.fixtures import make_long_ohlcv_df
from pathlib import Path

SYMBOLS = ("AAPL", "MSFT", "JPM", "XOM", "SPY")


class TestRobustnessReport:
    """Test the RobustnessReport dataclass serialization."""

    def test_report_serialization(self, tmp_path: Path):
        """Test that RobustnessReport can be serialized and deserialized."""
        cfg = ExperimentConfig(run_id="test", outputs_dir=tmp_path)
        report = RobustnessReport(
            config=cfg,
            reproduction={"run_id": "test", "determinism": {"match": True, "diffs": []}, "metric_summary": {}},
            per_symbol={"per_symbol": {}, "symbol_consistency": {}, "n_symbols": 5},
            temporal_stability={"quartiles": {}, "drift": {}},
            regimes={"regimes": {}, "regime_counts": {}},
            walk_forward_stability={},
            calibration={},
            feature_importance={"importances": {}, "rank_correlation": 0.0, "top_features": []},
            class_imbalance={"class_counts": {}, "shares": {}, "imbalance_ratio": 0.0, "per_phase_distribution": {}, "minority_recall": {}},
            overfitting={},
            probability_distribution={},
            confidence_monotonicity={},
            prediction_concentration={},
            significance={"available": False, "per_model": {}, "best_model": None, "best_macro_f1": 0.0},
        )

        out_path = save_robustness_report(report, outputs_dir=tmp_path)
        assert out_path.exists()
        assert out_path.is_dir()
        assert (out_path / "robustness.json").exists()

        with open(out_path / "robustness.json") as f:
            loaded = json.load(f)

        assert loaded["config"]["run_id"] == "test"
        assert loaded["reproduction"]["determinism"]["match"] is True

    def test_report_to_dict_excludes_internal(self):
        """Verify to_dict() strips private '_result' key."""
        cfg = ExperimentConfig(run_id="test_to_dict")
        report = RobustnessReport(
            config=cfg,
            reproduction={"_result": "internal", "determinism": {"match": True}},
            per_symbol={},
            temporal_stability={},
            regimes={},
            walk_forward_stability={},
            calibration={},
            feature_importance={},
            class_imbalance={},
            overfitting={},
            probability_distribution={},
            confidence_monotonicity={},
            prediction_concentration={},
            significance={},
        )
        d = report.to_dict()
        assert "_result" not in d["reproduction"]
        assert "determinism" in d["reproduction"]


class TestAnalyzeReproduction:
    """Tests for analyze_reproduction: determinism check."""

    def test_reproduction_runs(self):
        """Verify reproduction analysis runs and returns expected keys."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_repro", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        repro = result.reproduction
        assert "determinism" in repro
        assert "match" in repro["determinism"]
        assert isinstance(repro["determinism"]["match"], bool)
        assert "run_id" in repro
        assert "dataset_rows" in repro
        assert "metric_summary" in repro
        assert len(repro["metric_summary"]) > 0


class TestAnalyzePerSymbol:
    """Tests for analyze_per_symbol: hold-one-symbol-out."""

    def test_per_symbol_runs(self):
        """Verify per-symbol analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_per_symbol", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        per_sym = result.per_symbol
        assert "per_symbol" in per_sym
        assert "symbol_consistency" in per_sym
        assert "n_symbols" in per_sym
        assert per_sym["n_symbols"] == 5

        for sym in SYMBOLS:
            assert sym in per_sym["per_symbol"]
            assert len(per_sym["per_symbol"][sym]) > 0

        for est, cv in per_sym["symbol_consistency"].items():
            assert isinstance(cv, float)
            assert cv >= 0.0


class TestAnalyzeTemporalStability:
    """Tests for analyze_temporal_stability: test-region quartile drift."""

    def test_temporal_stability_runs(self):
        """Verify temporal stability analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_temporal", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        temp = result.temporal_stability
        assert "quartiles" in temp
        assert "drift" in temp

        for est, quartile_list in temp["quartiles"].items():
            assert isinstance(quartile_list, list)
            for q in quartile_list:
                assert "period" in q
                assert "n" in q
                assert "accuracy" in q
                assert "macro_f1" in q

        for est, drift in temp["drift"].items():
            assert isinstance(drift, float)


class TestAnalyzeRegimes:
    """Tests for analyze_regimes: SPY trend × volatility buckets."""

    def test_regimes_runs(self):
        """Verify regime analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_regimes", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        regimes = result.regimes
        assert "regimes" in regimes
        assert "regime_counts" in regimes

        # Synthetic data may not produce all 4 regimes; check at least some exist
        assert len(regimes["regimes"]) > 0
        for reg in regimes["regimes"]:
            assert reg in regimes["regime_counts"]
            assert isinstance(regimes["regime_counts"][reg], int)
            assert regimes["regime_counts"][reg] >= 0
            # Each regime should have results for each estimator
            for est in regimes["regimes"][reg]:
                assert "accuracy" in regimes["regimes"][reg][est]
                assert "macro_f1" in regimes["regimes"][reg][est]
                assert "n" in regimes["regimes"][reg][est]


class TestAnalyzeWalkForwardStability:
    """Tests for analyze_walk_forward_stability: fold-to-fold OOS dispersion."""

    def test_walk_forward_runs(self):
        """Verify walk-forward stability returns expected structure."""
        # Use smaller min_train_rows to ensure folds exist with synthetic data
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(
            run_id="test_wf",
            outputs_dir=Path("outputs/ml_experiments/test"),
            min_train_rows=15,
            test_block=10,
            step=10,
        )
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        wf = result.walk_forward_stability
        assert len(wf) > 0

        for est, metrics in wf.items():
            assert "n_folds" in metrics
            assert "fold_accuracy" in metrics
            assert "mean" in metrics
            assert "std" in metrics
            assert "cv" in metrics
            assert isinstance(metrics["n_folds"], int)
            assert metrics["n_folds"] >= 0
            assert isinstance(metrics["fold_accuracy"], list)
            if metrics["n_folds"] > 0:
                assert len(metrics["fold_accuracy"]) == metrics["n_folds"]
                assert isinstance(metrics["mean"], float)
                assert isinstance(metrics["std"], float)
                assert isinstance(metrics["cv"], float)
                assert metrics["cv"] >= 0.0
            else:
                # No folds case - metrics are None
                assert metrics["mean"] is None
                assert metrics["std"] is None
                assert metrics["cv"] is None


class TestAnalyzeCalibration:
    """Tests for analyze_calibration: validation probability reliability."""

    def test_calibration_runs(self):
        """Verify calibration analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_cal", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        cal = result.calibration
        assert len(cal) > 0

        for est, metrics in cal.items():
            assert "available" in metrics
            assert "reliability" in metrics
            assert "mean_realized_by_confidence_slope" in metrics
            assert "n_deciles" in metrics
            if metrics["available"]:
                assert isinstance(metrics["reliability"], list)
                assert len(metrics["reliability"]) <= 10
                assert isinstance(metrics["mean_realized_by_confidence_slope"], float)
                assert metrics["n_deciles"] > 0


class TestAnalyzeFeatureImportance:
    """Tests for analyze_feature_importance: tree model importance rank correlation."""

    def test_feature_importance_runs(self):
        """Verify feature importance analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_fi", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        fi = result.feature_importance
        assert "importances" in fi
        assert "rank_correlation" in fi
        assert "top_features" in fi

        assert isinstance(fi["rank_correlation"], float)
        assert -1.0 <= fi["rank_correlation"] <= 1.0
        assert isinstance(fi["top_features"], list)
        assert len(fi["top_features"]) > 0
        assert isinstance(fi["importances"], dict)
        assert len(fi["importances"]) > 0


class TestAnalyzeClassImbalance:
    """Tests for analyze_class_imbalance: class shares & minority recall."""

    def test_class_imbalance_runs(self):
        """Verify class imbalance analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_imb", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        imb = result.class_imbalance
        assert "class_counts" in imb
        assert "shares" in imb
        assert "imbalance_ratio" in imb
        assert "per_phase_distribution" in imb
        assert "minority_recall" in imb

        assert isinstance(imb["imbalance_ratio"], float)
        assert imb["imbalance_ratio"] >= 1.0

        for est, phases in imb["minority_recall"].items():
            assert "train" in phases
            assert "validation" in phases
            assert "test" in phases
            for phase, recall in phases.items():
                assert isinstance(recall, float)
                assert 0.0 <= recall <= 1.0


class TestAnalyzeOverfitting:
    """Tests for analyze_overfitting: train vs test degradation."""

    def test_overfitting_runs(self):
        """Verify overfitting analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_overfit", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        of = result.overfitting
        assert len(of) > 0

        for est, metrics in of.items():
            assert "train_accuracy" in metrics
            assert "test_accuracy" in metrics
            assert "degradation" in metrics
            assert "overfit_flag" in metrics

            assert isinstance(metrics["train_accuracy"], float)
            assert isinstance(metrics["test_accuracy"], float)
            assert isinstance(metrics["degradation"], float)
            assert isinstance(metrics["overfit_flag"], bool)

            expected_deg = metrics["train_accuracy"] - metrics["test_accuracy"]
            assert abs(metrics["degradation"] - expected_deg) < 1e-6


class TestAnalyzeProbabilityDistribution:
    """Tests for analyze_probability_distribution: confidence concentration."""

    def test_prob_dist_runs(self):
        """Verify probability distribution analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_pd", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        pdist = result.probability_distribution
        assert len(pdist) > 0

        for est, metrics in pdist.items():
            assert "mean_max_prob" in metrics
            assert "frac_max_prob_gt_0_9" in metrics
            assert "entropy_mean" in metrics
            assert "entropy_std" in metrics

            assert isinstance(metrics["mean_max_prob"], float)
            assert 0.0 <= metrics["mean_max_prob"] <= 1.0
            assert isinstance(metrics["frac_max_prob_gt_0_9"], float)
            assert 0.0 <= metrics["frac_max_prob_gt_0_9"] <= 1.0
            assert isinstance(metrics["entropy_mean"], float)
            # Allow small negative due to floating point precision
            assert metrics["entropy_mean"] >= -1e-10


class TestAnalyzeConfidenceMonotonicity:
    """Tests for analyze_confidence_monotonicity: confidence vs |realized return|."""

    def test_confidence_monotonicity_runs(self):
        """Verify confidence monotonicity analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_cm", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        cm = result.confidence_monotonicity
        assert len(cm) > 0

        for est, metrics in cm.items():
            assert "spearman" in metrics
            assert "monotonic_flag" in metrics
            assert "n" in metrics

            if metrics["spearman"] is not None:
                assert isinstance(metrics["spearman"], float)
                assert -1.0 <= metrics["spearman"] <= 1.0
            assert isinstance(metrics["monotonic_flag"], bool)
            assert isinstance(metrics["n"], int)
            assert metrics["n"] >= 0


class TestAnalyzePredictionConcentration:
    """Tests for analyze_prediction_concentration: label domination / collapse."""

    def test_prediction_concentration_runs(self):
        """Verify prediction concentration analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_pc", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        pc = result.prediction_concentration
        assert len(pc) > 0

        for est, metrics in pc.items():
            assert "predicted_class_share" in metrics
            assert "dominance" in metrics
            assert "gini" in metrics

            assert isinstance(metrics["predicted_class_share"], dict)
            assert isinstance(metrics["dominance"], float)
            assert 0.0 <= metrics["dominance"] <= 1.0
            assert isinstance(metrics["gini"], float)
            assert 0.0 <= metrics["gini"] <= 1.0


class TestAnalyzeSignificance:
    """Tests for analyze_significance: McNemar vs majority baseline."""

    def test_significance_runs(self):
        """Verify significance analysis returns expected structure."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_sig", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        sig = result.significance
        assert "available" in sig
        assert "per_model" in sig
        assert "best_model" in sig
        assert "best_macro_f1" in sig

        if sig["available"]:
            for model, metrics in sig["per_model"].items():
                assert "mcnemar_chi2" in metrics
                assert "p_value" in metrics
                assert "beats_baseline" in metrics
                assert "b" in metrics
                assert "c" in metrics

                assert isinstance(metrics["mcnemar_chi2"], float)
                assert isinstance(metrics["p_value"], float)
                assert 0.0 <= metrics["p_value"] <= 1.0
                assert isinstance(metrics["beats_baseline"], bool)

        assert isinstance(sig["best_macro_f1"], float)
        assert sig["best_macro_f1"] >= 0.0


class TestRunRobustnessAnalysis:
    """Integration tests for the full orchestrator."""

    def test_full_analysis_runs(self):
        """Verify full robustness analysis completes and returns all sections."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_full", outputs_dir=Path("outputs/ml_experiments/test"))
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        assert hasattr(result, "reproduction")
        assert hasattr(result, "per_symbol")
        assert hasattr(result, "temporal_stability")
        assert hasattr(result, "regimes")
        assert hasattr(result, "walk_forward_stability")
        assert hasattr(result, "calibration")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "class_imbalance")
        assert hasattr(result, "overfitting")
        assert hasattr(result, "probability_distribution")
        assert hasattr(result, "confidence_monotonicity")
        assert hasattr(result, "prediction_concentration")
        assert hasattr(result, "significance")

    def test_full_analysis_is_deterministic(self):
        """Verify two runs with same seed produce identical results."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_det", outputs_dir=Path("outputs/ml_experiments/test"))

        result1 = run_robustness_analysis(long_ohlcv, config=cfg)
        result2 = run_robustness_analysis(long_ohlcv, config=cfg)

        dict1 = result1.to_dict()
        dict2 = result2.to_dict()

        assert dict1 == dict2


class TestSaveRobustnessReport:
    """Tests for save_robustness_report persistence."""

    def test_save_creates_file(self, tmp_path: Path):
        """Verify save_robustness_report creates a valid JSON file."""
        long_ohlcv = make_long_ohlcv_df(symbols=SYMBOLS, n=200, seed=42)
        cfg = ExperimentConfig(run_id="test_save", outputs_dir=tmp_path)
        result = run_robustness_analysis(long_ohlcv, config=cfg)

        out_path = save_robustness_report(result, outputs_dir=tmp_path)

        assert out_path.exists()
        assert out_path.is_dir()
        assert (out_path / "robustness.json").exists()

        with open(out_path / "robustness.json") as f:
            loaded = json.load(f)

        assert loaded["config"]["run_id"] == "test_save"
        assert "reproduction" in loaded
        assert "significance" in loaded
