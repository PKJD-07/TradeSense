# TradeSense — ML Robustness Analysis (V1 Diagnostic)

This document describes the **13-step diagnostic suite** for the V1 ML pipeline.
It is a **read-only analysis** of the frozen V1 experiment; it never fits a model
on the test set, never imports the backtester / execution / risk layers, and
never claims profitability. Its purpose is to surface fragility, leakage risk,
and overfitting before any capital is risked.

---

## 1. Quick Start

```python
from src.ml.robustness import run_robustness_analysis, save_robustness_report
from src.ml.experiment import ExperimentConfig
from tests.features.fixtures import make_long_ohlcv_df
from pathlib import Path

ohlcv = make_long_ohlcv_df(symbols=('AAPL', 'MSFT', 'JPM', 'XOM', 'SPY'), n=1200, seed=3)
cfg = ExperimentConfig(run_id='robustness_v1_full', outputs_dir=Path('outputs/ml_experiments'))
report = run_robustness_analysis(ohlcv, config=cfg)
save_robustness_report(report)  # writes outputs/<run_id>/robustness.json
```

Output: `outputs/ml_experiments/<run_id>/robustness.json` (full JSON record).

---

## 2. The 13 Analytical Steps

Each step is a pure function taking the V1 `ExperimentResult` and/or the causal
dataset, returning a structured dict. The orchestrator `run_robustness_analysis`
runs all 13 and returns a `RobustnessReport`.

| # | Function | What it measures | Key outputs |
|---|----------|------------------|-------------|
| 1 | `analyze_reproduction` | Determinism: re-run V1 and compare | `determinism.match`, per-estimator test metrics |
| 2 | `analyze_per_symbol` | Hold-one-symbol-out: train on 4, test on 1 | `per_symbol`, `symbol_consistency` (CV of accuracy) |
| 3 | `analyze_temporal_stability` | Test-region quartiles over time | `quartiles`, `drift` (slope of accuracy) |
| 4 | `analyze_regimes` | SPY trend × volatility buckets | `regimes`, `regime_counts` |
| 5 | `analyze_walk_forward_stability` | Fold-to-fold OOS dispersion | `mean`, `std`, `cv` per estimator |
| 6 | `analyze_calibration` | Validation probability reliability | `reliability` deciles, `mean_realized_by_confidence_slope` |
| 7 | `analyze_feature_importance` | Tree model importance rank correlation | `importances`, `rank_correlation`, `top_features` |
| 8 | `analyze_class_imbalance` | Class shares & minority recall | `imbalance_ratio`, `minority_recall` per estimator/phase |
| 9 | `analyze_overfitting` | Train vs test degradation | `degradation`, `overfit_flag` (deg > 0.15) |
| 10 | `analyze_probability_distribution` | Confidence concentration | `mean_max_prob`, `entropy_mean`, `frac_max_prob_gt_0_9` |
| 11 | `analyze_confidence_monotonicity` | Confidence vs |realized return| rank corr | `spearman`, `monotonic_flag` |
| 12 | `analyze_prediction_concentration` | Label domination / collapse | `dominance`, `gini` of predicted shares |
| 13 | `analyze_significance` | McNemar vs majority baseline | `chi2`, `p_value`, `beats_baseline`, `best_model` |

---

## 3. Step-by-Step Interpretation Guide

### Step 1: Reproduction (Determinism)
- **Pass**: `determinism.match == true`, no diffs.
- **Fail**: Any metric differs beyond `atol=1e-12` → seed leak, nondeterministic op, or data mutation.

### Step 2: Per-Symbol Robustness
- **CV < 0.10** (low): signal generalizes across symbols.
- **CV > 0.20** (high): pooled model is driven by one instrument; inspect `per_symbol` table.
- **Action**: If one symbol dominates, consider symbol-specific models or dropping it.

### Step 3: Temporal Stability
- **Drift near 0**: performance stable across test period.
- **Negative drift (accuracy decreasing)**: model decays; may indicate regime shift or data quality issue.
- **Positive drift**: unusual; inspect for look-ahead or label leakage in later test regions.

### Step 4: Regime Analysis
- Compare `accuracy` / `macro_f1` across `bull_high_vol`, `bull_low_vol`, `bear_high_vol`, `bear_low_vol`.
- **Red flag**: Model only works in one regime (e.g., bull_low_vol) → will fail in production.
- **Action**: Report regime-specific metrics; consider regime-aware ensemble.

### Step 5: Walk-Forward Stability
- **CV < 0.05**: OOS performance consistent across folds.
- **CV > 0.10**: fold-to-fold variance high; model unstable; may be overfit to specific windows.
- **Inspect**: `fold_accuracy` list for catastrophic drops in specific quarters.

### Step 6: Calibration Assessment
- Uses validation confidence deciles (`mean_realized_return` per bin).
- **Slope > 0**: higher confidence → higher realized return (good).
- **Slope ≈ 0**: confidence uninformative.
- **Slope < 0**: perverse (overconfidence hurts).

### Step 7: Feature Importance Stability
- **Rank correlation > 0.7**: RF and GBM agree on what matters.
- **Rank correlation < 0.3**: models use different features → ensemble diversity (good) OR instability (bad).
- **Top features**: should be causal (e.g., `intraday_return`, `relative_return_1d`), not spurious.

### Step 8: Class Imbalance Diagnostics
- **Imbalance ratio > 10**: severe (minority class < 10%).
- **Minority recall ≈ 0** in test: model never learns the rare class.
- **Action**: Try `class_weight="balanced"` on validation; report both.

### Step 9: Overfitting Diagnostics
- **Degradation = train_acc - test_acc**.
- **Flag if degradation > 0.15** (configurable heuristic).
- **GBM often flags**: trees can memorize; `max_depth=3` and `n_estimators=200` are capped but not immune.

### Step 10: Probability Distribution
- **mean_max_prob near 1.0**: model is overconfident (e.g., baselines at 1.0).
- **mean_max_prob near 1/3 ≈ 0.33**: model is uncertain (good for calibration).
- **Entropy**: higher = more uncertain; `logistic_regression` typically ~0.9, GBM ~0.86.

### Step 11: Confidence Monotonicity
- **Spearman > 0.05**: higher confidence correlates with higher |realized return|.
- **Spearman ≤ 0**: confidence uninformative or inversely related.
- **Baselines**: return `None` (constant confidence).

### Step 12: Prediction Concentration
- **Dominance = max predicted class share**.
- **Dominance > 0.8**: model collapses to one class (e.g., Logistic Regression often predicts only +1).
- **Gini of predicted shares**: 0 = uniform, 1 = single class.

### Step 13: Statistical Significance vs Baselines
- **McNemar test** (chi-square approx) on test set disagreements.
- **p < 0.05**: model statistically beats majority baseline.
- **Typical V1 result**: `beats_baseline == false` for all models → no statistical edge over trivial baseline.
- **Best model**: highest test `macro_f1` (often `persistence` for this target).

---

## 4. V1 Baseline Findings (from `robustness_v1_full`)

| Diagnostic | Key Finding | Implication |
|------------|-------------|-------------|
| **Determinism** | ✅ Pass | Pipeline fully reproducible |
| **Per-symbol CV** | 0.03–0.08 | Signal generalizes moderately across symbols |
| **Temporal drift** | -0.017 to +0.004 | Mild negative drift for GBM/RF |
| **Regimes** | Best in `bull_low_vol` | Models fail in `bear_low_vol` |
| **Walk-forward CV** | 0.03–0.07 | Moderate fold-to-fold variance |
| **Calibration slope** | -0.0003 to +0.019 | Logistic Regression slightly positive; GBM negative |
| **Feature rank corr** | 0.42 | RF and GBM disagree on feature importance |
| **Imbalance ratio** | 6.37 | Moderate imbalance; class 0 is ~8% |
| **Minority recall** | Near 0 for most models | Class 0 effectively unlearned |
| **Overfitting** | GBM (0.34), RF (0.16) flagged | Trees overfit; linear model OK |
| **Max prob** | GBM 0.55, LR 0.49 | GBM more confident but not calibrated |
| **Monotonicity** | All < 0.05 | Confidence ≠ realized return |
| **Dominance** | LR 0.83, RF 0.82 | Linear/tree models collapse to bullish |
| **Significance** | No model beats majority (p > 0.05) | **No statistical edge** |

**Bottom line for V1**: The ML models show **no statistically significant advantage** over trivial baselines, exhibit **overfitting** (GBM/RF), **fail to learn the neutral class**, and **lack calibration**. The best test macro-F1 belongs to the **persistence baseline** (0.352). This is the expected V1 outcome: the pipeline is correctly built but the signal is weak/absent — exactly what a robustness analysis should reveal.

---

## 5. Output Schema (`robustness.json`)

```json
{
  "config": { "run_id", "epsilon", "models", "baselines" },
  "reproduction": {
    "run_id", "dataset_rows", "metric_summary", "determinism": { "match", "diffs" }
  },
  "per_symbol": {
    "per_symbol": { "SYM": { "estimator": { "accuracy", "macro_f1", ... } } },
    "symbol_consistency": { "estimator": cv_float },
    "n_symbols"
  },
  "temporal_stability": {
    "quartiles": { "estimator": [ { "period", "n", "accuracy", "macro_f1" } ] },
    "drift": { "estimator": slope_float }
  },
  "regimes": {
    "regimes": { "REGIME": { "estimator": { "accuracy", "macro_f1", "n" } } },
    "regime_counts": { "REGIME": count }
  },
  "walk_forward_stability": {
    "estimator": { "n_folds", "fold_accuracy", "mean", "std", "cv" }
  },
  "calibration": {
    "estimator": { "available", "reliability", "mean_realized_by_confidence_slope", "n_deciles" }
  },
  "feature_importance": {
    "importances": { "model": { "feature": importance } },
    "rank_correlation": float,
    "top_features": [ "feature", ... ]
  },
  "class_imbalance": {
    "class_counts", "shares", "imbalance_ratio", "per_phase_distribution",
    "minority_recall": { "estimator": { "phase": recall } }
  },
  "overfitting": {
    "estimator": { "train_accuracy", "test_accuracy", "degradation", "overfit_flag" }
  },
  "probability_distribution": {
    "estimator": { "mean_max_prob", "frac_max_prob_gt_0_9", "entropy_mean", "entropy_std" }
  },
  "confidence_monotonicity": {
    "estimator": { "spearman", "monotonic_flag", "n" }
  },
  "prediction_concentration": {
    "estimator": { "predicted_class_share", "dominance", "gini" }
  },
  "significance": {
    "available", "per_model": { "mcnemar_chi2", "p_value", "beats_baseline", "b", "c" },
    "best_model", "best_macro_f1"
  }
}
```

---

## 6. Reproducibility Checklist

To reproduce the exact V1 robustness report:

1. **Data**: 5 symbols (AAPL, MSFT, JPM, XOM, SPY) ~4 years of daily OHLCV.
2. **Config**: `ExperimentConfig()` defaults (see `ml_pipeline.md` §13).
3. **Seed**: `SEED = 42` (propagated to all estimators).
4. **Order**: Canonical `(symbol, timestamp)` sort everywhere.
5. **Output**: `outputs/ml_experiments/<run_id>/robustness.json`.

Any deviation in data, config, or library versions will change results — the
report records library versions and config for auditability.

---

## 7. Extending the Analysis

The module is designed for easy extension. To add a 14th step:

```python
# In src/ml/robustness.py
def analyze_my_new_diagnostic(result, long_ohlcv, config) -> dict:
    # ... compute metrics ...
    return {"my_metric": value}

# In run_robustness_analysis:
my_new = analyze_my_new_diagnostic(result, long_ohlcv, config)
return RobustnessReport(..., my_new_diagnostic=my_new, ...)
```

Then add `my_new_diagnostic` to `RobustnessReport` dataclass and `to_dict()`.

---

## 8. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Using test data in training | Reproduction fails, overfit flag | Ensure `run_experiment` is the only entry point; analysis is read-only |
| Ignoring per-symbol CV | Pooled model looks good but fails on new symbol | Always check Step 2 |
| Trusting accuracy alone | Imbalanced classes mask failure | Use balanced accuracy + macro-F1 (Step 8, 9) |
| Assuming calibration | High confidence ≠ high return | Check Step 6 slope; if ≤ 0, probabilities are not actionable |
| Ignoring significance | "Model beats baseline on accuracy" | McNemar test (Step 13) is the correct comparison |

---

## 9. Files

```
src/ml/
├── robustness.py           # 13 analysis functions + orchestrator
└── __init__.py             # exports RobustnessReport, run_robustness_analysis, save_robustness_report

docs/
└── ml_robustness.md        # this file

outputs/ml_experiments/
└── <run_id>/
    ├── experiment.json     # V1 experiment result
    ├── predictions/        # per-estimator CSVs
    └── robustness.json     # this analysis (gitignored)
```

---

## 10. Integration with CI

The robustness analysis can be run in CI as a **diagnostic gate** (not a pass/fail
gate — the signal may genuinely be absent). Example:

```yaml
# .github/workflows/ml-robustness.yml
- name: Run V1 robustness analysis
  run: |
    python -c "
    from src.ml.robustness import run_robustness_analysis
    from src.ml.experiment import ExperimentConfig
    from tests.features.fixtures import make_long_ohlcv_df
    ohlcv = make_long_ohlcv_df(symbols=('AAPL','MSFT','JPM','XOM','SPY'), n=1200, seed=3)
    cfg = ExperimentConfig(run_id='ci_robustness')
    report = run_robustness_analysis(ohlcv, config=cfg)
    # Fail only on determinism or leakage flags
    assert report.reproduction['determinism']['match'], 'Non-deterministic!'
    for name, of in report.overfitting.items():
        if of['overfit_flag'] and name in ('logistic_regression', 'random_forest', 'gradient_boosting'):
            print(f'WARNING: {name} overfit_flag=True (deg={of[\"degradation\"]:.3f})')
    "
```

This ensures the pipeline remains deterministic and flags overfitting without
failing the build on weak signal (which is a research outcome, not a bug).

---

*End of document*