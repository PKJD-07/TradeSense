"""
TradeSense ML prediction layer.

Converts the causal feature matrix and target into temporally separated
datasets, train-only preprocessing, baseline models, candidate ML models,
out-of-sample predictions, and reproducible evaluation results.

This layer does NOT:
- implement trading strategies, backtesting, transaction costs, slippage,
  position sizing, portfolio construction, or risk management
- tune hyperparameters on the final test set
- claim profitability or predictive signal

Modules:
    dataset         MLDataset construction: alignment, per-symbol targets,
                    warm-up / NaN accounting
    preprocessing   leakage-safe sklearn Pipelines (train-fitted only)
    baselines       majority-class, persistence, prior-probability baselines
    models          model registry: Logistic Regression, Random Forest, GBM
    evaluation      per-phase metric blocks, probability metrics, financial
                    interpretation diagnostics
    validation      date-based split, purge gap, expanding walk-forward
    experiment      ExperimentConfig/Result, run_experiment()
"""

from src.ml.constants import (
    WARMUP_ROWS,
    DEFAULT_EPSILON,
    DEFAULT_HORIZON,
    SEED,
    DEFAULT_MAX_NAN_FRACTION,
    TARGET_COLUMN,
)
from src.ml.dataset import build_ml_dataset, MLDataset, DatasetReport
from src.ml.baselines import (
    MajorityClass,
    Persistence,
    PriorProbability,
    build_baseline,
    BASELINE_NAMES,
    INTRADAY_RETURN,
)
from src.ml.models import (
    build_model,
    make_model_pipeline,
    model_family,
    MODEL_NAMES,
    MODEL_CONFIGS,
    MODEL_FAMILY,
)
from src.ml.evaluation import evaluate_phase, PhaseResult
from src.ml.validation import split_by_date, walk_forward_folds
from src.ml.experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    save_experiment,
)

__all__ = [
    "WARMUP_ROWS",
    "DEFAULT_EPSILON",
    "DEFAULT_HORIZON",
    "SEED",
    "DEFAULT_MAX_NAN_FRACTION",
    "TARGET_COLUMN",
    "build_ml_dataset",
    "MLDataset",
    "DatasetReport",
    "MajorityClass",
    "Persistence",
    "PriorProbability",
    "build_baseline",
    "BASELINE_NAMES",
    "INTRADAY_RETURN",
    "build_model",
    "make_model_pipeline",
    "model_family",
    "MODEL_NAMES",
    "MODEL_CONFIGS",
    "MODEL_FAMILY",
    "evaluate_phase",
    "PhaseResult",
    "split_by_date",
    "walk_forward_folds",
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
    "save_experiment",
]
