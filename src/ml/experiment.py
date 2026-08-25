"""Experiment orchestration: reproduce-and-run the whole V1 pipeline.

``run_experiment`` wires dataset construction -> date-based split -> static
train/val/test evaluation -> expanding walk-forward validation over
``train ∪ val``, for every baseline and candidate model, and records everything
into an :class:`ExperimentResult`. ``save_experiment`` writes the JSON summary
plus per-phase prediction CSVs to a gitignored ``outputs/`` directory
(docs/ml_pipeline.md, §11).

Discipline (all enforced by design, each with a regression test):
    - One ``SEED`` in :class:`ExperimentConfig`, passed as ``random_state`` to
      every estimator.
    - Canonical ``(symbol, timestamp)`` ordering everywhere.
    - Preprocessing is train-fitted only (models fit through
      :func:`src.ml.models.make_model_pipeline`).
    - The final test region is touched exactly once, only for the frozen-config
      evaluation — it never enters a walk-forward fold.
    - Baselines run on RAW features (persistence's epsilon threshold is defined
      on the raw OC move); models scale internally.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sklearn

from src.ml.baselines import BASELINE_NAMES, build_baseline
from src.ml.constants import (
    DEFAULT_EPSILON,
    DEFAULT_MAX_NAN_FRACTION,
    SEED,
    TARGET_COLUMN,
    WARMUP_ROWS,
)
from src.ml.dataset import build_ml_dataset
from src.ml.evaluation import (
    LABELS,
    evaluate_phase,
    next_session_oc_return,
)
from src.ml.models import MODEL_NAMES, make_model_pipeline
from src.ml.validation import split_by_date, walk_forward_folds

#: Default gitignored outputs directory.
DEFAULT_OUTPUTS_DIR = Path("outputs") / "ml_experiments"

#: Default estimator sets.
DEFAULT_BASELINES = BASELINE_NAMES
DEFAULT_MODELS = MODEL_NAMES

#: Phase name for the walk-forward OOS block in results.
WALK_FORWARD_PHASE = "walk_forward"


def _library_versions() -> dict:
    import numpy as _np
    import pandas as _pd

    return {
        "python": platform.python_version(),
        "numpy": _np.__version__,
        "pandas": _pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


@dataclass
class ExperimentConfig:
    """Every knob of a V1 experiment.

    All values are frozen V1 defaults unless explicitly overridden. The
    materialized split boundaries are recorded in the result so appending data
    never silently redraws them.

    Attributes:
        seed: Global reproducibility seed (``random_state`` for every model).
        epsilon: Target epsilon (neutral-zone half-width), NOT tuned on test.
        include_market_context: Include SPY market-context features.
        warmup_rows: Structural warm-up rows dropped per symbol.
        max_nan_fraction: Guard on post-warm-up per-feature NaN fraction.
        train_fraction / val_fraction: Split percentiles (70th / 85th).
        purge_gap: Sessions dropped between train block end and test block
            start in the walk-forward (execution-boundary buffer).
        test_block / step / min_train_rows: Walk-forward geometry
            (63 / 63 / 504).
        models: Candidate model names (fixed config, no tuning).
        baselines: Baseline names.
        compute_calibration: Per-class calibration on validation only.
        run_id: Identifier stamped into outputs (defaults to a UTC timestamp).
        outputs_dir: Directory under which ``<run_id>/`` outputs are written.
    """

    seed: int = SEED
    epsilon: float = DEFAULT_EPSILON
    include_market_context: bool = True
    warmup_rows: int = WARMUP_ROWS
    max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    purge_gap: int = 1
    test_block: int = 63
    step: int = 63
    min_train_rows: int = 504

    models: tuple[str, ...] = DEFAULT_MODELS
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    compute_calibration: bool = False

    run_id: str = field(
        default_factory=lambda: time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    )
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR


@dataclass
class ExperimentResult:
    """Full record of one experiment run.

    Attributes:
        config: The (possibly defaulted) config used.
        dataset: Dataset metadata (date ranges, features, drops, classes).
        split: Materialized split boundaries + phase sizes + walk-forward
            geometry.
        library_versions: Python / NumPy / pandas / scikit-learn versions.
        estimator_results: ``{estimator_name: {...}}`` with ``kind``,
            ``phase_metrics`` (per phase), and ``walk_forward`` (OOS summary or
            ``None``).
        predictions: ``{estimator_name: {phase: DataFrame}}`` of
            ``symbol, timestamp, y_true, y_pred, y_prob_*, phase`` rows
            (walk-forward OOS under ``WALK_FORWARD_PHASE``).
    """

    config: ExperimentConfig
    dataset: dict
    split: dict
    library_versions: dict
    estimator_results: dict
    predictions: dict

    def to_dict(self) -> dict:
        """JSON-serializable summary (predictions are written as CSVs)."""
        config_dict = asdict(self.config)
        config_dict["outputs_dir"] = str(config_dict["outputs_dir"])
        return {
            "run_id": self.config.run_id,
            "config": config_dict,
            "dataset": self.dataset,
            "split": self.split,
            "library_versions": self.library_versions,
            "estimators": self.estimator_results,
        }


def _new_estimator(name: str, config: ExperimentConfig):
    """Fresh estimator instance for a baseline or model name."""
    if name in config.baselines:
        return build_baseline(name, epsilon=config.epsilon)
    if name in config.models:
        return make_model_pipeline(name)
    raise ValueError(f"'{name}' is neither a configured baseline nor a model")


def _prediction_frame(
    meta: pd.DataFrame,
    y_true,
    y_pred,
    y_prob,
    phase: str,
    realized=None,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    """One (symbol, timestamp)-traceable prediction block."""
    frame = pd.DataFrame(
        {
            "symbol": meta["symbol"].to_numpy(),
            "timestamp": meta["timestamp"].to_numpy(),
            TARGET_COLUMN: np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )
    if y_prob is not None:
        for i, c in enumerate(LABELS):
            frame[f"y_prob_{int(c)}"] = np.asarray(y_prob)[:, i]
    if realized is not None:
        frame["realized_return"] = np.asarray(realized)
    frame["phase"] = phase
    if fold is not None:
        frame["fold"] = fold
    return frame


def _run_walk_forward(
    name: str,
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    realized: np.ndarray,
    meta: pd.DataFrame,
    folds,
) -> tuple[Optional[dict], Optional[pd.DataFrame]]:
    """Expanding walk-forward over ``train ∪ val`` for one estimator.

    Each fold fits a FRESH estimator on that fold's train block only and
    predicts its test block. Returns ``(summary, oos_frame)``; ``(None, None)``
    when there are no folds.
    """
    if not folds:
        return None, None
    blocks = []
    for f in folds:
        est = _new_estimator(name, config)
        est.fit(X.iloc[f.train_index], y[f.train_index])
        y_pred = est.predict(X.iloc[f.test_index])
        y_prob = est.predict_proba(X.iloc[f.test_index])
        blocks.append(
            _prediction_frame(
                meta.iloc[f.test_index],
                y[f.test_index],
                y_pred,
                y_prob,
                phase=WALK_FORWARD_PHASE,
                realized=realized[f.test_index],
                fold=f.fold,
            )
        )
    oos = pd.concat(blocks, ignore_index=True)
    metrics = evaluate_phase(
        oos[TARGET_COLUMN].to_numpy(),
        oos["y_pred"].to_numpy(),
        phase=WALK_FORWARD_PHASE,
        y_prob=oos[[f"y_prob_{int(c)}" for c in LABELS]].to_numpy(),
        realized_return=oos["realized_return"].to_numpy(),
    )
    summary = {
        "n_folds": int(len(folds)),
        "n_oos_rows": int(len(oos)),
        "oos_metrics": metrics.to_dict(),
    }
    return summary, oos


def run_experiment(
    long_ohlcv: pd.DataFrame,
    config: Optional[ExperimentConfig] = None,
) -> ExperimentResult:
    """Run the full V1 pipeline on long-form OHLCV data.

    Args:
        long_ohlcv: DataFrame with [symbol, timestamp, open, high, low, close,
            volume].
        config: Experiment configuration (defaults to the frozen V1 config).

    Returns:
        An :class:`ExperimentResult` with per-estimator metrics and prediction
        frames. No files are written here; call :func:`save_experiment`.

    Raises:
        ValueError: If the dataset build or split fails (see the respective
            modules), or an estimator name is not recognized.
    """
    config = config or ExperimentConfig()

    # 1. Dataset (leakage-safe construction; warm-up + NaN accounting).
    ds = build_ml_dataset(
        long_ohlcv,
        epsilon=config.epsilon,
        include_market_context=config.include_market_context,
        warmup_rows=config.warmup_rows,
        max_nan_fraction=config.max_nan_fraction,
    )

    # Realized next-session returns for the financial-interpretation
    # diagnostics (NOT strategy returns). Left merge preserves panel row order.
    realized = (
        ds.df.merge(
            next_session_oc_return(long_ohlcv),
            on=["symbol", "timestamp"],
            how="left",
        )["realized_return"].to_numpy(dtype=float)
    )

    # 2. Date-based split (boundaries materialized in the config/result).
    split = split_by_date(
        ds.timestamp.to_numpy(),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        purge_gap=config.purge_gap,
    )

    X, y = ds.X, ds.y.to_numpy()
    meta = ds.df[["symbol", "timestamp"]]

    phases = {
        "train": (split.train_index, "train"),
        "validation": (split.val_index, "validation"),
        "test": (split.test_index, "test"),
    }

    # 3. Walk-forward universe = train ∪ val only (test excluded).
    universe_idx = np.concatenate([split.train_index, split.val_index])
    folds = walk_forward_folds(
        ds.timestamp.to_numpy()[universe_idx],
        purge_gap=config.purge_gap,
        test_block=config.test_block,
        step=config.step,
        min_train_rows=config.min_train_rows,
    )

    # 4. Per-estimator evaluation.
    estimator_results: dict = {}
    predictions: dict = {}
    for name in (*config.baselines, *config.models):
        est = _new_estimator(name, config)
        est.fit(X.iloc[split.train_index], y[split.train_index])

        phase_metrics: dict = {}
        pred_blocks: dict = {}
        for _, (idx, phase_name) in phases.items():
            y_pred = est.predict(X.iloc[idx])
            y_prob = est.predict_proba(X.iloc[idx])
            result = evaluate_phase(
                y[idx],
                y_pred,
                phase=phase_name,
                y_prob=y_prob,
                realized_return=realized[idx],
                compute_calibration=(
                    phase_name == "validation" and config.compute_calibration
                ),
            )
            phase_metrics[phase_name] = result.to_dict()
            pred_blocks[phase_name] = _prediction_frame(
                meta.iloc[idx],
                y[idx],
                y_pred,
                y_prob,
                phase=phase_name,
                realized=realized[idx],
            )

        wf, oos_frame = _run_walk_forward(
            name,
            config,
            X.iloc[universe_idx],
            y[universe_idx],
            realized[universe_idx],
            meta.iloc[universe_idx],
            folds,
        )
        if oos_frame is not None:
            pred_blocks[WALK_FORWARD_PHASE] = oos_frame

        estimator_results[name] = {
            "kind": "baseline" if name in config.baselines else "model",
            "phase_metrics": phase_metrics,
            "walk_forward": wf,
        }
        predictions[name] = pred_blocks

    dataset_record = {
        "symbols": ds.report.symbols,
        "date_range_by_symbol": {
            s: [str(lo), str(hi)]
            for s, (lo, hi) in ds.report.date_range_by_symbol.items()
        },
        "feature_names": ds.report.feature_names,
        "target": {"column": TARGET_COLUMN, "epsilon": config.epsilon},
        "warmup_rows": ds.report.warmup_rows,
        "max_nan_fraction": ds.report.max_nan_fraction,
        "dropped": {
            "warmup": ds.report.dropped_warmup,
            "no_target": ds.report.dropped_no_target,
            "nan_features": ds.report.dropped_nan_features,
        },
        "class_counts": {str(k): v for k, v in ds.report.class_counts.items()},
        "final_rows": ds.report.final_rows,
    }

    split_record = {
        "train_end": str(split.config.train_end),
        "val_end": str(split.config.val_end),
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "purge_gap": config.purge_gap,
        "n_train_rows": int(len(split.train_index)),
        "n_val_rows": int(len(split.val_index)),
        "n_test_rows": int(len(split.test_index)),
        "walk_forward": {
            "test_block": config.test_block,
            "step": config.step,
            "min_train_rows": config.min_train_rows,
            "n_folds": len(folds),
        },
    }

    return ExperimentResult(
        config=config,
        dataset=dataset_record,
        split=split_record,
        library_versions=_library_versions(),
        estimator_results=estimator_results,
        predictions=predictions,
    )


def save_experiment(result: ExperimentResult, outputs_dir=None) -> Path:
    """Write the JSON summary + per-phase prediction CSVs to disk.

    Layout: ``<outputs_dir>/<run_id>/experiment.json`` and
    ``<outputs_dir>/<run_id>/predictions/<estimator>_<phase>.csv``.

    Args:
        result: The experiment result to persist.
        outputs_dir: Override for ``result.config.outputs_dir``.

    Returns:
        The run directory that was written.
    """
    outputs_dir = Path(outputs_dir or result.config.outputs_dir)
    run_dir = outputs_dir / result.config.run_id
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    import json

    (run_dir / "experiment.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    # Per-phase prediction CSVs
    for name, phase_frames in result.predictions.items():
        for phase, frame in phase_frames.items():
            frame.to_csv(pred_dir / f"{name}_{phase}.csv", index=False)
    return run_dir
