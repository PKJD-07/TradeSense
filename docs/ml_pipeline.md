# TradeSense — ML Pipeline (V1 Design)

This document describes the machine-learning prediction layer for TradeSense. It
is the approved V1 design: a leakage-resistant pipeline that converts the causal
feature matrix and target into temporally separated datasets, train-only
preprocessing, baseline models, candidate ML models, out-of-sample predictions,
and reproducible evaluation results.

Scope: this layer **builds the ML prediction pipeline only**. It does **not**
implement trading strategies, backtesting, transaction costs, slippage, position
sizing, portfolio construction, or risk management. It does **not** claim that
models are profitable or that predictive signal exists merely because
classification metrics exceed a baseline — trading performance is evaluated
later in the backtesting/risk phase.

## 1. Execution / Timestamp Convention (inherited)

Every sample is indexed by a daily session `t`:

| Event | Allowed data |
|---|---|
| **Features** for sample `t` | Information available at or before the close of session `t` |
| **Signal** | Generated after the close of session `t` |
| **Execution** | At the open of session `t+1` — the earliest actionable moment |
| **Targets** for sample `t` | Information strictly after `close_t`, anchored at the `t+1` open |

The primary classification target is `next_session_direction`:

```
y_t = +1  if close_{t+1}/open_{t+1} − 1 >  ε
y_t = −1  if close_{t+1}/open_{t+1} − 1 < −ε
y_t =  0  otherwise (neutral zone)
ε = 0.001 (default, configurable, NOT tuned on the test set)
```

The **3-class target is the actual training target**: `{-1, 0, +1}`. Neutral rows
are kept and fit. A binary `y != 0` evaluation lens is included **as a
diagnostic only**; it is never a silent transformation of the training set.

## 2. Architecture

```
src/ml/
├── __init__.py          # Public exports
├── constants.py         # WARMUP_ROWS=21, DEFAULT_EPSILON, DEFAULT_HORIZON, SEED
├── dataset.py           # MLDataset: alignment, per-symbol targets, NaN accounting
├── preprocessing.py     # Leakage-safe sklearn Pipelines; NaN + scaling policy
├── baselines.py         # MajorityClass, Persistence, PriorProbability
├── models.py            # Model registry: LogReg / RF / GBM with fixed config
├── evaluation.py        # Per-phase metric blocks, probability metrics, financial diagnostics
├── validation.py        # Date-based split, purge gap, walk-forward blocks
└── experiment.py        # ExperimentConfig/Result, run_experiment(), run_walk_forward()
```

`src/ml/` is the **only** new code. The existing layers
(`src/data/`, `src/analysis/`, `src/features/`) are consumed as read-only
dependencies.

## 3. Dataset alignment (`dataset.py`)

One row = one `(symbol, timestamp=t)` session. The builder produces a single
long-form panel:

```
MLDataset:
  df                # long-form, sorted by (symbol, timestamp)
                    # columns: timestamp, symbol, <20 features>, target
  feature_columns   # the 20 feature names from src.features
  target_column     # "target_direction"
  metadata_columns  # ["timestamp", "symbol"]
  report            # DatasetReport
```

Construction order:

1. `src.features.build_features(long_ohlcv)` → features panel. Causal guarantees
   are covered by `tests/features/test_leakage.py`.
2. **Targets are computed per-symbol, never on the pooled frame.** The target
   functions use `.shift(-1)` internally; applied to a long-form multi-symbol
   frame they would cross symbol boundaries at the last row of each symbol.
   `dataset.py` applies them with `groupby("symbol")` and asserts each symbol's
   target series aligns to its feature rows on `timestamp`.
3. Features and per-symbol targets are index-aligned into one panel.
4. **Warm-up drop:** the first `WARMUP_ROWS = 21` rows of every symbol are
   dropped (the true max lookback — `volatility_20d` is first valid at row
   index 21). This is a fixed structural property applied pre-split; each drop
   decision uses only information through row `t`, so it is causally safe.
5. **Remaining-NaN drop:** rows with any feature NaN are dropped and recorded.
   A guard fails loudly if any feature's post-warmup NaN fraction exceeds
   `max_allowed_nan_fraction` (default 2%).
6. Sanity gates: no target-like column in the feature matrix (reuses
   `src.features.validation.check_no_target_leakage`), finite targets on the
   surviving interior, timezone-aware UTC timestamps, canonical ordering.

Alignment invariants (each has a regression test):

- `X_t` uses only data through `close_t`.
- `y_t = f(session t+1 only)`.
- `timestamp` and `symbol` are traceable on every surviving row.
- The target is never included as a feature.
- The neutral class is retained.

## 4. Splitting (`validation.py`)

### 4.1 Date-based pooled-panel split

`src/analysis/split.py` provides row-count-based, single-index helpers. For a
pooled multi-symbol panel these would assign different calendar windows to
symbols with unequal histories. `src/ml/validation.py` therefore implements a
**date-based** split:

- `train_end` = 70th percentile date, `val_end` = 85th percentile date of the
  panel's sorted unique session dates.
- Both dates are **materialized into the config** and recorded, so appending new
  data never silently redraws the boundaries.
- Assignment (identical for every symbol):
  - `train = {timestamp <= train_end}`
  - `val   = {train_end < timestamp <= val_end}`
  - `test  = {timestamp > val_end}`

The test region is **never touched** until the frozen-config final evaluation.

### 4.2 Purge gap

The primary classification target uses exactly one future session per label
(`y_t = f(session t+1)`), so labels of adjacent rows are disjoint — there is
**zero intrinsic label overlap**. `purge_gap = 1` is therefore a **conservative
execution-boundary buffer** (clean separation of "signal generated after
close_T" from "first executable validation row"), not a mathematically required
label-overlap purge.

For N-period targets (e.g. `forward_return(n=5)`, whose labels span sessions
`t+1..t+n`), the required purge is `gap = n − 1` (first clean test row at index
`k+n`). The config stores `purge_gap`; if an N-period target is ever used, the
horizon-based formula applies and is documented in the result.

### 4.3 Walk-forward validation

Walk-forward runs over **train ∪ val only** (everything `<= val_end`); the
untouched test region is excluded from every fold.

- **Expanding** training window (grows with each fold), requiring a minimum
  warm start `min_train_rows = 504` sessions (~2 years).
- Test block = **63 sessions (one quarter)**, `step = 63` (non-overlapping test
  blocks).
- `gap = purge_gap` dropped between each train block end and test block start.
- **Preprocessing is refit on each fold's training block only.**
- Folds are a deterministic function of the canonical sorted panel and config.

Why temporal ordering matters: label overlap (N-period targets), feature
windows, non-stationarity / regime drift, and honest model selection all require
that validation simulate deployment — the model only ever sees the past.

## 5. Preprocessing (`preprocessing.py`)

Policy:

1. **Warm-up rows** dropped at dataset construction (§3).
2. **Remaining feature NaNs:** drop with explicit accounting; **no imputer by
   default**. An optional train-fitted `SimpleImputer` exists but is off by
   default — missing SPY is a data gap, not missing-at-random. No forward-fill,
   backward-fill, or full-series statistics.
3. **Scaling:** `StandardScaler` for linear models only (train-fitted). Tree
   models are split-based and scale-invariant, so **no scaler** for RF/GBM.
4. All fitted parameters on the feature matrix live in an sklearn `Pipeline`
   learned only through `.fit(X_train, y_train)`. `src/features/` never scales.

## 6. Baselines (`baselines.py`)

All baselines implement `fit(X, y)` / `predict(X)` / `predict_proba(X)`.

1. **MajorityClass:** predicts the most frequent class from **training labels
   only**. No future information.
2. **Persistence:** predicts `y_hat_t = sign(OC_t)` thresholded by the same
   epsilon, where `OC_t = close_t/open_t − 1` is session `t`'s own open-to-close
   move ("tomorrow continues today's direction"). Uses only data available after
   `close_t` — causally valid. This is the `intraday_return` feature; the
   baseline computes it from the panel, it is not added to `X`.
3. **PriorProbability:** predicts training-label priors as probabilities. A
   calibration reference point.

## 7. Candidate models (`models.py`)

Fixed, documented hyperparameters. **No hyperparameter tuning in V1.** No
XGBoost/LightGBM.

| Model | Configuration | Scaling | Notes |
|---|---|---|---|
| Logistic Regression | `C=1.0` | StandardScaler | Linear interpretable baseline; well-behaved probabilities |
| Random Forest | `max_depth=6`, `min_samples_leaf=20`, `n_estimators=100`, `random_state=SEED` | none | Nonlinear; depth/leaf bounds limit overfit |
| Gradient Boosting | `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `early_stopping=False`, `random_state=SEED` | none | `early_stopping=False` because sklearn's internal early-stopping split is random and violates temporal discipline |

## 8. Evaluation (`evaluation.py`)

Per **phase** (`train`, `validation`, `test`) — never conflated:

- Accuracy **and Balanced Accuracy**.
- Per-class Precision / Recall / F1 and **macro** Precision / Recall / F1.
- Confusion matrix (3×3).
- ROC-AUC where mathematically appropriate: macro one-vs-rest AUC for the
  3-class problem (secondary diagnostic); standard binary ROC-AUC for the
  `y != 0` lens.
- Class distribution per phase.
- Probability metrics: multi-class Log Loss, multi-class Brier, per-class
  calibration (validation only).

Financial-interpretation diagnostics (clearly labeled **not** strategy returns):

- Mean realized next-session return conditional on predicted class.
- Mean return of predicted-bullish and predicted-bearish subsets; long−short
  proxy spread.
- Confidence vs realized return (probability deciles vs mean realized return).

## 9. Multi-symbol strategy

**Global pooled model** across all symbols. `symbol` is a traceable metadata
column, **not** a default feature (no one-hot by default — the model would latch
onto symbol ID as a level/regime proxy). Cross-symbol leakage is structurally
impossible: features are per-symbol causal and the only cross-asset feature is
SPY's past return (LEFT-JOINed at exact timestamps). Per-symbol evaluation and
symbol-context variants are validation diagnostics only.

## 10. Class imbalance

- Report class distribution per phase.
- Model-selection metrics: **balanced accuracy** and **macro-F1**.
- Compare unweighted vs `class_weight="balanced"` (LogReg/RF) or per-class
  `sample_weight` (GBM) on validation; V1 default is unweighted.
- **No SMOTE / oversampling** — synthetic neighbors interpolate across time,
  break autocorrelation structure, and can fabricate signal.
- Threshold analysis is deferred to the strategy layer; V1 uses the argmax
  decision rule and reports calibration.

## 11. Reproducibility (`experiment.py`)

- One `SEED` (42) in `ExperimentConfig`, passed as `random_state` to every
  estimator.
- Canonical `(symbol, timestamp)` ordering everywhere.
- `ExperimentResult` records: dataset date range per symbol, feature names,
  target definition (epsilon), split dates, purge gap, walk-forward geometry,
  NaN policy + dropped rows, model/baseline configs, seeds, library versions,
  per-phase metrics, per-phase prediction CSVs
  (`symbol, timestamp, y_true, y_pred, y_prob_*, phase`).
- Outputs (JSON + CSVs) go to a gitignored `outputs/` directory. No
  experiment-tracking system in V1.

## 12. Leakage risks (each maps to a regression test)

1. Target in features.
2. Imputer/scaler fitted on validation/test.
3. Future observations affecting earlier feature rows.
4. Target values present in X.
5. Test data influencing model fitting.
6. Chronological ordering not preserved.
7. Walk-forward training windows containing future observations.
8. Purge gaps not respected.
9. Cross-symbol leakage (target shift crossing a symbol boundary).
10. Evaluation metrics using the wrong predictions/targets.

These are implemented as regression tests in `tests/ml/test_leakage.py`.

## 13. V1 configuration summary

| Concern | V1 choice |
|---|---|
| Target | `next_session_direction`, `ε=0.001`, **3 classes retained** |
| Dataset | Pooled 5-symbol panel; 20 causal features; per-symbol targets; warm-up drop 21 rows/symbol; drop remaining feature NaNs with reporting |
| Split | Date-based; boundaries at 70th/85th percentile of sorted session dates, materialized + recorded; `purge_gap=1` |
| Walk-forward | Expanding; test block 63 sessions; step 63; min train 504; gap 1; over train∪val only; preprocessing refit per fold |
| NaN policy | Drop warm-up + drop remaining; no imputer by default (optional train-fitted `SimpleImputer`); 2% NaN guard |
| Scaling | `StandardScaler` for LogReg only; no scaler for RF/GBM; all transforms train-fitted pipelines |
| Baselines | MajorityClass, Persistence (session-t OC direction), PriorProbability |
| Models | `LogisticRegression(C=1.0)`; `RandomForestClassifier(max_depth=6, min_samples_leaf=20)`; `GradientBoostingClassifier(n_estimators=200, max_depth=3, early_stopping=False)`; fixed seeds; no tuning |
| Selection metrics | Balanced accuracy + macro-F1 on validation; accuracy only alongside |
| Multi-symbol | Global pooled model; symbol as metadata; per-symbol eval as diagnostic; no one-hot |
| Imbalance | Report class shares; unweighted vs balanced compared on validation; no SMOTE |
| Reproducibility | Fixed SEED; canonical ordering; full config recorded; JSON + per-phase CSV to gitignored `outputs/` |
| Discipline | Test set touched once, only at frozen-config final evaluation |
