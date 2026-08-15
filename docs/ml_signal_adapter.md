# TradeSense — ML Signal Adapter (V1 Design)

This document describes the **ML Signal Adapter**, the bridge between the ML
prediction layer and the common `TradingSignal` abstraction shared with the
backtester/trading layer.

---

## 1. Purpose

The ML Signal Adapter converts raw ML model predictions into a structured
`TradingSignal` that the backtester can consume **without knowing the signal
came from ML**. The same `TradingSignal` contract is used by rule-based
strategies (RSI, MACD, etc.) so the backtester is strategy-agnostic.

```
ML Model → ML Prediction → ML Signal Adapter → TradingSignal → Backtester
Rule-Based Strategy → TradingSignal → Backtester
```

**Architectural guarantees:**
- The ML package does NOT import the backtester.
- The adapter does NOT call the backtester.
- The adapter does NOT know about positions, cash, orders, execution,
  slippage, transaction costs, portfolio allocation, risk management, or
  equity curves.
- The adapter operates only on the model prediction + information available
  at signal-generation time (after close of session t).
- No future data (realized_return, y_true, future prices) is used to
  generate the signal.

---

## 2. TradingSignal Contract

Defined in `src/ml/signal_types.py`.

```python
class TradingSignal(BaseModel):
    timestamp: datetime          # When signal generated (after close t)
    symbol: str                  # Instrument (e.g., "AAPL")
    action: SignalAction         # BUY / SELL / HOLD
    confidence: float            # [0, 1]
    source: SignalSource         # ML / RULE_BASED / MANUAL
    model_name: Optional[str]    # e.g., "gradient_boosting" (when source=ML)
    probability_down: float      # P(target = -1 | features)  [0, 1]
    probability_neutral: float   # P(target = 0  | features)  [0, 1]
    probability_up: float        # P(target = +1 | features)  [0, 1]
```

**Key points:**
- All three class probabilities are **preserved** — not discarded after
  action selection.
- `confidence` = probability of the predicted class for BUY/SELL;
  for HOLD it is `max(P_down, P_neutral, P_up)`.
- A `TradingSignal` is a **signal representation**, NOT an executed trade.

---

## 3. BUY / SELL / HOLD Rules

The adapter uses **explicit, configurable thresholds**.

### Default thresholds (V1)
| Threshold | Value | Meaning |
|-----------|-------|---------|
| `buy_threshold` | 0.55 | Minimum P(+1) for BUY when y_pred = +1 |
| `sell_threshold` | 0.55 | Minimum P(-1) for SELL when y_pred = -1 |
| `min_confidence` | 0.50 | Absolute floor for confidence |

### Decision logic
```
BUY:
    y_pred == +1
    AND P(+1) >= buy_threshold

SELL:
    y_pred == -1
    AND P(-1) >= sell_threshold

HOLD:
    otherwise
```

### Important behaviors
- **P(+1) = 0.51 does NOT mean BUY** just because it exceeds 0.50.
  The default `buy_threshold = 0.55` requires stronger conviction.
- Thresholds are **configurable** (see §4) and **validated** at construction.
- The neutral class (0) never triggers BUY or SELL; it always yields HOLD.

---

## 4. Configurable Thresholds

```python
from src.ml.signal_adapter import MLSignalAdapter, SignalThresholds

# Custom thresholds
thresholds = SignalThresholds(
    buy_threshold=0.60,
    sell_threshold=0.60,
    min_confidence=0.55,
)
adapter = MLSignalAdapter(
    thresholds=thresholds,
    model_name="gradient_boosting",
)
```

### Validation rules
- All thresholds must be in `[0, 1]`.
- `buy_threshold >= min_confidence`
- `sell_threshold >= min_confidence`
- Invalid configuration raises `ValueError` immediately.

---

## 5. Confidence Semantics

| Action | Confidence = |
|--------|--------------|
| BUY    | P(+1)        |
| SELL   | P(-1)        |
| HOLD   | max(P(-1), P(0), P(+1)) |

This means confidence always reflects the model's **highest conviction**
probability. For directional actions it equals the relevant class probability;
for HOLD it equals the maximum probability (which could be neutral).

---

## 6. Probability Preservation

All three probabilities `P(-1)`, `P(0)`, `P(+1)` are preserved in the
output `TradingSignal`. They are **not discarded** after determining the
action. This enables downstream calibration analysis, threshold sweeps,
and diagnostic evaluation.

---

## 7. Probability Validation Policy

The adapter enforces strict validation on input probabilities:

| Condition | Behavior |
|-----------|----------|
| `NaN` | `ValueError` |
| `inf` / `-inf` | `ValueError` |
| Outside `[0, 1]` | `ValueError` |
| Sum ≠ 1.0 (beyond 1e-6 tolerance) | `ValueError` |
| Missing `y_pred` or invalid class | `ValueError` |
| Missing `symbol` or `timestamp` | `ValueError` |

**No silent fabrication** of probabilities occurs. Invalid inputs fail loudly.

---

## 8. Causality / Leakage Guarantees

The adapter **never** uses:
- `realized_return` (next-session return) — provided only for downstream
  diagnostics, ignored for signal generation.
- `y_true` (test labels).
- Future prices.
- Any information not available at signal-generation time (after close t).

This is enforced by code review (the adapter source has no access to these
fields in its decision logic) and tested in `tests/ml/test_signal_adapter.py`.

---

## 9. Determinism

- Identical inputs → identical `TradingSignal` outputs.
- Timestamps and symbols are preserved exactly.
- Batch conversion preserves input row order.
- No random state or non-deterministic operations.

---

## 10. Usage Example

```python
from src.ml.signal_adapter import DEFAULT_ADAPTER

# Single prediction
signal = DEFAULT_ADAPTER.convert(
    symbol="AAPL",
    timestamp=pd.Timestamp("2024-01-02"),
    y_pred=1,
    prob_down=0.09,
    prob_neutral=0.18,
    prob_up=0.73,
)
# signal.action == SignalAction.BUY
# signal.confidence == 0.73

# Batch from experiment predictions DataFrame
signals = DEFAULT_ADAPTER.convert_batch(predictions_df)
```

The `DEFAULT_ADAPTER` uses V1 defaults (`buy=0.55`, `sell=0.55`,
`min_confidence=0.50`, `model_name=None`).

---

## 11. Integration with Experiment Output

The ML experiment (`src/ml/experiment.py`) produces per-phase prediction
DataFrames with columns:
```
symbol, timestamp, target_direction, y_pred,
y_prob_-1, y_prob_0, y_prob_1, realized_return, phase
```

These are directly compatible with `MLSignalAdapter.convert_batch()`:

```python
from src.ml.experiment import run_experiment, save_experiment
from src.ml.signal_adapter import MLSignalAdapter

result = run_experiment(long_ohlcv)
adapter = MLSignalAdapter(model_name="gradient_boosting")

# Convert test-phase predictions to signals
test_signals = adapter.convert_batch(result.predictions["gradient_boosting"]["test"])
```

---

## 12. Architectural Separation

| Layer | Knows About |
|-------|-------------|
| **ML Signal Adapter** | ML predictions, thresholds, TradingSignal |
| **Backtester** | TradingSignal, execution, risk, portfolio |
| **ML Models** | Features, targets, probabilities |

The ML package has **zero imports** from `backend.app.backtesting` or any
trading/execution module. This is verified by test
`test_adapter_does_not_import_backtester`.

---

## 13. Files

| File | Purpose |
|------|---------|
| `src/ml/signal_types.py` | `TradingSignal`, `SignalAction`, `SignalSource` |
| `src/ml/signal_adapter.py` | `MLSignalAdapter`, `SignalThresholds`, `DEFAULT_ADAPTER` |
| `tests/ml/test_signal_adapter.py` | 46 regression tests |
| `docs/ml_signal_adapter.md` | This document |

---

## 14. What This Is NOT

- **Not a trading strategy.** It only classifies predictions.
- **Not a backtester.** No positions, cash, orders, or equity curves.
- **Not a claim of profitability.** The ML pipeline intentionally found
  weak/no meaningful out-of-sample signal in V1. Thresholds are
  configurable defaults, not tuned for profit.
- **Not a risk manager.** No position sizing, stop-loss, or portfolio logic.

---

## 15. Future Extensions

When the friend's side implements the V1 backtester interface, it will
consume `TradingSignal` objects from any source (ML, rule-based, manual).
The adapter will remain unchanged — only the downstream consumer evolves.