# TradeSense — Feature Engineering

This document describes the causal feature-engineering layer for TradeSense.

## Overview

The feature layer transforms validated OHLCV data into an ML-ready feature matrix. Every feature is **causal** — each value at timestamp `t` uses only information available at or before `close_t`. No feature uses future data.

**This layer does NOT:**
- Train ML models
- Perform feature scaling (deferred to ML pipeline)
- Include target columns in the feature matrix
- Implement trading strategies, backtesting, or risk management

## Causal Requirement

Every feature must satisfy:

```
feature[t] = f(data[<=t])
```

No feature may use information from `t+1` or later. This is enforced by:
1. Feature functions use only trailing windows (`rolling`, `shift`)
2. Leakage regression tests mutate future data and verify earlier features remain unchanged
3. Cross-asset alignment uses exact timestamp matching with LEFT JOIN (no forward-fill)

## Feature Groups

### 1. Return / Price Features (6 features)

| Feature | Formula | Lookback | Information Used |
|---------|---------|----------|------------------|
| `return_1d` | `close[t] / close[t-1] - 1` | 1 day | close[t], close[t-1] |
| `return_5d` | `close[t] / close[t-5] - 1` | 5 days | close[t], close[t-5] |
| `return_10d` | `close[t] / close[t-10] - 1` | 10 days | close[t], close[t-10] |
| `return_20d` | `close[t] / close[t-20] - 1` | 20 days | close[t], close[t-20] |
| `log_return_1d` | `ln(close[t] / close[t-1])` | 1 day | close[t], close[t-1] |
| `intraday_return` | `close[t] / open[t] - 1` | 0 (intra-session) | open[t], close[t] |

**Rationale:** Multi-horizon returns capture different momentum regimes. Log returns standardize for volatility comparisons. Intraday return captures session-specific price action.

### 2. Momentum Features (4 features)

| Feature | Formula | Lookback | Information Used |
|---------|---------|----------|------------------|
| `price_sma_ratio_10` | `close[t] / SMA(close, 10)[t]` | 10 days | close[t-9:t] |
| `price_sma_ratio_20` | `close[t] / SMA(close, 20)[t]` | 20 days | close[t-19:t] |
| `price_ema_ratio_10` | `close[t] / EMA(close, 10)[t]` | 10 days | close[t-9:t] (exp-weighted) |
| `sma_cross_10_20` | `SMA(10)[t] / SMA(20)[t] - 1` | 20 days | close[t-19:t] |

**Rationale:** Price relative to moving averages captures trend strength. SMA/EMA ratio captures short vs long trend alignment. SMA crossover is a classic trend-following signal.

### 3. Volatility Features (5 features)

| Feature | Formula | Lookback | Information Used |
|---------|---------|----------|------------------|
| `volatility_10d` | `std(log_return[t-9:t]) × √252` | 10 days (11 prices) | close[t-10:t] |
| `volatility_20d` | `std(log_return[t-19:t]) × √252` | 20 days (21 prices) | close[t-20:t] |
| `high_low_range` | `(high[t] - low[t]) / close[t]` | 0 (intra-session) | high[t], low[t], close[t] |
| `atr_ratio_14` | `ATR(14)[t] / close[t]` | 14 days | high[t-13:t], low[t-13:t], close[t-14:t] |
| `volatility_ratio` | `volatility_10d[t] / volatility_20d[t]` | 20 days | close[t-20:t] |

**Rationale:** Rolling volatility captures risk regime. High-low range and ATR capture intraday volatility. Volatility ratio detects regime shifts (short vol spiking above long vol).

**Important:** Rolling volatility requires `n` log returns, which requires `n+1` price observations:
- `volatility_10d[t]` requires prices through `t-10` (first non-NaN at index 11)
- `volatility_20d[t]` requires prices through `t-20` (first non-NaN at index 21)

### 4. Volume Features (3 features)

| Feature | Formula | Lookback | Information Used |
|---------|---------|----------|------------------|
| `volume_change_1d` | `volume[t] / volume[t-1] - 1` | 1 day | volume[t], volume[t-1] |
| `relative_volume_10d` | `volume[t] / SMA(volume, 10)[t]` | 10 days | volume[t-9:t] |
| `volume_trend_5d` | `SMA(volume, 5)[t] / SMA(volume, 20)[t] - 1` | 20 days | volume[t-19:t] |

**Rationale:** Volume changes signal conviction. Relative volume detects unusual activity. Volume trend captures sustained interest or withdrawal.

### 5. Market-Context Features (2 features)

| Feature | Formula | Lookback | Information Used |
|---------|---------|----------|------------------|
| `spy_return_1d` | `SPY_close[t] / SPY_close[t-1] - 1` | 1 day | SPY close[t-1:t] |
| `relative_return_1d` | `return_1d_stock - return_1d_spy` | 1 day | stock close[t-1:t], SPY close[t-1:t] |

**Rationale:** Market return provides context for individual stock moves. Relative return isolates stock-specific signal from market beta.

**Total: 20 features**

## NaN / Warm-Up Policy

**Policy: Preserve NaNs, defer handling to ML pipeline.**

1. Rolling features with insufficient lookback produce NaN.
2. No forward-fill, backward-fill, or imputation at the feature layer.
3. Zero denominators produce NaN (never +inf or -inf).
4. Missing SPY data produces NaN market-context features (no interpolation).

**Warm-up period:**
- Maximum lookback is 20 days.
- Rows 0–19 may have NaNs depending on the feature.
- Row 21 onward has all features defined (assuming complete data).

**Why no imputation:**
- Forward-filling uses future information.
- Backward-filling is non-causal.
- Mean/median imputation uses statistics from the entire series.
- The ML pipeline is responsible for handling NaNs after feature/target alignment.

## Multi-Symbol Architecture

### Input Format
Long-form DataFrame with columns: `symbol, timestamp, open, high, low, close, volume`

### Per-Symbol Isolation
- All rolling features computed independently per symbol using `groupby('symbol')`
- No rolling window ever crosses symbol boundaries
- Each symbol may have its own timestamp series (no common calendar assumed)

### Cross-Asset Alignment
- Market-context features use **LEFT JOIN** from stock timestamps onto SPY
- Stock observations are always preserved
- If SPY is missing at a stock timestamp, market-context features are NaN
- No forward-fill or interpolation of SPY values
- Exact timestamp equality required for alignment

### Output Format
Long-form DataFrame with columns: `timestamp, symbol, feature_1, feature_2, ..., feature_20`

## Feature Naming Convention

```
{category}_{descriptor}_{lookback}
```

| Category | Prefix | Examples |
|----------|--------|----------|
| Returns | `return_`, `log_return_` | `return_1d`, `log_return_1d` |
| Momentum | `price_sma_ratio_`, `price_ema_ratio_`, `sma_cross_` | `price_sma_ratio_10`, `sma_cross_10_20` |
| Volatility | `volatility_`, `high_low_range`, `atr_ratio_` | `volatility_10d`, `atr_ratio_14` |
| Volume | `volume_`, `relative_volume_` | `volume_change_1d`, `relative_volume_10d` |
| Market | `spy_`, `relative_` | `spy_return_1d`, `relative_return_1d` |
| Intraday | (no lookback) | `intraday_return`, `high_low_range` |

## Zero-Denominator Handling

All ratio features handle zero denominators safely:

```python
# Returns NaN instead of inf when denominator is zero or NaN
def _safe_ratio(numerator, denominator):
    result = NaN where denominator == 0 or NaN
    result = numerator / denominator elsewhere
    return result
```

Affected features:
- `intraday_return` (open = 0)
- All SMA/EMA ratios (SMA = 0)
- `volume_change_1d` (prior volume = 0)
- `relative_volume_10d` (average volume = 0)
- `volume_trend_5d` (SMA(20) volume = 0)
- `high_low_range` (close = 0)
- `atr_ratio_14` (close = 0)

## Scaling (Deferred to ML Pipeline)

The feature layer outputs **raw, meaningful features**. Scaling is NOT performed here because:

1. Scalers must be fit **only on training data** to avoid leakage
2. The ML pipeline determines the split before fitting scalers
3. Different ML algorithms have different scaling requirements

The feature matrix should be scaled by the ML pipeline using a scaler fit exclusively on the training fold.

## Candidate Features

All features in this layer are **candidate features**. Predictive value must be established through out-of-sample ML evaluation. No feature is claimed to be predictive until demonstrated empirically.

## Leakage Prevention

### Design Principles
1. All rolling windows use trailing data only (`rolling`, `shift`)
2. No centered or forward-looking windows
3. Cross-asset alignment uses exact timestamp matching
4. No imputation using future statistics

### Regression Tests
Five leakage regression tests are implemented in `tests/features/test_leakage.py`:

1. **Future-data mutation**: Mutate data at `t+5`, verify features at `t` unchanged
2. **Rolling-window boundary**: Outlier in one symbol doesn't affect another
3. **Cross-asset alignment**: No forward-fill, missing SPY produces NaN
4. **Target isolation**: No target columns in feature matrix
5. **Timestamp stability**: Shuffled input produces same sorted timestamps

## Architecture

```
src/features/
├── __init__.py           # Public exports
├── price.py              # Return and price features
├── momentum.py           # SMA, EMA, crossover features
├── volatility.py         # Volatility, ATR, range features
├── volume.py             # Volume-based features
├── market_context.py     # SPY-relative cross-asset features
├── builder.py            # Orchestrator combining feature groups
└── validation.py         # Feature-level validation utilities

tests/features/
├── fixtures.py           # Synthetic multi-symbol OHLCV data
├── test_price.py
├── test_momentum.py
├── test_volatility.py
├── test_volume.py
├── test_market_context.py
├── test_builder.py
└── test_leakage.py       # Critical leakage regression tests
```

## Usage

```python
from src.features import build_features, get_feature_names

# Build feature matrix from long-form OHLCV data
features = build_features(ohlcv_df, include_market_context=True)

# Get feature column names
feature_names = get_feature_names(include_market_context=True)
```

## Dependencies

The feature layer uses:
- `pandas` for DataFrame operations
- `numpy` for numerical computations
- Existing `src.analysis` utilities where appropriate

No external API calls. No ML model dependencies.
