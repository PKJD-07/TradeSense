# TradeSense Data Pipeline

This document describes the historical market data ingestion pipeline for TradeSense.

## Overview

The data pipeline provides a clean, modular system for fetching, validating, and preprocessing historical OHLCV (Open, High, Low, Close, Volume) market data. It exposes two validation stages — raw (pre-preprocessing) and final (post-preprocessing) — so callers always see accurate diagnostics for the data they actually receive.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Data Provider  │ ──▶ │  Normalization   │ ──▶ │   Validation   │
│  (Yahoo, etc.)  │     │  (Candle model)  │     │  (raw candles) │
└─────────────────┘     └──────────────────┘     └───────┬────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Preprocessing (sort, dedup, missing-value policy)                │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│   Validation   │  final candles ──▶ IngestionResult
│  (final, clean)│                      (pre_validation,
└────────────────┘                       final_validation,
                                         candles, rejected_rows)
```

## Components

### 1. Data Model (`src/data/models.py`)

**Candle**: Represents a single OHLCV data point.

- **Timezone policy:** All internal Candle timestamps are timezone-aware UTC timestamps. Naive timestamps are rejected because their timezone cannot be determined safely. Aware timestamps (from any timezone) are normalized to UTC when stored, so ordering and duplicate detection are deterministic regardless of the source timezone. `datetime.now(timezone.utc)` produces a UTC timestamp directly; a non-UTC aware value (e.g. exchange-local time from a provider) is converted to UTC via `astimezone(timezone.utc)`.
- **Symbol policy:** Symbols are stripped and uppercased at the model boundary. `Candle(symbol="aapl")` stores `"AAPL"`.

```python
from src.data.models import Candle
from datetime import datetime, timezone

candle = Candle(
    symbol="AAPL",           # or "aapl" — normalized to "AAPL"
    timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc),
    open=210.15,
    high=212.30,
    low=209.80,
    close=211.75,
    volume=1523400,
)
```

**CandleCollection**: A collection of candles for a single symbol. Symbol is normalized to uppercase.

```python
from src.data.models import CandleCollection

collection = CandleCollection(symbol="AAPL", candles=[candle1, candle2])
print(len(collection))  # 2
```

### 2. Data Providers (`src/data/providers/`)

**Provider Abstraction:** The `HistoricalDataProvider` protocol defines the interface for fetching data. Any class that exposes a `name` property and a `fetch_historical` method matching the required signature satisfies it — no inheritance is required.

```python
from datetime import date

class MyProvider:
    """Satisfies the HistoricalDataProvider protocol via duck typing."""
    @property
    def name(self) -> str:
        return "My Provider"

    def fetch_historical(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        # Each dict must have: timestamp, open, high, low, close, volume
        # Timestamps MUST be timezone-aware. Never guess a timezone for a naive
        # value — reject it. Candle normalizes aware timestamps to UTC.
        return [
            {
                "symbol": symbol,
                "timestamp": some_aware_datetime,  # timezone-aware required
                "open": float_val,
                "high": float_val,
                "low": float_val,
                "close": float_val,
                "volume": int_val,
            }
        ]
```

**Yahoo Finance Provider:** Free provider using the `yfinance` library. No API key is required.

```python
from src.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()            # auto_adjust=True (default)
raw = provider.fetch_historical("AAPL", start_date, end_date)
```

#### Adjusted vs unadjusted prices

The `auto_adjust` constructor parameter controls whether yfinance restates historical prices:

| Mode | `auto_adjust=True` (default) | `auto_adjust=False` |
|---|---|---|
| **What it does** | OHLC prices are restated so returns are continuous across dividends and stock splits | Actual traded prices on each day |
| **Use for** | Return computation, feature engineering, backtesting — the safe default for quant/ML | Auditing raw market data |
| **Gotcha** | Historical prices no longer match published closing prices on split/dividend dates | Price series contains discontinuities that create phantom returns if used as-is for return computation |
| **Volume** | Not adjusted by yfinance in either mode | Same |

In both modes timestamps are timezone-aware (in the exchange's local timezone as delivered by Yahoo). The `Candle` model normalizes these to UTC when the data is ingested, so the internal representation is always UTC.

### 3. Validation (`src/data/validation.py`)

**Single Candle Validation** (`CandleValidator`):

- Symbol is not empty
- Timestamp is valid (datetime, year >= 1900)
- **All prices are finite and positive** (NaN, +Inf, -Inf rejected; 0 rejected)
- **Volume is finite and non-negative**
- OHLC relationships: `high >= open`, `high >= close`, `low <= open`, `low <= close`, `high >= low`

```python
from src.data.validation import CandleValidator

validator = CandleValidator(strict_mode=False)
result = validator.validate(candle)

if not result.is_valid:
    print(result.errors)     # e.g. ["open price must be a finite number, got nan"]
```

**Dataset Validation** (`DatasetValidator`):

- Duplicate timestamps
- Chronological order
- Multiple symbols in same dataset
- Missing expected trading days (opt-in via `expected_trading_days` callback)

```python
from src.data.validation import DatasetValidator

validator = DatasetValidator(strict_mode=False)
result = validator.validate(candles)

if not result.is_valid:
    print(result.issues)
```

### 4. Preprocessing (`src/data/preprocessing.py`)

**Operations:**

- **Sorting:** Chronological ordering by timestamp
- **Deduplication:** Configurable policy for duplicate timestamps

**`PreprocessingResult` fields:**

| Field | Meaning |
|---|---|
| `candles` | The cleaned candles |
| `original_count` | Number of candles before preprocessing |
| `final_count` | Number after preprocessing |
| `duplicates_removed` | Number removed by the duplicate policy |
| `was_unsorted` | `True` if the preprocessor had to sort the input |
| `removed_indices` | Original indices of removed candles |
| `warnings` | Non-fatal issues (e.g. "Removed 3 duplicate candles") |

**Duplicate Policies:**

| Policy | Behavior |
|--------|----------|
| `KEEP_FIRST` | Keep first occurrence, remove others |
| `KEEP_LAST` | Keep last occurrence, remove others |
| `REMOVE_ALL` | Remove all candles with duplicate timestamps |
| `ERROR` | Raise an error if duplicates found |

```python
from src.data.preprocessing import Preprocessor, DuplicatePolicy

preprocessor = Preprocessor(
    duplicate_policy=DuplicatePolicy.KEEP_FIRST,
    sort=True,
)

result = preprocessor.preprocess(candles)
print(f"Removed {result.duplicates_removed} duplicates")
```

### 5. Pipeline (`src/data/pipeline.py`)

The pipeline orchestrates fetching, normalization, validation, preprocessing, and a second round of validation.

**Two validation stages:**

| Stage | Attribute | What it reflects |
|---|---|---|
| Raw | `result.pre_validation` | Validation of the data exactly as returned by the provider |
| Final | `result.validation_result` / `result.final_validation` | Validation of the post-preprocessing data — this drives `result.is_valid` |

If preprocessing resolves an issue (e.g. an unsorted input is sorted), `is_valid` is `True` — the caller receives clean data and an accurate verdict.

```python
from src.data import DataPipeline
from src.data.providers import YahooFinanceProvider
from datetime import date

# Setup
provider = YahooFinanceProvider()
pipeline = DataPipeline(provider)

# Fetch and process
result = pipeline.fetch_historical(
    symbol="AAPL",
    start_date=date(2026, 1, 1),
    end_date=date(2026, 8, 10),
)

print(result.summary())
print(f"Candles: {result.candle_count}")
print(f"Valid: {result.is_valid}")

# Pre vs final validation — useful for diagnostics
if not result.pre_validation.is_valid and result.final_validation.is_valid:
    print("Raw data had issues that preprocessing resolved.")
```

#### Rejected rows

Provider rows that fail normalization (e.g. a missing `close` column) are **never silently discarded**. They are recorded as `RejectedRow` objects on `IngestionResult.rejected_rows` for full traceability:

```python
for row in result.rejected_rows:
    print(f"Row {row.index} rejected: {row.reason}")
```

#### Error policy

| `error_policy` | When provider returns malformed rows | When final validation fails |
|---|---|---|
| `FAIL_FAST` (default) | Raises `DataProviderError` on the first bad row | Raises `DataQualityError` listing all remaining issues |
| `COLLECT_ALL` | Records bad rows in `rejected_rows`; continues | Returns result with `is_valid=False`; issues on `final_validation` |

## Error Handling

The pipeline uses specific exceptions for different failure modes:

| Exception | When Raised |
|-----------|-------------|
| `DataProviderError` | External API failures, malformed provider responses, or symbol validation errors (raised by the pipeline's normalization step under `FAIL_FAST`) |
| `DataQualityError` | Final data-quality validation fails under `FAIL_FAST` (e.g. invalid OHLC relationships that preprocessing cannot fix, empty dataset) |
| `ValidationError` | Used by `CandleValidator` in `strict_mode=True`; the pipeline itself uses non-strict validation and raises `DataQualityError` directly |
| `ConfigurationError` | Missing API keys, invalid provider configuration |

```python
from src.data.exceptions import DataProviderError, DataQualityError

try:
    result = pipeline.fetch_historical("AAPL", start_date, end_date)
except DataProviderError as e:
    print(f"Provider error: {e}")
except DataQualityError as e:
    for issue in e.issues:
        print(f"  - {issue}")
```

## Quick Start

```python
from datetime import date
from src.data import DataPipeline
from src.data.providers import YahooFinanceProvider

pipeline = DataPipeline(YahooFinanceProvider())

result = pipeline.fetch_historical(
    symbol="AAPL",
    start_date=date(2026, 1, 1),
    end_date=date(2026, 8, 10),
)

if result.is_valid:
    print(f"Fetched {result.candle_count} candles")
    for candle in result.candles[:5]:
        print(f"  {candle.timestamp.date()}: Close {candle.close}")
else:
    print("Validation issues in final data:")
    for issue in result.validation_result.issues:
        print(f"  - {issue}")
```

## Running Tests

```bash
# Run all data layer tests
pytest tests/data/

# Run specific test file
pytest tests/data/test_validation.py

# Run with verbose output
pytest tests/data/ -v
```

## Adding a New Provider

1. Create a new file in `src/data/providers/`.
2. Write a class that satisfies the `HistoricalDataProvider` protocol (duck typing — no base class required).
3. Each dict must have keys: `timestamp` (timezone-aware), `open`, `high`, `low`, `close` (floats), `volume` (int).
4. Register in `src/data/providers/__init__.py`.

```python
# src/data/providers/my_provider.py
from datetime import date, datetime, timezone

class MyProvider:
    @property
    def name(self) -> str:
        return "My Provider"

    def fetch_historical(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        raw_data = self._call_api(symbol, start_date, end_date)
        candles = []
        for row in raw_data:
            timestamp = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                # Never guess a timezone: naive timestamps are rejected.
                raise ValueError(f"Naive timestamp for {row['date']}")
            candles.append(
                {
                    "symbol": row["ticker"],
                    "timestamp": timestamp,  # aware; Candle normalizes to UTC
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": int(row["v"]),
                }
            )
        return candles
```

## Data Flow Summary

1. **Provider** fetches raw data from an external source; the pipeline normalizes the symbol.
2. **Pipeline** normalizes raw dicts to `Candle` objects (uppercase symbols, aware timestamps normalized to UTC). Malformed rows are recorded as `RejectedRow`.
3. **CandleValidator** validates each candle (finite prices, OHLC relationships). `DatasetValidator` validates the collection (duplicates, order). This produces `pre_validation`.
4. **Preprocessor** sorts chronologically and applies the duplicate policy. A second `DatasetValidator` pass produces `final_validation` and drives `is_valid`.
5. **`IngestionResult`** returns everything: `candles`, `pre_validation`, `final_validation`, `preprocessing_result`, `rejected_rows`.

## Design Principles

- **Provider isolation:** External APIs are isolated from the rest of the system.
- **Model independence:** The Candle model doesn't depend on any provider.
- **Validation separation:** Validation is independent from data acquisition.
- **Preprocessing isolation:** Preprocessing doesn't touch ML-specific operations.
- **No silent mutations:** Invalid data is rejected; preprocessing only applies explicit policies.
- **No silent discards:** Malformed provider rows are recorded as rejected, never dropped.
- **Testability:** All components can be tested with mocks, no live API needed.
