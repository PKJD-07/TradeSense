"""
Validation module for OHLCV candles and datasets.

Provides single-candle validation and collection-level validation
to ensure data quality before processing.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from src.data.exceptions import ValidationError, DataQualityError
from src.data.models import Candle


@dataclass
class ValidationResult:
    """
    Result of validating a single candle.

    Attributes:
        is_valid: Whether the candle passed all validations
        errors: List of error messages (empty if valid)
        candle: The candle that was validated
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    candle: Candle | None = None


@dataclass
class DatasetValidationResult:
    """
    Result of validating a collection of candles.

    Attributes:
        is_valid: Whether the dataset passed all validations
        total_candles: Total number of candles checked
        valid_candles: Number of candles that passed validation
        invalid_candles: Number of candles that failed validation
        issues: List of dataset-level issues (duplicates, gaps, etc.)
        candle_errors: Dict mapping candle index to error messages
    """

    is_valid: bool
    total_candles: int = 0
    valid_candles: int = 0
    invalid_candles: int = 0
    issues: list[str] = field(default_factory=list)
    candle_errors: dict[int, list[str]] = field(default_factory=dict)


class CandleValidator:
    """
    Validator for individual OHLCV candles.

    Validates:
    - Symbol is not empty
    - Timestamp is valid
    - All prices are finite and positive (NaN/Inf rejected)
    - Volume is finite and non-negative
    - OHLC relationships (high >= open/close, low <= open/close, high >= low)
    """

    def __init__(
        self,
        strict_mode: bool = True,
    ):
        """
        Initialize the candle validator.

        Args:
            strict_mode: If True, raise ValidationError on invalid data.
                        If False, return ValidationResult with errors.
        """
        self.strict_mode = strict_mode

    def validate(self, candle: Candle) -> ValidationResult:
        """
        Validate a single candle.

        Args:
            candle: The candle to validate

        Returns:
            ValidationResult with is_valid and errors
        """
        errors: list[str] = []

        # Check symbol
        if not candle.symbol or not candle.symbol.strip():
            errors.append("Symbol is empty or contains only whitespace")

        # Check timestamp
        if not isinstance(candle.timestamp, datetime):
            errors.append(f"Invalid timestamp type: {type(candle.timestamp)}")
        elif candle.timestamp.year < 1900:
            errors.append(f"Timestamp year {candle.timestamp.year} is unreasonably old")

        # Check prices are finite and positive
        for price_name in ("open", "high", "low", "close"):
            price = getattr(candle, price_name)
            if not math.isfinite(price):
                errors.append(f"{price_name} price must be a finite number, got {price}")
            elif price <= 0:
                errors.append(f"{price_name} price must be positive, got {price}")

        # Check volume is finite and non-negative
        if not math.isfinite(candle.volume):
            errors.append(f"Volume must be a finite number, got {candle.volume}")
        elif candle.volume < 0:
            errors.append(f"Volume must be non-negative, got {candle.volume}")

        # Check OHLC relationships
        if candle.high < candle.open:
            errors.append(
                f"High ({candle.high}) must be >= open ({candle.open})"
            )
        if candle.high < candle.close:
            errors.append(
                f"High ({candle.high}) must be >= close ({candle.close})"
            )
        if candle.low > candle.open:
            errors.append(
                f"Low ({candle.low}) must be <= open ({candle.open})"
            )
        if candle.low > candle.close:
            errors.append(
                f"Low ({candle.low}) must be <= close ({candle.close})"
            )
        if candle.high < candle.low:
            errors.append(
                f"High ({candle.high}) must be >= low ({candle.low})"
            )

        is_valid = len(errors) == 0

        if not is_valid and self.strict_mode:
            raise ValidationError(
                message="; ".join(errors),
                field="candle",
                value=candle.symbol,
            )

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            candle=candle,
        )

    def validate_batch(self, candles: list[Candle]) -> list[ValidationResult]:
        """
        Validate multiple candles.

        Args:
            candles: List of candles to validate

        Returns:
            List of ValidationResult objects, one per candle
        """
        return [self.validate(candle) for candle in candles]


class DatasetValidator:
    """
    Validator for collections of candles.

    Validates:
    - Duplicate timestamps
    - Chronological order
    - Detectable gaps in data
    - All candles belong to the same symbol
    """

    def __init__(
        self,
        strict_mode: bool = True,
        check_gaps: bool = True,
        expected_trading_days: Callable[[datetime, datetime], list[datetime]] | None = None,
    ):
        """
        Initialize the dataset validator.

        Args:
            strict_mode: If True, raise DataQualityError on issues.
                        If False, return DatasetValidationResult with issues.
            check_gaps: Whether to check for gaps in timestamps.
                        Set to False if you don't have a trading calendar.
            expected_trading_days: Optional function that takes (start, end) and returns
                                  expected trading days. Used for gap detection.
        """
        self.strict_mode = strict_mode
        self.check_gaps = check_gaps
        self.expected_trading_days = expected_trading_days

    def validate(self, candles: list[Candle]) -> DatasetValidationResult:
        """
        Validate a collection of candles.

        Args:
            candles: List of candles to validate

        Returns:
            DatasetValidationResult with validation status and issues
        """
        issues: list[str] = []
        candle_errors: dict[int, list[str]] = {}

        if not candles:
            return DatasetValidationResult(
                is_valid=False,
                total_candles=0,
                valid_candles=0,
                invalid_candles=0,
                issues=["Empty dataset"],
            )

        total_candles = len(candles)

        # Check all candles have the same symbol
        symbols = {c.symbol for c in candles}
        if len(symbols) > 1:
            issues.append(f"Multiple symbols in dataset: {symbols}")

        # Check for duplicate timestamps
        timestamps = [c.timestamp for c in candles]
        seen: set[datetime] = set()
        duplicates: list[datetime] = []

        for ts in timestamps:
            if ts in seen:
                duplicates.append(ts)
            else:
                seen.add(ts)

        if duplicates:
            dup_count = len(duplicates)
            issues.append(f"Found {dup_count} duplicate timestamp(s)")
            # Record which candles have duplicates
            ts_to_indices: dict[datetime, list[int]] = {}
            for i, ts in enumerate(timestamps):
                if ts not in ts_to_indices:
                    ts_to_indices[ts] = []
                ts_to_indices[ts].append(i)

            for ts, indices in ts_to_indices.items():
                if len(indices) > 1:
                    for idx in indices[1:]:  # Skip first occurrence
                        if idx not in candle_errors:
                            candle_errors[idx] = []
                        candle_errors[idx].append(f"Duplicate timestamp: {ts}")

        # Check chronological order
        is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
        if not is_sorted:
            issues.append("Timestamps are not in chronological order")

        # Check for gaps if we have a trading calendar
        if self.check_gaps and self.expected_trading_days and len(timestamps) > 1:
            sorted_timestamps = sorted(timestamps)
            start = min(sorted_timestamps)
            end = max(sorted_timestamps)
            expected = set(self.expected_trading_days(start, end))
            actual = set(sorted_timestamps)
            missing = expected - actual

            if missing:
                # Don't report too many missing days
                missing_list = sorted(missing)[:5]
                missing_str = ", ".join(str(d.date()) for d in missing_list)
                if len(missing) > 5:
                    missing_str += f" (and {len(missing) - 5} more)"
                issues.append(f"Missing {len(missing)} expected trading day(s): {missing_str}")

        # Validate individual candles and collect errors
        candle_validator = CandleValidator(strict_mode=False)
        valid_count = 0
        invalid_count = 0

        for i, candle in enumerate(candles):
            result = candle_validator.validate(candle)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if i not in candle_errors:
                    candle_errors[i] = []
                candle_errors[i].extend(result.errors)

        is_valid = len(issues) == 0 and invalid_count == 0

        if not is_valid and self.strict_mode:
            raise DataQualityError(
                message="Dataset validation failed",
                issues=issues + [
                    f"Candle {idx}: {'; '.join(errs)}"
                    for idx, errs in candle_errors.items()
                ],
            )

        return DatasetValidationResult(
            is_valid=is_valid,
            total_candles=total_candles,
            valid_candles=valid_count,
            invalid_candles=invalid_count,
            issues=issues,
            candle_errors=candle_errors,
        )
