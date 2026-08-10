"""
Tests for candle and dataset validation.
"""

from datetime import datetime, timezone

import pytest

from src.data.models import Candle
from src.data.validation import CandleValidator, DatasetValidator, ValidationResult, DatasetValidationResult
from src.data.exceptions import ValidationError, DataQualityError
from tests.data.fixtures import (
    make_valid_candle,
    make_valid_candles,
    make_candle_with_invalid_ohlc,
    make_candle_with_negative_price,
    make_candle_with_negative_volume,
    make_candle_with_missing_symbol,
    make_duplicate_timestamp_candles,
    make_unsorted_candles,
)


class TestCandleValidator:
    """Tests for single-candle validation."""

    def test_valid_candle_passes_validation(self):
        """Test that a valid candle passes validation."""
        validator = CandleValidator(strict_mode=False)
        candle = make_valid_candle()

        result = validator.validate(candle)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_ohlc_relationship(self):
        """Test that invalid OHLC relationship is detected."""
        validator = CandleValidator(strict_mode=False)
        data = make_candle_with_invalid_ohlc()

        candle = Candle(**data)
        result = validator.validate(candle)

        assert not result.is_valid
        assert any("High" in e for e in result.errors)

    def test_negative_price_detected(self):
        """Test that negative prices are detected."""
        validator = CandleValidator(strict_mode=False)

        # Use object.__setattr__ to bypass model validation and test the
        # validator's own checks (model would reject this at construction)
        candle = make_valid_candle()
        object.__setattr__(candle, "open", -210.0)

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("positive" in e.lower() for e in result.errors)

    def test_negative_volume_detected(self):
        """Test that negative volume is detected."""
        validator = CandleValidator(strict_mode=False)

        # Bypass model validation to test validator's own checks
        candle = make_valid_candle()
        object.__setattr__(candle, "volume", -1000000)

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("volume" in e.lower() for e in result.errors)

    def test_missing_symbol_detected(self):
        """Test that empty symbol is detected."""
        validator = CandleValidator(strict_mode=False)
        data = make_candle_with_missing_symbol()

        # Note: Candle model already validates this, but validation catches it too
        # This test validates the validator would catch it if model didn't
        candle = make_valid_candle()
        candle.symbol = ""  # Bypass model validation for this test

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("symbol" in e.lower() for e in result.errors)

    def test_high_less_than_open_detected(self):
        """Test that high < open is detected."""
        validator = CandleValidator(strict_mode=False)
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=212.0,
            high=210.0,  # Invalid: high < open
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("High" in e and "open" in e for e in result.errors)

    def test_low_greater_than_close_detected(self):
        """Test that low > close is detected."""
        validator = CandleValidator(strict_mode=False)
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=210.0,
            high=212.0,
            low=215.0,  # Invalid: low > close
            close=211.0,
            volume=1000000,
        )

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("Low" in e and "close" in e for e in result.errors)

    def test_zero_price_detected(self):
        """Test that zero price is detected as invalid."""
        validator = CandleValidator(strict_mode=False)
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=0.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("open" in e and "positive" in e for e in result.errors)

    def test_strict_mode_raises_exception(self):
        """Test that strict mode raises ValidationError."""
        validator = CandleValidator(strict_mode=True)
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=0.0,  # Invalid
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        with pytest.raises(ValidationError):
            validator.validate(candle)

    def test_validate_batch(self):
        """Test validating multiple candles at once."""
        validator = CandleValidator(strict_mode=False)
        valid_candle = make_valid_candle()
        invalid_candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=0.0,  # Invalid
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        results = validator.validate_batch([valid_candle, invalid_candle])

        assert len(results) == 2
        assert results[0].is_valid
        assert not results[1].is_valid

    @pytest.mark.parametrize(
        "field,value",
        [
            ("open", float("nan")),
            ("high", float("nan")),
            ("low", float("nan")),
            ("close", float("nan")),
            ("open", float("inf")),
            ("high", float("inf")),
            ("low", float("-inf")),
            ("close", float("-inf")),
        ],
    )
    def test_non_finite_price_detected(self, field, value):
        """Test that NaN and ±Inf prices are rejected."""
        validator = CandleValidator(strict_mode=False)
        candle = make_valid_candle()
        object.__setattr__(candle, field, value)

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("finite" in e for e in result.errors)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_volume_detected(self, value):
        """Test that NaN and ±Inf volume are rejected."""
        validator = CandleValidator(strict_mode=False)
        candle = make_valid_candle()
        object.__setattr__(candle, "volume", value)

        result = validator.validate(candle)

        assert not result.is_valid
        assert any("finite" in e for e in result.errors)

    def test_non_finite_strict_mode_raises(self):
        """Test that strict mode raises on non-finite values."""
        validator = CandleValidator(strict_mode=True)
        candle = make_valid_candle()
        object.__setattr__(candle, "open", float("nan"))

        with pytest.raises(ValidationError):
            validator.validate(candle)


class TestDatasetValidator:
    """Tests for dataset-level validation."""

    def test_valid_dataset_passes(self):
        """Test that a valid dataset passes validation."""
        validator = DatasetValidator(strict_mode=False)
        candles = make_valid_candles(count=5)

        result = validator.validate(candles)

        assert result.is_valid
        assert result.total_candles == 5

    def test_duplicate_timestamps_detected(self):
        """Test that duplicate timestamps are detected."""
        validator = DatasetValidator(strict_mode=False)
        candles = make_duplicate_timestamp_candles()

        result = validator.validate(candles)

        assert not result.is_valid
        assert any("duplicate" in issue.lower() for issue in result.issues)

    def test_unsorted_timestamps_detected(self):
        """Test that unsorted timestamps are detected."""
        validator = DatasetValidator(strict_mode=False)
        candles = make_unsorted_candles()

        result = validator.validate(candles)

        assert not result.is_valid
        assert any("chronological" in issue.lower() for issue in result.issues)

    def test_multiple_symbols_detected(self):
        """Test that multiple symbols in a dataset are detected."""
        validator = DatasetValidator(strict_mode=False)
        candles = [
            make_valid_candle(symbol="AAPL"),
            make_valid_candle(symbol="MSFT"),
        ]

        result = validator.validate(candles)

        assert not result.is_valid
        assert any("multiple symbols" in issue.lower() for issue in result.issues)

    def test_empty_dataset(self):
        """Test validation of empty dataset."""
        validator = DatasetValidator(strict_mode=False)

        result = validator.validate([])

        assert result.total_candles == 0
        assert "Empty dataset" in result.issues

    def test_strict_mode_raises_exception(self):
        """Test that strict mode raises DataQualityError."""
        validator = DatasetValidator(strict_mode=True)
        candles = make_unsorted_candles()

        with pytest.raises(DataQualityError):
            validator.validate(candles)

    def test_candle_errors_recorded(self):
        """Test that individual candle errors are recorded."""
        validator = DatasetValidator(strict_mode=False)

        # Create candles with validation issues
        candles = [
            Candle(
                symbol="AAPL",
                timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                open=0.0,  # Invalid
                high=212.0,
                low=209.0,
                close=211.0,
                volume=1000000,
            ),
            make_valid_candle(),
        ]

        result = validator.validate(candles)

        assert not result.is_valid
        assert 0 in result.candle_errors  # First candle has errors

    def test_duplicate_indices_recorded(self):
        """Test that indices of duplicate candles are recorded."""
        validator = DatasetValidator(strict_mode=False)
        candles = make_duplicate_timestamp_candles()

        result = validator.validate(candles)

        # Should have recorded which indices have duplicates
        assert len(result.candle_errors) > 0
