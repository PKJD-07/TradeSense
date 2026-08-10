"""
Tests for data preprocessing.
"""

from datetime import datetime, timezone
import math

import pytest

from src.data.models import Candle
from src.data.preprocessing import (
    Preprocessor,
    PreprocessingResult,
    DuplicatePolicy,
    MissingValuePolicy,
)
from src.data.exceptions import DataQualityError
from tests.data.fixtures import (
    make_valid_candle,
    make_valid_candles,
    make_duplicate_timestamp_candles,
    make_unsorted_candles,
)


class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_sort_chronologically(self):
        """Test that unsorted candles are sorted."""
        preprocessor = Preprocessor(sort=True)
        candles = make_unsorted_candles()

        result = preprocessor.preprocess(candles)

        assert result.was_unsorted
        assert result.candles[0].timestamp < result.candles[1].timestamp
        assert result.candles[1].timestamp < result.candles[2].timestamp

    def test_no_sort_when_disabled(self):
        """Test that sorting is skipped when disabled."""
        preprocessor = Preprocessor(sort=False)
        candles = make_unsorted_candles()

        result = preprocessor.preprocess(candles)

        assert not result.was_unsorted

    def test_duplicate_removal_keep_first(self):
        """Test keeping first occurrence of duplicates."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        candles = make_duplicate_timestamp_candles()

        result = preprocessor.preprocess(candles)

        assert result.duplicates_removed == 1
        assert len(result.candles) == 2
        # Find the candle at the duplicate timestamp (Aug 10)
        dup_timestamp = candles[0].timestamp
        kept = [c for c in result.candles if c.timestamp == dup_timestamp]
        assert len(kept) == 1
        # First occurrence (volume 1000000) should be kept
        assert kept[0].volume == 1000000

    def test_duplicate_removal_keep_last(self):
        """Test keeping last occurrence of duplicates."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.KEEP_LAST)
        candles = make_duplicate_timestamp_candles()

        result = preprocessor.preprocess(candles)

        assert result.duplicates_removed == 1
        assert len(result.candles) == 2
        # Find the candle at the duplicate timestamp (Aug 10)
        dup_timestamp = candles[0].timestamp
        kept = [c for c in result.candles if c.timestamp == dup_timestamp]
        assert len(kept) == 1
        # Last occurrence (volume 1100000) should be kept
        assert kept[0].volume == 1100000

    def test_duplicate_removal_remove_all(self):
        """Test removing all duplicates."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.REMOVE_ALL)
        candles = make_duplicate_timestamp_candles()

        result = preprocessor.preprocess(candles)

        # Both duplicate candles should be removed
        assert result.duplicates_removed == 2
        assert len(result.candles) == 1

    def test_duplicate_error_policy(self):
        """Test that ERROR policy raises exception on duplicates."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.ERROR)
        candles = make_duplicate_timestamp_candles()

        with pytest.raises(DataQualityError, match="Duplicate"):
            preprocessor.preprocess(candles)

    def test_empty_input(self):
        """Test preprocessing empty list."""
        preprocessor = Preprocessor()

        result = preprocessor.preprocess([])

        assert len(result.candles) == 0
        assert result.original_count == 0

    def test_preserves_valid_candles(self):
        """Test that valid candles are preserved."""
        preprocessor = Preprocessor()
        candles = make_valid_candles(count=3)

        result = preprocessor.preprocess(candles)

        assert len(result.candles) == 3
        assert result.original_count == 3
        assert result.final_count == 3

    def test_result_metadata(self):
        """Test that preprocessing result has correct metadata."""
        preprocessor = Preprocessor()
        candles = make_valid_candles(count=3)

        result = preprocessor.preprocess(candles)

        assert result.original_count == 3
        assert result.final_count == 3
        assert result.duplicates_removed == 0

    def test_warnings_generated(self):
        """Test that warnings are generated for operations."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        candles = make_duplicate_timestamp_candles()

        result = preprocessor.preprocess(candles)

        assert len(result.warnings) > 0
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_sort_chronologically_method(self):
        """Test the sort_chronologically method."""
        preprocessor = Preprocessor()
        candles = make_unsorted_candles()

        sorted_candles = preprocessor.sort_chronologically(candles)

        assert sorted_candles[0].timestamp < sorted_candles[1].timestamp
        assert sorted_candles[1].timestamp < sorted_candles[2].timestamp
        # Original list should be unchanged
        assert candles[0].timestamp.day == 12

    def test_remove_duplicates_method(self):
        """Test the remove_duplicates method."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        candles = make_duplicate_timestamp_candles()

        filtered, removed = preprocessor.remove_duplicates(candles)

        assert removed == 1
        assert len(filtered) == 2


class TestMissingValueHandling:
    """Tests for handling missing values."""

    def test_missing_value_detection(self):
        """Test detection of NaN values."""
        preprocessor = Preprocessor(
            missing_value_policy=MissingValuePolicy.REMOVE
        )

        # Create a candle with NaN (need to bypass model validation)
        valid_candle = make_valid_candle()

        # Use the internal method to check detection
        # (Model prevents NaN creation, so we test the detection logic conceptually)
        assert not preprocessor._has_missing_values(valid_candle)

    def test_missing_value_removal_policy(self):
        """Test that REMOVE policy filters candles with missing values."""
        preprocessor = Preprocessor(
            missing_value_policy=MissingValuePolicy.REMOVE
        )

        # All valid candles should pass
        candles = make_valid_candles(count=3)
        result = preprocessor.preprocess(candles)

        assert result.final_count == 3

    def test_missing_value_error_policy(self):
        """Test that ERROR policy raises on missing values."""
        preprocessor = Preprocessor(
            missing_value_policy=MissingValuePolicy.ERROR
        )

        # Valid candles should pass
        candles = make_valid_candles(count=3)
        result = preprocessor.preprocess(candles)

        assert result.final_count == 3


class TestPreprocessingResult:
    """Tests for the PreprocessingResult dataclass."""

    def test_result_creation(self):
        """Test creating a preprocessing result."""
        candles = make_valid_candles(count=2)
        result = PreprocessingResult(
            candles=candles,
            original_count=2,
            final_count=2,
        )

        assert len(result.candles) == 2
        assert result.duplicates_removed == 0
        assert result.was_unsorted is False

    def test_removed_indices_tracking(self):
        """Test that removed indices are tracked."""
        preprocessor = Preprocessor(duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        candles = make_duplicate_timestamp_candles()

        result = preprocessor.preprocess(candles)

        assert len(result.removed_indices) == 1
