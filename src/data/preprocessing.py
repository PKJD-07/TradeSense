"""
Preprocessing module for OHLCV candle data.

Handles data cleaning operations like sorting, deduplication,
and missing value handling. Does NOT perform feature engineering,
interpolation, or ML-specific transformations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.data.models import Candle
from src.data.exceptions import DataQualityError


class DuplicatePolicy(Enum):
    """Policy for handling duplicate timestamps in candle data."""

    KEEP_FIRST = "keep_first"  # Keep the first occurrence, remove others
    KEEP_LAST = "keep_last"    # Keep the last occurrence, remove others
    REMOVE_ALL = "remove_all"  # Remove all candles with duplicate timestamps
    ERROR = "error"            # Raise an error if duplicates are found


class MissingValuePolicy(Enum):
    """Policy for handling missing values in candle data."""

    KEEP = "keep"      # Keep candles with missing/invalid values (handled by validation)
    REMOVE = "remove"  # Remove candles with missing/invalid values
    ERROR = "error"    # Raise an error if missing values are found


@dataclass
class PreprocessingResult:
    """
    Result of preprocessing a collection of candles.

    Attributes:
        candles: The preprocessed candles
        original_count: Number of candles before preprocessing
        final_count: Number of candles after preprocessing
        duplicates_removed: Number of duplicate candles removed
        was_unsorted: True if the input was not in chronological order and the
            preprocessor had to sort it
        removed_indices: Original indices of removed candles
        warnings: Non-fatal issues encountered during preprocessing
    """

    candles: list[Candle]
    original_count: int = 0
    final_count: int = 0
    duplicates_removed: int = 0
    was_unsorted: bool = False
    removed_indices: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Preprocessor:
    """
    Preprocessor for OHLCV candle data.

    Provides operations for:
    - Sorting candles chronologically
    - Removing duplicates based on configurable policy
    - Handling missing values

    Does NOT perform:
    - Feature engineering
    - Price interpolation
    - Technical indicators
    - Normalization/scaling for ML
    """

    def __init__(
        self,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.KEEP_FIRST,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.KEEP,
        sort: bool = True,
    ):
        """
        Initialize the preprocessor.

        Args:
            duplicate_policy: How to handle duplicate timestamps
            missing_value_policy: How to handle missing values
            sort: Whether to sort candles chronologically
        """
        self.duplicate_policy = duplicate_policy
        self.missing_value_policy = missing_value_policy
        self.sort = sort

    def preprocess(self, candles: list[Candle]) -> PreprocessingResult:
        """
        Preprocess a list of candles.

        Operations are applied in order:
        1. Remove candles with missing values (if policy is REMOVE)
        2. Handle duplicates
        3. Sort chronologically (if enabled)

        Args:
            candles: List of candles to preprocess

        Returns:
            PreprocessingResult with cleaned candles and metadata
        """
        if not candles:
            return PreprocessingResult(
                candles=[],
                original_count=0,
                final_count=0,
            )

        original_count = len(candles)
        working_candles = list(enumerate(candles))  # (original_index, candle)
        removed_indices: list[int] = []
        warnings: list[str] = []
        duplicates_removed = 0
        was_unsorted = False

        # Step 1: Handle missing values
        if self.missing_value_policy == MissingValuePolicy.REMOVE:
            valid_candles = []
            for orig_idx, candle in working_candles:
                if self._has_missing_values(candle):
                    removed_indices.append(orig_idx)
                else:
                    valid_candles.append((orig_idx, candle))
            working_candles = valid_candles
            removed_count = original_count - len(working_candles)
            if removed_count > 0:
                warnings.append(f"Removed {removed_count} candles with missing values")

        elif self.missing_value_policy == MissingValuePolicy.ERROR:
            for i, candle in enumerate(candles):
                if self._has_missing_values(candle):
                    raise DataQualityError(
                        message=f"Candle at index {i} has missing values",
                        issues=[f"Candle {i}: missing values detected"],
                    )

        # Step 2: Handle duplicates
        if self.duplicate_policy != DuplicatePolicy.ERROR:
            working_candles, dup_removed, dup_indices = self._handle_duplicates(
                working_candles
            )
            duplicates_removed = dup_removed
            removed_indices.extend(dup_indices)
            if dup_removed > 0:
                warnings.append(f"Removed {dup_removed} duplicate candles ({self.duplicate_policy.value} policy)")
        else:
            # Check for duplicates and error if found
            timestamps = [c.timestamp for _, c in working_candles]
            if len(timestamps) != len(set(timestamps)):
                dup_ts = [ts for ts in timestamps if timestamps.count(ts) > 1]
                raise DataQualityError(
                    message="Duplicate timestamps found",
                    issues=[f"Duplicate timestamps: {set(dup_ts)}"],
                )

        # Step 3: Sort if enabled
        if self.sort:
            timestamps = [c.timestamp for _, c in working_candles]
            is_sorted = all(
                timestamps[i] <= timestamps[i + 1]
                for i in range(len(timestamps) - 1)
            )
            if not is_sorted:
                working_candles.sort(key=lambda x: x[1].timestamp)
                was_unsorted = True

        # Extract final candles
        final_candles = [candle for _, candle in working_candles]

        return PreprocessingResult(
            candles=final_candles,
            original_count=original_count,
            final_count=len(final_candles),
            duplicates_removed=duplicates_removed,
            was_unsorted=was_unsorted,
            removed_indices=removed_indices,
            warnings=warnings,
        )

    def _has_missing_values(self, candle: Candle) -> bool:
        """Check if a candle has missing or invalid values."""
        # Check for None values (shouldn't happen with dataclass, but be safe)
        if any(
            getattr(candle, field) is None
            for field in ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        ):
            return True

        # Check for NaN or Inf in numeric fields
        import math
        for field in ["open", "high", "low", "close"]:
            value = getattr(candle, field)
            if math.isnan(value) or math.isinf(value):
                return True

        # Volume can be 0 (valid) but not NaN/Inf
        if math.isnan(candle.volume) or math.isinf(candle.volume):
            return True

        return False

    def _handle_duplicates(
        self,
        indexed_candles: list[tuple[int, Candle]],
    ) -> tuple[list[tuple[int, Candle]], int, list[int]]:
        """
        Handle duplicate timestamps according to the configured policy.

        Args:
            indexed_candles: List of (original_index, candle) tuples

        Returns:
            Tuple of (filtered_candles, count_removed, removed_indices)
        """
        if not indexed_candles:
            return indexed_candles, 0, []

        # Group by timestamp
        ts_groups: dict[datetime, list[tuple[int, Candle]]] = {}
        for orig_idx, candle in indexed_candles:
            ts = candle.timestamp
            if ts not in ts_groups:
                ts_groups[ts] = []
            ts_groups[ts].append((orig_idx, candle))

        result: list[tuple[int, Candle]] = []
        removed_indices: list[int] = []
        duplicates_removed = 0

        for ts, group in ts_groups.items():
            if len(group) == 1:
                # No duplicate
                result.append(group[0])
            else:
                # Duplicate found
                if self.duplicate_policy == DuplicatePolicy.KEEP_FIRST:
                    result.append(group[0])
                    for orig_idx, _ in group[1:]:
                        removed_indices.append(orig_idx)
                        duplicates_removed += 1

                elif self.duplicate_policy == DuplicatePolicy.KEEP_LAST:
                    result.append(group[-1])
                    for orig_idx, _ in group[:-1]:
                        removed_indices.append(orig_idx)
                        duplicates_removed += 1

                elif self.duplicate_policy == DuplicatePolicy.REMOVE_ALL:
                    for orig_idx, _ in group:
                        removed_indices.append(orig_idx)
                        duplicates_removed += 1

        return result, duplicates_removed, removed_indices

    def sort_chronologically(self, candles: list[Candle]) -> list[Candle]:
        """
        Sort candles by timestamp in ascending order.

        Args:
            candles: List of candles to sort

        Returns:
            New list of candles sorted by timestamp
        """
        return sorted(candles, key=lambda c: c.timestamp)

    def remove_duplicates(
        self,
        candles: list[Candle],
        policy: DuplicatePolicy | None = None,
    ) -> tuple[list[Candle], int]:
        """
        Remove duplicate candles based on timestamp.

        Args:
            candles: List of candles
            policy: Policy to use (defaults to instance policy)

        Returns:
            Tuple of (filtered_candles, count_removed)
        """
        if policy is None:
            policy = self.duplicate_policy

        indexed = list(enumerate(candles))
        filtered, removed, _ = self._handle_duplicates(indexed)
        return [c for _, c in filtered], removed
