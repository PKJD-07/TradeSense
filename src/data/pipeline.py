"""
Data pipeline for orchestrating market data ingestion.

Coordinates the flow:
    Provider → Normalization → Validation → Preprocessing → Result

The pipeline exposes two validation verdicts:

- ``pre_validation``: validation of the candles exactly as normalized from the
  provider, *before* preprocessing. This reports issues that preprocessing may
  be about to fix (e.g. unsorted timestamps, duplicates).
- ``final_validation`` (also ``validation_result``): validation of the final,
  post-preprocessing candles. ``IngestionResult.is_valid`` reflects this, so a
  result is never reported invalid for a problem that preprocessing resolved.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum

from src.data.models import Candle, CandleCollection
from src.data.providers.base import HistoricalDataProvider
from src.data.validation import DatasetValidator, DatasetValidationResult
from src.data.preprocessing import Preprocessor, PreprocessingResult, DuplicatePolicy, MissingValuePolicy
from src.data.exceptions import DataProviderError, DataQualityError


class ErrorPolicy(Enum):
    """Policy for handling errors during data ingestion."""

    FAIL_FAST = "fail_fast"      # Stop on first error; raise the documented exception
    COLLECT_ALL = "collect_all"  # Collect all errors before reporting


@dataclass
class RejectedRow:
    """
    A provider row that could not be normalized into a Candle.

    Rejected rows are never silently discarded: they are recorded here for
    traceability.

    Attributes:
        index: Position of the row in the provider's response
        reason: Why normalization failed (e.g. missing field, bad type)
        raw: The raw provider row, when available
    """

    index: int
    reason: str
    raw: dict | None = None


@dataclass
class IngestionResult:
    """
    Complete result of the data ingestion pipeline.

    Attributes:
        candles: List of validated and preprocessed candles
        symbol: The ticker symbol (uppercase)
        start_date: Requested start date
        end_date: Requested end date
        pre_validation: Validation of the raw (pre-preprocessing) candles
        final_validation: Validation of the final (post-preprocessing) candles
        preprocessing_result: Result of preprocessing
        provider_name: Name of the data provider used
        fetch_timestamp: When the data was fetched
        rejected_rows: Provider rows that could not be normalized
    """

    candles: list[Candle]
    symbol: str
    start_date: date
    end_date: date
    pre_validation: DatasetValidationResult
    final_validation: DatasetValidationResult
    preprocessing_result: PreprocessingResult
    provider_name: str
    fetch_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rejected_rows: list[RejectedRow] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the final (post-preprocessing) data passed validation."""
        return self.final_validation.is_valid

    @property
    def validation_result(self) -> DatasetValidationResult:
        """Alias for ``final_validation`` (the post-preprocessing verdict)."""
        return self.final_validation

    @property
    def candle_count(self) -> int:
        """Number of candles in the result."""
        return len(self.candles)

    @property
    def rejected_count(self) -> int:
        """Number of provider rows that could not be normalized."""
        return len(self.rejected_rows)

    def to_candle_collection(self) -> CandleCollection:
        """Convert result to a CandleCollection."""
        return CandleCollection(symbol=self.symbol, candles=self.candles)

    def summary(self) -> str:
        """Return a human-readable summary of the ingestion."""
        lines = [
            "Data Ingestion Summary",
            f"  Symbol: {self.symbol}",
            f"  Date Range: {self.start_date} to {self.end_date}",
            f"  Provider: {self.provider_name}",
            f"  Candles: {self.candle_count}",
            f"  Valid: {self.is_valid}",
        ]

        if self.preprocessing_result.duplicates_removed > 0:
            lines.append(f"  Duplicates Removed: {self.preprocessing_result.duplicates_removed}")

        if self.preprocessing_result.was_unsorted:
            lines.append("  Data was sorted chronologically")

        if self.rejected_rows:
            lines.append(f"  Rejected Rows: {self.rejected_count}")

        if self.pre_validation.issues:
            lines.append(f"  Raw Data Issues: {len(self.pre_validation.issues)}")

        if self.final_validation.issues:
            lines.append(f"  Final Issues: {len(self.final_validation.issues)}")

        return "\n".join(lines)


class DataPipeline:
    """
    Pipeline for fetching and processing historical market data.

    Orchestrates:
    1. Fetching data from a provider
    2. Normalizing to Candle objects
    3. Validating the raw (pre-processing) candles
    4. Preprocessing (sorting, deduplication, missing-value policy)
    5. Validating the final (post-processing) candles

    Example:
        >>> from src.data.providers import YahooFinanceProvider
        >>> provider = YahooFinanceProvider()
        >>> pipeline = DataPipeline(provider)
        >>> result = pipeline.fetch_historical("AAPL", date(2026, 1, 1), date(2026, 8, 10))
        >>> print(result.summary())
    """

    def __init__(
        self,
        provider: HistoricalDataProvider,
        error_policy: ErrorPolicy = ErrorPolicy.FAIL_FAST,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.KEEP_FIRST,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.KEEP,
        auto_preprocess: bool = True,
    ):
        """
        Initialize the data pipeline.

        Args:
            provider: The data provider to use
            error_policy: How to handle errors (fail fast or collect all)
            duplicate_policy: How to handle duplicate timestamps
            missing_value_policy: How to handle missing values
            auto_preprocess: Whether to automatically preprocess after fetching
        """
        self.provider = provider
        self.error_policy = error_policy
        self.duplicate_policy = duplicate_policy
        self.missing_value_policy = missing_value_policy
        self.auto_preprocess = auto_preprocess

    def fetch_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> IngestionResult:
        """
        Fetch and process historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            start_date: Start date inclusive
            end_date: End date inclusive

        Returns:
            IngestionResult with candles, validation, and preprocessing info

        Raises:
            DataProviderError: If the provider fails, or a provider row cannot
                be normalized (under FAIL_FAST).
            DataQualityError: If the final (post-preprocessing) data still fails
                validation and ``error_policy`` is FAIL_FAST.
        """
        # Normalize the symbol at the boundary so every downstream component
        # sees the canonical uppercase form.
        symbol = symbol.strip().upper()
        if not symbol:
            raise DataProviderError("Symbol cannot be empty", provider=self.provider.name)

        # Step 1: Fetch raw data from provider
        raw_candles = self.provider.fetch_historical(symbol, start_date, end_date)

        # Step 2: Normalize to Candle objects; record any rejected rows
        candles, rejected_rows = self._normalize_candles(raw_candles, symbol)

        # Step 3: Validate the raw (pre-processing) candles
        pre_validation = self._validate(candles)

        # Step 4: Preprocess (sort, dedup, missing-value policy)
        if self.auto_preprocess:
            preprocessor = Preprocessor(
                duplicate_policy=self.duplicate_policy,
                missing_value_policy=self.missing_value_policy,
                sort=True,
            )
            preprocessing_result = preprocessor.preprocess(candles)
        else:
            preprocessing_result = PreprocessingResult(
                candles=candles,
                original_count=len(candles),
                final_count=len(candles),
            )

        # Step 5: Validate the final (post-preprocessing) candles
        final_validation = self._validate(preprocessing_result.candles)

        result = IngestionResult(
            candles=preprocessing_result.candles,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            pre_validation=pre_validation,
            final_validation=final_validation,
            preprocessing_result=preprocessing_result,
            provider_name=self.provider.name,
            rejected_rows=rejected_rows,
        )

        # Honest FAIL_FAST: raise if the final data is still invalid after
        # preprocessing resolved everything it could.
        if self.error_policy == ErrorPolicy.FAIL_FAST and not final_validation.is_valid:
            issues = list(final_validation.issues)
            issues += [
                f"Candle {idx}: {'; '.join(errs)}"
                for idx, errs in final_validation.candle_errors.items()
            ]
            raise DataQualityError(
                message="Data quality validation failed after preprocessing",
                issues=issues,
            )

        return result

    def _normalize_candles(
        self,
        raw_candles: list[dict],
        symbol: str,
    ) -> tuple[list[Candle], list[RejectedRow]]:
        """
        Normalize raw provider data to Candle objects.

        Under FAIL_FAST, the first malformed row raises ``DataProviderError``.
        Under COLLECT_ALL, malformed rows are recorded as ``RejectedRow`` and
        never silently discarded.

        Args:
            raw_candles: List of dictionaries from the provider
            symbol: The symbol being fetched (already normalized)

        Returns:
            Tuple of (valid candles, rejected rows)
        """
        candles: list[Candle] = []
        rejected_rows: list[RejectedRow] = []

        for i, raw in enumerate(raw_candles):
            try:
                candle = Candle(
                    symbol=raw.get("symbol", symbol),
                    timestamp=raw["timestamp"],
                    open=raw["open"],
                    high=raw["high"],
                    low=raw["low"],
                    close=raw["close"],
                    volume=raw["volume"],
                )
                candles.append(candle)
            except (KeyError, ValueError, TypeError) as e:
                if self.error_policy == ErrorPolicy.FAIL_FAST:
                    raise DataProviderError(
                        f"Failed to normalize candle at index {i}: {str(e)}",
                        provider=self.provider.name,
                        original_error=e,
                    )
                rejected_rows.append(RejectedRow(index=i, reason=str(e), raw=raw))

        return candles, rejected_rows

    def _validate(self, candles: list[Candle]) -> DatasetValidationResult:
        """
        Validate a collection of candles.

        Validation is always advisory (non-strict): it reports issues rather
        than raising, so preprocessing has a chance to resolve fixable problems
        before FAIL_FAST enforcement happens in :meth:`fetch_historical`.

        Args:
            candles: List of candles to validate

        Returns:
            DatasetValidationResult
        """
        dataset_validator = DatasetValidator(strict_mode=False)
        return dataset_validator.validate(candles)

    def fetch_and_validate(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[CandleCollection, DatasetValidationResult]:
        """
        Fetch data and return as a CandleCollection with validation results.

        This is a convenience method for when you want the data
        as a CandleCollection.

        Args:
            symbol: Ticker symbol
            start_date: Start date inclusive
            end_date: End date inclusive

        Returns:
            Tuple of (CandleCollection, final DatasetValidationResult)
        """
        result = self.fetch_historical(symbol, start_date, end_date)
        collection = result.to_candle_collection()
        return collection, result.final_validation
