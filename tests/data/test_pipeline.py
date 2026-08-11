"""
Tests for the data ingestion pipeline.

Uses mocked providers to test the pipeline without external API calls.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from src.data.pipeline import DataPipeline, IngestionResult, ErrorPolicy, RejectedRow
from src.data.providers.yahoo import YahooFinanceProvider
from src.data.preprocessing import DuplicatePolicy, MissingValuePolicy
from src.data.exceptions import DataProviderError, DataQualityError
from tests.data.fixtures import (
    make_mock_provider_response,
    make_malformed_provider_response,
    make_valid_candles,
    make_candle_with_invalid_ohlc,
)


class MockProvider:
    """Mock provider for testing."""

    def __init__(self, response=None, should_fail=False):
        self._response = response or make_mock_provider_response()
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return "Mock Provider"

    def fetch_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ):
        if self._should_fail:
            raise DataProviderError("Mock provider failure", provider=self.name)
        return self._response


class TestDataPipeline:
    """Tests for the DataPipeline class."""

    def test_fetch_historical_success(self):
        """Test successful data fetch through pipeline."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert isinstance(result, IngestionResult)
        assert result.symbol == "AAPL"
        assert result.provider_name == "Mock Provider"
        assert len(result.candles) == 3

    def test_pipeline_validates_candles(self):
        """Test that pipeline validates fetched candles."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert result.validation_result is not None
        assert isinstance(result.validation_result.is_valid, bool)

    def test_pipeline_preprocesses_candles(self):
        """Test that pipeline preprocesses candles."""
        provider = MockProvider()
        pipeline = DataPipeline(provider, auto_preprocess=True)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert result.preprocessing_result is not None

    def test_pipeline_provider_error(self):
        """Test that provider errors are propagated."""
        provider = MockProvider(should_fail=True)
        pipeline = DataPipeline(provider)

        with pytest.raises(DataProviderError):
            pipeline.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

    def test_result_to_candle_collection(self):
        """Test converting result to CandleCollection."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        collection = result.to_candle_collection()

        assert collection.symbol == "AAPL"
        assert len(collection) == len(result.candles)

    def test_result_summary(self):
        """Test result summary generation."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        summary = result.summary()

        assert "AAPL" in summary
        assert "Mock Provider" in summary
        assert "Candles:" in summary

    def test_result_is_valid_property(self):
        """Test is_valid property on result."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        # Mock provider returns valid data
        assert isinstance(result.is_valid, bool)

    def test_error_policy_fail_fast(self):
        """Test fail fast error policy."""
        provider = MockProvider(response=make_malformed_provider_response())
        pipeline = DataPipeline(
            provider,
            error_policy=ErrorPolicy.FAIL_FAST,
        )

        with pytest.raises(DataProviderError):
            pipeline.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

    def test_duplicate_policy_applied(self):
        """Test that duplicate policy is applied."""
        # Create response with duplicate timestamps
        response = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                "open": 210.0,
                "high": 212.0,
                "low": 209.0,
                "close": 211.0,
                "volume": 1000000,
            },
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),  # Duplicate!
                "open": 211.0,
                "high": 213.0,
                "low": 210.0,
                "close": 212.0,
                "volume": 1100000,
            },
        ]

        provider = MockProvider(response=response)
        pipeline = DataPipeline(
            provider,
            duplicate_policy=DuplicatePolicy.KEEP_FIRST,
            auto_preprocess=True,
        )

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        assert len(result.candles) == 1
        assert result.preprocessing_result.duplicates_removed == 1

    def test_fetch_and_validate(self):
        """Test fetch_and_validate convenience method."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        collection, validation = pipeline.fetch_and_validate(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert collection.symbol == "AAPL"
        assert len(collection) == 3
        assert validation is not None

    def test_pre_validation_flags_unsorted_but_final_is_clean(self):
        """Test that sortable issues appear only in pre_validation.

        The default mock response is descending (newest first), so raw
        validation should flag it unsorted; after preprocessing sorts it, the
        final validation and is_valid should be clean.
        """
        provider = MockProvider()  # default response is descending
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert not result.pre_validation.is_valid
        assert any("chronological" in i.lower() for i in result.pre_validation.issues)
        assert result.preprocessing_result.was_unsorted
        assert result.final_validation.is_valid
        assert result.is_valid

    def test_fail_fast_raises_on_final_validation_failure(self):
        """Test that FAIL_FAST raises DataQualityError when preprocessing
        cannot fix a validation problem (e.g. an invalid OHLC relationship)."""
        response = [make_candle_with_invalid_ohlc()]
        provider = MockProvider(response=response)
        pipeline = DataPipeline(provider, error_policy=ErrorPolicy.FAIL_FAST)

        with pytest.raises(DataQualityError, match="validation failed"):
            pipeline.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

    def test_collect_all_records_rejected_rows(self):
        """Test that COLLECT_ALL records malformed rows instead of dropping them."""
        provider = MockProvider(response=make_malformed_provider_response())
        pipeline = DataPipeline(provider, error_policy=ErrorPolicy.COLLECT_ALL)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        # The malformed row (missing 'close') is recorded, not silently dropped
        assert result.rejected_count == 1
        assert isinstance(result.rejected_rows[0], RejectedRow)
        assert result.rejected_rows[0].index == 1
        assert "close" in result.rejected_rows[0].reason
        assert result.rejected_rows[0].raw is not None
        # The valid row is still ingested
        assert len(result.candles) == 1
        # And the rejection is surfaced in the summary
        assert "Rejected Rows: 1" in result.summary()

    def test_fail_fast_does_not_raise_for_fixable_duplicates(self):
        """Test that FAIL_FAST does not raise when preprocessing resolves duplicates."""
        response = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                "open": 210.0,
                "high": 212.0,
                "low": 209.0,
                "close": 211.0,
                "volume": 1000000,
            },
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                "open": 211.0,
                "high": 213.0,
                "low": 210.0,
                "close": 212.0,
                "volume": 1100000,
            },
        ]
        provider = MockProvider(response=response)
        pipeline = DataPipeline(provider, error_policy=ErrorPolicy.FAIL_FAST)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        # Duplicates were resolvable -> no raise, final data is clean
        assert result.pre_validation.is_valid is False
        assert result.preprocessing_result.duplicates_removed == 1
        assert result.final_validation.is_valid
        assert result.is_valid

    def test_lowercase_provider_symbol_does_not_break_collection(self):
        """Test that a provider returning lowercase symbols works end to end."""
        response = make_mock_provider_response()
        for row in response:
            row["symbol"] = "aapl"
        provider = MockProvider(response=response)
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="aapl",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        collection = result.to_candle_collection()
        assert collection.symbol == "AAPL"
        assert len(collection) == 3

    def test_auto_preprocess_disabled(self):
        """Test disabling auto preprocessing."""
        # Provide already-sorted data so the default FAIL_FAST policy passes
        sorted_response = list(reversed(make_mock_provider_response()))
        provider = MockProvider(response=sorted_response)
        pipeline = DataPipeline(provider, auto_preprocess=False)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        # Preprocessing result should show no changes
        assert result.preprocessing_result.original_count == result.preprocessing_result.final_count

    def test_pipeline_with_yahoo_provider(self):
        """Test pipeline with real Yahoo Finance provider (mocked)."""
        provider = YahooFinanceProvider()
        pipeline = DataPipeline(provider)

        # Mock the yfinance call
        mock_df = Mock()
        mock_df.empty = False
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        mock_df.iterrows.return_value = [
            (
                datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                {"Open": 210.0, "High": 212.0, "Low": 209.0, "Close": 211.0, "Volume": 1000000},
            ),
        ]
        mock_df.__contains__ = lambda self, key: key in mock_df.columns

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            result = pipeline.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

            assert result.provider_name == "Yahoo Finance"
            assert len(result.candles) == 1


class TestTimezoneNormalization:
    """Tests for UTC normalization of timestamps through the pipeline."""

    def test_exchange_local_timestamp_normalized_to_utc(self):
        """Test that an exchange-local aware timestamp becomes UTC in candles."""
        response = [
            {
                "symbol": "AAPL",
                # 2026-08-10 09:30 New York (EDT, -04:00) == 13:30 UTC
                "timestamp": datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone(timedelta(hours=-4))),
                "open": 210.0,
                "high": 212.0,
                "low": 209.0,
                "close": 211.0,
                "volume": 1000000,
            },
        ]
        provider = MockProvider(response=response)
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        assert result.candles[0].timestamp == datetime(2026, 8, 10, 13, 30, 0, tzinfo=timezone.utc)
        assert result.candles[0].timestamp.utcoffset() == timedelta(0)

    def test_duplicate_detection_across_timezones(self):
        """Test that equivalent instants in different timezones are deduped."""
        response = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone(timedelta(hours=-4))),  # 13:30 UTC
                "open": 210.0,
                "high": 212.0,
                "low": 209.0,
                "close": 211.0,
                "volume": 1000000,
            },
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 19, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),  # 13:30 UTC
                "open": 211.0,
                "high": 213.0,
                "low": 210.0,
                "close": 212.0,
                "volume": 1100000,
            },
        ]
        provider = MockProvider(response=response)
        pipeline = DataPipeline(
            provider,
            duplicate_policy=DuplicatePolicy.KEEP_FIRST,
            auto_preprocess=True,
        )

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        assert len(result.candles) == 1
        assert result.preprocessing_result.duplicates_removed == 1

    def test_ordering_correct_after_timezone_normalization(self):
        """Test that chronological ordering is correct across mixed timezones."""
        response = [
            # 2026-08-10 09:30 EDT == 13:30 UTC
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone(timedelta(hours=-4))),
                "open": 210.0,
                "high": 212.0,
                "low": 209.0,
                "close": 211.0,
                "volume": 1000000,
            },
            # 2026-08-10 12:00 UTC
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc),
                "open": 211.0,
                "high": 213.0,
                "low": 210.0,
                "close": 212.0,
                "volume": 1100000,
            },
            # 2026-08-10 14:00 IST == 08:30 UTC
            {
                "symbol": "AAPL",
                "timestamp": datetime(2026, 8, 10, 14, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),
                "open": 209.0,
                "high": 211.0,
                "low": 208.0,
                "close": 210.0,
                "volume": 900000,
            },
        ]
        # Shuffle so the preprocessor must sort
        response = [response[1], response[2], response[0]]
        provider = MockProvider(response=response)
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        timestamps = [c.timestamp for c in result.candles]
        assert timestamps == sorted(timestamps)
        # Earliest is 08:30 UTC (the IST candle), latest is 13:30 UTC (the EDT candle)
        assert timestamps[0] == datetime(2026, 8, 10, 8, 30, 0, tzinfo=timezone.utc)
        assert timestamps[-1] == datetime(2026, 8, 10, 13, 30, 0, tzinfo=timezone.utc)


class TestIngestionResult:
    """Tests for the IngestionResult dataclass."""

    def test_candle_count_property(self):
        """Test candle_count property."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert result.candle_count == len(result.candles)

    def test_fetch_timestamp_set(self):
        """Test that fetch_timestamp is set."""
        provider = MockProvider()
        pipeline = DataPipeline(provider)

        result = pipeline.fetch_historical(
            symbol="AAPL",
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 10),
        )

        assert result.fetch_timestamp is not None
        assert isinstance(result.fetch_timestamp, datetime)
