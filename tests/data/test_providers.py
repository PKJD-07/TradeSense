"""
Tests for data providers.

Uses mocks to avoid external API calls.
"""

from datetime import date, datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.data.providers.yahoo import YahooFinanceProvider
from src.data.exceptions import DataProviderError
from tests.data.fixtures import make_mock_provider_response


class TestYahooFinanceProvider:
    """Tests for the Yahoo Finance provider."""

    def test_provider_name(self):
        """Test provider name property."""
        provider = YahooFinanceProvider()
        assert provider.name == "Yahoo Finance"

    def test_fetch_historical_success(self):
        """Test successful historical data fetch."""
        provider = YahooFinanceProvider()

        # Mock yfinance
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iterrows.return_value = [
            (
                datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                {
                    "Open": 210.15,
                    "High": 212.30,
                    "Low": 209.80,
                    "Close": 211.75,
                    "Volume": 1523400,
                },
            ),
            (
                datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
                {
                    "Open": 208.50,
                    "High": 210.20,
                    "Low": 207.90,
                    "Close": 209.80,
                    "Volume": 1452000,
                },
            ),
        ]
        mock_df.__contains__ = lambda self, key: key in ["Open", "High", "Low", "Close", "Volume"]
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            result = provider.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

            assert len(result) == 2
            assert result[0]["symbol"] == "AAPL"
            assert result[0]["open"] == 210.15

    def test_auto_adjust_default_true(self):
        """Test that adjusted prices are the default and passed to yfinance."""
        provider = YahooFinanceProvider()
        assert provider.auto_adjust is True

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        mock_df.iterrows.return_value = []

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            provider.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

            call_kwargs = mock_ticker.return_value.history.call_args.kwargs
            assert call_kwargs.get("auto_adjust") is True

    def test_auto_adjust_false_passthrough(self):
        """Test that auto_adjust=False requests raw/unadjusted prices."""
        provider = YahooFinanceProvider(auto_adjust=False)

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        mock_df.iterrows.return_value = []

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            provider.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

            call_kwargs = mock_ticker.return_value.history.call_args.kwargs
            assert call_kwargs.get("auto_adjust") is False

    def test_fetch_historical_empty_response(self):
        """Test handling of empty response from Yahoo Finance."""
        provider = YahooFinanceProvider()

        mock_df = MagicMock()
        mock_df.empty = True

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            with pytest.raises(DataProviderError, match="No data returned"):
                provider.fetch_historical(
                    symbol="INVALID",
                    start_date=date(2026, 8, 1),
                    end_date=date(2026, 8, 10),
                )

    def test_fetch_historical_empty_symbol(self):
        """Test that empty symbol raises error."""
        provider = YahooFinanceProvider()

        with pytest.raises(DataProviderError, match="Symbol cannot be empty"):
            provider.fetch_historical(
                symbol="",
                start_date=date(2026, 8, 1),
                end_date=date(2026, 8, 10),
            )

    def test_fetch_historical_invalid_date_range(self):
        """Test that invalid date range raises error."""
        provider = YahooFinanceProvider()

        with pytest.raises(DataProviderError, match="start_date.*cannot be after end_date"):
            provider.fetch_historical(
                symbol="AAPL",
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 1),
            )

    def test_fetch_historical_missing_yfinance(self):
        """Test error when yfinance is not installed."""
        provider = YahooFinanceProvider()

        with patch.dict("sys.modules", {"yfinance": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(DataProviderError, match="yfinance library not installed"):
                    provider.fetch_historical(
                        symbol="AAPL",
                        start_date=date(2026, 8, 1),
                        end_date=date(2026, 8, 10),
                    )

    def test_fetch_historical_naive_timestamp_rejected(self):
        """Test that naive timestamps from Yahoo are rejected, never assumed UTC."""
        provider = YahooFinanceProvider()

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        mock_df.iterrows.return_value = [
            (
                datetime(2026, 8, 10, 0, 0, 0),  # naive
                {"Open": 210.0, "High": 212.0, "Low": 209.0, "Close": 211.0, "Volume": 1000000},
            ),
        ]
        mock_df.__contains__ = lambda self, key: key in mock_df.columns

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df

            with pytest.raises(DataProviderError, match="Naive timestamp"):
                provider.fetch_historical(
                    symbol="AAPL",
                    start_date=date(2026, 8, 1),
                    end_date=date(2026, 8, 10),
                )

    def test_normalize_dataframe(self):
        """Test normalization of DataFrame to dict list."""
        provider = YahooFinanceProvider()

        mock_df = MagicMock()
        mock_df.columns = ["Open", "High", "Low", "Close", "Volume"]

        rows = [
            (
                datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                {"Open": 210.0, "High": 212.0, "Low": 209.0, "Close": 211.0, "Volume": 1000000},
            ),
        ]
        mock_df.iterrows.return_value = rows
        mock_df.__contains__ = lambda self, key: key in mock_df.columns

        result = provider._normalize_dataframe(mock_df, "AAPL")

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["open"] == 210.0
        assert result[0]["volume"] == 1000000

    def test_normalize_dataframe_missing_columns(self):
        """Test error when DataFrame is missing required columns."""
        provider = YahooFinanceProvider()

        mock_df = MagicMock()
        mock_df.columns = ["Open", "High"]  # Missing Low, Close, Volume

        with pytest.raises(DataProviderError, match="Missing required columns"):
            provider._normalize_dataframe(mock_df, "AAPL")

    def test_provider_repr(self):
        """Test string representation of provider."""
        provider = YahooFinanceProvider()

        assert "YahooFinanceProvider" in repr(provider)
        assert "Yahoo Finance" in repr(provider)


class TestProviderContract:
    """Tests for the provider contract."""

    def test_protocol_compliance(self):
        """Test that YahooFinanceProvider satisfies the HistoricalDataProvider protocol."""
        from src.data.providers.base import HistoricalDataProvider

        provider = YahooFinanceProvider()

        # Structural check - the provider must have the required members
        assert hasattr(provider, "name")
        assert hasattr(provider, "fetch_historical")
        assert callable(provider.fetch_historical)
        # Duck-typing check: the provider matches the protocol structurally.
        assert isinstance(provider, HistoricalDataProvider)
