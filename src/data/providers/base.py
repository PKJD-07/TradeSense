"""
Abstract interface for historical market data providers.

Defines the contract that all providers must satisfy, allowing the rest
of the system to remain decoupled from any specific data source.
"""

from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class HistoricalDataProvider(Protocol):
    """
    Protocol defining the interface for historical market data providers.

    This is a structural interface: any object with a ``name`` property and a
    ``fetch_historical`` method matching the signatures below satisfies it,
    without needing to subclass. The pipeline type-hints against this protocol
    so providers can be swapped in and out freely.

    The provider is responsible for:
    - Connecting to the external data source
    - Fetching raw OHLCV data
    - Returning data in a provider-agnostic format (list of dicts)

    The pipeline is responsible for:
    - Normalizing the raw data to Candle objects
    - Validation
    - Preprocessing
    """

    @property
    def name(self) -> str:
        """
        Provider name for logging and error messages.

        Returns:
            A human-readable provider name (e.g., "Yahoo Finance", "Alpha Vantage")
        """
        ...

    def fetch_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """
        Fetch historical OHLCV data for a symbol within a date range.

        Args:
            symbol: The ticker symbol (e.g., "AAPL")
            start_date: Start date inclusive
            end_date: End date inclusive

        Returns:
            List of dictionaries with the following keys:
            - timestamp: timezone-aware datetime object
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: int

        Raises:
            DataProviderError: If the provider fails to fetch data
            ConfigurationError: If required configuration (API key) is missing
        """
        ...
