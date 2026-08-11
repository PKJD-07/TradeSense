"""
Yahoo Finance data provider for TradeSense.

Uses the yfinance library to fetch historical OHLCV data.
Yahoo Finance is free and doesn't require an API key, making it
ideal for educational purposes and prototyping.
"""

from datetime import date, datetime
from typing import Any

from src.data.exceptions import DataProviderError


class YahooFinanceProvider:
    """
    Yahoo Finance provider for historical market data.

    This provider fetches daily OHLCV data from Yahoo Finance using
    the yfinance library. No API key is required.

    By default it returns *adjusted* prices, which is the appropriate choice
    for TradeSense's future return computation, feature engineering, and
    backtesting. See ``auto_adjust`` for the exact semantics.

    Args:
        auto_adjust: Whether to fetch adjusted or raw/unadjusted OHLCV.
            - ``True`` (default): historical prices are restated so returns are
              continuous across dividends and stock splits. Returns computed
              from adjusted prices are economically correct (total return).
              This is the safe default for quantitative/ML use.
            - ``False``: the actual traded prices on each day. The series
              contains discontinuities at dividend/split dates, which create
              phantom returns if used as-is for return/backtest computation.
            In both modes yfinance reports *raw* volume (volume is not adjusted
            by Yahoo Finance).

    Example:
        >>> provider = YahooFinanceProvider()
        >>> data = provider.fetch_historical("AAPL", date(2026, 1, 1), date(2026, 8, 10))
        >>> len(data)
        150
    """

    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"auto_adjust={self.auto_adjust!r})"
        )

    def fetch_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: The ticker symbol (e.g., "AAPL")
            start_date: Start date inclusive
            end_date: End date inclusive

        Returns:
            List of dictionaries with OHLCV data. Each dict contains:
            - timestamp: timezone-aware datetime (in the exchange's local
              timezone, as delivered by yfinance; naive timestamps are
              rejected, never guessed)
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: int

        Raises:
            DataProviderError: If yfinance fails or returns invalid data
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataProviderError(
                "yfinance library not installed. Install with: pip install yfinance",
                provider=self.name,
                original_error=e,
            )

        symbol = symbol.strip().upper()
        if not symbol:
            raise DataProviderError(
                "Symbol cannot be empty",
                provider=self.name,
            )

        if start_date > end_date:
            raise DataProviderError(
                f"start_date ({start_date}) cannot be after end_date ({end_date})",
                provider=self.name,
            )

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=self.auto_adjust,
            )

            if df.empty:
                raise DataProviderError(
                    f"No data returned for symbol '{symbol}' between {start_date} and {end_date}",
                    provider=self.name,
                )

            return self._normalize_dataframe(df, symbol)

        except DataProviderError:
            raise
        except Exception as e:
            raise DataProviderError(
                f"Failed to fetch data for '{symbol}': {str(e)}",
                provider=self.name,
                original_error=e,
            )

    def _normalize_dataframe(self, df: Any, symbol: str) -> list[dict]:
        """
        Normalize a pandas DataFrame from yfinance to a list of dictionaries.

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume
            symbol: The ticker symbol

        Returns:
            List of dictionaries with normalized OHLCV data
        """
        candles = []

        # yfinance returns columns: Open, High, Low, Close, Volume (and, when
        # auto_adjust=False, Dividends and Stock Splits). We only need OHLCV.
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Verify required columns exist
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise DataProviderError(
                f"Missing required columns in Yahoo Finance response: {missing}",
                provider=self.name,
            )

        for idx, row in df.iterrows():
            # idx is typically a pandas Timestamp; may also be a plain datetime
            timestamp = idx
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()
            elif not isinstance(timestamp, datetime):
                raise DataProviderError(
                    f"Unexpected timestamp type from provider: {type(idx)}",
                    provider=self.name,
                )

            # Canonical policy: timestamps are always timezone-aware. yfinance
            # returns aware timestamps in the exchange's local timezone. A naive
            # timestamp is rejected rather than guessed — the Candle model later
            # normalizes aware timestamps to UTC.
            if timestamp.tzinfo is None:
                raise DataProviderError(
                    f"Naive timestamp returned by Yahoo Finance at index {idx}: "
                    f"{timestamp!r}. Timestamps must be timezone-aware so they "
                    "can be normalized to UTC.",
                    provider=self.name,
                )

            candle = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
            candles.append(candle)

        return candles
