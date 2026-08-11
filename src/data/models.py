"""
OHLCV candle data model for TradeSense.

Provides a clean, type-hinted representation of a single candle
and operations on collections of candles.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator


@dataclass
class Candle:
    """
    A single OHLCV candle representing price and volume data for a symbol at a point in time.

    All prices must be positive. Volume must be non-negative.
    Timestamps are always timezone-aware UTC timestamps; naive datetimes are
    rejected because their timezone cannot be determined safely.

    Attributes:
        symbol: The ticker symbol (e.g., "AAPL"), normalized to uppercase
        timestamp: The candle timestamp (timezone-aware, normalized to UTC)
        open: Opening price
        high: Highest price during the candle period
        low: Lowest price during the candle period
        close: Closing price
        volume: Trading volume (number of shares/contracts)
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        """Normalize symbol and validate types and basic constraints."""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        # Canonical symbol form: uppercase, no surrounding whitespace.
        self.symbol = self.symbol.strip().upper()

        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware (e.g. datetime.now(timezone.utc))"
            )
        # Canonical internal representation: every timestamp is normalized to
        # UTC. This keeps ordering, duplicate detection, and any future
        # cross-series math deterministic regardless of the source timezone.
        self.timestamp = self.timestamp.astimezone(timezone.utc)

        for price_name in ("open", "high", "low", "close"):
            price = getattr(self, price_name)
            if not isinstance(price, (int, float)):
                raise ValueError(f"{price_name} must be a number")
            if price < 0:
                raise ValueError(f"{price_name} must be non-negative")

        if not isinstance(self.volume, (int, float)):
            raise ValueError("volume must be a number")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

        # Convert finite float volume to int. NaN/Inf volume is intentionally
        # left as-is so the validation layer can reject it with a clear error
        # rather than raising OverflowError/ValueError here.
        if isinstance(self.volume, float) and math.isfinite(self.volume):
            self.volume = int(self.volume)

    def to_dict(self) -> dict:
        """Convert candle to a dictionary representation."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candle":
        """
        Create a Candle from a dictionary.

        The timestamp can be a datetime object or an ISO format string. The
        timestamp must be timezone-aware: naive timestamps are rejected because
        their timezone cannot be determined safely. Aware timestamps (e.g.
        "2026-08-10T09:30:00-04:00" or "2026-08-10T09:30:00Z") are normalized to
        UTC.
        """
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            # "Z" is not accepted by fromisoformat on older Pythons; map it to
            # an explicit UTC offset before parsing.
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError(
                    "ISO timestamp must include a timezone offset "
                    "(e.g. 2026-08-10T09:30:00-04:00 or 2026-08-10T09:30:00Z); "
                    "naive timestamps are rejected because their timezone "
                    "cannot be determined safely"
                )

        return cls(
            symbol=data["symbol"],
            timestamp=timestamp,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
        )


@dataclass
class CandleCollection:
    """
    A collection of candles for a single symbol.

    Provides utility methods for working with multiple candles,
    including iteration, length, and basic statistics.

    Attributes:
        symbol: The ticker symbol for all candles in this collection (uppercase)
        candles: List of Candle objects
    """

    symbol: str
    candles: list[Candle] = field(default_factory=list)

    def __post_init__(self):
        """Normalize symbol and ensure all candles match it."""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("collection symbol must be a non-empty string")
        self.symbol = self.symbol.strip().upper()

        for candle in self.candles:
            if candle.symbol != self.symbol:
                raise ValueError(
                    f"Candle symbol '{candle.symbol}' does not match "
                    f"collection symbol '{self.symbol}'"
                )

    def __len__(self) -> int:
        """Return the number of candles."""
        return len(self.candles)

    def __iter__(self) -> Iterator[Candle]:
        """Iterate over candles."""
        return iter(self.candles)

    def __getitem__(self, index: int) -> Candle:
        """Get candle by index."""
        return self.candles[index]

    def add(self, candle: Candle) -> None:
        """Add a candle to the collection."""
        if candle.symbol != self.symbol:
            raise ValueError(
                f"Cannot add candle with symbol '{candle.symbol}' to "
                f"collection for '{self.symbol}'"
            )
        self.candles.append(candle)

    def sort_chronologically(self) -> None:
        """Sort candles by timestamp in ascending order (in-place)."""
        self.candles.sort(key=lambda c: c.timestamp)

    def get_timestamp_range(self) -> tuple[datetime | None, datetime | None]:
        """
        Get the earliest and latest timestamps in the collection.

        Returns:
            Tuple of (earliest, latest) timestamps, or (None, None) if empty.
        """
        if not self.candles:
            return None, None

        timestamps = [c.timestamp for c in self.candles]
        return min(timestamps), max(timestamps)

    def to_dict_list(self) -> list[dict]:
        """Convert all candles to a list of dictionaries."""
        return [c.to_dict() for c in self.candles]

    @classmethod
    def from_candles(cls, candles: list[Candle]) -> "CandleCollection":
        """
        Create a CandleCollection from a list of candles.

        The symbol is inferred from the first candle.
        """
        if not candles:
            raise ValueError("Cannot create collection from empty candle list")

        symbol = candles[0].symbol
        return cls(symbol=symbol, candles=candles)
