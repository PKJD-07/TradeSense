"""
Tests for the OHLCV candle models.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.data.models import Candle, CandleCollection
from tests.data.fixtures import (
    make_valid_candle,
    make_valid_candles,
    make_candle_with_invalid_timestamp,
)


class TestCandle:
    """Tests for the Candle dataclass."""

    def test_valid_candle_creation(self):
        """Test creating a valid candle."""
        candle = make_valid_candle()

        assert candle.symbol == "AAPL"
        assert candle.open == 210.15
        assert candle.high == 212.30
        assert candle.low == 209.80
        assert candle.close == 211.75
        assert candle.volume == 1523400

    def test_utc_timestamp_remains_utc(self):
        """Test that a UTC timestamp is stored unchanged (normalization is a no-op)."""
        ts = datetime(2026, 8, 10, 14, 30, 0, tzinfo=timezone.utc)
        candle = make_valid_candle(timestamp=ts)

        assert candle.timestamp == ts
        assert candle.timestamp.tzinfo == timezone.utc

    def test_aware_timestamp_normalized_to_utc(self):
        """Test that a non-UTC aware timestamp is converted to UTC."""
        ts = datetime(2026, 8, 10, 14, 30, 0, tzinfo=timezone(timedelta(hours=-4)))
        candle = make_valid_candle(timestamp=ts)

        assert candle.timestamp == datetime(2026, 8, 10, 18, 30, 0, tzinfo=timezone.utc)
        assert candle.timestamp.tzinfo == timezone.utc
        assert candle.timestamp.utcoffset() == timedelta(0)

    def test_candle_to_dict(self):
        """Test converting a candle to a dictionary."""
        candle = make_valid_candle()
        result = candle.to_dict()

        assert isinstance(result, dict)
        assert result["symbol"] == "AAPL"
        assert result["open"] == 210.15
        assert result["high"] == 212.30
        assert result["low"] == 209.80
        assert result["close"] == 211.75
        assert result["volume"] == 1523400
        assert "timestamp" in result

    def test_candle_from_dict(self):
        """Test creating a candle from a dictionary."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T00:00:00+00:00",
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        candle = Candle.from_dict(data)

        assert candle.symbol == "MSFT"
        assert candle.open == 380.0
        assert candle.high == 385.0
        assert candle.low == 378.0
        assert candle.close == 382.0
        assert candle.volume == 2000000

    def test_candle_from_dict_with_datetime_object(self):
        """Test creating a candle from a dict with datetime object."""
        ts = datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            "symbol": "GOOGL",
            "timestamp": ts,
            "open": 140.0,
            "high": 142.0,
            "low": 139.0,
            "close": 141.0,
            "volume": 800000,
        }

        candle = Candle.from_dict(data)

        assert candle.timestamp == ts

    def test_candle_equality(self):
        """Test that two candles with same values are equal."""
        ts = datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc)

        candle1 = Candle(
            symbol="AAPL",
            timestamp=ts,
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        candle2 = Candle(
            symbol="AAPL",
            timestamp=ts,
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        assert candle1 == candle2

    def test_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol must be a non-empty string"):
            Candle(
                symbol="",
                timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                open=210.0,
                high=212.0,
                low=209.0,
                close=211.0,
                volume=1000000,
            )

    def test_negative_price_raises_error(self):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Candle(
                symbol="AAPL",
                timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                open=-210.0,
                high=212.0,
                low=209.0,
                close=211.0,
                volume=1000000,
            )

    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError."""
        with pytest.raises(ValueError, match="volume must be non-negative"):
            Candle(
                symbol="AAPL",
                timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
                open=210.0,
                high=212.0,
                low=209.0,
                close=211.0,
                volume=-1000000,
            )

    def test_float_volume_converted_to_int(self):
        """Test that float volume is converted to int."""
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000.5,
        )

        assert candle.volume == 1000000
        assert isinstance(candle.volume, int)

    def test_zero_price_is_allowed(self):
        """Test that zero price is technically allowed (validation catches it)."""
        # The model allows non-negative (>= 0), validation catches <= 0 as invalid
        candle = Candle(
            symbol="AAPL",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=1000000,
        )

        assert candle.open == 0.0

    def test_symbol_normalized_to_uppercase(self):
        """Test that symbols are normalized to uppercase at the model boundary."""
        candle = Candle(
            symbol="aapl",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        assert candle.symbol == "AAPL"

    def test_symbol_whitespace_stripped(self):
        """Test that surrounding whitespace is stripped from symbols."""
        candle = Candle(
            symbol="  msft  ",
            timestamp=datetime(2026, 8, 10, 0, 0, 0, tzinfo=timezone.utc),
            open=210.0,
            high=212.0,
            low=209.0,
            close=211.0,
            volume=1000000,
        )

        assert candle.symbol == "MSFT"

    def test_naive_timestamp_rejected(self):
        """Test that a naive (timezone-less) timestamp is rejected."""
        with pytest.raises(ValueError, match="timezone-aware"):
            Candle(
                symbol="AAPL",
                timestamp=datetime(2026, 8, 10, 0, 0, 0),  # naive
                open=210.0,
                high=212.0,
                low=209.0,
                close=211.0,
                volume=1000000,
            )

    def test_invalid_timestamp_type_rejected(self):
        """Test that a non-datetime timestamp is rejected at construction."""
        data = make_candle_with_invalid_timestamp()

        with pytest.raises(ValueError, match="timestamp must be a datetime object"):
            Candle(**data)

    def test_from_dict_naive_timestamp_rejected(self):
        """Test that a naive ISO timestamp (no offset) is rejected."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T09:30:00",  # no timezone offset
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        with pytest.raises(ValueError, match="timezone"):
            Candle.from_dict(data)

    def test_from_dict_utc_offset_iso_accepted(self):
        """Test that an explicit UTC (+00:00) ISO timestamp is accepted."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T09:30:00+00:00",
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        candle = Candle.from_dict(data)

        assert candle.timestamp == datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone.utc)
        assert candle.timestamp.tzinfo == timezone.utc

    def test_from_dict_z_suffix_iso_accepted(self):
        """Test that a 'Z'-suffixed ISO timestamp is accepted as UTC."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T09:30:00Z",
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        candle = Candle.from_dict(data)

        assert candle.timestamp == datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone.utc)
        assert candle.timestamp.tzinfo == timezone.utc

    def test_from_dict_positive_offset_normalized_to_utc(self):
        """Test that a +05:30 ISO timestamp is normalized to UTC."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T15:00:00+05:30",
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        candle = Candle.from_dict(data)

        assert candle.timestamp == datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone.utc)
        assert candle.timestamp.tzinfo == timezone.utc

    def test_from_dict_negative_offset_normalized_to_utc(self):
        """Test that a -04:00 ISO timestamp is normalized to UTC."""
        data = {
            "symbol": "MSFT",
            "timestamp": "2026-08-10T09:30:00-04:00",
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        candle = Candle.from_dict(data)

        assert candle.timestamp == datetime(2026, 8, 10, 13, 30, 0, tzinfo=timezone.utc)
        assert candle.timestamp.tzinfo == timezone.utc

    def test_from_dict_naive_datetime_object_rejected(self):
        """Test that a naive datetime object in from_dict is rejected."""
        data = {
            "symbol": "MSFT",
            "timestamp": datetime(2026, 8, 10, 9, 30, 0),  # naive object
            "open": 380.0,
            "high": 385.0,
            "low": 378.0,
            "close": 382.0,
            "volume": 2000000,
        }

        with pytest.raises(ValueError, match="timezone"):
            Candle.from_dict(data)

    def test_internal_invariant_is_aware_utc(self):
        """Test the canonical invariant: stored timestamps are aware and in UTC."""
        ts = datetime(2026, 8, 10, 9, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        candle = make_valid_candle(timestamp=ts)

        assert candle.timestamp.tzinfo is not None
        assert candle.timestamp.utcoffset() == timedelta(0)
        assert candle.timestamp.tzinfo == timezone.utc


class TestCandleCollection:
    """Tests for the CandleCollection class."""

    def test_empty_collection(self):
        """Test creating an empty collection."""
        collection = CandleCollection(symbol="AAPL")

        assert len(collection) == 0
        assert collection.symbol == "AAPL"

    def test_collection_symbol_normalized(self):
        """Test that collection symbols are normalized to uppercase."""
        collection = CandleCollection(
            symbol="aapl",
            candles=[make_valid_candle()],
        )

        assert collection.symbol == "AAPL"

    def test_collection_with_candles(self):
        """Test creating a collection with candles."""
        candles = make_valid_candles(count=3)
        collection = CandleCollection.from_candles(candles)

        assert len(collection) == 3
        assert collection.symbol == "AAPL"

    def test_collection_iteration(self):
        """Test iterating over candles in a collection."""
        candles = make_valid_candles(count=3)
        collection = CandleCollection.from_candles(candles)

        count = 0
        for candle in collection:
            assert isinstance(candle, Candle)
            count += 1

        assert count == 3

    def test_collection_indexing(self):
        """Test accessing candles by index."""
        candles = make_valid_candles(count=3)
        collection = CandleCollection.from_candles(candles)

        assert collection[0] == candles[0]
        assert collection[1] == candles[1]
        assert collection[2] == candles[2]

    def test_add_candle_to_collection(self):
        """Test adding a candle to a collection."""
        collection = CandleCollection(symbol="AAPL")
        candle = make_valid_candle()

        collection.add(candle)

        assert len(collection) == 1
        assert collection[0] == candle

    def test_add_wrong_symbol_raises_error(self):
        """Test that adding a candle with wrong symbol raises error."""
        collection = CandleCollection(symbol="AAPL")
        wrong_candle = make_valid_candle(symbol="MSFT")

        with pytest.raises(ValueError, match="Cannot add candle with symbol"):
            collection.add(wrong_candle)

    def test_sort_chronologically(self):
        """Test sorting collection chronologically."""
        candles = [
            make_valid_candle(timestamp=datetime(2026, 8, 12, tzinfo=timezone.utc)),
            make_valid_candle(timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc)),
            make_valid_candle(timestamp=datetime(2026, 8, 11, tzinfo=timezone.utc)),
        ]
        collection = CandleCollection(symbol="AAPL", candles=candles)

        collection.sort_chronologically()

        assert collection[0].timestamp.day == 10
        assert collection[1].timestamp.day == 11
        assert collection[2].timestamp.day == 12

    def test_get_timestamp_range(self):
        """Test getting timestamp range of a collection."""
        candles = [
            make_valid_candle(timestamp=datetime(2026, 8, 15, tzinfo=timezone.utc)),
            make_valid_candle(timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc)),
            make_valid_candle(timestamp=datetime(2026, 8, 12, tzinfo=timezone.utc)),
        ]
        collection = CandleCollection(symbol="AAPL", candles=candles)

        earliest, latest = collection.get_timestamp_range()

        assert earliest.day == 10
        assert latest.day == 15

    def test_timestamp_range_empty_collection(self):
        """Test timestamp range on empty collection."""
        collection = CandleCollection(symbol="AAPL")

        earliest, latest = collection.get_timestamp_range()

        assert earliest is None
        assert latest is None

    def test_to_dict_list(self):
        """Test converting collection to list of dicts."""
        candles = make_valid_candles(count=2)
        collection = CandleCollection.from_candles(candles)

        result = collection.to_dict_list()

        assert len(result) == 2
        assert all(isinstance(d, dict) for d in result)
