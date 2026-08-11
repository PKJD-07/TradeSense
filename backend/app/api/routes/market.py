from fastapi import APIRouter

from backend.app.api.schemas.market import (
    MarketCandle,
    HistoricalMarketData,
)
from backend.app.data.validator import validate_historical_data
from backend.app.data.fetcher import fetch_historical_data
from backend.app.analysis.service import analyze_market
from backend.app.analysis.signals import generate_signal


router = APIRouter()


@router.post("/market/candle", response_model=MarketCandle)
def validate_candle(candle: MarketCandle):
    return candle


@router.post("/market/historical")
def validate_historical_market_data(data: HistoricalMarketData):
    validate_historical_data(data.candles)

    return {
        "message": "Historical market data is valid",
        "candles": len(data.candles),
        "symbol": data.candles[0].symbol,
    }


@router.get("/market/historical/{symbol}")
def get_historical_market_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
):
    candles = fetch_historical_data(
        symbol=symbol.upper(),
        period=period,
        interval=interval,
    )

    validate_historical_data(candles)

    return {
        "message": "NSE historical market data fetched and validated successfully",
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles": candles,
    }


@router.get("/market/analyze/{symbol}")
def analyze_stock(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
):
    candles = fetch_historical_data(
        symbol=symbol.upper(),
        period=period,
        interval=interval,
    )

    validate_historical_data(candles)

    analysis = analyze_market(candles)

    return analysis


@router.get("/market/signal/{symbol}")
def get_market_signal(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
):
    candles = fetch_historical_data(
        symbol=symbol.upper(),
        period=period,
        interval=interval,
    )

    validate_historical_data(candles)

    analysis = analyze_market(candles)
    signal = generate_signal(analysis)

    return {
        "symbol": symbol.upper(),
        "analysis": analysis,
        "signal": signal,
    }