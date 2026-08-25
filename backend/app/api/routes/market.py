from datetime import datetime, time
from zoneinfo import ZoneInfo

from fastapi import APIRouter
import yfinance as yf

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


@router.get("/market/overview")
def get_market_overview():
    """
    Returns major Indian and US market indices
    used by the dashboard ticker.
    """

    indices = [
        {
            "symbol": "^NSEI",
            "name": "NIFTY 50",
        },
        {
            "symbol": "^BSESN",
            "name": "SENSEX",
        },
        {
            "symbol": "^GSPC",
            "name": "S&P 500",
        },
        {
            "symbol": "^IXIC",
            "name": "NASDAQ",
        },
        {
            "symbol": "^DJI",
            "name": "DOW",
        },
    ]

    markets = []

    for index in indices:
        try:
            ticker = yf.Ticker(index["symbol"])

            history = ticker.history(
                period="5d",
                interval="1d",
                auto_adjust=False,
            )

            if history.empty or len(history) < 2:
                print(
                    f"No data available for {index['name']}"
                )
                continue

            latest = float(history["Close"].iloc[-1])
            previous = float(history["Close"].iloc[-2])

            change_value = latest - previous

            change_percent = (
                (change_value / previous) * 100
                if previous != 0
                else 0
            )

            markets.append(
                {
                    "symbol": index["symbol"],
                    "name": index["name"],
                    "price": round(latest, 2),
                    "change_value": round(change_value, 2),
                    "change": round(change_percent, 2),
                }
            )

        except Exception as exc:
            print(
                f"Unable to fetch {index['name']} market data: {exc}"
            )

    # =========================================
    # NSE MARKET STATUS
    # =========================================

    india_timezone = ZoneInfo("Asia/Kolkata")
    now = datetime.now(india_timezone)

    market_open_time = time(9, 15)
    market_close_time = time(15, 30)

    is_weekday = now.weekday() < 5

    is_market_hours = (
        market_open_time <= now.time() < market_close_time
    )

    market_is_open = is_weekday and is_market_hours

    status = "OPEN" if market_is_open else "CLOSED"

    return {
        "markets": markets,
        "exchange": "NSE INDIA",
        "status": status,
        "timestamp": now.isoformat(),
    }