from fastapi import APIRouter, HTTPException

from backend.app.data.fetcher import fetch_historical_data
from backend.app.data.validator import validate_historical_data
from backend.app.backtesting.engine import backtest_strategy


router = APIRouter()


@router.get("/backtest/{symbol}")
def run_backtest(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    initial_capital: float = 100000.0,
    stop_loss_percent: float = 2.0,
    take_profit_percent: float = 4.0,
):
    try:
        candles = fetch_historical_data(
            symbol=symbol.upper(),
            period=period,
            interval=interval,
        )

        validate_historical_data(candles)

        result = backtest_strategy(
            historical_candles=candles,
            initial_capital=initial_capital,
            stop_loss_percent=stop_loss_percent,
            take_profit_percent=take_profit_percent,
        )

        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "results": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {str(exc)}",
        )