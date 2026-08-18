from datetime import datetime, timedelta

from backend.app.api.schemas.market import MarketCandle
from backend.app.backtesting.engine import backtest_strategy
from backend.app.signals.schema import SignalAction, TradingSignal


def create_test_candles(count: int = 52) -> list[MarketCandle]:
    candles = []

    start = datetime(2026, 1, 1)

    for i in range(count):
        price = 100.0 if i < 51 else 101.0

        candles.append(
            MarketCandle(
                symbol="TEST",
                timestamp=start + timedelta(days=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1000.0,
            )
        )

    return candles


def test_ml_trading_signal_drives_backtester():
    candles = create_test_candles()

    call_count = 0

    def ml_signal_provider(
        historical_data: list[MarketCandle],
    ) -> TradingSignal:

        nonlocal call_count
        call_count += 1

        if call_count == 1:
            action = SignalAction.BUY
            probabilities = (0.05, 0.05, 0.90)
        else:
            action = SignalAction.SELL
            probabilities = (0.05, 0.90, 0.05)

        return TradingSignal(
            timestamp=historical_data[-1].timestamp,
            symbol="TEST",
            action=action,
            confidence=max(probabilities),
            source="ML",
            model="test_model",
            probability_down=probabilities[0],
            probability_neutral=probabilities[1],
            probability_up=probabilities[2],
        )

    result = backtest_strategy(
        historical_candles=candles,
        initial_capital=100000.0,
        signal_provider=ml_signal_provider,
    )

    assert call_count == 2
    assert result["total_trades"] == 1
    assert result["winning_trades"] == 1
    assert result["losing_trades"] == 0
    assert result["trades"][0]["exit_reason"] == "SIGNAL"
    assert result["trades"][0]["profit"] > 0


def test_backtester_output_works_with_robustness_analysis():
    from src.analysis.robustness import analyze_trade_robustness

    trades = [
        {
            "entry_date": "2026-01-01",
            "entry_price": 100,
            "exit_date": "2026-01-03",
            "exit_price": 110,
            "profit": 1000,
            "exit_reason": "SIGNAL",
        },
        {
            "entry_date": "2026-01-04",
            "entry_price": 110,
            "exit_date": "2026-01-06",
            "exit_price": 105,
            "profit": -500,
            "exit_reason": "STOP_LOSS",
        },
    ]

    equity_curve = [
        {"date": "2026-01-01", "equity": 100000},
        {"date": "2026-01-02", "equity": 100500},
        {"date": "2026-01-03", "equity": 101000},
        {"date": "2026-01-04", "equity": 100750},
        {"date": "2026-01-05", "equity": 100500},
        {"date": "2026-01-06", "equity": 100500},
    ]

    result = analyze_trade_robustness(
        trades,
        equity_curve,
    )

    assert result["total_trades"] == 2
    assert result["win_rate"] == 0.5
    assert result["profit_factor"] == 2.0
    assert result["average_trade"] == 250
    assert result["maximum_drawdown"] < 0