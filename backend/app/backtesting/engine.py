from backend.app.api.schemas.market import MarketCandle
from backend.app.analysis.service import analyze_market
from backend.app.analysis.signals import generate_signal
from backend.app.backtesting.risk import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit,
)


def backtest_strategy(
    historical_candles: list[MarketCandle],
    initial_capital: float = 100000.0,
    stop_loss_percent: float = 2.0,
    take_profit_percent: float = 4.0,
) -> dict:

    if len(historical_candles) < 50:
        raise ValueError("Not enough historical data for backtesting")

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    entry_price = None
    entry_date = None
    position_size = 0

    for i in range(50, len(historical_candles)):

        historical_data = historical_candles[: i + 1]
        current_candle = historical_candles[i]

        analysis = analyze_market(historical_data)
        signal = generate_signal(analysis)

        current_price = current_candle.close
        current_date = current_candle.timestamp

        # --------------------------------------------------
        # Manage existing position
        # --------------------------------------------------

        if position == "LONG":

            stop_loss = calculate_stop_loss(
                entry_price,
                stop_loss_percent,
            )

            take_profit = calculate_take_profit(
                entry_price,
                take_profit_percent,
            )

            # Stop loss hit
            if current_candle.low <= stop_loss:

                exit_price = stop_loss

                profit = (
                    exit_price - entry_price
                ) * position_size

                capital += profit

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": exit_price,
                    "profit": profit,
                    "exit_reason": "STOP_LOSS",
                })

                position = None
                entry_price = None
                position_size = 0

            # Take profit hit
            elif current_candle.high >= take_profit:

                exit_price = take_profit

                profit = (
                    exit_price - entry_price
                ) * position_size

                capital += profit

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": exit_price,
                    "profit": profit,
                    "exit_reason": "TAKE_PROFIT",
                })

                position = None
                entry_price = None
                position_size = 0

            # Signal says SELL
            elif signal["decision"] == "SELL":

                exit_price = current_price

                profit = (
                    exit_price - entry_price
                ) * position_size

                capital += profit

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": exit_price,
                    "profit": profit,
                    "exit_reason": "SIGNAL",
                })

                position = None
                entry_price = None
                position_size = 0

        # --------------------------------------------------
        # Open new position
        # --------------------------------------------------

        if position is None and signal["decision"] == "BUY":

            entry_price = current_price
            entry_date = current_date

            position_size = calculate_position_size(
                capital=capital,
                entry_price=entry_price,
                stop_loss_percent=stop_loss_percent,
            )

            position = "LONG"

        # --------------------------------------------------
        # Record equity
        # --------------------------------------------------

        current_equity = capital

        if position == "LONG":

            unrealized_profit = (
                current_price - entry_price
            ) * position_size

            current_equity += unrealized_profit

        equity_curve.append({
            "date": current_date,
            "equity": current_equity,
        })

    # ------------------------------------------------------
    # Close remaining position at final price
    # ------------------------------------------------------

    if position == "LONG":

        final_candle = historical_candles[-1]

        exit_price = final_candle.close

        profit = (
            exit_price - entry_price
        ) * position_size

        capital += profit

        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": final_candle.timestamp,
            "exit_price": exit_price,
            "profit": profit,
            "exit_reason": "END_OF_DATA",
        })

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    winning_trades = [
        trade for trade in trades
        if trade["profit"] > 0
    ]

    losing_trades = [
        trade for trade in trades
        if trade["profit"] < 0
    ]

    total_trades = len(trades)

    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    total_profit = sum(
        trade["profit"] for trade in winning_trades
    )

    total_loss = sum(
        trade["profit"] for trade in losing_trades
    )

    total_return_percent = (
        (capital - initial_capital)
        / initial_capital
        * 100
    )

    win_rate = (
        win_count / total_trades * 100
        if total_trades > 0
        else 0
    )

    profit_factor = (
        total_profit / abs(total_loss)
        if total_loss != 0
        else float("inf")
    )

    average_win = (
        total_profit / win_count
        if win_count > 0
        else 0
    )

    average_loss = (
        total_loss / loss_count
        if loss_count > 0
        else 0
    )

    largest_win = (
        max(
            trade["profit"]
            for trade in winning_trades
        )
        if winning_trades
        else 0
    )

    largest_loss = (
        min(
            trade["profit"]
            for trade in losing_trades
        )
        if losing_trades
        else 0
    )

    # ------------------------------------------------------
    # Buy & hold
    # ------------------------------------------------------

    first_price = historical_candles[0].close
    last_price = historical_candles[-1].close

    buy_and_hold_return_percent = (
        (last_price - first_price)
        / first_price
        * 100
    )

    # ------------------------------------------------------
    # Maximum drawdown
    # ------------------------------------------------------

    peak = initial_capital
    maximum_drawdown = 0

    for point in equity_curve:

        equity = point["equity"]

        if equity > peak:
            peak = equity

        drawdown = (
            (equity - peak)
            / peak
            * 100
        )

        if drawdown < maximum_drawdown:
            maximum_drawdown = drawdown

    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_percent": round(
            total_return_percent, 2
        ),
        "buy_and_hold_return_percent": round(
            buy_and_hold_return_percent, 2
        ),
        "maximum_drawdown_percent": round(
            maximum_drawdown, 2
        ),
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate_percent": round(
            win_rate, 2
        ),
        "profit_factor": round(
            profit_factor, 2
        ),
        "average_winning_trade": round(
            average_win, 2
        ),
        "average_losing_trade": round(
            average_loss, 2
        ),
        "largest_winning_trade": round(
            largest_win, 2
        ),
        "largest_losing_trade": round(
            largest_loss, 2
        ),
        "trades": trades,
        "equity_curve": equity_curve,
    }