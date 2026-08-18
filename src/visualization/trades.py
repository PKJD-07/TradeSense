"""Trade and equity-curve visualizations for TradeSense."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity_curve: list[dict],
    *,
    title: str = "Strategy Equity Curve",
):
    """Plot portfolio equity over time.

    Args:
        equity_curve: List of dictionaries containing ``date`` and ``equity``.

    Returns:
        Matplotlib Figure.
    """
    if not equity_curve:
        raise ValueError("equity_curve cannot be empty")

    df = pd.DataFrame(equity_curve)

    required_columns = {"date", "equity"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "equity_curve must contain 'date' and 'equity' columns"
        )

    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df["date"].dt.to_pydatetime(),
        df["equity"],
        label="Equity",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_trades(
    candles: list,
    trades: list[dict],
    *,
    title: str = "Strategy Trades",
):
    """Plot price data with trade entry and exit markers.

    Args:
        candles: List of MarketCandle-like objects containing timestamp
            and close price.
        trades: List of completed trade dictionaries from the backtester.

    Returns:
        Matplotlib Figure.
    """
    if not candles:
        raise ValueError("candles cannot be empty")

    candle_dates = [
        pd.to_datetime(candle.timestamp).to_pydatetime()
        for candle in candles
    ]
    close_prices = [candle.close for candle in candles]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        candle_dates,
        close_prices,
        label="Close",
    )

    for trade in trades:
        entry_date = (
            pd.to_datetime(trade["entry_date"])
            .to_pydatetime()
        )
        entry_price = trade["entry_price"]

        exit_date = (
            pd.to_datetime(trade["exit_date"])
            .to_pydatetime()
        )
        exit_price = trade["exit_price"]

        ax.scatter(
            entry_date,
            entry_price,
            marker="^",
            s=80,
            label="Buy",
        )

        ax.scatter(
            exit_date,
            exit_price,
            marker="v",
            s=80,
            label="Sell",
        )

    # Remove duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig