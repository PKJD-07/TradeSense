"""Strategy robustness analysis for TradeSense."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def analyze_trade_robustness(
    trades: list[dict],
    equity_curve: list[dict],
) -> dict[str, Any]:
    """Calculate robustness metrics from backtest trades and equity.

    Args:
        trades: Completed trades from the backtester.
        equity_curve: Equity history from the backtester.

    Returns:
        Dictionary containing trade-quality and equity-curve robustness metrics.
    """
    if not equity_curve:
        raise ValueError("equity_curve cannot be empty")

    equity = pd.DataFrame(equity_curve)

    if not {"date", "equity"}.issubset(equity.columns):
        raise ValueError(
            "equity_curve must contain 'date' and 'equity' columns"
        )

    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)

    equity_values = equity["equity"].astype(float)

    # --------------------------------------------------
    # Returns
    # --------------------------------------------------

    returns = equity_values.pct_change().dropna()

    if len(returns) > 0 and returns.std(ddof=1) > 0:
        sharpe_ratio = (
            returns.mean() / returns.std(ddof=1)
        ) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # --------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------

    running_peak = equity_values.cummax()

    drawdowns = (
        (equity_values - running_peak)
        / running_peak
    )

    maximum_drawdown = float(drawdowns.min())

    # --------------------------------------------------
    # Recovery factor
    # --------------------------------------------------

    initial_equity = float(equity_values.iloc[0])
    final_equity = float(equity_values.iloc[-1])

    total_return = final_equity - initial_equity

    recovery_factor = (
        total_return / abs(initial_equity * maximum_drawdown)
        if maximum_drawdown < 0
        else float("inf")
    )

    # --------------------------------------------------
    # Trade statistics
    # --------------------------------------------------

    profits = np.array(
        [float(trade["profit"]) for trade in trades],
        dtype=float,
    )

    winning = profits[profits > 0]
    losing = profits[profits < 0]

    total_trades = len(profits)

    win_rate = (
        len(winning) / total_trades
        if total_trades > 0
        else 0.0
    )

    profit_factor = (
        winning.sum() / abs(losing.sum())
        if len(losing) > 0
        else float("inf")
    )

    average_trade = (
        profits.mean()
        if total_trades > 0
        else 0.0
    )

    # --------------------------------------------------
    # Consecutive losses
    # --------------------------------------------------

    max_consecutive_losses = 0
    current_losses = 0

    for profit in profits:
        if profit < 0:
            current_losses += 1
            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses,
            )
        else:
            current_losses = 0

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "average_trade": round(float(average_trade), 2),
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "maximum_drawdown": round(maximum_drawdown, 4),
        "recovery_factor": round(float(recovery_factor), 4),
        "max_consecutive_losses": max_consecutive_losses,
    }