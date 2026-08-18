"""Tests for strategy robustness analysis."""

from src.analysis.robustness import analyze_trade_robustness


def test_robustness_metrics():
    trades = [
        {"profit": 100},
        {"profit": -50},
        {"profit": 200},
        {"profit": -75},
    ]

    equity_curve = [
        {"date": "2026-01-01", "equity": 100000},
        {"date": "2026-01-02", "equity": 100100},
        {"date": "2026-01-03", "equity": 100050},
        {"date": "2026-01-04", "equity": 100250},
        {"date": "2026-01-05", "equity": 100175},
    ]

    result = analyze_trade_robustness(
        trades,
        equity_curve,
    )

    assert result["total_trades"] == 4
    assert result["win_rate"] == 0.5
    assert result["profit_factor"] == 2.4
    assert result["average_trade"] == 43.75
    assert result["max_consecutive_losses"] == 1


def test_empty_trades():
    equity_curve = [
        {"date": "2026-01-01", "equity": 100000},
        {"date": "2026-01-02", "equity": 101000},
    ]

    result = analyze_trade_robustness(
        [],
        equity_curve,
    )

    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["average_trade"] == 0.0


def test_empty_equity_curve():
    trades = [{"profit": 100}]

    try:
        analyze_trade_robustness(trades, [])
        assert False
    except ValueError:
        pass