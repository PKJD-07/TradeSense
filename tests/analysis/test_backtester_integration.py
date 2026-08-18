from src.analysis.robustness import analyze_trade_robustness


def test_backtester_output_works_with_robustness_analysis():
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