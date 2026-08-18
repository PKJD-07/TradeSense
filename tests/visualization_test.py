from src.visualization.trades import plot_equity_curve, plot_trades


def test_plot_equity_curve():
    equity_curve = [
        {"date": "2026-01-01", "equity": 100000},
        {"date": "2026-01-02", "equity": 101000},
        {"date": "2026-01-03", "equity": 99500},
    ]

    fig = plot_equity_curve(equity_curve)

    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_trades():
    class Candle:
        def __init__(self, timestamp, close):
            self.timestamp = timestamp
            self.close = close

    candles = [
        Candle("2026-01-01", 100),
        Candle("2026-01-02", 105),
        Candle("2026-01-03", 102),
    ]

    trades = [
        {
            "entry_date": "2026-01-01",
            "entry_price": 100,
            "exit_date": "2026-01-03",
            "exit_price": 102,
            "profit": 2,
            "exit_reason": "SIGNAL",
        }
    ]

    fig = plot_trades(candles, trades)

    assert fig is not None
    assert len(fig.axes) == 1