"""Smoke tests for the plotting helpers (Agg backend, save-to-file)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from src.analysis import plots
from src.analysis.returns import log_returns
from src.analysis.statistics import autocorrelation, cross_asset_correlation
from src.analysis.volatility import rolling_volatility
from src.analysis.drawdown import drawdown_series
from tests.analysis.fixtures import make_candle_df, make_long_dataframe


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "func,key",
    [
        ("plot_price", "close"),
        ("plot_log_price", "close"),
        ("plot_returns", "returns"),
        ("plot_return_distribution", "returns"),
        ("plot_rolling_volatility", "rolling_vol"),
        ("plot_volume", "volume"),
        ("plot_drawdown", "dd"),
        ("plot_acf", "acf"),
    ],
)
def test_simple_plot_saves_file(tmp_path, func, key):
    df = make_candle_df(n=60, seed=1)
    ret = log_returns(df["close"])
    values = {
        "close": df["close"],
        "returns": ret,
        "rolling_vol": rolling_volatility(ret, window=21, annualize=True),
        "volume": df["volume"],
        "dd": drawdown_series(df["close"]),
        "acf": autocorrelation(ret, lags=10),
    }
    out_path = tmp_path / f"{func}.png"
    fig = getattr(plots, func)(**{key: values[key]}, out_path=out_path)
    assert out_path.exists()
    assert fig is not None


def test_plot_correlation_heatmap(tmp_path):
    long_df = make_long_dataframe()
    wide = long_df.pivot(index="timestamp", columns="symbol", values="close")
    corr = cross_asset_correlation(wide)
    out_path = tmp_path / "corr.png"
    fig = plots.plot_correlation_heatmap(corr, out_path=out_path)
    assert out_path.exists()
    assert fig is not None
