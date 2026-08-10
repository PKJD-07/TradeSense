"""Visualization helpers for the EDA phase.

Each function accepts an optional matplotlib ``ax`` and an optional ``out_path``
and returns the matplotlib Figure it produced. When ``out_path`` is provided the
figure is saved there and closed.

The non-interactive ``Agg`` backend is forced so figures can be produced in
scripts and tests without opening a window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _new_figure(ax):
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def _finish(fig, out_path):
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_price(close, title="Adjusted Close Price", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.plot(close.index, close.to_numpy(), color="#1f77b4", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_log_price(close, title="Log Adjusted Close Price", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.plot(close.index, np.log(close).to_numpy(), color="#2ca02c", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Price")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_returns(returns, title="Daily Log Returns", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.plot(returns.index, returns.to_numpy(), color="#9467bd", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_return_distribution(returns, bins=60, title="Return Distribution", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    values = returns.dropna().to_numpy()
    ax.hist(values, bins=bins, color="#ff7f0e", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_rolling_volatility(rolling_vol, title="Rolling Volatility (annualized)", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.plot(rolling_vol.index, rolling_vol.to_numpy(), color="#d62728", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_volume(volume, title="Volume", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.bar(volume.index, volume.to_numpy(), width=1.0, color="#8c564b", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_drawdown(dd, title="Drawdown (underwater curve)", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="#e377c2", alpha=0.5)
    ax.plot(dd.index, dd.to_numpy(), color="#e377c2", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_acf(acf, title="Return Autocorrelation (ACF)", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    lags = acf.index.to_numpy(dtype=int)
    ax.bar(lags, acf.to_numpy(), width=0.8, color="#17becf", edgecolor="white")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(lags)
    ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


def plot_correlation_heatmap(corr, title="Cross-Asset Return Correlation", ax=None, out_path=None):
    fig, ax = _new_figure(ax)
    data = corr.to_numpy()
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    return _finish(fig, out_path)
