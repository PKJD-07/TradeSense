const API_BASE_URL = "https://tradesenseapp-backend.vercel.app";

export async function getHistoricalData(
  symbol,
  period = "1mo",
  interval = "1d"
) {
  const response = await fetch(
    `${API_BASE_URL}/market/historical/${encodeURIComponent(
      symbol
    )}?period=${encodeURIComponent(period)}&interval=${encodeURIComponent(
      interval
    )}`
  );

  if (!response.ok) {
    const error = await response.json().catch(() => null);

    throw new Error(
      error?.detail || `Failed to fetch ${symbol} market data`
    );
  }

  return response.json();
}

export async function getMarketSignal(
  symbol,
  period = "3mo",
  interval = "1d"
) {
  const response = await fetch(
    `${API_BASE_URL}/market/signal/${encodeURIComponent(
      symbol
    )}?period=${encodeURIComponent(period)}&interval=${encodeURIComponent(
      interval
    )}`
  );

  if (!response.ok) {
    const error = await response.json().catch(() => null);

    throw new Error(
      error?.detail || `Failed to fetch ${symbol} market signal`
    );
  }

  return response.json();
}

export async function getBacktest(
  symbol,
  period = "1y",
  interval = "1d",
  initialCapital = 100000,
  stopLossPercent = 2,
  takeProfitPercent = 4
) {
  const params = new URLSearchParams({
    period,
    interval,
    initial_capital: initialCapital,
    stop_loss_percent: stopLossPercent,
    take_profit_percent: takeProfitPercent,
  });

  const response = await fetch(
    `${API_BASE_URL}/backtest/${encodeURIComponent(symbol)}?${params.toString()}`
  );

  if (!response.ok) {
    const error = await response.json().catch(() => null);

    throw new Error(
      error?.detail || `Failed to run backtest for ${symbol}`
    );
  }

  return response.json();
}