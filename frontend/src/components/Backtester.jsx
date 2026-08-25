import { useState } from "react";
import { getBacktest } from "../services/marketApi";
import "./Backtester.css";

function Backtester({ symbol }) {
  const [period, setPeriod] = useState("1y");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [stopLoss, setStopLoss] = useState(2);
  const [takeProfit, setTakeProfit] = useState(4);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runBacktest() {
    try {
      setLoading(true);
      setError(null);

      const response = await getBacktest(
        symbol,
        period,
        "1d",
        Number(initialCapital),
        Number(stopLoss),
        Number(takeProfit)
      );

      setResult(response.results);
    } catch (err) {
      setError(err.message || "Failed to run backtest");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const formatCurrency = (value) => {
    if (value === undefined || value === null) return "—";

    return `₹${Number(value).toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  const formatPercent = (value) => {
    if (value === undefined || value === null) return "—";

    const number = Number(value);

    return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
  };

  const tradeCount = Array.isArray(result?.trades)
    ? result.trades.length
    : result?.trades ?? "—";

  const equityCurve = Array.isArray(result?.equity_curve)
    ? result.equity_curve
    : [];

  const buildChartPoints = () => {
    if (equityCurve.length < 2) return "";

    const width = 1000;
    const height = 180;
    const padding = 8;

    const values = equityCurve.map((point) =>
      Number(point.equity)
    );

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    return equityCurve
      .map((point, index) => {
        const x =
          padding +
          (index / (equityCurve.length - 1)) *
            (width - padding * 2);

        const y =
          height -
          padding -
          ((Number(point.equity) - min) / range) *
            (height - padding * 2);

        return `${x},${y}`;
      })
      .join(" ");
  };

  return (
    <section className="backtester">
      <div className="backtester-header">
        <div>
          <div className="backtester-title-row">
            <h2>Backtester</h2>
            <span>{symbol} · NSE</span>
          </div>

          <p>
            Evaluate TradeSense strategy performance against
            historical market data.
          </p>
        </div>
      </div>

      <div className="backtester-grid">
        {/* CONFIGURATION */}

        <div className="backtest-panel">
          <div className="panel-label">CONFIGURATION</div>

          <div className="form-group">
            <label>TIME PERIOD</label>

            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
            >
              <option value="6mo">6 Months</option>
              <option value="1y">1 Year</option>
              <option value="2y">2 Years</option>
              <option value="5y">5 Years</option>
            </select>
          </div>

          <div className="form-group">
            <label>INITIAL CAPITAL</label>

            <input
              type="number"
              value={initialCapital}
              onChange={(e) =>
                setInitialCapital(e.target.value)
              }
              min="1000"
            />
          </div>

          <div className="form-group">
            <label>STOP LOSS (%)</label>

            <input
              type="number"
              value={stopLoss}
              onChange={(e) =>
                setStopLoss(e.target.value)
              }
              min="0"
              step="0.1"
            />
          </div>

          <div className="form-group">
            <label>TAKE PROFIT (%)</label>

            <input
              type="number"
              value={takeProfit}
              onChange={(e) =>
                setTakeProfit(e.target.value)
              }
              min="0"
              step="0.1"
            />
          </div>

          <button
            className="run-backtest"
            onClick={runBacktest}
            disabled={loading}
          >
            {loading ? "RUNNING..." : "RUN BACKTEST"}
          </button>

          {error && (
            <div className="backtest-error">
              {error}
            </div>
          )}
        </div>

        {/* RESULTS */}

        <div className="results-panel">
          <div className="panel-label">PERFORMANCE</div>

          {!result && !loading && (
            <div className="empty-results">
              <span>BACKTEST RESULTS</span>

              <h3>Run a strategy</h3>

              <p>
                Configure the parameters and run the backtest
                to see performance metrics.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-results">
              <span>ANALYZING</span>

              <h3>Running backtest...</h3>

              <p>
                Processing historical market data and evaluating
                strategy performance.
              </p>
            </div>
          )}

          {result && !loading && (
            <div className="results-grid">
              <div className="metric">
                <span>FINAL CAPITAL</span>

                <strong>
                  {formatCurrency(result.final_capital)}
                </strong>
              </div>

              <div className="metric">
                <span>STRATEGY RETURN</span>

                <strong
                  className={
                    Number(result.total_return_percent) >= 0
                      ? "metric-positive"
                      : "metric-negative"
                  }
                >
                  {formatPercent(
                    result.total_return_percent
                  )}
                </strong>
              </div>

              <div className="metric">
                <span>BUY & HOLD</span>

                <strong
                  className={
                    Number(
                      result.buy_and_hold_return_percent
                    ) >= 0
                      ? "metric-positive"
                      : "metric-negative"
                  }
                >
                  {formatPercent(
                    result.buy_and_hold_return_percent
                  )}
                </strong>
              </div>

              <div className="metric">
                <span>MAX DRAWDOWN</span>

                <strong className="metric-negative">
                  {formatPercent(
                    result.maximum_drawdown_percent
                  )}
                </strong>
              </div>

              <div className="metric">
                <span>TRADES</span>

                <strong>{tradeCount}</strong>
              </div>

              <div className="metric">
                <span>WIN RATE</span>

                <strong>
                  {result.win_rate_percent !== undefined &&
                  result.win_rate_percent !== null
                    ? `${Number(
                        result.win_rate_percent
                      ).toFixed(2)}%`
                    : "—"}
                </strong>
              </div>

              <div className="metric">
                <span>PROFIT FACTOR</span>

                <strong>
                  {result.profit_factor !== undefined &&
                  result.profit_factor !== null
                    ? Number(
                        result.profit_factor
                      ).toFixed(2)
                    : "—"}
                </strong>
              </div>

              {/* EQUITY CURVE — SAME GRID CELL SIZE */}

              <div className="metric equity-mini">
                <span>EQUITY CURVE</span>

                {equityCurve.length > 1 ? (
                  <div className="equity-mini-chart">
                    <svg
                      viewBox="0 0 1000 180"
                      preserveAspectRatio="none"
                      aria-label="Backtest equity curve"
                    >
                      <polyline
                        points={buildChartPoints()}
                        fill="none"
                        className="equity-line"
                      />
                    </svg>
                  </div>
                ) : (
                  <strong>—</strong>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default Backtester;
