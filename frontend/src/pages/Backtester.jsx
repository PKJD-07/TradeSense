import { useState } from "react";
import { getBacktest } from "../services/marketApi";
import "./Backtester.css";

function Backtester() {
  const [symbol, setSymbol] = useState("RELIANCE");
  const [period, setPeriod] = useState("1y");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [stopLoss, setStopLoss] = useState(2);
  const [takeProfit, setTakeProfit] = useState(4);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runBacktest = async () => {
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

      setResult(response);
    } catch (err) {
      setError(err.message || "Unable to run backtest");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const results = result?.results;

  const formatCurrency = (value) =>
    typeof value === "number"
      ? `₹${value.toLocaleString("en-IN", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}`
      : "—";

  const formatPercent = (value) =>
    typeof value === "number"
      ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`
      : "—";

  return (
    <main className="backtester-page">
      <section className="backtester-header">
        <div>
          <div className="backtester-eyebrow">STRATEGY ANALYSIS</div>

          <h1>Backtester</h1>

          <p>
            Test the TradeSense strategy against historical market data.
          </p>
        </div>

        <div className="backtester-status">
          <span />
          HISTORICAL DATA
        </div>
      </section>

      <section className="backtester-panel">
        <div className="panel-heading">
          <div>
            <span>CONFIGURATION</span>
            <h2>Backtest parameters</h2>
          </div>
        </div>

        <div className="backtester-form">
          <label>
            <span>STOCK</span>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="RELIANCE"
            />
          </label>

          <label>
            <span>PERIOD</span>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
            >
              <option value="6mo">6 Months</option>
              <option value="1y">1 Year</option>
              <option value="2y">2 Years</option>
              <option value="5y">5 Years</option>
            </select>
          </label>

          <label>
            <span>INITIAL CAPITAL</span>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(e.target.value)}
              min="1000"
            />
          </label>

          <label>
            <span>STOP LOSS</span>
            <input
              type="number"
              value={stopLoss}
              onChange={(e) => setStopLoss(e.target.value)}
              min="0"
              step="0.5"
            />
          </label>

          <label>
            <span>TAKE PROFIT</span>
            <input
              type="number"
              value={takeProfit}
              onChange={(e) => setTakeProfit(e.target.value)}
              min="0"
              step="0.5"
            />
          </label>

          <button
            className="run-backtest"
            onClick={runBacktest}
            disabled={loading || !symbol.trim()}
          >
            {loading ? "RUNNING..." : "RUN BACKTEST"}
          </button>
        </div>
      </section>

      {error && (
        <section className="backtester-error">
          <strong>BACKTEST FAILED</strong>
          <span>{error}</span>
        </section>
      )}

      {results && (
        <section className="backtest-results">
          <div className="results-header">
            <div>
              <span>RESULTS</span>
              <h2>
                {result.symbol} · {result.period}
              </h2>
            </div>

            <div className="results-badge">COMPLETED</div>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <span>FINAL CAPITAL</span>
              <strong>{formatCurrency(results.final_capital)}</strong>
            </div>

            <div className="metric">
              <span>STRATEGY RETURN</span>
              <strong>
                {formatPercent(results.strategy_return)}
              </strong>
            </div>

            <div className="metric">
              <span>BUY & HOLD</span>
              <strong>
                {formatPercent(results.buy_and_hold_return)}
              </strong>
            </div>

            <div className="metric">
              <span>MAX DRAWDOWN</span>
              <strong>
                {formatPercent(results.max_drawdown)}
              </strong>
            </div>

            <div className="metric">
              <span>TRADES</span>
              <strong>{results.trades ?? "—"}</strong>
            </div>

            <div className="metric">
              <span>WIN RATE</span>
              <strong>
                {typeof results.win_rate === "number"
                  ? `${results.win_rate.toFixed(2)}%`
                  : "—"}
              </strong>
            </div>

            <div className="metric">
              <span>WINS</span>
              <strong>{results.wins ?? "—"}</strong>
            </div>

            <div className="metric">
              <span>LOSSES</span>
              <strong>{results.losses ?? "—"}</strong>
            </div>
          </div>

          {typeof results.profit_factor === "number" && (
            <div className="profit-factor">
              <span>PROFIT FACTOR</span>
              <strong>{results.profit_factor.toFixed(2)}</strong>
            </div>
          )}
        </section>
      )}

      {!results && !loading && !error && (
        <section className="backtester-empty">
          <div>RUN A BACKTEST TO SEE RESULTS</div>
          <p>
            Configure the strategy above and run it against historical
            market data.
          </p>
        </section>
      )}
    </main>
  );
}

export default Backtester;
