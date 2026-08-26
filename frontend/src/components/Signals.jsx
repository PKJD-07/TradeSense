
import { useEffect, useState } from "react";
import StockSelector from "./StockSelector";
import "./Signals.css";

const API_BASE_URL = "https://tradesenseapp-backend.vercel.app";

function Signals({ symbol, onSelect }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchSignal() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `${API_BASE_URL}/market/signal/${symbol}?period=3mo&interval=1d`
        );

        if (!response.ok) {
          throw new Error("Failed to fetch signal");
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message || "Failed to load signal");
        setData(null);
      } finally {
        setLoading(false);
      }
    }

    fetchSignal();
  }, [symbol]);

  const analysis = data?.analysis;
  const signal = data?.signal;

  const decision = signal?.decision || "—";
  const score = signal?.score ?? "—";

  // Make technical indicator names clear in the SIGNAL REASONS.
  const formatReason = (reason) => {
    if (!reason) return "";

    return reason
      .replace(
        /20-period SMA/gi,
        "20-period Simple Moving Average (SMA)"
      )
      .replace(
        /20-period EMA/gi,
        "20-period Exponential Moving Average (EMA)"
      )
      .replace(
        /50-period SMA/gi,
        "50-period Simple Moving Average (SMA)"
      )
      .replace(
        /\bRSI\b/gi,
        "Relative Strength Index (RSI)"
      )
      .replace(
        /\bMACD\b/gi,
        "Moving Average Convergence Divergence (MACD)"
      )
      .replace(
        /\bADX\b/gi,
        "Average Directional Index (ADX)"
      );
  };

  return (
    <section className="signals-page">
      <div className="signals-header">
        <div>
          <div className="signals-title-row">
            <h1>Signals</h1>
            <span>{symbol} · NSE</span>
          </div>

          <p>
            Monitor TradeSense trading signals, technical
            indicators and market direction.
          </p>
        </div>
      </div>

      <StockSelector
        selectedSymbol={symbol}
        onSelect={onSelect}
      />

      {loading && (
        <div className="signal-loading">
          <span>ANALYZING MARKET</span>

          <h2>Generating signal...</h2>

          <p>
            Processing technical indicators and evaluating
            current market conditions.
          </p>
        </div>
      )}

      {error && !loading && (
        <div className="signal-error">
          {error}
        </div>
      )}

      {data && !loading && !error && analysis && signal && (
        <>
          {/* CURRENT SIGNAL */}
          <section className="signals-grid">
            <div className="signal-overview-panel">
              <div className="panel-label">
                CURRENT SIGNAL
              </div>

              <div className="signal-overview-content">
                <span className="signal-symbol">
                  {symbol}
                </span>

                <h2
                  className={`signal-decision signal-${decision.toLowerCase()}`}
                >
                  {decision}
                </h2>

                <p>
                  TradeSense currently assesses {symbol} as{" "}
                  <strong>{decision}</strong> based on the
                  current market indicators.
                </p>
              </div>
            </div>

            <div className="signal-confidence-panel">
              <div className="panel-label">
                SIGNAL SCORE
              </div>

              <div className="confidence-value">
                {score > 0 ? "+" : ""}
                {score}
              </div>

              <span className="confidence-label">
                TECHNICAL SCORE / 5
              </span>
            </div>
          </section>

          {/* SIGNAL BREAKDOWN */}
          <section className="signal-breakdown">
            <div className="panel-label">
              SIGNAL BREAKDOWN
            </div>

            <div className="signal-breakdown-grid">

              {/* PRICE TREND */}
              <div className="signal-factor">
                <span>PRICE TREND</span>

                <strong>
                  {analysis.latest_price > analysis.sma_20
                    ? "UPWARD"
                    : "DOWNWARD"}
                </strong>

                <small>
                  SMA 20 · EMA 20 · SMA 50
                </small>
              </div>

              {/* MOMENTUM */}
              <div className="signal-factor">
                <span>MOMENTUM</span>

                <strong>
                  {analysis.rsi_14 > 50 &&
                  analysis.macd > analysis.macd_signal
                    ? "UPWARD"
                    : "WEAK"}
                </strong>

                <small>
                  RSI: {Number(analysis.rsi_14).toFixed(1)}
                </small>
              </div>

              {/* TREND STRENGTH */}
              <div className="signal-factor">
                <span>TREND STRENGTH</span>

                <strong>
                  {analysis.adx_14 >= 20
                    ? "STRONG TREND"
                    : "WEAK TREND"}
                </strong>

                <small>
                  ADX: {Number(analysis.adx_14).toFixed(1)}
                </small>
              </div>

            </div>
          </section>

          {/* SIGNAL REASONS */}
          <section className="signal-reasons">
            <div className="panel-label">
              SIGNAL REASONS
            </div>

            <div className="reasons-list">
              {signal.reasons?.map((reason, index) => (
                <div
                  className="reason"
                  key={`${reason}-${index}`}
                >
                  <span className="reason-index">
                    {String(index + 1).padStart(2, "0")}
                  </span>

                  <span>{formatReason(reason)}</span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </section>
  );
}

export default Signals;
