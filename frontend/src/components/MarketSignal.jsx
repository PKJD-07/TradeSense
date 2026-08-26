import { useEffect, useState } from "react";
import { getMarketSignal } from "../services/marketApi";
import "./MarketSignal.css";

function MarketSignal({ symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadSignal() {
      try {
        setLoading(true);
        setError(null);

        const response = await getMarketSignal(
          symbol,
          "3mo",
          "1d"
        );

        if (cancelled) return;

        setData(response);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Unable to load market signal");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadSignal();

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (loading) {
    return (
      <section className="market-signal">
        <div className="signal-state">
          Loading market signal...
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="market-signal">
        <div className="signal-state signal-error">
          {error}
        </div>
      </section>
    );
  }

  if (!data || !data.analysis || !data.signal) {
    return (
      <section className="market-signal">
        <div className="signal-state">
          No signal data available.
        </div>
      </section>
    );
  }

  const { analysis, signal } = data;

  const formatNumber = (value, decimals = 2) =>
    typeof value === "number"
      ? value.toFixed(decimals)
      : "—";

  const decision = signal.decision;

  const decisionClass =
    decision === "BUY"
      ? "signal-buy"
      : decision === "SELL"
        ? "signal-sell"
        : "signal-hold";

  const explainHold = () => {
    if (decision !== "HOLD") return [];

    const reasons = [];

    if (signal.score < 4 && signal.score > -4) {
      reasons.push(
        `The indicator score is ${signal.score}, which is not strong enough to trigger a BUY or SELL signal.`
      );
    }

    if (analysis.adx_14 < 20) {
      reasons.push(
        `ADX is ${formatNumber(
          analysis.adx_14
        )}, indicating a weak trend.`
      );
    }

    if (analysis.latest_price < analysis.ema_20) {
      reasons.push(
        "Price is below the 20-period EMA, which adds some bearish pressure."
      );
    }

    if (analysis.rsi_14 < 50) {
      reasons.push(
        `RSI is ${formatNumber(
          analysis.rsi_14
        )}, slightly below the neutral 50 level.`
      );
    }

    if (
      analysis.latest_price > analysis.sma_20 &&
      analysis.latest_price > analysis.sma_50
    ) {
      reasons.push(
        "However, price remains above both the 20-period and 50-period SMA."
      );
    }

    if (analysis.macd > analysis.macd_signal) {
      reasons.push(
        "MACD remains above its signal line, providing a positive momentum signal."
      );
    }

    return reasons;
  };

  const holdExplanation = explainHold();

  return (
    <section className="market-signal">
      <div className="signal-header">
        <div>
          <div className="signal-eyebrow">
            MARKET SIGNAL
          </div>

          <div className="signal-title-row">
            <h3>Trading assessment</h3>
            <span>{symbol} · 3M</span>
          </div>
        </div>

        <div className={`signal-badge ${decisionClass}`}>
          <span className="signal-badge-dot" />
          {decision}
        </div>
      </div>

      <div className="signal-main">
        <div className="signal-decision">
          <span className="signal-label">
            CURRENT ASSESSMENT
          </span>

          <strong className={decisionClass}>
            {decision}
          </strong>

          <div className="signal-score">
            Score{" "}
            <span>
              {signal.score > 0 ? "+" : ""}
              {signal.score}
            </span>
            {" "} / 5
          </div>
        </div>

        <div className="signal-indicators">
          <div className="indicator">
            <span>PRICE</span>
            <strong>
              ₹
              {analysis.latest_price.toLocaleString(
                "en-IN",
                {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                }
              )}
            </strong>
          </div>

          <div className="indicator">
            <span>SMA 20</span>
            <strong>
              ₹{formatNumber(analysis.sma_20)}
            </strong>
          </div>

          <div className="indicator">
            <span>SMA 50</span>
            <strong>
              ₹{formatNumber(analysis.sma_50)}
            </strong>
          </div>

          <div className="indicator">
            <span>EMA 20</span>
            <strong>
              ₹{formatNumber(analysis.ema_20)}
            </strong>
          </div>

          <div className="indicator">
            <span>RSI 14</span>
            <strong>
              {formatNumber(analysis.rsi_14)}
            </strong>
          </div>

          <div className="indicator">
            <span>MACD</span>
            <strong>
              {formatNumber(analysis.macd)}
            </strong>
          </div>

          <div className="indicator">
            <span>MACD SIGNAL</span>
            <strong>
              {formatNumber(analysis.macd_signal)}
            </strong>
          </div>

          <div className="indicator">
            <span>ADX 14</span>
            <strong>
              {formatNumber(analysis.adx_14)}
            </strong>
          </div>
        </div>
      </div>

      {decision === "HOLD" && (
        <div className="signal-explanation">
          <div className="explanation-header">
            <span className="explanation-icon">i</span>

            <div>
              <strong>Why HOLD?</strong>
              <p>
                The indicators are giving mixed signals, so
                the system is not confident enough to recommend
                entering or exiting the position.
              </p>
            </div>
          </div>

          <div className="explanation-reasons">
            {holdExplanation.map((reason, index) => (
              <div
                className="explanation-reason"
                key={index}
              >
                <span />
                {reason}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default MarketSignal;
