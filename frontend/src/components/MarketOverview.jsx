
import { useEffect, useState } from "react";
import { getHistoricalData } from "../services/marketApi";
import "./MarketOverview.css";

function MarketOverview({ symbol }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadMarketData() {
      try {
        setLoading(true);
        setError(null);

        const response = await getHistoricalData(symbol, "1mo", "1d");

        if (cancelled) return;

        setData(response.candles || []);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Unable to load market data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadMarketData();

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  const latest = data[data.length - 1];
  const previous = data[data.length - 2];

  const price = latest?.close ?? null;

  const change =
    latest && previous
      ? latest.close - previous.close
      : null;

  const changePercent =
    change !== null && previous?.close
      ? (change / previous.close) * 100
      : null;

  const formatPrice = (value) =>
    value !== null
      ? `₹${value.toLocaleString("en-IN", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}`
      : "—";

  const isPositive = change !== null && change >= 0;

  return (
    <section className="market-overview">
      <div className="market-header">
        <div>
          <div className="market-eyebrow">MARKET OVERVIEW</div>

          <div className="market-symbol-row">
            <h3>{symbol}</h3>
          </div>
        </div>

        <div className="market-live">
          <span className="market-live-dot" />
          LIVE MARKET DATA
        </div>
      </div>

      <div className="market-main">
        <div className="price-section">
          {loading ? (
            <div className="price">—</div>
          ) : error ? (
            <div className="price">—</div>
          ) : (
            <>
              <div className="price">
                {formatPrice(price)}
              </div>

              {changePercent !== null && (
                <div
                  className={
                    isPositive
                      ? "price-change positive"
                      : "price-change negative"
                  }
                >
                  {isPositive ? "+" : ""}
                  {change?.toFixed(2)}

                  <span>
                    ({isPositive ? "+" : ""}
                    {changePercent.toFixed(2)}%)
                  </span>
                </div>
              )}
            </>
          )}
        </div>

        <div className="market-meta">
          <div>
            <span>TIMEFRAME</span>
            <strong>1D</strong>
          </div>
        </div>
      </div>

      {error && (
        <div className="market-error">
          {error}
        </div>
      )}
    </section>
  );
}

export default MarketOverview;

