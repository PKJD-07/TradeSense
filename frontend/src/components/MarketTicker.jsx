import { useEffect, useState } from "react";
import "./MarketTicker.css";

const API_BASE_URL = "https://tradesense-backend-6ojk.onrender.com";

function MarketTicker() {
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchMarkets() {
      try {
        const response = await fetch(
          `${API_BASE_URL}/market/overview`
        );

        if (!response.ok) {
          throw new Error("Failed to fetch market overview");
        }

        const data = await response.json();

        setMarkets(data.markets || []);
      } catch (error) {
        console.error("Market overview error:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchMarkets();

    const interval = setInterval(fetchMarkets, 60000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="market-ticker">
      <div className="market-ticker-list">
        {loading ? (
          <div className="market-loading">
            Loading...
          </div>
        ) : markets.length > 0 ? (
          markets.map((market) => (
            <div
              className="market-ticker-item"
              key={market.symbol}
            >
              <span className="market-ticker-name">
                {market.name}
              </span>

              <strong>
                {Number(market.price).toLocaleString("en-IN", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </strong>

              <span
                className={
                  market.change >= 0
                    ? "market-change positive"
                    : "market-change negative"
                }
              >
                {market.change >= 0 ? "▲" : "▼"}{" "}
                {Math.abs(market.change).toFixed(2)}%
              </span>
            </div>
          ))
        ) : (
          <div className="market-loading">
            Market data unavailable
          </div>
        )}
      </div>
    </section>
  );
}

export default MarketTicker;
