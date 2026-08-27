import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { getHistoricalData } from "../services/marketApi";
import "./PriceChart.css";

const TIMEFRAMES = {
  "1M": "1mo",
  "3M": "3mo",
  "6M": "6mo",
  "1Y": "1y",
  "2Y": "2y",
  "5Y": "5y",
};

const LABEL_COUNTS = {
  "1M": 6,
  "3M": 6,
  "6M": 6,
  "1Y": 6,
  "2Y": 6,
  "5Y": 6,
};

function PriceChart({ symbol }) {
  const [timeframe, setTimeframe] = useState("1M");
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [isMobile, setIsMobile] = useState(
    window.innerWidth <= 600
  );

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 600);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadMarketData() {
      try {
        setLoading(true);
        setError(null);

        const response = await getHistoricalData(
          symbol,
          TIMEFRAMES[timeframe],
          "1d"
        );

        if (cancelled) return;

        const candles = response.candles || [];

        const formattedData = candles.map((candle, index) => ({
          index,
          timestamp: new Date(candle.timestamp).getTime(),
          date: new Date(candle.timestamp),
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume,
        }));

        setData(formattedData);
      } catch (err) {
        if (!cancelled) {
          setError(
            err.message || "Unable to load market data"
          );
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
  }, [symbol, timeframe]);

  const latestPrice =
    data.length > 0 ? data[data.length - 1].close : null;

  const previousPrice =
    data.length > 1 ? data[data.length - 2].close : null;

  const change =
    latestPrice !== null && previousPrice !== null
      ? latestPrice - previousPrice
      : null;

  const changePercent =
    change !== null && previousPrice !== 0
      ? (change / previousPrice) * 100
      : null;

  /*
   * Keep a maximum of 6 labels for every timeframe.
   * Labels are distributed evenly from the first
   * data point to the last data point.
   */
  const getTickIndexes = () => {
    if (data.length === 0) {
      return [];
    }

    const targetLabels = Math.min(
      LABEL_COUNTS[timeframe],
      data.length
    );

    if (targetLabels === 1) {
      return [0];
    }

    const indexes = [];

    for (let i = 0; i < targetLabels; i++) {
      const index = Math.round(
        (i * (data.length - 1)) /
          (targetLabels - 1)
      );

      indexes.push(index);
    }

    return [...new Set(indexes)];
  };

  const tickIndexes = getTickIndexes();

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);

    if (
      timeframe === "1Y" ||
      timeframe === "2Y" ||
      timeframe === "5Y"
    ) {
      return date.toLocaleDateString("en-IN", {
        month: "short",
        year: "numeric",
      });
    }

    return date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
    });
  };

  return (
    <section className="price-chart-section">
      <div className="chart-header">
        <div>
          <div className="chart-eyebrow">
            PRICE PERFORMANCE
          </div>

          <div className="chart-title-row">
            <h3>{symbol}</h3>
          </div>
        </div>

        <div className="chart-price">
          {latestPrice !== null ? (
            <>
              <strong>
                ₹
                {latestPrice.toLocaleString("en-IN", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </strong>

              {changePercent !== null && (
                <span
                  className={
                    changePercent >= 0
                      ? "chart-positive"
                      : "chart-negative"
                  }
                >
                  {changePercent >= 0 ? "+" : ""}
                  {changePercent.toFixed(2)}%
                </span>
              )}
            </>
          ) : (
            <strong>—</strong>
          )}
        </div>
      </div>

      <div className="chart-toolbar">
        <div className="chart-timeframes">
          {Object.keys(TIMEFRAMES).map((period) => (
            <button
              key={period}
              className={
                timeframe === period ? "active" : ""
              }
              onClick={() => setTimeframe(period)}
            >
              {period}
            </button>
          ))}
        </div>

        <div className="chart-info">
          <span className="chart-dot" />
          Daily close
        </div>
      </div>

      <div className="chart-container">
        {loading ? (
          <div className="chart-state">
            <span>Loading market data...</span>
          </div>
        ) : error ? (
          <div className="chart-state chart-error">
            <span>{error}</span>
          </div>
        ) : data.length === 0 ? (
          <div className="chart-state">
            <span>No market data available.</span>
          </div>
        ) : (
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <AreaChart
              data={data}
              margin={{
                top: 20,
                right: isMobile ? 15 : 35,
                left: isMobile ? 0 : 5,
                bottom: 5,
              }}
            >
              <defs>
                <linearGradient
                  id="priceGradient"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor="#b19d3c"
                    stopOpacity={0.28}
                  />

                  <stop
                    offset="100%"
                    stopColor="#b19d3c"
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>

              <CartesianGrid
                stroke="rgba(255,255,255,0.06)"
                vertical={false}
              />

              <XAxis
                dataKey="index"
                type="category"
                axisLine={false}
                tickLine={false}
                ticks={tickIndexes}
                interval={0}
                padding={{
                  left: 20,
                  right: 20,
                }}
                tick={{
                  fill: "rgba(255,255,255,0.45)",
                  fontSize: isMobile ? 9 : 11,
                }}
                tickMargin={8}
                tickFormatter={(index) => {
                  const point = data[index];

                  return point
                    ? formatDate(point.timestamp)
                    : "";
                }}
              />

              <YAxis
                domain={[
                  "dataMin - 10",
                  "dataMax + 10",
                ]}
                axisLine={false}
                tickLine={false}
                width={isMobile ? 58 : 78}
                tick={{
                  fill: "rgba(255,255,255,0.45)",
                  fontSize: isMobile ? 9 : 11,
                }}
                tickFormatter={(value) =>
                  `₹${Number(value).toLocaleString(
                    "en-IN",
                    {
                      maximumFractionDigits: 0,
                    }
                  )}`
                }
              />

              <Tooltip
                contentStyle={{
                  background: "rgba(8,8,8,0.94)",
                  border:
                    "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  color: "#f5f2e9",
                  fontSize: "12px",
                }}
                labelStyle={{
                  color: "rgba(255,255,255,0.45)",
                  marginBottom: "4px",
                }}
                labelFormatter={(index) => {
                  const point = data[index];

                  return point
                    ? new Date(
                        point.timestamp
                      ).toLocaleDateString("en-IN", {
                        day: "2-digit",
                        month: "short",
                        year: "numeric",
                      })
                    : "";
                }}
                formatter={(value) => [
                  `₹${Number(value).toFixed(2)}`,
                  "Close",
                ]}
              />

              <Area
                type="monotone"
                dataKey="close"
                stroke="#b19d3c"
                strokeWidth={2}
                fill="url(#priceGradient)"
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#b19d3c",
                  stroke: "#050505",
                  strokeWidth: 2,
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

export default PriceChart;