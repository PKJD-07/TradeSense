import { useEffect, useState } from "react";
import "./Navbar.css";

function Navbar({ activePage, onNavigate }) {
  const [marketStatus, setMarketStatus] = useState("CLOSED");

  useEffect(() => {
    let cancelled = false;

    async function fetchMarketStatus() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/market/overview"
        );

        if (!response.ok) {
          throw new Error("Failed to fetch market status");
        }

        const data = await response.json();

        if (!cancelled) {
          setMarketStatus(
            data.status === "OPEN" ? "OPEN" : "CLOSED"
          );
        }
      } catch (error) {
        if (!cancelled) {
          setMarketStatus("CLOSED");
        }
      }
    }

    fetchMarketStatus();

    const interval = setInterval(fetchMarketStatus, 60000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const isOpen = marketStatus === "OPEN";

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <img
          src="/tradesense-logo.png"
          alt="TradeSense"
          className="navbar-logo"
        />

        <div>
          <div className="navbar-title">TradeSense</div>

          <div className="navbar-subtitle">
            Quantitative market intelligence
          </div>
        </div>
      </div>

      <div className="navbar-links">
        <button
          className={`nav-link ${
            activePage === "dashboard" ? "active" : ""
          }`}
          onClick={() => onNavigate("dashboard")}
        >
          Dashboard
        </button>

        <button
          className={`nav-link ${
            activePage === "backtester" ? "active" : ""
          }`}
          onClick={() => onNavigate("backtester")}
        >
          Backtester
        </button>

        <button
          className={`nav-link ${
            activePage === "signals" ? "active" : ""
          }`}
          onClick={() => onNavigate("signals")}
        >
          Signals
        </button>
      </div>

      <div
        className={`navbar-status ${
          isOpen ? "market-open" : "market-closed"
        }`}
      >
        <span className="status-dot" />

        <span>NSE INDIA</span>

        <span className="navbar-market-status">
          {marketStatus}
        </span>
      </div>
    </nav>
  );
}

export default Navbar;