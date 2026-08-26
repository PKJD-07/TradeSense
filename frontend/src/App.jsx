import { useState } from "react";

import Plasma from "./components/Plasma";
import Navbar from "./components/Navbar";
import MarketTicker from "./components/MarketTicker";
import StockSelector from "./components/StockSelector";
import MarketOverview from "./components/MarketOverview";
import PriceChart from "./components/PriceChart";
import MarketSignal from "./components/MarketSignal";
import Backtester from "./components/Backtester";
import Signals from "./components/Signals";

import "./App.css";

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState("RELIANCE");
  const [activePage, setActivePage] = useState("dashboard");

  return (
    <div className="app">
      {/* PLASMA BACKGROUND */}
      <div className="plasma-background">
        <Plasma
          color="#b19d3c"
          speed={0.3}
          direction="forward"
          scale={1}
          opacity={1}
          mouseInteractive={false}
          iterations={60}
          renderScale={0.55}
          targetFps={60}
          maxDpr={1.5}
        />
      </div>

      {/* UI */}
      <div className="ui-layer">
        {/* NAVBAR */}
        <Navbar
          activePage={activePage}
          onNavigate={setActivePage}
        />

        {/* MARKET TICKER */}
        <MarketTicker />

        {/* DASHBOARD */}
        {activePage === "dashboard" && (
          <main className="content">
            <section className="hero-copy">
              <h2>
                Read the Market.
                <br />
                Before it moves.
              </h2>

              <p>
                Analyze market conditions, trading signals and
                historical performance through one quantitative
                dashboard.
              </p>
            </section>

            <StockSelector
              selectedSymbol={selectedSymbol}
              onSelect={setSelectedSymbol}
            />

            <MarketOverview symbol={selectedSymbol} />

            <PriceChart symbol={selectedSymbol} />

            <MarketSignal symbol={selectedSymbol} />
          </main>
        )}

        {/* BACKTESTER */}
        {activePage === "backtester" && (
          <main className="content page-content">
            <Backtester symbol={selectedSymbol} />
          </main>
        )}

        {/* SIGNALS */}
        {activePage === "signals" && (
          <main className="content page-content">
            <Signals
              symbol={selectedSymbol}
              onSelect={setSelectedSymbol}
            />
          </main>
        )}
      </div>
    </div>
  );
}

export default App;
