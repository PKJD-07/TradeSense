import { useState } from "react";
import "./StockSelector.css";

const POPULAR_STOCKS = [
  { symbol: "RELIANCE", name: "Reliance Industries" },
  { symbol: "ADANIENT", name: "Adani Enterprises" },
  { symbol: "ADANIPORTS", name: "Adani Ports & SEZ" },

  { symbol: "TCS", name: "Tata Consultancy Services" },
  { symbol: "INFY", name: "Infosys" },
  { symbol: "HCLTECH", name: "HCL Technologies" },
  { symbol: "WIPRO", name: "Wipro" },
  { symbol: "TECHM", name: "Tech Mahindra" },
  { symbol: "LTIM", name: "LTIMindtree" },

  { symbol: "HDFCBANK", name: "HDFC Bank" },
  { symbol: "ICICIBANK", name: "ICICI Bank" },
  { symbol: "SBIN", name: "State Bank of India" },
  { symbol: "AXISBANK", name: "Axis Bank" },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank" },
  { symbol: "INDUSINDBK", name: "IndusInd Bank" },
  { symbol: "BAJFINANCE", name: "Bajaj Finance" },
  { symbol: "BAJAJFINSV", name: "Bajaj Finserv" },

  { symbol: "ITC", name: "ITC" },
  { symbol: "HINDUNILVR", name: "Hindustan Unilever" },
  { symbol: "NESTLEIND", name: "Nestle India" },
  { symbol: "BRITANNIA", name: "Britannia Industries" },
  { symbol: "TATACONSUM", name: "Tata Consumer Products" },

  { symbol: "MARUTI", name: "Maruti Suzuki" },
  { symbol: "M&M", name: "Mahindra & Mahindra" },
  { symbol: "TATAMOTORS", name: "Tata Motors" },
  { symbol: "EICHERMOT", name: "Eicher Motors" },
  { symbol: "HEROMOTOCO", name: "Hero MotoCorp" },
  { symbol: "BAJAJ-AUTO", name: "Bajaj Auto" },

  { symbol: "BHARTIARTL", name: "Bharti Airtel" },

  { symbol: "LT", name: "Larsen & Toubro" },
  { symbol: "ULTRACEMCO", name: "UltraTech Cement" },
  { symbol: "GRASIM", name: "Grasim Industries" },

  { symbol: "SUNPHARMA", name: "Sun Pharmaceutical" },
  { symbol: "DRREDDY", name: "Dr. Reddy's Laboratories" },
  { symbol: "CIPLA", name: "Cipla" },
  { symbol: "DIVISLAB", name: "Divi's Laboratories" },
  { symbol: "APOLLOHOSP", name: "Apollo Hospitals" },

  { symbol: "ONGC", name: "Oil & Natural Gas Corporation" },
  { symbol: "NTPC", name: "NTPC" },
  { symbol: "POWERGRID", name: "Power Grid Corporation" },
  { symbol: "COALINDIA", name: "Coal India" },

  { symbol: "TATASTEEL", name: "Tata Steel" },
  { symbol: "JSWSTEEL", name: "JSW Steel" },
  { symbol: "HINDALCO", name: "Hindalco Industries" },

  { symbol: "ASIANPAINT", name: "Asian Paints" },
  { symbol: "TITAN", name: "Titan Company" },
  { symbol: "TRENT", name: "Trent" },
  { symbol: "ZOMATO", name: "Zomato" },
  { symbol: "BEL", name: "Bharat Electronics" },
  { symbol: "HAL", name: "Hindustan Aeronautics" },
];

function StockSelector({ selectedSymbol, onSelect }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filteredStocks = POPULAR_STOCKS.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
      stock.name.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (symbol) => {
    onSelect(symbol);
    setQuery("");
    setOpen(false);
  };

  return (
    <section className="stock-selector">
      <div className="stock-selector-header">
        <div>
          <h3>Select a stock</h3>
        </div>
      </div>

      <div className="stock-search-wrapper">
        <div className="stock-search">
          <span className="search-icon" aria-hidden="true">
            ⌕
          </span>

          <input
            type="text"
            placeholder="Search NSE stocks..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
          />

          <span className="selected-stock">{selectedSymbol}</span>
        </div>

        {open && (
          <>
            <div
              className="stock-search-backdrop"
              onClick={() => setOpen(false)}
            />

            <div className="stock-dropdown">
              {filteredStocks.length > 0 ? (
                filteredStocks.map((stock) => (
                  <button
                    key={stock.symbol}
                    className={
                      selectedSymbol === stock.symbol
                        ? "stock-option active"
                        : "stock-option"
                    }
                    onClick={() => handleSelect(stock.symbol)}
                  >
                    <div>
                      <strong>{stock.symbol}</strong>
                      <span>{stock.name}</span>
                    </div>

                    {selectedSymbol === stock.symbol && (
                      <span className="stock-check">✓</span>
                    )}
                  </button>
                ))
              ) : (
                <div className="stock-empty">
                  No matching stock found.
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <div className="popular-stocks">
        {POPULAR_STOCKS.slice(0, 6).map((stock) => (
          <button
            key={stock.symbol}
            className={
              selectedSymbol === stock.symbol
                ? "popular-stock active"
                : "popular-stock"
            }
            onClick={() => handleSelect(stock.symbol)}
          >
            {stock.symbol}
          </button>
        ))}
      </div>
    </section>
  );
}

export default StockSelector;