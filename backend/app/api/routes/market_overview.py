from datetime import datetime, time
from zoneinfo import ZoneInfo

import yfinance as yf
from fastapi import APIRouter

router = APIRouter(
    prefix="/market",
    tags=["Market Overview"],
)

IST = ZoneInfo("Asia/Kolkata")

# NSE equity-market holidays for 2026.
# Update this list when NSE publishes the next year's calendar.
NSE_HOLIDAYS_2026 = {
    "2026-01-26",
    "2026-03-03",
    "2026-03-26",
    "2026-03-31",
    "2026-04-03",
    "2026-04-14",
    "2026-05-01",
    "2026-05-28",
    "2026-06-26",
    "2026-09-14",
    "2026-10-02",
    "2026-10-20",
    "2026-11-10",
    "2026-11-24",
    "2026-12-25",
}


def get_nse_status():
    now = datetime.now(IST)

    current_date = now.strftime("%Y-%m-%d")

    is_weekday = now.weekday() < 5
    is_holiday = current_date in NSE_HOLIDAYS_2026

    market_open = time(9, 15)
    market_close = time(15, 30)

    is_open = (
        is_weekday
        and not is_holiday
        and market_open <= now.time() < market_close
    )

    if is_open:
        return {
            "status": "OPEN",
            "message": "Closes at 3:30 PM IST",
        }

    return {
        "status": "CLOSED",
        "message": "Opens at 9:15 AM IST",
    }


def get_market_data(ticker, name):
    try:
        data = yf.Ticker(ticker).history(
            period="2d",
            interval="1d",
            auto_adjust=False,
        )

        if data.empty:
            return None

        latest = data.iloc[-1]

        price = float(latest["Close"])

        if len(data) >= 2:
            previous = float(data.iloc[-2]["Close"])
            change = ((price - previous) / previous) * 100
        else:
            change = 0.0

        return {
            "symbol": ticker,
            "name": name,
            "price": round(price, 2),
            "change": round(change, 2),
        }

    except Exception:
        return None


@router.get("/overview")
def market_overview():
    instruments = [
        ("^NSEI", "NIFTY 50"),
        ("^BSESN", "SENSEX"),
        ("^GSPC", "S&P 500"),
        ("^IXIC", "NASDAQ"),
        ("^DJI", "DOW"),
        ("BTC-USD", "BITCOIN"),
    ]

    markets = []

    for ticker, name in instruments:
        result = get_market_data(ticker, name)

        if result:
            markets.append(result)

    return {
        "status": get_nse_status(),
        "markets": markets,
    }