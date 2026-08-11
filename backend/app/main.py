from fastapi import FastAPI

from backend.app.api.routes.market import router as market_router
from backend.app.api.routes.history import router as history_router

app = FastAPI(
    title="TradeSense API",
    description="Backend API for the TradeSense quantitative trading system",
    version="0.1.0",
)

app.include_router(market_router)
app.include_router(history_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "TradeSense API",
        "version": "0.1.0",
    }