from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.market import router as market_router
from backend.app.api.routes.backtest import router as backtest_router


app = FastAPI(title="TradeSense API")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",

        # Production frontend
        "https://tradesenseapi.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "TradeSense API is running",
        "docs": "/docs",
    }


@app.get("/api")
def api_root():
    return {
        "message": "TradeSense API is running",
        "docs": "/docs",
    }


app.include_router(market_router)
app.include_router(backtest_router)