"""
FastAPI entry point — Agentic Forex Trading System.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import TradingAgent
from app.config import get_settings
from app.trader import MT5Trader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

trader = MT5Trader()
agent = TradingAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic Forex Trading System...")
    if settings.mt5_enabled:
        connected = trader.connect()
        logger.info("MT5 connected: %s", connected)
    yield
    trader.disconnect()
    logger.info("System shutdown.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Forex Trading System",
    description="SMC-based AI trading system for XAUUSD",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="H1")


class SignalRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="H1")
    execute: bool = Field(default=False, description="Execute trade via MT5 if signal is valid")
    account_balance: float = Field(default=10000.0)


class SwingRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")


class StatusRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")


class ManualSignalRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(default="H1")


class BriefingRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "symbol": settings.symbol, "auto_trade": settings.auto_trade}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Full multi-timeframe SMC analysis with AI narrative."""
    try:
        result = await agent.analyze(req.symbol, req.timeframe)
        return result
    except Exception as exc:
        logger.exception("Error in /analyze")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/signal")
async def signal(req: SignalRequest):
    """Generate a validated SMC signal. Optionally execute via MT5."""
    try:
        sig = await agent.get_signal(req.symbol, req.timeframe)

        if req.execute and settings.auto_trade and sig.get("action") in ("BUY", "SELL"):
            from app.trader import TradeRequest
            trade_req = TradeRequest(
                symbol=req.symbol,
                direction=sig["action"],
                entry=sig.get("entry", 0),
                stop_loss=sig.get("stop_loss", 0),
                take_profit=sig.get("take_profit", 0),
                account_balance=req.account_balance,
            )
            trade_result = trader.execute_trade(trade_req)
            sig["execution"] = {
                "attempted": True,
                "success":   trade_result.success,
                "ticket":    trade_result.ticket,
                "lot_size":  trade_result.lot_size,
                "message":   trade_result.message,
            }
        else:
            sig["execution"] = {"attempted": False}

        return sig
    except Exception as exc:
        logger.exception("Error in /signal")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/swing")
async def swing(req: SwingRequest):
    """Generate a multi-day swing trade idea."""
    try:
        result = await agent.get_swing_idea(req.symbol)
        return result
    except Exception as exc:
        logger.exception("Error in /swing")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/status")
async def status(req: StatusRequest):
    """Get active trade status and account info."""
    try:
        result = await agent.get_status(req.symbol)
        if settings.mt5_enabled and trader._connected:
            result["active_trades"] = trader.get_open_positions(req.symbol)
            result["account"]       = trader.get_account_info()
        return result
    except Exception as exc:
        logger.exception("Error in /status")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/manual_signal")
async def manual_signal(req: ManualSignalRequest):
    """Manual trade signal card — full reasoning, no auto-execution."""
    try:
        result = await agent.get_manual_signal(req.symbol, req.timeframe)
        return result
    except Exception as exc:
        logger.exception("Error in /manual_signal")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/briefing")
async def morning_briefing(req: BriefingRequest):
    """Morning briefing — previous day recap + today's trade plan."""
    try:
        result = await agent.get_morning_briefing(req.symbol)
        return result
    except Exception as exc:
        logger.exception("Error in /briefing")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chart/mtf")
async def chart_mtf(symbol: str = "XAUUSDm"):
    """Generate multi-timeframe chart image (D1+H4+H1)."""
    try:
        from app.chart import chart_mtf as _chart_mtf
        from app.tools import get_market_data, get_current_price
        df_d1 = get_market_data(symbol, "D1")
        df_h4 = get_market_data(symbol, "H4")
        df_h1 = get_market_data(symbol, "H1")
        price = get_current_price(symbol).get("mid", 0.0)
        buf = _chart_mtf(df_d1, df_h4, df_h1, symbol, price)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as exc:
        logger.exception("Error in /chart/mtf")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chart/signal")
async def chart_signal_endpoint(req: SignalRequest):
    """Generate signal chart with entry/SL/TP marked."""
    try:
        from app.chart import chart_signal as _chart_signal
        from app.tools import get_market_data
        sig = await agent.get_signal(req.symbol, req.timeframe)
        if sig.get("action") == "NO_TRADE":
            raise HTTPException(status_code=204, detail="No trade signal")
        df = get_market_data(req.symbol, req.timeframe)
        buf = _chart_signal(df, req.symbol, req.timeframe, sig)
        return Response(content=buf.read(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /chart/signal")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
