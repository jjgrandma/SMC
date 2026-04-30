"""
Market data and news tools.

Live data sources:
  - PRIMARY:   yfinance  (free, no API key, Gold Futures GC=F)
  - SECONDARY: Twelve Data (free tier 800 req/day, needs API key)
  - FALLBACK:  synthetic data (dev/offline use only)

News:
  - PLACEHOLDER — connect ForexFactory or Investing.com calendar here
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

# yfinance tickers for common Forex/metals
YFINANCE_SYMBOL_MAP: dict[str, str] = {
    "XAUUSD": "GC=F",    # Gold Futures (most liquid, real-time delayed ~10min)
    "XAGUSD": "SI=F",    # Silver Futures
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
}

# yfinance interval strings per timeframe
TIMEFRAME_MAP: dict[str, dict[str, str]] = {
    "M1":  {"interval": "1m",  "period": "1d"},
    "M5":  {"interval": "5m",  "period": "5d"},
    "M15": {"interval": "15m", "period": "5d"},
    "M30": {"interval": "30m", "period": "10d"},
    "H1":  {"interval": "1h",  "period": "60d"},
    "H4":  {"interval": "4h",  "period": "60d"},
    "D1":  {"interval": "1d",  "period": "1y"},
    "W1":  {"interval": "1wk", "period": "5y"},
}


# ---------------------------------------------------------------------------
# Primary: yfinance
# ---------------------------------------------------------------------------

def _fetch_yfinance(symbol: str, timeframe: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed. Run: pip install yfinance")
        return None

    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        logger.error("Unknown timeframe: %s", timeframe)
        return None

    ticker = YFINANCE_SYMBOL_MAP.get(symbol.upper(), symbol)

    try:
        df = yf.download(
            ticker,
            period=tf["period"],
            interval=tf["interval"],
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if df is None or df.empty:
        logger.warning("yfinance returned empty data for %s %s", ticker, timeframe)
        return None

    # Flatten multi-level columns (yfinance v0.2+ returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        logger.error("yfinance missing columns: %s", df.columns.tolist())
        return None

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    logger.info(
        "yfinance: %s %s — %d candles (latest: %s)",
        symbol, timeframe, len(df), df.index[-1]
    )
    return df


# ---------------------------------------------------------------------------
# Secondary: Twelve Data
# ---------------------------------------------------------------------------

def _fetch_twelvedata(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """
    Requires TWELVEDATA_API_KEY in .env
    Free tier: 800 requests/day, 8 requests/minute
    Sign up: https://twelvedata.com
    """
    try:
        from twelvedata import TDClient
    except ImportError:
        return None

    import os
    api_key = os.getenv("TWELVEDATA_API_KEY", "")
    if not api_key:
        return None

    td_interval_map = {
        "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
        "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week",
    }
    interval = td_interval_map.get(timeframe.upper())
    if not interval:
        return None

    try:
        td = TDClient(apikey=api_key)
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=200,
            timezone="UTC",
        ).as_pandas()

        ts.columns = [c.lower() for c in ts.columns]
        ts = ts[["open", "high", "low", "close", "volume"]].copy()
        ts = ts.apply(pd.to_numeric, errors="coerce")
        ts.dropna(inplace=True)
        ts.sort_index(inplace=True)

        logger.info("TwelveData: %s %s — %d candles", symbol, timeframe, len(ts))
        return ts
    except Exception as exc:
        logger.error("TwelveData failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Fallback: synthetic data (offline / dev only)
# ---------------------------------------------------------------------------

def _synthetic_data(symbol: str, timeframe: str) -> pd.DataFrame:
    logger.warning(
        "Using SYNTHETIC data for %s %s — NOT suitable for live trading!", symbol, timeframe
    )
    tf_minutes = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080}
    mins = tf_minutes.get(timeframe.upper(), 60)
    now = datetime.utcnow()
    price = 2320.0
    rows = []
    for i in range(200, 0, -1):
        ts = now - timedelta(minutes=mins * i)
        o = price + random.uniform(-5, 5)
        h = o + random.uniform(0, 8)
        l = o - random.uniform(0, 8)
        c = random.uniform(l, h)
        rows.append({"open": round(o,2), "high": round(h,2),
                     "low": round(l,2), "close": round(c,2), "volume": random.randint(100,5000)})
        price = c
    df = pd.DataFrame(rows, index=pd.date_range(
        end=now, periods=200, freq=f"{mins}min", tz="UTC"
    ))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_market_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for the given symbol and timeframe.

    Priority:
      1. MT5 terminal  (real broker data, zero delay — when MT5_ENABLED=true)
      2. yfinance      (free, no key needed, ~10min delay)
      3. Twelve Data   (free tier, needs TWELVEDATA_API_KEY in .env)
      4. Synthetic     (dev/offline only)

    Returns a DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC)
    """
    # 1. MT5 — real broker data (best source)
    if settings.mt5_enabled:
        try:
            from app.trader import MT5Trader
            trader = MT5Trader()
            df = trader.get_ohlcv(symbol, timeframe)
            if df is not None and len(df) >= 20:
                return df
        except Exception as exc:
            logger.warning("MT5 data fetch failed, falling back: %s", exc)

    # 2. yfinance
    df = _fetch_yfinance(symbol, timeframe)
    if df is not None and len(df) >= 20:
        return df

    # 3. Twelve Data
    df = _fetch_twelvedata(symbol, timeframe)
    if df is not None and len(df) >= 20:
        return df

    # 4. Synthetic fallback
    return _synthetic_data(symbol, timeframe)


def get_current_price(symbol: str) -> dict[str, Any]:
    """
    Get the latest bid/ask/mid price.

    Priority:
      1. MT5 terminal  (true real-time tick)
      2. yfinance      (1-min candle close, ~10min delay)
      3. Synthetic     (fallback)
    """
    # 1. MT5 real-time tick
    if settings.mt5_enabled:
        try:
            from app.trader import MT5Trader
            trader = MT5Trader()
            price = trader.get_live_price(symbol)
            if price:
                return price
        except Exception as exc:
            logger.warning("MT5 price fetch failed, falling back: %s", exc)

    # 2. yfinance
    try:
        import yfinance as yf
        ticker = YFINANCE_SYMBOL_MAP.get(symbol.upper(), symbol)
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0].lower() for c in data.columns]
            else:
                data.columns = [c.lower() for c in data.columns]
            last = data.iloc[-1]
            mid = float(last["close"])
            spread = mid * 0.0001
            return {
                "symbol":    symbol,
                "bid":       round(mid - spread, 3),
                "ask":       round(mid + spread, 3),
                "mid":       round(mid, 3),
                "spread":    round(spread * 2, 3),
                "timestamp": str(data.index[-1]),
                "source":    "yfinance",
            }
    except Exception as exc:
        logger.error("get_current_price yfinance failed: %s", exc)

    # 3. Fallback
    mid = 2320.0 + random.uniform(-10, 10)
    return {
        "symbol":    symbol,
        "bid":       round(mid - 0.15, 2),
        "ask":       round(mid + 0.15, 2),
        "mid":       round(mid, 2),
        "spread":    0.30,
        "timestamp": datetime.utcnow().isoformat(),
        "source":    "synthetic",
    }


# ---------------------------------------------------------------------------
# Economic Calendar (PLACEHOLDER — connect real feed here)
# ---------------------------------------------------------------------------

def get_economic_calendar() -> list[dict[str, Any]]:
    """
    PLACEHOLDER FOR:
    - ForexFactory API  (scrape or use community JSON feed)
    - Investing.com economic calendar
    - Myfxbook calendar

    High-impact events block trade execution.
    Currently returns synthetic upcoming events for development.
    """
    now = datetime.utcnow()
    return [
        {
            "time": (now + timedelta(hours=2)).isoformat(),
            "currency": "USD",
            "event": "Non-Farm Payrolls",
            "impact": "HIGH",
            "forecast": "180K",
            "previous": "175K",
        },
        {
            "time": (now + timedelta(hours=5)).isoformat(),
            "currency": "USD",
            "event": "CPI m/m",
            "impact": "HIGH",
            "forecast": "0.3%",
            "previous": "0.4%",
        },
    ]


def is_high_impact_news_window(symbol: str, window_minutes: int = 30) -> bool:
    """
    Returns True if a HIGH-impact news event is within `window_minutes`.
    Trading is blocked during this window.
    """
    now = datetime.utcnow()
    for event in get_economic_calendar():
        if event.get("impact") != "HIGH":
            continue
        event_time = datetime.fromisoformat(event["time"])
        delta = abs((event_time - now).total_seconds() / 60)
        if delta <= window_minutes:
            return True
    return False
