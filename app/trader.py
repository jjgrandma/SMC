"""
MetaTrader 5 Execution + Data Layer.

Responsibilities:
  - Connect / disconnect to MT5 terminal
  - Fetch live OHLCV data (replaces yfinance when MT5 is connected)
  - Get real-time bid/ask price
  - Execute market orders with risk management
  - Manage open positions (close, modify SL/TP)
  - Return account info and trade history
  - Validate spread, margin, symbol availability

Requires:
  - MetaTrader 5 terminal installed and running on this PC
  - MT5_ENABLED=true in .env
  - MT5_LOGIN, MT5_PASSWORD, MT5_SERVER filled in .env
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from app.config import get_settings
from app.risk import RiskManager, RiskParams

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Conditional MT5 import
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 library not installed. Run: pip install MetaTrader5")


# ---------------------------------------------------------------------------
# Timeframe map
# ---------------------------------------------------------------------------
MT5_TIMEFRAME_MAP: dict[str, Any] = {}
if MT5_AVAILABLE:
    MT5_TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TradeRequest:
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    account_balance: float
    risk_percent: float | None = None
    comment: str = "SMC-Agent"


@dataclass
class TradeResult:
    success: bool
    ticket: int | None = None
    lot_size: float = 0.0
    message: str = ""
    raw: dict | None = None


@dataclass
class ConnectionStatus:
    connected: bool
    account_login: int = 0
    account_name: str = ""
    broker: str = ""
    server: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0
    leverage: int = 0
    trade_mode: str = ""   # DEMO | REAL | CONTEST
    terminal_version: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# MT5 Trader
# ---------------------------------------------------------------------------

class MT5Trader:
    def __init__(self):
        self.risk_manager = RiskManager()
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> ConnectionStatus:
        if not MT5_AVAILABLE:
            return ConnectionStatus(
                connected=False,
                error="MetaTrader5 library not installed. Run: pip install MetaTrader5",
            )
        if not settings.mt5_enabled:
            return ConnectionStatus(
                connected=False,
                error="MT5 disabled. Set MT5_ENABLED=true in .env",
            )

        # Build init kwargs
        kwargs: dict[str, Any] = {}
        if settings.mt5_login:
            kwargs["login"] = settings.mt5_login
        if settings.mt5_password:
            kwargs["password"] = settings.mt5_password
        if settings.mt5_server:
            kwargs["server"] = settings.mt5_server
        if settings.mt5_path:
            kwargs["path"] = settings.mt5_path

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize failed: %s", err)
            return ConnectionStatus(connected=False, error=f"MT5 init failed: {err}")

        account = mt5.account_info()
        if account is None:
            err = mt5.last_error()
            mt5.shutdown()
            return ConnectionStatus(connected=False, error=f"MT5 account_info failed: {err}")

        terminal = mt5.terminal_info()
        ver = mt5.version()

        trade_modes = {
            mt5.ACCOUNT_TRADE_MODE_DEMO:    "DEMO",
            mt5.ACCOUNT_TRADE_MODE_REAL:    "REAL",
            mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
        }

        self._connected = True
        status = ConnectionStatus(
            connected=True,
            account_login=account.login,
            account_name=account.name,
            broker=account.company,
            server=account.server,
            balance=account.balance,
            equity=account.equity,
            margin_free=account.margin_free,
            leverage=account.leverage,
            trade_mode=trade_modes.get(account.trade_mode, "UNKNOWN"),
            terminal_version=str(ver[0]) if ver else "unknown",
        )

        logger.info(
            "MT5 connected: %s | %s | balance=%.2f | mode=%s",
            account.login, account.server, account.balance, status.trade_mode,
        )
        return status

    def disconnect(self):
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def ensure_connected(self) -> bool:
        if self._connected:
            return True
        status = self.connect()
        return status.connected

    def get_connection_status(self) -> ConnectionStatus:
        if not self._connected:
            return ConnectionStatus(connected=False, error="Not connected")
        account = mt5.account_info()
        if account is None:
            self._connected = False
            return ConnectionStatus(connected=False, error="Connection lost")
        trade_modes = {
            mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
            mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
            mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
        }
        return ConnectionStatus(
            connected=True,
            account_login=account.login,
            account_name=account.name,
            broker=account.company,
            server=account.server,
            balance=account.balance,
            equity=account.equity,
            margin_free=account.margin_free,
            leverage=account.leverage,
            trade_mode=trade_modes.get(account.trade_mode, "UNKNOWN"),
        )

    # ------------------------------------------------------------------
    # Live market data from MT5 (replaces yfinance when connected)
    # ------------------------------------------------------------------

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame | None:
        """
        Fetch OHLCV candles directly from MT5 terminal.
        This is true broker data — no delay, no external API needed.
        """
        if not self.ensure_connected():
            return None

        tf = MT5_TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            logger.error("Unknown timeframe: %s", timeframe)
            return None

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            logger.error("Symbol %s not available in MT5", symbol)
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.error("MT5 copy_rates_from_pos failed for %s %s: %s",
                         symbol, timeframe, mt5.last_error())
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "open", "high": "high",
            "low": "low", "close": "close",
            "tick_volume": "volume",
        }, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.sort_index(inplace=True)

        logger.info("MT5 data: %s %s — %d candles", symbol, timeframe, len(df))
        return df

    def get_live_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get real-time bid/ask from MT5 — true live price, zero delay.
        """
        if not self.ensure_connected():
            return None

        if not mt5.symbol_select(symbol, True):
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "symbol":    symbol,
            "bid":       round(tick.bid, 3),
            "ask":       round(tick.ask, 3),
            "mid":       round((tick.bid + tick.ask) / 2, 3),
            "spread":    round(tick.ask - tick.bid, 3),
            "timestamp": datetime.utcfromtimestamp(tick.time).isoformat(),
            "source":    "MT5",
        }

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, request: TradeRequest) -> TradeResult:
        if not settings.auto_trade:
            return TradeResult(
                success=False,
                message="AUTO_TRADE=false. Set AUTO_TRADE=true in .env to execute.",
            )

        if not self.ensure_connected():
            return TradeResult(success=False, message="MT5 not connected.")

        # --- Risk validation ---
        risk_result = self.risk_manager.validate_and_size(RiskParams(
            symbol=request.symbol,
            direction=request.direction,
            entry=request.entry,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            account_balance=request.account_balance,
            risk_percent=request.risk_percent,
        ))

        if not risk_result.approved:
            return TradeResult(
                success=False,
                message=f"Risk check failed: {risk_result.rejection_reason}",
            )

        # --- No overtrading ---
        if self._has_open_position(request.symbol, request.direction):
            return TradeResult(
                success=False,
                message=f"Already have an open {request.direction} on {request.symbol}.",
            )

        # --- Symbol check ---
        if not mt5.symbol_select(request.symbol, True):
            return TradeResult(success=False, message=f"Symbol {request.symbol} not available.")

        # --- Spread check ---
        tick = mt5.symbol_info_tick(request.symbol)
        sym  = mt5.symbol_info(request.symbol)
        if tick is None or sym is None:
            return TradeResult(success=False, message="Cannot get symbol tick data.")

        spread_pts = round((tick.ask - tick.bid) / sym.point)
        if spread_pts > settings.max_spread_points:
            return TradeResult(
                success=False,
                message=f"Spread too wide: {spread_pts} pts (max {settings.max_spread_points}).",
            )

        # --- Determine filling mode ---
        filling_mode = self._get_filling_mode(request.symbol)

        # --- Build order ---
        order_type = mt5.ORDER_TYPE_BUY if request.direction == "BUY" else mt5.ORDER_TYPE_SELL
        price      = tick.ask if request.direction == "BUY" else tick.bid

        mt5_req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      request.symbol,
            "volume":      risk_result.lot_size,
            "type":        order_type,
            "price":       price,
            "sl":          request.stop_loss,
            "tp":          request.take_profit,
            "deviation":   20,
            "magic":       20240101,
            "comment":     request.comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        # --- Pre-check ---
        check = mt5.order_check(mt5_req)
        if check and check.retcode != 0:
            logger.warning("Order pre-check warning: %s", check.comment)

        # --- Send ---
        result = mt5.order_send(mt5_req)

        if result is None:
            return TradeResult(success=False, message=f"order_send returned None: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("MT5 order failed: retcode=%d %s", result.retcode, result.comment)
            return TradeResult(
                success=False,
                message=f"Order failed: {result.comment} (code {result.retcode})",
                raw=result._asdict(),
            )

        logger.info(
            "Trade executed: ticket=%d %s %s lots=%.2f entry=%.5f SL=%.5f TP=%.5f",
            result.order, request.direction, request.symbol,
            risk_result.lot_size, price, request.stop_loss, request.take_profit,
        )
        return TradeResult(
            success=True,
            ticket=result.order,
            lot_size=risk_result.lot_size,
            message=(
                f"Trade opened: #{result.order} | "
                f"{request.direction} {risk_result.lot_size} lots | "
                f"Entry: {price} | SL: {request.stop_loss} | TP: {request.take_profit} | "
                f"Risk: ${risk_result.risk_amount:.2f} | R:R: 1:{risk_result.rr_ratio}"
            ),
            raw=result._asdict(),
        )

    # ------------------------------------------------------------------
    # Modify SL/TP
    # ------------------------------------------------------------------

    def modify_position(
        self,
        ticket: int,
        new_sl: float | None = None,
        new_tp: float | None = None,
    ) -> TradeResult:
        if not self.ensure_connected():
            return TradeResult(success=False, message="MT5 not connected.")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return TradeResult(success=False, message=f"Position #{ticket} not found.")

        pos = positions[0]
        sl = new_sl if new_sl is not None else pos.sl
        tp = new_tp if new_tp is not None else pos.tp

        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       sl,
            "tp":       tp,
        }
        result = mt5.order_send(req)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                message=f"Modify failed: {result.comment}",
                raw=result._asdict(),
            )
        return TradeResult(
            success=True,
            ticket=ticket,
            message=f"Position #{ticket} modified: SL={sl} TP={tp}",
        )

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def close_position(self, ticket: int) -> TradeResult:
        if not self.ensure_connected():
            return TradeResult(success=False, message="MT5 not connected.")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return TradeResult(success=False, message=f"Position #{ticket} not found.")

        pos      = positions[0]
        tick     = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return TradeResult(success=False, message="Cannot get tick data.")

        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        filling    = self._get_filling_mode(pos.symbol)

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        20240101,
            "comment":      "SMC-Agent close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(req)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                message=f"Close failed: {result.comment} (code {result.retcode})",
                raw=result._asdict(),
            )

        logger.info("Position #%d closed at %.5f", ticket, price)
        return TradeResult(
            success=True,
            ticket=ticket,
            message=f"Position #{ticket} closed at {price}",
            raw=result._asdict(),
        )

    def close_all_positions(self, symbol: str | None = None) -> list[TradeResult]:
        positions = self.get_open_positions(symbol)
        results = []
        for pos in positions:
            results.append(self.close_position(pos["ticket"]))
        return results

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        if not MT5_AVAILABLE or not self._connected:
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            d = p._asdict()
            d["direction"] = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            d["open_time_dt"] = datetime.utcfromtimestamp(p.time).isoformat()
            result.append(d)
        return result

    def get_pending_orders(self, symbol: str | None = None) -> list[dict]:
        if not MT5_AVAILABLE or not self._connected:
            return []
        orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if orders is None:
            return []
        return [o._asdict() for o in orders]

    def get_trade_history(self, days: int = 7) -> list[dict]:
        if not MT5_AVAILABLE or not self._connected:
            return []
        from datetime import timedelta, timezone
        date_from = datetime.now(timezone.utc) - timedelta(days=days)
        date_to   = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return []
        return [d._asdict() for d in deals]

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        if not MT5_AVAILABLE or not self._connected:
            return {"error": "MT5 not connected"}
        info = mt5.account_info()
        if info is None:
            return {"error": str(mt5.last_error())}
        d = info._asdict()
        trade_modes = {
            mt5.ACCOUNT_TRADE_MODE_DEMO:    "DEMO",
            mt5.ACCOUNT_TRADE_MODE_REAL:    "REAL",
            mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
        }
        d["trade_mode_name"] = trade_modes.get(info.trade_mode, "UNKNOWN")
        return d

    def get_symbol_info(self, symbol: str) -> dict:
        if not MT5_AVAILABLE or not self._connected:
            return {"error": "MT5 not connected"}
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Symbol {symbol} not found"}
        info = mt5.symbol_info(symbol)
        if info is None:
            return {"error": str(mt5.last_error())}
        return {
            "symbol":       info.name,
            "bid":          info.bid,
            "ask":          info.ask,
            "spread":       info.spread,
            "digits":       info.digits,
            "point":        info.point,
            "volume_min":   info.volume_min,
            "volume_max":   info.volume_max,
            "volume_step":  info.volume_step,
            "trade_mode":   info.trade_mode,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_open_position(self, symbol: str, direction: str) -> bool:
        for pos in self.get_open_positions(symbol):
            if direction == "BUY"  and pos.get("type") == 0:
                return True
            if direction == "SELL" and pos.get("type") == 1:
                return True
        return False

    def _get_filling_mode(self, symbol: str) -> int:
        """
        Detect the correct order filling mode for the broker.
        Different brokers support different modes.
        """
        if not MT5_AVAILABLE:
            return mt5.ORDER_FILLING_IOC
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        filling = info.filling_mode
        # filling_mode is a bitmask: 1=FOK, 2=IOC, 4=RETURN
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN
