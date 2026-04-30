"""
Price Alert System — notifies users BEFORE price reaches key SMC levels.

How it works:
  1. Analyzes SMC structure to find key levels (OBs, FVGs, Liquidity)
  2. Sets proximity alerts (e.g. 20 pips away from an Order Block)
  3. Background watcher checks price every minute
  4. When price enters the alert zone → sends Telegram notification
  5. Alert is marked as triggered (no repeat spam)

Alert types:
  - ORDER_BLOCK   : Price approaching an unmitigated OB
  - FVG           : Price approaching an unfilled FVG
  - LIQUIDITY     : Price approaching a liquidity zone (equal highs/lows)
  - CUSTOM        : User-defined price level
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.pip_utils import price_to_pips, get_pip_size

logger = logging.getLogger(__name__)

ALERTS_FILE = Path("data/price_alerts.json")

# How many pips away from the level to trigger the alert
DEFAULT_PROXIMITY_PIPS = 30   # alert when price is within 30 pips


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PriceAlert:
    id: str
    user_id: int
    symbol: str
    timeframe: str
    alert_type: Literal["ORDER_BLOCK", "FVG", "LIQUIDITY", "CUSTOM"]
    direction: Literal["BUY", "SELL", "BOTH"]   # which direction to watch
    level_price: float                           # the exact SMC level
    level_top: float                             # zone top (for OB/FVG)
    level_bottom: float                          # zone bottom
    proximity_pips: float                        # alert when this close
    description: str                             # e.g. "Bullish OB H1 @ 4595"
    created_at: str = ""
    triggered: bool = False
    triggered_at: str = ""
    trigger_price: float = 0.0
    active: bool = True


# ---------------------------------------------------------------------------
# Alert store
# ---------------------------------------------------------------------------

class AlertStore:
    def __init__(self):
        ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._alerts: list[PriceAlert] = []
        self._load()

    def add(self, alert: PriceAlert) -> str:
        self._alerts.append(alert)
        self._save()
        return alert.id

    def get_active(self, user_id: int | None = None) -> list[PriceAlert]:
        alerts = [a for a in self._alerts if a.active and not a.triggered]
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        return alerts

    def get_all(self, user_id: int | None = None) -> list[PriceAlert]:
        alerts = list(reversed(self._alerts))
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        return alerts[:20]

    def mark_triggered(self, alert_id: str, price: float):
        for a in self._alerts:
            if a.id == alert_id:
                a.triggered    = True
                a.triggered_at = datetime.now(timezone.utc).isoformat()
                a.trigger_price = price
                a.active       = False
                self._save()
                return

    def delete(self, alert_id: str, user_id: int) -> bool:
        for a in self._alerts:
            if a.id == alert_id and a.user_id == user_id:
                a.active = False
                self._save()
                return True
        return False

    def delete_all(self, user_id: int) -> int:
        count = 0
        for a in self._alerts:
            if a.user_id == user_id and a.active:
                a.active = False
                count += 1
        if count:
            self._save()
        return count

    def _load(self):
        if not ALERTS_FILE.exists():
            return
        try:
            data = json.loads(ALERTS_FILE.read_text(encoding="utf-8"))
            self._alerts = [PriceAlert(**a) for a in data.get("alerts", [])]
        except Exception as exc:
            logger.error("Failed to load alerts: %s", exc)

    def _save(self):
        try:
            ALERTS_FILE.write_text(
                json.dumps({"alerts": [asdict(a) for a in self._alerts]}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save alerts: %s", exc)


# ---------------------------------------------------------------------------
# Alert builder — creates alerts from SMC analysis
# ---------------------------------------------------------------------------

def build_alerts_from_smc(
    user_id: int,
    symbol: str,
    timeframe: str,
    mtf_data: dict,
    current_price: float,
    proximity_pips: float = DEFAULT_PROXIMITY_PIPS,
) -> list[PriceAlert]:
    """
    Scans SMC data and creates proximity alerts for:
    - Unmitigated Order Blocks
    - Unfilled FVGs
    - Liquidity zones
    Returns list of new alerts (not yet saved).
    """
    alerts: list[PriceAlert] = []
    now = datetime.now(timezone.utc).isoformat()
    pip_size = get_pip_size(symbol)
    proximity_price = proximity_pips * pip_size

    tfs_data = mtf_data.get("timeframes", {})
    tf_data  = tfs_data.get(timeframe, {})
    overall_bias = mtf_data.get("overall_bias", "ranging")

    # Get SMC data for the timeframe
    smc = tf_data.get("smc_data", {}) if tf_data else {}

    # --- Order Blocks ---
    for ob in smc.get("order_blocks", []):
        if ob.get("mitigated"):
            continue
        ob_top    = float(ob.get("top", 0))
        ob_bottom = float(ob.get("bottom", 0))
        ob_mid    = (ob_top + ob_bottom) / 2
        direction = ob.get("direction", "")

        # Only alert for OBs aligned with HTF bias
        if direction == "bullish" and overall_bias not in ("bullish", "ranging"):
            continue
        if direction == "bearish" and overall_bias not in ("bearish", "ranging"):
            continue

        # Only alert if price is NOT already inside the OB
        if ob_bottom <= current_price <= ob_top:
            continue

        # Only alert if price is approaching (within 5x proximity)
        dist = abs(current_price - ob_mid)
        if dist > proximity_price * 5:
            continue

        alert_id = f"OB_{symbol}_{timeframe}_{user_id}_{int(ob_mid*100)}"
        alerts.append(PriceAlert(
            id=alert_id,
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
            alert_type="ORDER_BLOCK",
            direction="BUY" if direction == "bullish" else "SELL",
            level_price=ob_mid,
            level_top=ob_top,
            level_bottom=ob_bottom,
            proximity_pips=proximity_pips,
            description=(
                f"{'Bullish' if direction=='bullish' else 'Bearish'} OB {timeframe} "
                f"@ {ob_bottom:.2f}–{ob_top:.2f}"
            ),
            created_at=now,
        ))

    # --- FVGs ---
    for fvg in smc.get("fvg", []):
        if fvg.get("filled"):
            continue
        fvg_top    = float(fvg.get("top", 0))
        fvg_bottom = float(fvg.get("bottom", 0))
        fvg_mid    = (fvg_top + fvg_bottom) / 2
        direction  = fvg.get("direction", "")

        if fvg_bottom <= current_price <= fvg_top:
            continue

        dist = abs(current_price - fvg_mid)
        if dist > proximity_price * 5:
            continue

        alert_id = f"FVG_{symbol}_{timeframe}_{user_id}_{int(fvg_mid*100)}"
        alerts.append(PriceAlert(
            id=alert_id,
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
            alert_type="FVG",
            direction="BUY" if direction == "bullish" else "SELL",
            level_price=fvg_mid,
            level_top=fvg_top,
            level_bottom=fvg_bottom,
            proximity_pips=proximity_pips,
            description=(
                f"{'Bullish' if direction=='bullish' else 'Bearish'} FVG {timeframe} "
                f"@ {fvg_bottom:.2f}–{fvg_top:.2f}"
            ),
            created_at=now,
        ))

    # --- Liquidity zones ---
    for lz in smc.get("liquidity_zones", []):
        lz_price = float(lz.get("price", 0))
        lz_kind  = lz.get("kind", "")

        dist = abs(current_price - lz_price)
        if dist > proximity_price * 5:
            continue

        alert_id = f"LIQ_{symbol}_{timeframe}_{user_id}_{int(lz_price*100)}"
        alerts.append(PriceAlert(
            id=alert_id,
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
            alert_type="LIQUIDITY",
            direction="SELL" if lz_kind == "BSL" else "BUY",
            level_price=lz_price,
            level_top=lz_price + proximity_price,
            level_bottom=lz_price - proximity_price,
            proximity_pips=proximity_pips,
            description=(
                f"{'Buy-Side' if lz_kind=='BSL' else 'Sell-Side'} Liquidity {timeframe} "
                f"@ {lz_price:.2f}"
            ),
            created_at=now,
        ))

    return alerts


# ---------------------------------------------------------------------------
# Price watcher — background task
# ---------------------------------------------------------------------------

class PriceWatcher:
    """
    Runs every minute, checks all active alerts against current price.
    Sends Telegram notification when price enters alert zone.
    """

    CHECK_INTERVAL = 60   # seconds

    def __init__(self, bot_app, alert_store: AlertStore):
        self.bot_app     = bot_app
        self.alert_store = alert_store
        self._running    = False

    async def start(self):
        self._running = True
        logger.info("Price watcher started — checking every %ds", self.CHECK_INTERVAL)
        while self._running:
            try:
                await self._check_all()
            except Exception as exc:
                logger.error("Price watcher error: %s", exc)
            await asyncio.sleep(self.CHECK_INTERVAL)

    def stop(self):
        self._running = False

    async def _check_all(self):
        from app.tools import get_current_price
        from app.config import get_settings
        settings = get_settings()

        active = self.alert_store.get_active()
        if not active:
            return

        # Group by symbol to minimize API calls
        symbols = set(a.symbol for a in active)
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                p = get_current_price(sym)
                prices[sym] = float(p.get("mid", 0))
            except Exception:
                pass

        for alert in active:
            price = prices.get(alert.symbol, 0)
            if not price:
                continue

            if self._is_triggered(alert, price):
                self.alert_store.mark_triggered(alert.id, price)
                await self._send_alert(alert, price)

    def _is_triggered(self, alert: PriceAlert, current_price: float) -> bool:
        """Check if price has entered the alert proximity zone."""
        pip_size  = get_pip_size(alert.symbol)
        prox_dist = alert.proximity_pips * pip_size
        dist      = abs(current_price - alert.level_price)
        return dist <= prox_dist

    async def _send_alert(self, alert: PriceAlert, current_price: float):
        from app.pip_utils import price_to_pips
        dist_pips = price_to_pips(abs(current_price - alert.level_price), alert.symbol)
        direction_emoji = "🟢" if alert.direction == "BUY" else "🔴"
        type_emoji = {
            "ORDER_BLOCK": "📦",
            "FVG":         "🕳",
            "LIQUIDITY":   "💧",
            "CUSTOM":      "🎯",
        }.get(alert.alert_type, "⚡")

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = (
            f"🚨 *PRICE ALERT — {alert.symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{type_emoji} *{alert.alert_type.replace('_',' ')}*\n"
            f"{direction_emoji} Direction: *{alert.direction}*\n"
            f"📍 Level: `{alert.level_price:.2f}`\n"
            f"💰 Current: `{current_price:.2f}`\n"
            f"📏 Distance: `{dist_pips:.0f} pips`\n"
            f"🕐 `{now}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_{alert.description}_\n\n"
            f"⚡ *Price is approaching your key level!*\n"
            f"_Use /signal {alert.timeframe} for entry details_"
        )

        try:
            await self.bot_app.bot.send_message(
                chat_id=alert.user_id,
                text=text,
                parse_mode="Markdown",
            )
            logger.info(
                "Price alert sent to user %d: %s @ %.2f",
                alert.user_id, alert.description, current_price,
            )
        except Exception as exc:
            logger.error("Failed to send price alert to %d: %s", alert.user_id, exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: AlertStore | None = None


def get_alert_store() -> AlertStore:
    global _store
    if _store is None:
        _store = AlertStore()
    return _store
