"""
Signal Scanner — background job.
Scans XAUUSD on configured timeframes every N minutes.
When a valid SMC setup is found AND risk management approves it,
pushes a formatted alert to all subscribed Telegram users.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.agent import TradingAgent
from app.config import get_settings
from app.risk import RiskManager, RiskParams
from app.user_profile import ProfileStore, UserProfile

logger = logging.getLogger(__name__)
settings = get_settings()

CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class SignalScanner:
    """
    Runs as an asyncio background task.
    Calls TradingAgent.get_signal() on each timeframe,
    validates with RiskManager per user's account settings,
    then pushes alerts via the Telegram bot application.
    """

    def __init__(self, bot_app, profile_store: ProfileStore):
        self.bot_app = bot_app          # telegram.ext.Application instance
        self.profile_store = profile_store
        self.agent = TradingAgent()
        self.risk_manager = RiskManager()
        self._running = False
        self._last_signals: dict[str, dict] = {}   # key: "symbol_tf" → last signal

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        self._running = True
        logger.info(
            "Scanner started — interval=%dmin timeframes=%s",
            settings.scanner_interval_minutes,
            settings.scanner_timeframe_list,
        )
        while self._running:
            try:
                await self._scan_all()
            except Exception as exc:
                logger.error("Scanner error: %s", exc, exc_info=True)
            await asyncio.sleep(settings.scanner_interval_minutes * 60)

    def stop(self):
        self._running = False
        logger.info("Scanner stopped.")

    # ------------------------------------------------------------------
    # Core scan loop
    # ------------------------------------------------------------------

    async def _scan_all(self):
        symbol = settings.symbol
        timeframes = settings.scanner_timeframe_list
        subscribers = self.profile_store.all_alert_subscribers()

        if not subscribers:
            logger.debug("No alert subscribers — skipping scan.")
            return

        logger.info("Scanning %s on %s for %d subscribers...", symbol, timeframes, len(subscribers))

        for tf in timeframes:
            try:
                signal = await self.agent.get_signal(symbol, tf, user_profile=None)
            except Exception as exc:
                logger.error("get_signal failed for %s %s: %s", symbol, tf, exc)
                continue

            action = signal.get("action", "NO_TRADE")
            if action == "NO_TRADE":
                logger.debug("%s %s — NO_TRADE", symbol, tf)
                continue

            # Deduplicate: skip if same direction signal was already sent for this TF
            cache_key = f"{symbol}_{tf}"
            last = self._last_signals.get(cache_key, {})
            if (
                last.get("action") == action
                and last.get("entry") == signal.get("entry")
            ):
                logger.debug("%s %s — duplicate signal, skipping.", symbol, tf)
                continue

            self._last_signals[cache_key] = signal

            # Push to each subscriber with their personal risk sizing
            for user in subscribers:
                await self._push_to_user(user, signal)

    # ------------------------------------------------------------------
    # Per-user risk sizing + push
    # ------------------------------------------------------------------

    async def _push_to_user(self, user: UserProfile, signal: dict):
        # Check user's minimum confidence filter
        signal_confidence = signal.get("confidence", "LOW")
        user_min = user.min_confidence
        if CONFIDENCE_RANK.get(signal_confidence, 0) < CONFIDENCE_RANK.get(user_min, 0):
            logger.debug(
                "User %d skipped — confidence %s < min %s",
                user.user_id, signal_confidence, user_min,
            )
            return

        # Risk sizing for this user's account
        action = signal.get("action")
        entry = signal.get("entry", 0.0)
        sl = signal.get("stop_loss", 0.0)
        tp = signal.get("take_profit", 0.0)

        if not all([entry, sl, tp]):
            return

        risk_result = self.risk_manager.validate_and_size(RiskParams(
            symbol=signal.get("symbol", settings.symbol),
            direction=action,
            entry=float(entry),
            stop_loss=float(sl),
            take_profit=float(tp),
            account_balance=user.account_balance,
            risk_percent=user.risk_percent,
        ))

        # Build and send the message
        text = _format_alert(signal, risk_result, user)

        try:
            await self.bot_app.bot.send_message(
                chat_id=user.user_id,
                text=text,
                parse_mode="Markdown",
            )
            logger.info(
                "Alert sent to user %d — %s %s %s",
                user.user_id, action, signal.get("symbol"), signal.get("timeframe"),
            )
        except Exception as exc:
            logger.error("Failed to send alert to user %d: %s", user.user_id, exc)


# ---------------------------------------------------------------------------
# Alert formatter
# ---------------------------------------------------------------------------

def _format_alert(signal: dict, risk_result, user: UserProfile) -> str:
    action = signal.get("action", "N/A")
    symbol = signal.get("symbol", "XAUUSD")
    tf = signal.get("timeframe", "H1")
    price_info = signal.get("current_price", {})
    confidence = signal.get("confidence", "N/A")

    action_emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL"
    conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}.get(confidence, "")

    entry = signal.get("entry", "N/A")
    sl = signal.get("stop_loss", "N/A")
    tp = signal.get("take_profit", "N/A")
    rr = signal.get("rr_ratio", "N/A")

    # Risk block — only shown if approved
    if risk_result.approved:
        pip_risk = risk_result.pip_risk
        dollar_risk = risk_result.risk_amount
        lot_size = risk_result.lot_size
        rr_actual = risk_result.rr_ratio

        # Convert price distance to pips
        from app.pip_utils import price_to_pips
        symbol = signal.get("symbol", "XAUUSDm")
        sl_pips = price_to_pips(pip_risk, symbol)
        tp_pips = price_to_pips(abs(
            float(signal.get("take_profit", 0) or 0) -
            float(signal.get("entry", 0) or 0)
        ), symbol)

        risk_block = (
            f"\n💼 *Risk Management* _(your account)_\n"
            f"  💰 Balance: `${user.account_balance:,.0f}`\n"
            f"  📉 Risk: `{user.risk_percent}%` = `${dollar_risk:,.2f}`\n"
            f"  📦 Lot Size: `{lot_size}` lots\n"
            f"  📏 SL: `{pip_risk:.2f}` pts = `{sl_pips:.0f} pips`\n"
            f"  📏 TP: `{tp_pips:.0f} pips`\n"
            f"  📐 R:R: `1:{rr_actual}`\n"
            f"  💵 Potential Profit: `${dollar_risk * rr_actual:,.2f}`"
        )
    else:
        risk_block = (
            f"\n⚠️ *Risk Check Failed*\n"
            f"  _{risk_result.rejection_reason}_"
        )

    key_levels = signal.get("key_levels", {})
    support = key_levels.get("support", "N/A")
    resistance = key_levels.get("resistance", "N/A")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"🚨 *SIGNAL ALERT* — {symbol} `{tf}`",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"{action_emoji}  {conf_emoji} Confidence: *{confidence}*",
        f"🕐 `{now}`",
        f"",
        f"📌 *Levels*",
        f"  🎯 Entry:       `{entry}`",
        f"  🛑 Stop Loss:   `{sl}`",
        f"  ✅ Take Profit: `{tp}`",
        f"  📐 R:R:         `1:{rr}`",
        f"",
        f"📊 *Key Levels*",
        f"  Support:    `{support}`",
        f"  Resistance: `{resistance}`",
        risk_block,
        f"",
        f"💡 *Reasoning*",
        f"_{signal.get('reasoning', 'N/A')}_",
        f"",
        f"❌ *Invalidation*",
        f"_{signal.get('invalidation', 'N/A')}_",
        f"",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"_Use /setrisk and /setbalance to personalise sizing_",
    ]

    # AI comparison block
    ai_cmp = signal.get("ai_comparison", {})
    if ai_cmp and ai_cmp.get("chosen") not in (None, "none", "both_no_trade"):
        g = ai_cmp.get("gemini", {})
        q = ai_cmp.get("groq", {})
        chosen = ai_cmp.get("chosen", "")
        g_ok = "✅" if g.get("success") else "❌"
        q_ok = "✅" if q.get("success") else "❌"
        lines += [
            f"",
            f"🤖 *AI Comparison*",
            f"  🔬 Gemini {g_ok}: `{g.get('action','?')}` "
            f"conf=`{g.get('confidence','?')}` "
            f"RR=`{g.get('rr_ratio','?')}` "
            f"score=`{g.get('score','?')}` "
            f"`{g.get('latency_ms','?')}ms`",
            f"  ⚡ Groq {q_ok}:   `{q.get('action','?')}` "
            f"conf=`{q.get('confidence','?')}` "
            f"RR=`{q.get('rr_ratio','?')}` "
            f"score=`{q.get('score','?')}` "
            f"`{q.get('latency_ms','?')}ms`",
            f"  ✅ Chosen: *{chosen.upper()}* — _{ai_cmp.get('reason','')}_",
        ]

    return "\n".join(lines)
