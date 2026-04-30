"""
Telegram Bot — UI layer for the Agentic Forex Trading System.
Calls FastAPI backend for analysis/signals.
Runs the SignalScanner as a background task for auto-alerts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.risk import RiskManager, RiskParams
from app.scanner import SignalScanner, _format_alert
from app.user_profile import get_profile_store
from app.price_alerts import get_alert_store, PriceAlert, PriceWatcher
from app.user_mt5 import connect_user_mt5, disconnect_user_mt5, get_user_mt5_status
from bot.ui import (
    Icon, DIV, DIV2, fmt_loading, fmt_error, fmt_no_trade,
    fmt_signal, fmt_analysis, fmt_performance, fmt_profile,
    fmt_status, fmt_history, fmt_main_menu,
    main_menu_kb, back_kb, signal_action_kb, analysis_action_kb, kb,
)

logger = logging.getLogger(__name__)
settings = get_settings()
profile_store = get_profile_store()

API_BASE = settings.effective_api_base_url
SYMBOL = settings.symbol

VALID_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}

# ---------------------------------------------------------------------------
# Reusable keyboard builders
# ---------------------------------------------------------------------------

def _back_button() -> list[list[InlineKeyboardButton]]:
    """Single back-to-menu row."""
    return [[InlineKeyboardButton("🏠 Back to Menu", callback_data="main_menu")]]


def _back_keyboard() -> InlineKeyboardMarkup:
    """Markup with just the back button."""
    return InlineKeyboardMarkup(_back_button())


def _action_keyboard(extra_buttons: list[list[InlineKeyboardButton]] | None = None) -> InlineKeyboardMarkup:
    """Keyboard with optional extra buttons + back button at bottom."""
    rows = extra_buttons or []
    rows += _back_button()
    return InlineKeyboardMarkup(rows)


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Analyze H1",  callback_data="analyze_H1"),
         InlineKeyboardButton("📊 Analyze H4",  callback_data="analyze_H4")],
        [InlineKeyboardButton("⚡ Signal H1",   callback_data="signal_H1"),
         InlineKeyboardButton("⚡ Signal H4",   callback_data="signal_H4")],
        [InlineKeyboardButton("📋 Trade Card",  callback_data="trade_H1"),
         InlineKeyboardButton("📈 Swing",       callback_data="swing")],
        [InlineKeyboardButton("🌅 Briefing",    callback_data="briefing"),
         InlineKeyboardButton("📋 Status",      callback_data="status")],
        [InlineKeyboardButton("📉 Performance", callback_data="performance"),
         InlineKeyboardButton("📋 History",     callback_data="history")],
        [InlineKeyboardButton("🧠 Memory",      callback_data="memory"),
         InlineKeyboardButton("👤 Profile",     callback_data="profile")],
        [InlineKeyboardButton("🔔 Alerts ON/OFF", callback_data="toggle_alerts"),
         InlineKeyboardButton("🖥 MT5 Status",  callback_data="mt5_status")],
    ])
# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def restricted(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        allowed = settings.allowed_user_ids
        if allowed and update.effective_user.id not in allowed:
            await update.message.reply_text("⛔ Unauthorized.")
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

async def _post(endpoint: str, payload: dict) -> dict:
    import os
    # Always resolve the URL at call time — not at import time
    port = os.environ.get("PORT") or os.environ.get("API_PORT", "8000")
    base = f"http://localhost:{port}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{base}{endpoint}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to API at {base}{endpoint}. "
                "Make sure run_api.py is running."
            )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_manual_signal(data: dict, user_id: int) -> str:
    """Format a manual trade signal card — detailed, educational."""
    action = data.get("action", "N/A")

    if action == "NO_TRADE":
        return (
            "⚪ *NO TRADE — Manual Signal*\n\n"
            f"_{data.get('reasoning', 'No valid setup found.')}_\n\n"
            f"{'⚠️ High-impact news window active.' if data.get('news_blocked') else ''}"
        )

    profile  = profile_store.get(user_id)
    rm       = RiskManager()
    entry    = float(data.get("entry") or 0)
    sl       = float(data.get("stop_loss") or 0)
    tp       = float(data.get("take_profit") or 0)

    risk_result = rm.validate_and_size(RiskParams(
        symbol=data.get("symbol", SYMBOL),
        direction=action,
        entry=entry, stop_loss=sl, take_profit=tp,
        account_balance=profile.account_balance,
        risk_percent=profile.risk_percent,
    ))

    action_emoji = "🟢" if action == "BUY" else "🔴"
    conf         = data.get("confidence", "N/A")
    conf_emoji   = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}.get(conf, "")
    pd_zone      = data.get("premium_discount", "N/A")
    pd_emoji     = "🔻" if pd_zone == "premium" else "🔺" if pd_zone == "discount" else "⚖️"
    ez           = data.get("entry_zone", {})
    tp2          = data.get("tp2", "")
    kl           = data.get("key_levels", {})

    lines = [
        f"📋 *MANUAL TRADE SIGNAL — {data.get('symbol', SYMBOL)} `{data.get('timeframe','H1')}`*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{action_emoji} *{action}*  {conf_emoji} Confidence: *{conf}*",
        f"💰 Price: `{data.get('current_price',{}).get('mid','N/A')}`",
        f"{pd_emoji} Zone: *{pd_zone.upper()}*  |  HTF Bias: *{data.get('htf_bias','N/A').upper()}*",
        f"",
        f"📌 *Entry Details*",
        f"  Type:        `{data.get('entry_type','market').upper()}`",
        f"  Entry:       `{entry}`",
    ]
    if ez and ez.get("from") and ez.get("to"):
        lines.append(f"  Entry Zone:  `{ez.get('from')} – {ez.get('to')}`")
    lines += [
        f"  Stop Loss:   `{sl}`",
        f"  Take Profit: `{tp}`",
    ]
    if tp2:
        lines.append(f"  TP2:         `{tp2}`")
    lines += [
        f"  R:R:         `1:{data.get('rr_ratio','N/A')}`",
        f"",
        f"📊 *Key Levels*",
        f"  Support:    `{kl.get('support','N/A')}`",
        f"  Resistance: `{kl.get('resistance','N/A')}`",
        f"",
    ]

    # Risk block
    if risk_result.approved:
        lines += [
            f"💼 *Your Risk* _(${profile.account_balance:,.0f} @ {profile.risk_percent}%)_",
            f"  Lot Size:   `{risk_result.lot_size}` lots",
            f"  Risk $:     `${risk_result.risk_amount:,.2f}`",
            f"  Reward $:   `${risk_result.risk_amount * risk_result.rr_ratio:,.2f}`",
            f"  SL Dist:    `{risk_result.pip_risk:.2f}` pts",
            f"",
        ]
    else:
        lines += [f"⚠️ Risk check: _{risk_result.rejection_reason}_", f""]

    lines += [
        f"🔬 *SMC Structure Used*",
        f"_{data.get('smc_structure_used', 'N/A')}_",
        f"",
        f"💡 *Why Enter Now*",
        f"_{data.get('why_enter_now', 'N/A')}_",
        f"",
        f"⏳ *Wait For Confirmation*",
        f"_{data.get('confirmation_needed', 'N/A')}_",
        f"",
        f"📖 *Full Reasoning*",
        f"_{data.get('reasoning', 'N/A')}_",
        f"",
        f"🛠 *Trade Management*",
        f"_{data.get('trade_management', 'N/A')}_",
        f"",
        f"❌ *Invalidation*",
        f"_{data.get('invalidation', 'N/A')}_",
    ]

    # Confluences
    confluences = data.get("confluences", [])
    if confluences:
        lines += [f"", f"✅ *Confluences ({len(confluences)})*"]
        for c in confluences[:6]:
            lines.append(f"  • {c}")

    # AI comparison
    ai_cmp = data.get("ai_comparison", {})
    if ai_cmp and ai_cmp.get("chosen") not in (None, "none"):
        g = ai_cmp.get("gemini", {})
        q = ai_cmp.get("groq", {})
        lines += [
            f"",
            f"🤖 *AI: Gemini `{g.get('action','?')}` {g.get('latency_ms','?')}ms  |  "
            f"Groq `{q.get('action','?')}` {q.get('latency_ms','?')}ms  |  "
            f"Chosen: *{ai_cmp.get('chosen','?').upper()}*",
        ]

    full = "\n".join(lines)
    if len(full) > 4000:
        full = full[:3950] + "\n\n_...truncated_"
    return full


def _fmt_briefing(data: dict) -> list[str]:
    """
    Format time-aware briefing — returns list of message parts.
    """
    if "error" in data:
        return [f"❌ Briefing failed: {data['error']}"]

    title     = data.get("title", "Market Briefing")
    gen_at    = data.get("generated_at", "")
    session   = data.get("session", "")
    next_sess = data.get("next_session", "")
    price     = data.get("price_info", {}).get("mid", "N/A")
    w_bias    = data.get("weekly_bias", "N/A").upper()
    d_bias    = data.get("daily_bias", "N/A").upper()
    aligned   = data.get("htf_aligned", False)
    symbol    = data.get("symbol", "XAUUSD")

    w_emoji   = "🟢" if w_bias == "BULLISH" else "🔴" if w_bias == "BEARISH" else "⚪"
    d_emoji   = "🟢" if d_bias == "BULLISH" else "🔴" if d_bias == "BEARISH" else "⚪"
    align_str = "✅ Aligned" if aligned else "⚠️ Not Aligned"

    kl        = data.get("key_levels", {})
    plan      = data.get("trade_plan", {})
    plan_bias = plan.get("bias", "WAIT")
    pb_emoji  = "🟢" if plan_bias == "BUY" else "🔴" if plan_bias == "SELL" else "⏸"
    prev      = data.get("prev_day", {})
    today_c   = data.get("today_candle", {})
    summary   = data.get("summary", "")

    # Part 1 — Header + bias + recaps
    part1 = [
        f"🌅 *{title}*",
        f"🕐 `{gen_at}`",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💰 `{price}` _{data.get('price_info',{}).get('source','')}_",
        f"📍 Session: *{session}* → Next: _{next_sess}_",
        f"",
        f"*HTF Bias*",
        f"  {w_emoji} Weekly: *{w_bias}*",
        f"  {d_emoji} Daily:  *{d_bias}*",
        f"  {align_str}",
    ]

    if summary:
        part1 += ["", f"*Summary*", f"_{summary}_"]

    if prev:
        dir_e = "🟢" if prev.get("direction") == "bullish" else "🔴"
        part1 += [
            "",
            f"*Yesterday* {dir_e}",
            f"  `{prev.get('date','')}` O:`{prev.get('open')}` H:`{prev.get('high')}` L:`{prev.get('low')}` C:`{prev.get('close')}`",
            f"  Range: `{prev.get('range')}` pts | Body: `{prev.get('body')}` pts",
            f"  _{data.get('yesterday_recap','')}_",
        ]

    if today_c:
        dir_e = "🟢" if today_c.get("direction") == "bullish" else "🔴"
        part1 += [
            "",
            f"*Today So Far* {dir_e}",
            f"  O:`{today_c.get('open')}` H:`{today_c.get('high')}` L:`{today_c.get('low')}` C:`{today_c.get('close')}`",
            f"  _{data.get('today_recap','')}_",
        ]

    # Part 2 — Key levels + structure
    part2 = [f"*Key Levels*"]
    if kl:
        res = kl.get("major_resistance", [])
        sup = kl.get("major_support", [])
        if res:
            part2.append("  🔴 Resistance: " + " | ".join([f"`{r}`" for r in res if r]))
        if sup:
            part2.append("  🟢 Support:    " + " | ".join([f"`{s}`" for s in sup if s]))
        for k, label in [
            ("today_high","Today H"), ("today_low","Today L"),
            ("yesterday_high","Yest H"), ("yesterday_low","Yest L"),
            ("weekly_high","Week H"), ("weekly_low","Week L"),
            ("equilibrium","EQ 50%"),
        ]:
            if kl.get(k):
                part2.append(f"  {label}: `{kl[k]}`")

    part2 += [
        "",
        f"*Order Blocks*",
        f"_{data.get('active_order_blocks','N/A')}_",
        "",
        f"*Fair Value Gaps*",
        f"_{data.get('active_fvgs','N/A')}_",
        "",
        f"*Liquidity Above*",
        f"_{data.get('liquidity_above','N/A')}_",
        "",
        f"*Liquidity Below*",
        f"_{data.get('liquidity_below','N/A')}_",
        "",
        f"*Current Zone*",
        f"_{data.get('premium_discount_now','N/A')}_",
    ]

    # Part 3 — Outlook + trade plan
    part3 = [
        f"*{session} Outlook*",
        f"_{data.get('current_session_outlook','N/A')}_",
        "",
        f"*{next_sess} Preview*",
        f"_{data.get('next_session_preview','N/A')}_",
        "",
        f"*Tomorrow's Expectation*",
        f"_{data.get('tomorrow_expectation','N/A')}_",
        "",
        f"*Trade Plan*",
        f"  {pb_emoji} Bias: *{plan_bias}*",
        f"  🎯 Entry Zone: _{plan.get('ideal_entry_zone','N/A')}_",
        f"  ⏳ Watch For: _{plan.get('watch_for','N/A')}_",
        f"  🚫 Avoid If: _{plan.get('avoid_if','N/A')}_",
        f"  ⏰ Best Session: _{plan.get('best_session_to_trade','N/A')}_",
        "",
        f"*News Impact*",
        f"_{data.get('news_impact','N/A')}_",
        "",
        f"*Experience Note*",
        f"_{data.get('experience_note','N/A')}_",
        "",
        f"*Risk Reminder*",
        f"_{data.get('risk_reminder','N/A')}_",
        "",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"_Generated: {gen_at}_",
    ]

    messages = []
    for part in [part1, part2, part3]:
        text = "\n".join(part)
        if len(text) > 4000:
            text = text[:3950] + "\n_...truncated_"
        messages.append(text)
    return messages
    """Format a signal with the user's personal risk sizing."""
    action = data.get("action", "N/A")
    if action == "NO_TRADE":
        return (
            "⚪ *NO TRADE*\n\n"
            f"_{data.get('reasoning', 'No valid setup found.')}_\n\n"
            f"{'⚠️ News window active.' if data.get('news_blocked') else ''}"
        )

    profile = profile_store.get(user_id)
    risk_manager = RiskManager()

    entry = data.get("entry", 0.0)
    sl = data.get("stop_loss", 0.0)
    tp = data.get("take_profit", 0.0)

    risk_result = risk_manager.validate_and_size(RiskParams(
        symbol=data.get("symbol", SYMBOL),
        direction=action,
        entry=float(entry),
        stop_loss=float(sl),
        take_profit=float(tp),
        account_balance=profile.account_balance,
        risk_percent=profile.risk_percent,
    ))

    return _format_alert(data, risk_result, profile)


def _fmt_analysis(data: dict) -> str:
    smc = data.get("smc_data", {})
    trend = smc.get("trend", "ranging")
    trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "⚪"
    price = data.get("current_price", {})

    bos_list = smc.get("bos", [])
    choch_list = smc.get("choch", [])
    fvg_list = smc.get("fvg", [])
    ob_list = smc.get("order_blocks", [])
    liq_list = smc.get("liquidity_zones", [])

    lines = [
        f"📊 *SMC ANALYSIS — {data.get('symbol', SYMBOL)} {data.get('timeframe', 'H1')}*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"{trend_emoji} Trend: *{trend.upper()}*",
        f"💰 Price: `{price.get('mid', 'N/A')}` _(source: {price.get('source', 'N/A')})_",
        f"📰 News Block: `{'YES ⚠️' if data.get('news_blocked') else 'NO'}`",
        f"",
        f"📐 *Market Structure*",
        f"  BOS signals:  `{len(bos_list)}`",
        f"  CHoCH signals: `{len(choch_list)}`",
    ]

    if bos_list:
        last_bos = bos_list[-1]
        lines.append(f"  Last BOS: `{last_bos.get('direction','').upper()}` @ `{last_bos.get('price','N/A')}`")
    if choch_list:
        last_choch = choch_list[-1]
        lines.append(f"  Last CHoCH: `{last_choch.get('direction','').upper()}` @ `{last_choch.get('price','N/A')}`")

    lines += [
        f"",
        f"📦 *Order Blocks* (active): `{len(ob_list)}`",
    ]
    for ob in ob_list[-3:]:
        lines.append(f"  • {ob.get('direction','').upper()} OB: `{ob.get('bottom')} – {ob.get('top')}`")

    lines += [
        f"",
        f"🕳 *Fair Value Gaps* (unfilled): `{len(fvg_list)}`",
    ]
    for fvg in fvg_list[-3:]:
        lines.append(f"  • {fvg.get('direction','').upper()} FVG: `{fvg.get('bottom')} – {fvg.get('top')}`")

    lines += [
        f"",
        f"💧 *Liquidity Zones*: `{len(liq_list)}`",
    ]
    for lz in liq_list[-4:]:
        lines.append(f"  • {lz.get('kind','')}: `{lz.get('price','N/A')}`")

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"🤖 *AI Narrative*",
        data.get("analysis", "No analysis available."),
    ]

    full = "\n".join(lines)
    if len(full) > 4000:
        full = full[:3950] + "\n\n_...truncated_"
    return full


def _fmt_profile(profile) -> str:
    conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}.get(profile.min_confidence, "")
    return (
        f"👤 *Your Profile*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Account Balance: `${profile.account_balance:,.2f}`\n"
        f"📉 Risk per Trade:  `{profile.risk_percent}%`\n"
        f"💵 Max Risk Amount: `${profile.account_balance * profile.risk_percent / 100:,.2f}`\n"
        f"📊 Default TF:      `{profile.timeframe}`\n"
        f"🔔 Alerts:          `{'ON ✅' if profile.alerts_enabled else 'OFF ❌'}`\n"
        f"🎯 Min Confidence:  {conf_emoji} `{profile.min_confidence}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Use /setbalance, /setrisk, /alerts to update_"
    )


def _fmt_swing(data: dict) -> str:
    action = data.get("action", "N/A")
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    price = data.get("current_price", {})
    targets = data.get("targets", [])
    tp_lines = "\n".join([f"  TP{i+1}: `{t}`" for i, t in enumerate(targets)]) if targets else "  N/A"
    lines = [
        f"📈 *SWING TRADE — {data.get('symbol', SYMBOL)}*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} Direction: *{action}*",
        f"💰 Price: `{price.get('mid', 'N/A')}`",
        f"",
        f"🎯 Entry Zone: `{data.get('entry', 'N/A')}`",
        f"🛑 Stop Loss:  `{data.get('stop_loss', 'N/A')}`",
        f"📐 R:R:        `1:{data.get('rr_ratio', 'N/A')}`",
        f"⏱ Duration:   `{data.get('duration', 'N/A')}`",
        f"",
        f"🎯 *Targets*",
        tp_lines,
        f"",
        f"💡 *Rationale*",
        f"_{data.get('reasoning', 'N/A')}_",
    ]
    risks = data.get("risk_factors", "")
    if risks:
        lines += [f"", f"⚠️ *Risk Factors*", f"_{risks}_"]
    return "\n".join(lines)


def _fmt_status(data: dict) -> str:
    price = data.get("current_price", {})
    trades = data.get("active_trades", [])
    account = data.get("account", {})
    lines = [
        f"📊 *SYSTEM STATUS — {data.get('symbol', SYMBOL)}*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"💰 Live Price: `{price.get('mid', 'N/A')}` _{price.get('source', '')}_",
        f"",
    ]
    if account and "balance" in account:
        lines += [
            f"🏦 *MT5 Account*",
            f"  Balance: `${account.get('balance', 0):,.2f}`",
            f"  Equity:  `${account.get('equity', 0):,.2f}`",
            f"  Margin:  `${account.get('margin', 0):,.2f}`",
            f"",
        ]
    if trades:
        lines.append(f"🔄 *Active Trades ({len(trades)})*")
        for t in trades:
            direction = "BUY" if t.get("type") == 0 else "SELL"
            pnl = t.get("profit", 0)
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"  {pnl_emoji} #{t.get('ticket')} {direction} "
                f"{t.get('volume')} lots @ {t.get('price_open')} "
                f"| P&L: `${pnl:.2f}`"
            )
    else:
        lines.append("💤 No active trades.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@restricted
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    profile = profile_store.get(user.id)
    profile.username = user.username or user.first_name or ""
    profile_store.update(profile)
    await update.message.reply_text(
        fmt_main_menu(SYMBOL, profile),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )


# ---------------------------------------------------------------------------
# /setbalance <amount>
# ---------------------------------------------------------------------------

@restricted
async def cmd_setbalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: `/setbalance 10000`\nEnter your account balance in USD.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    try:
        balance = float(args[0].replace(",", ""))
        if balance <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Invalid amount. Example: `/setbalance 5000`", parse_mode=ParseMode.MARKDOWN)
        return

    profile = profile_store.set_balance(user_id, balance)
    max_risk = balance * profile.risk_percent / 100
    await update.message.reply_text(
        f"✅ Balance updated to `${balance:,.2f}`\n"
        f"📉 Max risk per trade: `${max_risk:,.2f}` ({profile.risk_percent}%)",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /setrisk <percent>
# ---------------------------------------------------------------------------

@restricted
async def cmd_setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: `/setrisk 1.5`\nEnter risk percentage (0.1 – 5.0).",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    try:
        risk = float(args[0].replace("%", ""))
        if not (0.1 <= risk <= 5.0):
            raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Risk must be between 0.1% and 5.0%. Example: `/setrisk 1`", parse_mode=ParseMode.MARKDOWN)
        return

    profile = profile_store.set_risk(user_id, risk)
    max_risk = profile.account_balance * risk / 100
    await update.message.reply_text(
        f"✅ Risk updated to `{risk}%`\n"
        f"💵 Max risk per trade: `${max_risk:,.2f}` on `${profile.account_balance:,.0f}` balance",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /settf <timeframe>
# ---------------------------------------------------------------------------

@restricted
async def cmd_settf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args or args[0].upper() not in VALID_TIMEFRAMES:
        await update.message.reply_text(
            f"Usage: `/settf H1`\nValid: `{' | '.join(sorted(VALID_TIMEFRAMES))}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    tf = args[0].upper()
    profile_store.set_timeframe(user_id, tf)
    await update.message.reply_text(f"✅ Default timeframe set to `{tf}`", parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /setconfidence <level>
# ---------------------------------------------------------------------------

@restricted
async def cmd_setconfidence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args or args[0].upper() not in CONFIDENCE_LEVELS:
        await update.message.reply_text(
            "Usage: `/setconfidence MEDIUM`\nOptions: `LOW | MEDIUM | HIGH`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    level = args[0].upper()
    profile_store.set_min_confidence(user_id, level)
    await update.message.reply_text(
        f"✅ Minimum alert confidence set to `{level}`\n"
        f"_You will only receive alerts with {level} or higher confidence._",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /alerts — toggle
# ---------------------------------------------------------------------------

@restricted
async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)
    new_state = not profile.alerts_enabled
    profile_store.set_alerts(user_id, new_state)
    status = "✅ ON" if new_state else "❌ OFF"
    await update.message.reply_text(
        f"🔔 Auto-alerts: *{status}*\n\n"
        f"{'You will receive signals when valid setups are found.' if new_state else 'You will not receive automatic signals.'}",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /myprofile
# ---------------------------------------------------------------------------

@restricted
async def cmd_myprofile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    profile = profile_store.get(update.effective_user.id)
    await update.message.reply_text(
        fmt_profile(profile), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb()
    )


# ---------------------------------------------------------------------------
# /analyze [TF]
# ---------------------------------------------------------------------------

@restricted
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)
    args = context.args
    tf = args[0].upper() if args and args[0].upper() in VALID_TIMEFRAMES else profile.timeframe
    msg = await update.message.reply_text(
        fmt_loading("Analyzing", SYMBOL, tf), parse_mode=ParseMode.MARKDOWN
    )
    try:
        data = await _post("/analyze", {"symbol": SYMBOL, "timeframe": tf})
        await msg.edit_text(
            fmt_analysis(data), parse_mode=ParseMode.MARKDOWN,
            reply_markup=analysis_action_kb(tf),
        )
    except Exception as exc:
        await msg.edit_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /signal [TF]
# ---------------------------------------------------------------------------

@restricted
async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)
    args = context.args
    tf = args[0].upper() if args and args[0].upper() in VALID_TIMEFRAMES else profile.timeframe
    msg = await update.message.reply_text(
        fmt_loading("Scanning", SYMBOL, tf), parse_mode=ParseMode.MARKDOWN
    )
    try:
        import os
        port = os.environ.get("PORT") or os.environ.get("API_PORT", "8000")
        base = f"http://localhost:{port}"

        data = await _post("/signal", {"symbol": SYMBOL, "timeframe": tf, "execute": False})
        if data.get("action") == "NO_TRADE":
            text = fmt_no_trade(data.get("reasoning","No setup"), data.get("news_blocked", False))
            await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())
        else:
            rm = RiskManager()
            risk = rm.validate_and_size(RiskParams(
                symbol=SYMBOL, direction=data["action"],
                entry=float(data.get("entry") or 0),
                stop_loss=float(data.get("stop_loss") or 0),
                take_profit=float(data.get("take_profit") or 0),
                account_balance=profile.account_balance,
                risk_percent=profile.risk_percent,
            ))
            await msg.edit_text(
                fmt_signal(data, risk, profile), parse_mode=ParseMode.MARKDOWN,
                reply_markup=signal_action_kb(data["action"], tf, SYMBOL),
            )
            # Send signal chart
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{base}/chart/signal",
                        json={"symbol": SYMBOL, "timeframe": tf, "execute": False},
                    )
                    if resp.status_code == 200:
                        action = data.get("action","")
                        caption = (
                            f"{'🟢 BUY' if action=='BUY' else '🔴 SELL'} "
                            f"`{SYMBOL}` `{tf}`\n"
                            f"Entry: `{data.get('entry')}` | "
                            f"SL: `{data.get('stop_loss')}` | "
                            f"TP: `{data.get('take_profit')}` | "
                            f"R:R: `1:{data.get('rr_ratio')}`"
                        )
                        await update.message.reply_photo(
                            photo=resp.content,
                            caption=caption,
                            parse_mode=ParseMode.MARKDOWN,
                        )
            except Exception as chart_exc:
                logger.warning("Signal chart failed: %s", chart_exc)
    except Exception as exc:
        await msg.edit_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /swing
# ---------------------------------------------------------------------------

@restricted
async def cmd_swing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(f"📈 Building swing idea for {SYMBOL}...")
    try:
        data = await _post("/swing", {"symbol": SYMBOL})
        await msg.edit_text(_fmt_swing(data), parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.exception("Error in /swing")
        await msg.edit_text(f"❌ Error: {exc}")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@restricted
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(fmt_loading("Fetching", SYMBOL), parse_mode=ParseMode.MARKDOWN)
    try:
        data = await _post("/status", {"symbol": SYMBOL})
        await msg.edit_text(fmt_status(data), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())
    except Exception as exc:
        await msg.edit_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Inline keyboard callbacks
# ---------------------------------------------------------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    allowed = settings.allowed_user_ids
    if allowed and user_id not in allowed:
        await query.edit_message_text("⛔ Unauthorized.")
        return

    if data.startswith("analyze_"):
        tf = data.split("_")[1]
        await query.edit_message_text(fmt_loading("Analyzing", SYMBOL, tf), parse_mode=ParseMode.MARKDOWN)
        try:
            result = await _post("/analyze", {"symbol": SYMBOL, "timeframe": tf})
            await query.edit_message_text(
                fmt_analysis(result), parse_mode=ParseMode.MARKDOWN,
                reply_markup=analysis_action_kb(tf),
            )
        except Exception as exc:
            await query.edit_message_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data.startswith("signal_"):
        tf = data.split("_")[1]
        await query.edit_message_text(fmt_loading("Scanning", SYMBOL, tf), parse_mode=ParseMode.MARKDOWN)
        try:
            result = await _post("/signal", {"symbol": SYMBOL, "timeframe": tf, "execute": False})
            profile = profile_store.get(user_id)
            if result.get("action") == "NO_TRADE":
                text = fmt_no_trade(result.get("reasoning","No setup"), result.get("news_blocked", False))
                await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())
            else:
                rm = RiskManager()
                risk = rm.validate_and_size(RiskParams(
                    symbol=SYMBOL, direction=result["action"],
                    entry=float(result.get("entry") or 0),
                    stop_loss=float(result.get("stop_loss") or 0),
                    take_profit=float(result.get("take_profit") or 0),
                    account_balance=profile.account_balance,
                    risk_percent=profile.risk_percent,
                ))
                await query.edit_message_text(
                    fmt_signal(result, risk, profile), parse_mode=ParseMode.MARKDOWN,
                    reply_markup=signal_action_kb(result["action"], tf, SYMBOL),
                )
        except Exception as exc:
            await query.edit_message_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "swing":
        await query.edit_message_text(fmt_loading("Building swing idea", SYMBOL), parse_mode=ParseMode.MARKDOWN)
        try:
            result = await _post("/swing", {"symbol": SYMBOL})
            await query.edit_message_text(_fmt_swing(result), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())
        except Exception as exc:
            await query.edit_message_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "status":
        try:
            result = await _post("/status", {"symbol": SYMBOL})
            await query.edit_message_text(fmt_status(result), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())
        except Exception as exc:
            await query.edit_message_text(fmt_error(str(exc)), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "profile":
        profile = profile_store.get(user_id)
        await query.edit_message_text(fmt_profile(profile), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "performance":
        from app.journal import get_journal
        stats = get_journal().get_stats()
        await query.edit_message_text(fmt_performance(stats), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "history":
        from app.journal import get_journal
        records = get_journal().get_all()[:15]
        await query.edit_message_text(fmt_history(records), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "main_menu":
        profile = profile_store.get(user_id)
        await query.edit_message_text(
            fmt_main_menu(SYMBOL, profile), parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )

    elif data == "briefing":
        await query.edit_message_text("Generating morning briefing...\n_This takes ~30 seconds_", parse_mode=ParseMode.MARKDOWN)
        try:
            result = await _post("/briefing", {"symbol": SYMBOL})
            parts = _fmt_briefing(result)
            await query.edit_message_text(parts[0], parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard())
            for part in parts[1:]:
                await query.message.reply_text(part, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard())
        except Exception as exc:
            await query.edit_message_text(f"Error: {exc}", reply_markup=_back_keyboard())

    elif data == "memory":
        from app.memory import get_memory
        memory = get_memory()
        stats = memory.get_stats()
        wr = stats.win_rate
        wr_emoji = "green" if wr >= 60 else "yellow" if wr >= 45 else "red"
        text = (
            f"*BOT MEMORY*\n\n"
            f"Total signals: `{stats.total_signals}`\n"
            f"Win rate: `{stats.win_rate}%` ({stats.wins}W / {stats.losses}L)\n"
            f"Best TF: `{stats.best_timeframe or 'learning...'}` \n"
            f"Best session: `{stats.best_session or 'learning...'}` \n"
            f"Total lessons: `{stats.total_lessons}`"
        )
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard())

    elif data == "mt5_status":
        try:
            from app.trader import MT5Trader
            trader = MT5Trader()
            if not trader.ensure_connected():
                await query.edit_message_text("MT5 not connected. Use /mt5connect first.", reply_markup=_back_keyboard())
                return
            account = trader.get_account_info()
            sym_info = trader.get_symbol_info(SYMBOL)
            text = (
                f"*MT5 STATUS*\n\n"
                f"Balance: `${account.get('balance',0):,.2f}`\n"
                f"Equity: `${account.get('equity',0):,.2f}`\n"
                f"Mode: `{account.get('trade_mode_name','N/A')}`\n\n"
                f"*{SYMBOL}*\n"
                f"Bid: `{sym_info.get('bid','N/A')}` | Ask: `{sym_info.get('ask','N/A')}`\n"
                f"Spread: `{sym_info.get('spread','N/A')} pts`"
            )
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard())
        except Exception as exc:
            await query.edit_message_text(f"Error: {exc}", reply_markup=_back_keyboard())

    elif data.startswith("trade_"):
        tf = data.split("_")[1]
        await query.edit_message_text(f"Building manual trade card for {SYMBOL} {tf}...", reply_markup=_back_keyboard())
        try:
            result = await _post("/manual_signal", {"symbol": SYMBOL, "timeframe": tf})
            text = _fmt_manual_signal(result, user_id)
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard())
        except Exception as exc:
            await query.edit_message_text(f"Error: {exc}", reply_markup=_back_keyboard())

    elif data == "toggle_alerts":
        profile = profile_store.get(user_id)
        new_state = not profile.alerts_enabled
        profile_store.set_alerts(user_id, new_state)
        status = "✅ ON" if new_state else "❌ OFF"
        await query.edit_message_text(
            f"🔔 Auto-alerts: *{status}*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_back_keyboard(),
        )


# ---------------------------------------------------------------------------
# /trade [TF] — manual signal card
# ---------------------------------------------------------------------------

@restricted
async def cmd_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)
    args    = context.args
    tf      = args[0].upper() if args and args[0].upper() in VALID_TIMEFRAMES else profile.timeframe

    msg = await update.message.reply_text(
        f"📋 Building manual trade card for {SYMBOL} `{tf}`...",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        data = await _post("/manual_signal", {"symbol": SYMBOL, "timeframe": tf})
        text = _fmt_manual_signal(data, user_id)
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.exception("Error in /trade")
        await msg.edit_text(f"❌ Error: {exc}")


# ---------------------------------------------------------------------------
# /briefing — morning briefing on demand
# ---------------------------------------------------------------------------

@restricted
async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        f"🌅 Generating briefing for {SYMBOL}...\n_This takes ~30 seconds_",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        import os
        import httpx
        port = os.environ.get("PORT") or os.environ.get("API_PORT", "8000")
        base = f"http://localhost:{port}"

        # Get briefing data
        data  = await _post("/briefing", {"symbol": SYMBOL})
        parts = _fmt_briefing(data)

        # Send first text part
        await msg.edit_text(parts[0], parse_mode=ParseMode.MARKDOWN)

        # Generate and send chart
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(f"{base}/chart/mtf", params={"symbol": SYMBOL})
                if resp.status_code == 200:
                    caption = (
                        f"📊 *{SYMBOL} Multi-Timeframe Chart*\n"
                        f"D1 | H4 | H1  •  {data.get('generated_at','')}\n"
                        f"Session: *{data.get('session','')}*  →  Next: _{data.get('next_session','')}_"
                    )
                    await update.message.reply_photo(
                        photo=resp.content,
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN,
                    )
        except Exception as chart_exc:
            logger.warning("Chart generation failed: %s", chart_exc)

        # Send remaining text parts
        for part in parts[1:]:
            await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)

    except Exception as exc:
        logger.exception("Error in /briefing")
        await msg.edit_text(f"❌ Error: {exc}")


# ---------------------------------------------------------------------------
# /memory — show bot's learned stats and lessons
# ---------------------------------------------------------------------------

@restricted
async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.memory import get_memory
    memory = get_memory()
    stats  = memory.get_stats()
    lessons = memory.get_lessons()[:5]

    wr_emoji = "🟢" if stats.win_rate >= 60 else "🟡" if stats.win_rate >= 45 else "🔴"

    lines = [
        f"🧠 *BOT MEMORY & LEARNING*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 *Performance Stats*",
        f"  Total signals: `{stats.total_signals}`",
        f"  {wr_emoji} Win rate:    `{stats.win_rate}%` ({stats.wins}W / {stats.losses}L / {stats.breakevens}BE)",
        f"  Pending:       `{stats.pending}`",
        f"  Avg RR won:    `{stats.avg_rr_won}`",
        f"  Avg RR lost:   `{stats.avg_rr_lost}`",
        f"",
        f"🏆 *Best Patterns*",
        f"  Best TF:       `{stats.best_timeframe or 'learning...'}`",
        f"  Best session:  `{stats.best_session or 'learning...'}`",
        f"  Worst session: `{stats.worst_session or 'learning...'}`",
        f"  Best confluence: `{stats.best_confluence[:60] if stats.best_confluence else 'learning...'}`",
        f"",
        f"📚 *Top Lessons* ({stats.total_lessons} total)",
    ]

    for l in lessons:
        cat_emoji = "✅" if "WIN" in l.category else "❌" if "LOSS" in l.category else "📝"
        lines.append(
            f"  {cat_emoji} [{l.occurrences}x] _{l.title}_"
        )

    lines += [
        f"",
        f"_Use /outcome to mark a signal result_",
        f"_Use /addlesson to teach the bot manually_",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /outcome <memory_id> <WIN|LOSS|BREAKEVEN> [price] [note]
# ---------------------------------------------------------------------------

@restricted
async def cmd_outcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.memory import get_memory
    args = context.args

    if len(args) < 2:
        # Show pending signals
        memory  = get_memory()
        pending = memory.get_pending_signals()
        if not pending:
            await update.message.reply_text(
                "No pending signals.\n\nUsage: `/outcome <id> <WIN|LOSS|BREAKEVEN> [price] [note]`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = ["📋 *Pending Signals* (need outcome):"]
        for s in pending[-10:]:
            lines.append(
                f"  `{s.id}`\n"
                f"  {s.action} {s.symbol} {s.timeframe} | "
                f"Entry: {s.entry} | SL: {s.stop_loss} | TP: {s.take_profit}"
            )
        lines.append("\nUsage: `/outcome <id> WIN 4650.0 TP hit perfectly`")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    sig_id  = args[0]
    outcome = args[1].upper()
    price   = float(args[2]) if len(args) > 2 else 0.0
    note    = " ".join(args[3:]) if len(args) > 3 else ""

    if outcome not in ("WIN", "LOSS", "BREAKEVEN", "CANCELLED"):
        await update.message.reply_text(
            "❌ Outcome must be: `WIN | LOSS | BREAKEVEN | CANCELLED`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    memory = get_memory()
    ok = memory.update_outcome(sig_id, outcome, price, note)

    if ok:
        emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "➖"
        await update.message.reply_text(
            f"{emoji} *Outcome recorded*\n\n"
            f"Signal: `{sig_id}`\n"
            f"Result: *{outcome}*\n"
            f"Price: `{price}`\n"
            f"Note: _{note or 'none'}_\n\n"
            f"_Bot has learned from this trade._",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            f"❌ Signal `{sig_id}` not found.\nUse `/outcome` to see pending signals.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ---------------------------------------------------------------------------
# /addlesson <title> | <content>
# ---------------------------------------------------------------------------

@restricted
async def cmd_addlesson(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.memory import get_memory, Lesson
    from datetime import datetime, timezone

    text = " ".join(context.args) if context.args else ""
    if "|" not in text:
        await update.message.reply_text(
            "Usage: `/addlesson <title> | <content>`\n\n"
            "Example:\n`/addlesson Avoid NY session Mondays | "
            "Gold tends to fake out on Monday NY open — wait for London close first`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    parts   = text.split("|", 1)
    title   = parts[0].strip()
    content = parts[1].strip()

    lesson = Lesson(
        id=f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        category="GENERAL",
        title=title,
        content=content,
        confidence=0.8,   # manual lessons get high confidence
    )

    memory = get_memory()
    memory.add_lesson(lesson)

    await update.message.reply_text(
        f"✅ *Lesson saved*\n\n"
        f"📝 *{title}*\n"
        f"_{content}_\n\n"
        f"_This will be injected into future AI prompts._",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /performance — full performance dashboard with visual bars
# ---------------------------------------------------------------------------

@restricted
async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.journal import get_journal
    journal = get_journal()
    args    = context.args
    days    = int(args[0]) if args and args[0].isdigit() else None
    stats   = journal.get_stats(days=days)
    period  = f"last {days} days" if days else "all time"

    await update.message.reply_text(
        _fmt_performance(stats, period),
        parse_mode=ParseMode.MARKDOWN,
    )


def _fmt_performance(stats, period: str = "all time") -> str:
    """Full performance dashboard with visual progress bars."""

    def bar(value: float, total: float = 100, width: int = 10) -> str:
        filled = int((value / total) * width) if total > 0 else 0
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def pct_bar(wins: int, losses: int) -> str:
        total = wins + losses
        if total == 0:
            return "░░░░░░░░░░ 0%"
        w = int(wins / total * 10)
        l = 10 - w
        return "🟩" * w + "🟥" * l

    # Win rate color
    wr = stats.win_rate
    wr_emoji = "🟢" if wr >= 60 else "🟡" if wr >= 45 else "🔴"

    # Streak display
    streak_emoji = "🔥" if stats.current_streak_type == "WIN" else "❄️" if stats.current_streak_type == "LOSS" else ""
    streak_str   = f"{streak_emoji} {stats.current_streak} {stats.current_streak_type} streak" if stats.current_streak else "—"

    # Last 10 visual
    last10_str = ""
    for t in stats.last_10:
        if t["outcome"] == "WIN":
            last10_str += "✅"
        elif t["outcome"] == "LOSS":
            last10_str += "❌"
        else:
            last10_str += "➖"
    if not last10_str:
        last10_str = "No completed trades yet"

    lines = [
        f"📊 *PERFORMANCE DASHBOARD*",
        f"📅 Period: _{period}_",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🎯 *Overall Results*",
        f"  Total signals:  `{stats.total}`",
        f"  Completed:      `{stats.wins + stats.losses + stats.breakevens}`",
        f"  Pending:        `{stats.pending}`",
        f"",
        f"  {pct_bar(stats.wins, stats.losses)}",
        f"  ✅ Wins:        `{stats.wins}` ({stats.win_rate}%)",
        f"  ❌ Losses:      `{stats.losses}` ({stats.loss_rate}%)",
        f"  ➖ Breakeven:   `{stats.breakevens}`",
        f"",
        f"  {wr_emoji} Win Rate: `{stats.win_rate}%`  `{bar(stats.win_rate)}`",
        f"",
        f"� *Pip P&L*",
        f"  Net pips:        `{'+' if stats.total_pips >= 0 else ''}{stats.total_pips} pips`",
        f"  Pips won:        `+{stats.pips_won} pips`",
        f"  Pips lost:       `{stats.pips_lost} pips`",
        f"  Avg win:         `+{stats.avg_pips_win} pips`",
        f"  Avg loss:        `{stats.avg_pips_loss} pips`",
        f"  Best trade:      `+{stats.best_trade_pips} pips`",
        f"  Worst trade:     `{stats.worst_trade_pips} pips`",
        f"",
        f"�📐 *Risk/Reward*",
        f"  Avg RR target:   `1:{stats.avg_rr_target}`",
        f"  Avg RR achieved: `1:{stats.avg_rr_achieved}`",
        f"  Best RR:         `1:{stats.best_rr}`",
        f"  Worst RR:        `{stats.worst_rr}`",
        f"",
        f"🔥 *Streaks*",
        f"  Current:         {streak_str}",
        f"  Best win streak: `{stats.best_win_streak}`",
        f"  Worst loss run:  `{stats.worst_loss_streak}`",
        f"",
        f"📈 *Last 10 Trades*",
        f"  {last10_str}",
    ]

    # By timeframe
    if stats.by_timeframe:
        lines += [f"", f"⏱ *By Timeframe*"]
        for tf, d in sorted(stats.by_timeframe.items(), key=lambda x: -x[1]["win_rate"]):
            wr_e = "🟢" if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else "🔴"
            lines.append(
                f"  {wr_e} `{tf}`: {d['wins']}W/{d['losses']}L "
                f"= `{d['win_rate']}%` `{bar(d['win_rate'])}`"
            )

    # By session
    if stats.by_session:
        lines += [f"", f"🕐 *By Session*"]
        for sess, d in sorted(stats.by_session.items(), key=lambda x: -x[1]["win_rate"]):
            wr_e = "🟢" if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else "🔴"
            lines.append(
                f"  {wr_e} `{sess}`: {d['wins']}W/{d['losses']}L "
                f"= `{d['win_rate']}%` `{bar(d['win_rate'])}`"
            )

    # By confidence
    if stats.by_confidence:
        lines += [f"", f"🔥 *By Confidence*"]
        for conf, d in sorted(stats.by_confidence.items(), key=lambda x: -x[1]["win_rate"]):
            conf_e = "🔥" if conf == "HIGH" else "⚡" if conf == "MEDIUM" else "💡"
            lines.append(
                f"  {conf_e} `{conf}`: {d['wins']}W/{d['losses']}L "
                f"= `{d['win_rate']}%`"
            )

    # By direction
    if stats.by_direction:
        lines += [f"", f"↕️ *By Direction*"]
        for direction, d in stats.by_direction.items():
            d_e = "🟢" if direction == "BUY" else "🔴"
            lines.append(
                f"  {d_e} `{direction}`: {d['wins']}W/{d['losses']}L "
                f"= `{d['win_rate']}%`"
            )

    # By day
    if stats.by_day:
        lines += [f"", f"📅 *By Day of Week*"]
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        for day in day_order:
            if day in stats.by_day:
                d = stats.by_day[day]
                wr_e = "🟢" if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else "🔴"
                lines.append(
                    f"  {wr_e} `{day[:3]}`: {d['wins']}W/{d['losses']}L "
                    f"= `{d['win_rate']}%`"
                )

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"_/history — view trade list_",
        f"_/outcome — mark a trade result_",
        f"_/performance 30 — last 30 days_",
    ]

    full = "\n".join(lines)
    if len(full) > 4000:
        full = full[:3950] + "\n_...truncated_"
    return full


# ---------------------------------------------------------------------------
# /history [limit] — trade history list
# ---------------------------------------------------------------------------

@restricted
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.journal import get_journal
    journal = get_journal()
    args    = context.args
    limit   = int(args[0]) if args and args[0].isdigit() else 20
    records = journal.get_all()[:limit]

    if not records:
        await update.message.reply_text("📋 No trade history yet.")
        return

    lines = [f"📋 *TRADE HISTORY* (last {min(limit, len(records))})"]
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

    for r in records:
        if r.outcome == "WIN":
            emoji = "✅"
        elif r.outcome == "LOSS":
            emoji = "❌"
        elif r.outcome == "PENDING":
            emoji = "⏳"
        elif r.outcome == "BREAKEVEN":
            emoji = "➖"
        else:
            emoji = "🚫"

        action_e = "🟢" if r.action == "BUY" else "🔴"
        date_str = r.timestamp[:10]

        # Show pips if outcome known, else show target pips
        if r.outcome in ("WIN", "LOSS", "BREAKEVEN") and r.outcome_pips != 0:
            pip_str = f"`{'+' if r.outcome_pips >= 0 else ''}{r.outcome_pips:.0f}p`"
        else:
            pip_str = f"SL:`{r.sl_pips:.0f}p` TP:`{r.tp_pips:.0f}p`"

        rr_str = f"RR:`{r.outcome_rr_achieved}`" if r.outcome_rr_achieved else f"RR:`{r.rr_ratio}(t)`"

        lines.append(
            f"{emoji} {action_e} `{r.action}` `{r.timeframe}` "
            f"@ `{r.entry}` | {pip_str} | {rr_str} | "
            f"`{r.session}` | {date_str}"
        )
        if r.outcome_note:
            lines.append(f"   _{r.outcome_note}_")

    lines += [
        f"",
        f"_/performance — full stats dashboard_",
        f"_/outcome — mark pending trades_",
    ]

    full = "\n".join(lines)
    if len(full) > 4000:
        full = full[:3950] + "\n_...truncated_"
    await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /outcome — updated to use journal
# ---------------------------------------------------------------------------

@restricted
async def cmd_outcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.journal import get_journal
    from app.memory import get_memory
    journal = get_journal()
    memory  = get_memory()
    args    = context.args

    if len(args) < 2:
        pending = journal.get_pending()
        if not pending:
            await update.message.reply_text(
                "No pending trades.\n\nUsage: `/outcome <id> <WIN|LOSS|BREAKEVEN> [price] [note]`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = ["⏳ *Pending Trades* (need outcome):"]
        for r in pending[-10:]:
            lines.append(
                f"  `{r.id}`\n"
                f"  {r.action} {r.symbol} {r.timeframe} | "
                f"Entry:`{r.entry}` SL:`{r.stop_loss}` TP:`{r.take_profit}`"
            )
        lines.append("\nUsage: `/outcome <id> WIN 4650.0 TP hit`")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    trade_id = args[0]
    outcome  = args[1].upper()
    price    = float(args[2]) if len(args) > 2 else 0.0
    note     = " ".join(args[3:]) if len(args) > 3 else ""

    if outcome not in ("WIN", "LOSS", "BREAKEVEN", "CANCELLED"):
        await update.message.reply_text(
            "❌ Outcome must be: `WIN | LOSS | BREAKEVEN | CANCELLED`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    rec = journal.set_outcome(trade_id, outcome, price, note)
    if rec:
        # Also update memory
        memory.update_outcome(trade_id, outcome, price, note)

        emoji    = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "➖"
        pip_line = f"Pips: `{'+' if rec.outcome_pips >= 0 else ''}{rec.outcome_pips:.0f} pips`\n" if rec.outcome_pips else ""
        rr_line  = f"R:R achieved: `{rec.outcome_rr_achieved}`\n" if rec.outcome_rr_achieved else ""

        await update.message.reply_text(
            f"{emoji} *Outcome Recorded*\n\n"
            f"Trade: `{trade_id}`\n"
            f"Result: *{outcome}*\n"
            f"Price: `{price}`\n"
            f"{pip_line}"
            f"{rr_line}"
            f"Note: _{note or 'none'}_\n\n"
            f"_Use /performance to see updated stats_",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            f"❌ Trade `{trade_id}` not found.\nUse `/outcome` to see pending trades.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ---------------------------------------------------------------------------
# /setalert <price> [note]  — set a custom price alert
# ---------------------------------------------------------------------------

@restricted
async def cmd_setalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from datetime import datetime, timezone
    user_id = update.effective_user.id
    args    = context.args

    if not args:
        await update.message.reply_text(
            "*Set a Price Alert*\n\n"
            "Usage: `/setalert <price> [note]`\n\n"
            "Examples:\n"
            "`/setalert 4580` — alert when price reaches 4580\n"
            "`/setalert 4650 Resistance zone`\n\n"
            "_The bot will notify you when price gets within 30 pips of your level._",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        price = float(args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid price. Example: `/setalert 4580`", parse_mode=ParseMode.MARKDOWN)
        return

    note = " ".join(args[1:]) if len(args) > 1 else f"Custom level @ {price}"
    now  = datetime.now(timezone.utc)

    alert = PriceAlert(
        id=f"CUSTOM_{user_id}_{int(price*100)}_{now.strftime('%H%M%S')}",
        user_id=user_id,
        symbol=SYMBOL,
        timeframe="H1",
        alert_type="CUSTOM",
        direction="BOTH",
        level_price=price,
        level_top=price + 0.5,
        level_bottom=price - 0.5,
        proximity_pips=30,
        description=note,
        created_at=now.isoformat(),
    )

    store = get_alert_store()
    store.add(alert)

    await update.message.reply_text(
        f"✅ *Alert Set*\n\n"
        f"🎯 Level: `{price}`\n"
        f"📏 Trigger: within `30 pips`\n"
        f"📝 Note: _{note}_\n\n"
        f"_You'll be notified when {SYMBOL} approaches `{price}`_\n"
        f"_Use /myalerts to see all active alerts_",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /myalerts — show active alerts
# ---------------------------------------------------------------------------

@restricted
async def cmd_myalerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from app.tools import get_current_price
    from app.pip_utils import price_to_pips
    user_id = update.effective_user.id
    store   = get_alert_store()
    active  = store.get_active(user_id)
    price   = get_current_price(SYMBOL).get("mid", 0)

    if not active:
        await update.message.reply_text(
            "📭 *No Active Alerts*\n\n"
            "_Use /setalert to set a custom price alert_\n"
            "_Alerts are also set automatically when the scanner finds key SMC levels_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = [
        f"🔔 *Active Price Alerts* ({len(active)})",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💰 Current: `{price}`",
        "",
    ]

    type_icons = {
        "ORDER_BLOCK": "📦", "FVG": "🕳",
        "LIQUIDITY": "💧", "CUSTOM": "🎯",
    }

    for a in active[:15]:
        dist_pips = price_to_pips(abs(float(price) - a.level_price), a.symbol)
        d_icon    = "🟢" if a.direction == "BUY" else "🔴" if a.direction == "SELL" else "⚪"
        t_icon    = type_icons.get(a.alert_type, "⚡")
        lines.append(
            f"{t_icon} {d_icon} `{a.level_price:.2f}` "
            f"({dist_pips:.0f} pips away)\n"
            f"   _{a.description}_\n"
            f"   `{a.id}`"
        )

    lines += [
        "",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"_/cancelalert <id> — cancel specific alert_",
        f"_/cancelalerts — cancel all alerts_",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /cancelalert <id>
# ---------------------------------------------------------------------------

@restricted
async def cmd_cancelalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args    = context.args

    if not args:
        await update.message.reply_text(
            "Usage: `/cancelalert <alert_id>`\n_Use /myalerts to see IDs_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    store = get_alert_store()
    ok    = store.delete(args[0], user_id)

    if ok:
        await update.message.reply_text(f"✅ Alert `{args[0]}` cancelled.", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(f"❌ Alert not found: `{args[0]}`", parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# /cancelalerts — cancel all
# ---------------------------------------------------------------------------

@restricted
async def cmd_cancelalerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    store   = get_alert_store()
    count   = store.delete_all(user_id)
    await update.message.reply_text(
        f"✅ Cancelled `{count}` alert(s).",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# MT5 Setup — multi-step conversation
# ---------------------------------------------------------------------------

MT5_LOGIN, MT5_PASSWORD, MT5_SERVER = range(3)


@restricted
async def cmd_mt5setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start MT5 setup conversation."""
    await update.message.reply_text(
        "🖥 *MT5 Account Setup*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "I'll connect your personal MT5 account.\n"
        "Your credentials are stored locally and never shared.\n\n"
        "*Step 1 of 3*\n"
        "Enter your MT5 *account number* (login):\n\n"
        "_Example: `435557033`_\n\n"
        "Send /cancel to stop.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return MT5_LOGIN


async def mt5_get_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    try:
        login = int(text)
        if login <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid account number. Please enter a numeric login.\n_Example: `435557033`_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return MT5_LOGIN

    context.user_data["mt5_login"] = login
    await update.message.reply_text(
        f"✅ Login: `{login}`\n\n"
        f"*Step 2 of 3*\n"
        f"Enter your MT5 *password*:\n\n"
        f"_Your password is stored securely on this server only._",
        parse_mode=ParseMode.MARKDOWN,
    )
    return MT5_PASSWORD


async def mt5_get_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    password = update.message.text.strip()

    # Delete the message immediately — password must not stay in chat
    try:
        await update.message.delete()
    except Exception:
        pass

    if len(password) < 3:
        await update.effective_chat.send_message(
            "❌ Password too short. Please try again.\n_Your message was deleted for security._",
            parse_mode=ParseMode.MARKDOWN,
        )
        return MT5_PASSWORD

    context.user_data["mt5_password"] = password

    await update.effective_chat.send_message(
        "✅ Password received and deleted from chat.\n\n"
        "*Step 3 of 3*\n"
        "Enter your broker *server name*:\n\n"
        "Find it in MT5 → File → Login → Server field\n\n"
        "_Examples:_\n"
        "`Exness-MT5Trial9`\n"
        "`ICMarkets-Demo02`\n"
        "`XM-MT5`\n"
        "`Pepperstone-Demo`",
        parse_mode=ParseMode.MARKDOWN,
    )
    return MT5_SERVER


async def mt5_get_server(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    server = update.message.text.strip()
    if len(server) < 3:
        await update.message.reply_text("❌ Invalid server name. Try again.")
        return MT5_SERVER

    user_id  = update.effective_user.id
    login    = context.user_data.get("mt5_login", 0)
    password = context.user_data.get("mt5_password", "")

    msg = await update.message.reply_text(
        f"🔌 Testing connection to `{server}`...",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Save credentials
    store = get_profile_store()
    store.set_mt5_credentials(user_id, login, password, server)
    profile = store.get(user_id)

    # Test connection
    status = connect_user_mt5(profile)

    if status.connected:
        mode_emoji = "🟡" if status.trade_mode == "DEMO" else "🔴"
        await msg.edit_text(
            f"✅ *MT5 Connected Successfully!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 Account: `{status.account_login}` — {status.account_name}\n"
            f"🏦 Broker:  `{status.broker}`\n"
            f"🌐 Server:  `{status.server}`\n"
            f"{mode_emoji} Mode:    *{status.trade_mode}*\n"
            f"💰 Balance: `${status.balance:,.2f}`\n"
            f"📈 Equity:  `${status.equity:,.2f}`\n"
            f"⚡ Leverage: `1:{status.leverage}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Your account balance has been updated in your profile._\n"
            f"_Use /mt5status to check anytime._",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await msg.edit_text(
            f"❌ *Connection Failed*\n\n"
            f"`{status.error}`\n\n"
            f"*Common fixes:*\n"
            f"• Make sure MT5 terminal is open on your PC\n"
            f"• Check the server name is exact\n"
            f"• Verify login and password\n\n"
            f"_Use /mt5setup to try again_",
            parse_mode=ParseMode.MARKDOWN,
        )

    # Clear sensitive data from context
    context.user_data.clear()
    return ConversationHandler.END


async def mt5_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("❌ MT5 setup cancelled.")
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# /mt5connect — connect using saved credentials
# ---------------------------------------------------------------------------

@restricted
async def cmd_mt5connect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)

    if not profile.mt5_login:
        await update.message.reply_text(
            "⚙️ No MT5 credentials saved.\n\n"
            "Use /mt5setup to configure your MT5 account.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    msg = await update.message.reply_text(
        f"🔌 Connecting to MT5 account `{profile.mt5_login}`...",
        parse_mode=ParseMode.MARKDOWN,
    )

    status = connect_user_mt5(profile)

    if status.connected:
        mode_emoji = "🟡" if status.trade_mode == "DEMO" else "🔴"
        await msg.edit_text(
            f"✅ *MT5 Connected*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 `{status.account_login}` — {status.account_name}\n"
            f"🏦 `{status.broker}`\n"
            f"🌐 `{status.server}`\n"
            f"{mode_emoji} *{status.trade_mode}*\n"
            f"💰 `${status.balance:,.2f}`\n"
            f"📈 `${status.equity:,.2f}`\n"
            f"⚡ `1:{status.leverage}`",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await msg.edit_text(
            f"❌ *Connection Failed*\n\n`{status.error}`\n\n"
            f"_Make sure MT5 terminal is open on your Windows PC_",
            parse_mode=ParseMode.MARKDOWN,
        )


# ---------------------------------------------------------------------------
# /mt5status — show current MT5 status
# ---------------------------------------------------------------------------

@restricted
async def cmd_mt5status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = profile_store.get(user_id)

    if not profile.mt5_login:
        await update.message.reply_text(
            "⚙️ No MT5 credentials.\n\nUse /mt5setup to connect your MT5 account.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    msg = await update.message.reply_text("📊 Fetching MT5 status...")
    status = get_user_mt5_status(user_id)

    if not status.connected:
        await msg.edit_text(
            f"📊 *MT5 Status*\n\n"
            f"Account: `{profile.mt5_login}`\n"
            f"Server: `{profile.mt5_server}`\n"
            f"Status: ❌ Not connected\n\n"
            f"_{status.error}_\n\n"
            f"Use /mt5connect to reconnect.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    from app.user_mt5 import _user_traders
    trader = _user_traders.get(user_id)
    positions = trader.get_open_positions(SYMBOL) if trader else []
    sym_info  = trader.get_symbol_info(SYMBOL) if trader else {}

    mode_emoji = "🟡" if status.trade_mode == "DEMO" else "🔴"
    lines = [
        f"📊 *MT5 STATUS*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"👤 `{status.account_login}` — {status.account_name}",
        f"🏦 `{status.broker}`",
        f"{mode_emoji} *{status.trade_mode}*",
        f"💰 Balance:  `${status.balance:,.2f}`",
        f"📈 Equity:   `${status.equity:,.2f}`",
        f"🆓 Free:     `${status.margin_free:,.2f}`",
        f"⚡ Leverage: `1:{status.leverage}`",
    ]

    if sym_info and "bid" in sym_info:
        lines += [
            f"",
            f"💹 *{SYMBOL}*",
            f"  Bid: `{sym_info.get('bid')}` | Ask: `{sym_info.get('ask')}`",
            f"  Spread: `{sym_info.get('spread')} pts`",
        ]

    if positions:
        lines += [f"", f"🔄 *Open Positions ({len(positions)})*"]
        for p in positions:
            pnl   = p.get("profit", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            d_icon = "🟢" if p.get("direction") == "BUY" else "🔴"
            lines.append(
                f"  {emoji} #{p.get('ticket')} {d_icon} `{p.get('direction')}` "
                f"{p.get('volume')} lots @ `{p.get('price_open')}` "
                f"P&L: `${pnl:.2f}`"
            )
    else:
        lines += [f"", f"_No open positions_"]

    await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())


# ---------------------------------------------------------------------------
# /mt5disconnect — disconnect and clear credentials
# ---------------------------------------------------------------------------

@restricted
async def cmd_mt5disconnect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    disconnect_user_mt5(user_id)
    profile_store.clear_mt5_credentials(user_id)
    await update.message.reply_text(
        "✅ MT5 disconnected and credentials cleared.\n\n"
        "_Use /mt5setup to reconnect with new credentials._",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /mt5close <ticket> or /mt5close all
# ---------------------------------------------------------------------------

@restricted
async def cmd_mt5close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args    = context.args

    if not args:
        await update.message.reply_text(
            "Usage:\n`/mt5close 12345` — close specific ticket\n`/mt5close all` — close all",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    from app.user_mt5 import _user_traders
    trader = _user_traders.get(user_id)

    if not trader:
        await update.message.reply_text(
            "❌ MT5 not connected. Use /mt5connect first.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    msg = await update.message.reply_text("⏳ Closing position(s)...")

    try:
        if args[0].lower() == "all":
            results = trader.close_all_positions(SYMBOL)
            lines   = ["🔄 *Close All Results*"]
            for r in results:
                emoji = "✅" if r.success else "❌"
                lines.append(f"  {emoji} {r.message}")
            if not results:
                lines.append("  No open positions.")
            await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        else:
            ticket = int(args[0])
            result = trader.close_position(ticket)
            emoji  = "✅" if result.success else "❌"
            await msg.edit_text(
                f"{emoji} *Close #{ticket}*\n\n{result.message}",
                parse_mode=ParseMode.MARKDOWN,
            )
    except ValueError:
        await msg.edit_text("❌ Invalid ticket. Use `/mt5close 12345`", parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        await msg.edit_text(f"❌ Error: {exc}")


async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❓ Unknown command. Use /start to see all commands.")


# ---------------------------------------------------------------------------
# Morning briefing scheduler — runs daily at configured time
# ---------------------------------------------------------------------------

async def _morning_briefing_scheduler(app: Application):
    """
    Sends morning briefing to all subscribers every day at BRIEFING_HOUR UTC.
    Default: 07:00 UTC (London open).
    """
    import datetime as dt
    briefing_hour   = settings.briefing_hour
    briefing_minute = settings.briefing_minute

    logger.info("Morning briefing scheduler: daily at %02d:%02d UTC", briefing_hour, briefing_minute)

    while True:
        now  = dt.datetime.now(dt.timezone.utc)
        next_run = now.replace(hour=briefing_hour, minute=briefing_minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += dt.timedelta(days=1)

        wait_seconds = (next_run - now).total_seconds()
        logger.info("Next morning briefing in %.0f seconds (%s UTC)", wait_seconds, next_run.strftime("%H:%M"))
        await asyncio.sleep(wait_seconds)

        # Send to all subscribers
        subscribers = profile_store.all_alert_subscribers()
        if not subscribers:
            continue

        logger.info("Sending morning briefing to %d subscribers...", len(subscribers))
        try:
            data  = await _post("/briefing", {"symbol": SYMBOL})
            parts = _fmt_briefing(data)
        except Exception as exc:
            logger.error("Morning briefing generation failed: %s", exc)
            continue

        for user in subscribers:
            try:
                for part in parts:
                    await app.bot.send_message(
                        chat_id=user.user_id,
                        text=part,
                        parse_mode="Markdown",
                    )
                logger.info("Morning briefing sent to user %d", user.user_id)
            except Exception as exc:
                logger.error("Failed to send briefing to user %d: %s", user.user_id, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = Application.builder().token(settings.telegram_bot_token).build()

    # Commands
    app.add_handler(CommandHandler("start",         cmd_start))
    app.add_handler(CommandHandler("analyze",       cmd_analyze))
    app.add_handler(CommandHandler("signal",        cmd_signal))
    app.add_handler(CommandHandler("trade",         cmd_trade))
    app.add_handler(CommandHandler("briefing",      cmd_briefing))
    app.add_handler(CommandHandler("swing",         cmd_swing))
    app.add_handler(CommandHandler("status",        cmd_status))
    app.add_handler(CommandHandler("setbalance",    cmd_setbalance))
    app.add_handler(CommandHandler("setrisk",       cmd_setrisk))
    app.add_handler(CommandHandler("settf",         cmd_settf))
    app.add_handler(CommandHandler("setconfidence", cmd_setconfidence))
    app.add_handler(CommandHandler("alerts",        cmd_alerts))
    app.add_handler(CommandHandler("myprofile",     cmd_myprofile))
    app.add_handler(CommandHandler("mt5connect",    cmd_mt5connect))
    app.add_handler(CommandHandler("mt5status",     cmd_mt5status))
    app.add_handler(CommandHandler("mt5close",      cmd_mt5close))
    app.add_handler(CommandHandler("mt5disconnect", cmd_mt5disconnect))

    # MT5 setup conversation
    mt5_conv = ConversationHandler(
        entry_points=[CommandHandler("mt5setup", cmd_mt5setup)],
        states={
            MT5_LOGIN:    [MessageHandler(filters.TEXT & ~filters.COMMAND, mt5_get_login)],
            MT5_PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, mt5_get_password)],
            MT5_SERVER:   [MessageHandler(filters.TEXT & ~filters.COMMAND, mt5_get_server)],
        },
        fallbacks=[CommandHandler("cancel", mt5_cancel)],
    )
    app.add_handler(mt5_conv)
    app.add_handler(CommandHandler("memory",        cmd_memory))
    app.add_handler(CommandHandler("outcome",       cmd_outcome))
    app.add_handler(CommandHandler("addlesson",     cmd_addlesson))
    app.add_handler(CommandHandler("performance",   cmd_performance))
    app.add_handler(CommandHandler("history",       cmd_history))
    app.add_handler(CommandHandler("setalert",      cmd_setalert))
    app.add_handler(CommandHandler("myalerts",      cmd_myalerts))
    app.add_handler(CommandHandler("cancelalert",   cmd_cancelalert))
    app.add_handler(CommandHandler("cancelalerts",  cmd_cancelalerts))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))

    # Start background tasks
    async def post_init(application: Application):
        if settings.scanner_enabled:
            scanner = SignalScanner(application, profile_store)
            asyncio.create_task(scanner.start())
            logger.info("Signal scanner started.")

        # Price alert watcher
        from app.price_alerts import PriceWatcher, get_alert_store
        watcher = PriceWatcher(application, get_alert_store())
        asyncio.create_task(watcher.start())
        logger.info("Price alert watcher started.")

        # Morning briefing scheduler
        asyncio.create_task(_morning_briefing_scheduler(application))
        logger.info("Morning briefing scheduler started.")

    app.post_init = post_init

    logger.info("Telegram bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
