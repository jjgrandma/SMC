"""
AI Decision Engine — multi-provider AI trading agent.

Supports: Groq (free) | Google Gemini (free) | OpenAI (paid)
Set AI_PROVIDER in .env to switch.

Full pipeline:
  1. get_user_settings()     — user risk profile
  2. get_active_trades()     — exposure check
  3. get_economic_calendar() — news filter
  4. MTFEngine.analyze()     — 1W→1D→4H→1H→15M cascade + confluence gate
  5. get_current_price()     — live price
  6. AI reasoning            — structured signal JSON
  7. Hard validation gate    — confluence count, HTF alignment, P/D zone
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings
from app.mtf_analysis import MTFEngine, mtf_to_dict, get_active_trades, get_user_settings
from app.tools import get_current_price, is_high_impact_news_window, get_economic_calendar
from app.smc_engine import analysis_to_dict
from app.dual_ai import dual_chat_json, dual_chat_narrative, single_chat
from app.memory import get_memory, OutcomeEvaluator
from app.journal import get_journal

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Legacy single-provider _chat (used only when both keys missing)
# ---------------------------------------------------------------------------

async def _chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 800,
    json_mode: bool = False,
) -> str:
    """Fallback: routes to whichever single provider is configured."""
    provider = settings.ai_provider.lower()
    try:
        return await single_chat(messages, provider, temperature, max_tokens, json_mode)
    except Exception as exc:
        # Try the other free provider
        fallback = "groq" if provider == "gemini" else "gemini"
        logger.warning("Primary provider %s failed, trying %s: %s", provider, fallback, exc)
        return await single_chat(messages, fallback, temperature, max_tokens, json_mode)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an advanced autonomous agentic AI Forex trading system.
You specialize in XAUUSD (Gold) trading using Smart Money Concepts (SMC).
You are NOT a chatbot. You are a professional trading decision engine.

STRICT RULES:
- NEVER hallucinate prices, candles, news, or trades.
- ONLY use the data provided to you in the user message.
- NEVER trade against Weekly or Daily HTF bias.
- A trade is ONLY valid if ALL conditions are met:
    ✔ HTF + LTF alignment confirmed
    ✔ Strong FVG OR Order Block present
    ✔ Liquidity sweep or target exists
    ✔ Entry in premium (SELL) or discount (BUY) zone
    ✔ Minimum 3 SMC confluences
- If ANY condition is not met → action must be "NO_TRADE"
- If HTF bias is unclear → "No clear higher timeframe bias. Stay out."

Signal output MUST be valid JSON:
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "rr_ratio": float,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "string — cite specific SMC structures used",
  "confluences": ["list of SMC confluences that validated this trade"],
  "key_levels": {"support": float, "resistance": float},
  "premium_discount": "premium" | "discount" | "equilibrium",
  "htf_bias": "bullish" | "bearish" | "ranging",
  "entry_timeframe": "string",
  "invalidation": "string"
}
""".strip()


# ---------------------------------------------------------------------------
# Trading Agent
# ---------------------------------------------------------------------------

class TradingAgent:
    def __init__(self, trader=None):
        self.mtf_engine = MTFEngine()
        self.trader     = trader
        self.memory     = get_memory()
        self.evaluator  = OutcomeEvaluator(self.memory)
        self.journal    = get_journal()

    # ------------------------------------------------------------------
    # TOOL: get_user_settings
    # ------------------------------------------------------------------

    def _tool_user_settings(self, user_profile=None) -> dict:
        return get_user_settings(user_profile)

    # ------------------------------------------------------------------
    # TOOL: get_active_trades
    # ------------------------------------------------------------------

    def _tool_active_trades(self) -> dict:
        return get_active_trades(self.trader)

    # ------------------------------------------------------------------
    # TOOL: get_economic_calendar
    # ------------------------------------------------------------------

    def _tool_news(self) -> dict:
        events = get_economic_calendar()
        blocked = is_high_impact_news_window("XAUUSD")
        return {
            "news_blocked": blocked,
            "upcoming_events": events[:5],
        }

    # ------------------------------------------------------------------
    # Full MTF analysis (narrative)
    # ------------------------------------------------------------------

    async def analyze(
        self,
        symbol: str,
        timeframe: str = "H1",
        user_profile=None,
    ) -> dict[str, Any]:

        price_info   = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data    = self._tool_news()
        user_cfg     = self._tool_user_settings(user_profile)
        active_trades = self._tool_active_trades()
        mtf_result   = self.mtf_engine.analyze(symbol, current_price)
        mtf_data     = mtf_to_dict(mtf_result)

        prompt = f"""
Perform a full multi-timeframe SMC analysis for {symbol}.

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_active_trades ===
{json.dumps(active_trades, indent=2)}

=== TOOL: get_economic_calendar ===
{json.dumps(news_data, indent=2)}

=== TOOL: MTF SMC Analysis (1W→1D→4H→1H→15M) ===
{json.dumps(mtf_data, indent=2)}

Provide a structured narrative covering:
1. Weekly and Daily bias — are they aligned?
2. HTF key levels (OB, FVG, Liquidity)
3. Premium / Discount zone status
4. MTF confluence count and what they are
5. Entry timeframe setup (H1 or M15)
6. Overall trade recommendation
7. What would invalidate the bias
""".strip()

        # ── Dual AI: Gemini deep analysis + Groq quick take ──
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        narrative, ai_meta = await dual_chat_narrative(messages, temperature=0.3, max_tokens=1500)

        return {
            "symbol":        symbol,
            "timeframe":     timeframe,
            "current_price": price_info,
            "news_blocked":  news_data["news_blocked"],
            "mtf_data":      mtf_data,
            "analysis":      narrative,
            "ai_meta":       ai_meta,
        }

    # ------------------------------------------------------------------
    # Trade signal — full validation pipeline
    # ------------------------------------------------------------------

    async def get_signal(
        self,
        symbol: str,
        timeframe: str = "H1",
        user_profile=None,
    ) -> dict[str, Any]:

        price_info    = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data     = self._tool_news()
        user_cfg      = self._tool_user_settings(user_profile)
        active_trades = self._tool_active_trades()

        # --- Hard gate 1: News ---
        if news_data["news_blocked"]:
            return _no_trade(symbol, timeframe, price_info,
                             "High-impact news event within 30 minutes. Trading blocked.",
                             news_blocked=True)

        # --- Hard gate 2: Existing exposure ---
        if active_trades["position_count"] > 0:
            return _no_trade(symbol, timeframe, price_info,
                             f"Already {active_trades['position_count']} open position(s). No overtrading.")

        # --- MTF cascade + confluence gate ---
        mtf_result = self.mtf_engine.analyze(symbol, current_price)
        mtf_data   = mtf_to_dict(mtf_result)

        # --- Hard gate 3: HTF alignment ---
        if not mtf_result.htf_aligned:
            return _no_trade(symbol, timeframe, price_info,
                             mtf_result.block_reason or "HTF not aligned.")

        # --- Hard gate 4: Minimum confluences ---
        if not mtf_result.trade_allowed:
            return _no_trade(symbol, timeframe, price_info,
                             mtf_result.block_reason or "Insufficient SMC confluences.")

        # --- AI signal generation ---
        # Inject memory context — bot learns from past signals
        memory_context = self.memory.get_ai_context(symbol, timeframe)

        # Also evaluate any pending signals against current price
        resolved = self.evaluator.evaluate_pending(current_price)
        if resolved:
            logger.info("Auto-resolved %d pending signals", len(resolved))

        prompt = f"""
Generate a precise SMC trade signal for {symbol}.

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_active_trades ===
{json.dumps(active_trades, indent=2)}

=== TOOL: get_economic_calendar ===
{json.dumps(news_data, indent=2)}

=== TOOL: MTF SMC Analysis ===
{json.dumps(mtf_data, indent=2)}

CONFIRMED PRE-CONDITIONS:
- HTF Aligned: {mtf_result.htf_aligned}
- Overall Bias: {mtf_result.overall_bias.upper()}
- Weekly Bias: {mtf_result.weekly_bias.upper()}
- Daily Bias: {mtf_result.daily_bias.upper()}
- Confluences found ({mtf_result.confluence_count}):
{chr(10).join('  - ' + c for c in mtf_result.confluence_details)}

{memory_context}

ENTRY RULES:
- Direction MUST match overall bias: {mtf_result.overall_bias.upper()}
- Entry MUST be at an unmitigated OB or FVG on H1 or M15
- Entry MUST be in {'DISCOUNT zone (BUY)' if mtf_result.overall_bias == 'bullish' else 'PREMIUM zone (SELL)' if mtf_result.overall_bias == 'bearish' else 'appropriate P/D zone'}
- Stop Loss: beyond the OB/FVG that triggered entry
- Take Profit: at the next liquidity zone / swing high or low
- R:R must be ≥ {settings.min_rr_ratio}
- USE the memory above — avoid patterns that previously failed, favour patterns that worked
- If no precise entry exists right now → return action: "NO_TRADE"

Respond ONLY with valid JSON matching the system prompt format.
""".strip()

        # ── Dual AI: Gemini (deep) + Groq (quick) in parallel ──
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        signal, ai_meta = await dual_chat_json(messages, temperature=0.1, max_tokens=700)

        if not signal or "action" not in signal:
            return _no_trade(symbol, timeframe, price_info, "AI response parse error.")

        # Attach metadata
        signal["symbol"]           = symbol
        signal["timeframe"]        = timeframe
        signal["current_price"]    = price_info
        signal["news_blocked"]     = news_data["news_blocked"]
        signal["mtf_data"]         = mtf_data
        signal["confluence_count"] = mtf_result.confluence_count
        signal["confluences"]      = signal.get("confluences", mtf_result.confluence_details)
        signal["ai_comparison"]    = ai_meta

        # Log signal to memory + journal
        if signal.get("action") != "NO_TRADE":
            sig_id = self.memory.log_signal(signal)
            signal["memory_id"] = sig_id
            trade_id = self.journal.add(signal)
            signal["trade_id"] = trade_id
            # Auto-evaluate any pending trades
            self.journal.auto_evaluate(current_price)

        return signal

    # ------------------------------------------------------------------
    # Swing trade idea
    # ------------------------------------------------------------------

    async def get_swing_idea(
        self,
        symbol: str,
        user_profile=None,
    ) -> dict[str, Any]:

        price_info    = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data     = self._tool_news()
        user_cfg      = self._tool_user_settings(user_profile)
        mtf_result    = self.mtf_engine.analyze(symbol, current_price)
        mtf_data      = mtf_to_dict(mtf_result)

        prompt = f"""
Generate a multi-day swing trade idea for {symbol}.

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_economic_calendar ===
{json.dumps(news_data, indent=2)}

=== TOOL: MTF SMC Analysis (W1+D1+H4 focus) ===
{json.dumps(mtf_data, indent=2)}

Provide a swing trade JSON with:
- action, entry, stop_loss, take_profit, rr_ratio, confidence
- targets: [TP1, TP2, TP3]
- duration: expected hold time
- reasoning: cite W1/D1 structure
- confluences: list of SMC confluences
- risk_factors: what could invalidate
- htf_bias, premium_discount, invalidation

Respond ONLY with valid JSON.
""".strip()

        # ── Dual AI: Gemini (deep) + Groq (quick) in parallel ──
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        idea, ai_meta = await dual_chat_json(messages, temperature=0.2, max_tokens=900)

        if not idea or "action" not in idea:
            idea = {"action": "NO_TRADE", "reasoning": "AI response parse error."}

        idea["symbol"]        = symbol
        idea["current_price"] = price_info
        idea["mtf_data"]      = mtf_data
        idea["ai_comparison"] = ai_meta
        return idea

    # ------------------------------------------------------------------
    # Manual trade signal — clean entry card, no auto-execution
    # ------------------------------------------------------------------

    async def get_manual_signal(
        self,
        symbol: str,
        timeframe: str = "H1",
        user_profile=None,
    ) -> dict[str, Any]:
        """
        Returns a detailed manual trade card.
        Includes full entry reasoning, SMC structure used,
        what to watch for confirmation, and invalidation.
        No auto-execution — for manual trader decision making.
        """
        price_info    = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data     = self._tool_news()
        user_cfg      = self._tool_user_settings(user_profile)
        active_trades = self._tool_active_trades()

        if news_data["news_blocked"]:
            return _no_trade(symbol, timeframe, price_info,
                             "High-impact news event within 30 minutes. Avoid trading.",
                             news_blocked=True)

        mtf_result = self.mtf_engine.analyze(symbol, current_price)
        mtf_data   = mtf_to_dict(mtf_result)

        # Inject memory
        memory_context = self.memory.get_ai_context(symbol, timeframe)
        self.evaluator.evaluate_pending(current_price)

        prompt = f"""
You are generating a MANUAL TRADE SIGNAL CARD for a human trader.
The trader will decide whether to enter — you are NOT executing automatically.
Be precise, educational, and cite every SMC structure you use.

Symbol: {symbol} | Timeframe: {timeframe}

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_economic_calendar ===
{json.dumps(news_data, indent=2)}

=== TOOL: MTF SMC Analysis ===
{json.dumps(mtf_data, indent=2)}

HTF Bias: {mtf_result.overall_bias.upper()}
HTF Aligned: {mtf_result.htf_aligned}
Confluences ({mtf_result.confluence_count}): {', '.join(mtf_result.confluence_details[:5])}

{memory_context}

Generate a JSON signal card with these EXACT fields:
{{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "entry": float,
  "entry_type": "market" | "limit" | "stop",
  "entry_zone": {{"from": float, "to": float}},
  "stop_loss": float,
  "take_profit": float,
  "tp2": float,
  "rr_ratio": float,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "htf_bias": "bullish" | "bearish" | "ranging",
  "premium_discount": "premium" | "discount" | "equilibrium",
  "entry_timeframe": "string",
  "smc_structure_used": "string — exact structure: OB/FVG/BOS/CHoCH + price level",
  "why_enter_now": "string — specific reason this is the right moment",
  "confirmation_needed": "string — what the trader should wait to see before entering",
  "reasoning": "string — full SMC reasoning chain",
  "confluences": ["list"],
  "key_levels": {{"support": float, "resistance": float}},
  "invalidation": "string — exact price/candle that kills this setup",
  "trade_management": "string — how to manage the trade after entry"
}}

If no valid setup: action = "NO_TRADE" with clear reasoning.
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        signal, ai_meta = await dual_chat_json(messages, temperature=0.1, max_tokens=900)

        if not signal or "action" not in signal:
            return _no_trade(symbol, timeframe, price_info, "AI response parse error.")

        signal["symbol"]           = symbol
        signal["timeframe"]        = timeframe
        signal["current_price"]    = price_info
        signal["news_blocked"]     = news_data["news_blocked"]
        signal["mtf_data"]         = mtf_data
        signal["confluence_count"] = mtf_result.confluence_count
        signal["ai_comparison"]    = ai_meta
        signal["mode"]             = "MANUAL"

        # Log to memory + journal
        if signal.get("action") != "NO_TRADE":
            sig_id = self.memory.log_signal(signal)
            signal["memory_id"] = sig_id
            trade_id = self.journal.add(signal)
            signal["trade_id"] = trade_id

        return signal

    # ------------------------------------------------------------------
    # Morning briefing — daily analysis sent every morning
    # ------------------------------------------------------------------

    async def get_morning_briefing(
        self,
        symbol: str,
        user_profile=None,
    ) -> dict[str, Any]:
        """
        Full morning briefing:
        - Previous day recap (D1 candle analysis)
        - Weekly context (W1 bias)
        - Multi-timeframe structure: W1, D1, H4, H1
        - Key levels to watch today
        - AI expectation for the session
        - High-impact news today
        - Trade plan for the day
        """
        from datetime import datetime, timezone
        from app.tools import get_market_data

        price_info    = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data     = self._tool_news()
        user_cfg      = self._tool_user_settings(user_profile)

        # Fetch data for all key timeframes
        mtf_result = self.mtf_engine.analyze(symbol, current_price)
        mtf_data   = mtf_to_dict(mtf_result)

        # Get D1 candles for previous day recap
        df_d1 = get_market_data(symbol, "D1")
        prev_day = {}
        if df_d1 is not None and len(df_d1) >= 2:
            prev = df_d1.iloc[-2]
            curr = df_d1.iloc[-1]
            prev_day = {
                "date":        str(df_d1.index[-2].date()),
                "open":        round(float(prev["open"]), 3),
                "high":        round(float(prev["high"]), 3),
                "low":         round(float(prev["low"]), 3),
                "close":       round(float(prev["close"]), 3),
                "range":       round(float(prev["high"]) - float(prev["low"]), 3),
                "direction":   "bullish" if prev["close"] > prev["open"] else "bearish",
                "body_size":   round(abs(float(prev["close"]) - float(prev["open"])), 3),
                "today_open":  round(float(curr["open"]), 3),
            }

        today = datetime.now(timezone.utc)
        session = "London" if 7 <= today.hour < 16 else "New York" if 13 <= today.hour < 22 else "Asian"

        prompt = f"""
You are generating a MORNING BRIEFING for a professional Gold (XAUUSD) trader.
Today is {today.strftime('%A, %B %d, %Y')} — {session} session.

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_economic_calendar (today's events) ===
{json.dumps(news_data, indent=2)}

=== PREVIOUS DAY (D1) CANDLE ===
{json.dumps(prev_day, indent=2)}

=== TOOL: MTF SMC Analysis (W1→D1→H4→H1) ===
{json.dumps(mtf_data, indent=2)}

Generate a comprehensive morning briefing JSON:
{{
  "date": "string",
  "session": "string",
  "current_price": float,
  "weekly_bias": "bullish" | "bearish" | "ranging",
  "daily_bias": "bullish" | "bearish" | "ranging",
  "htf_aligned": bool,
  "previous_day_recap": "string — what happened yesterday, key moves, candle type",
  "weekly_context": "string — where are we in the weekly range",
  "key_levels_today": {{
    "major_resistance": [float, float],
    "major_support": [float, float],
    "daily_high": float,
    "daily_low": float,
    "weekly_high": float,
    "weekly_low": float,
    "equilibrium": float
  }},
  "active_order_blocks": "string — key OBs to watch",
  "active_fvgs": "string — key FVGs to watch",
  "liquidity_targets": "string — where liquidity sits above/below",
  "premium_discount_now": "string — is price in premium, discount or EQ",
  "session_expectation": "string — what you expect price to do today based on structure",
  "trade_plan": {{
    "bias": "BUY" | "SELL" | "WAIT",
    "ideal_entry_zone": "string",
    "watch_for": "string — what confirmation to look for",
    "avoid_if": "string — conditions that cancel the plan"
  }},
  "news_impact": "string — how today's news events affect the plan",
  "experience_note": "string — a professional insight based on Gold's typical behavior in this structure",
  "risk_reminder": "string — risk management reminder for today"
}}
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        briefing, ai_meta = await dual_chat_json(messages, temperature=0.3, max_tokens=1200)

        if not briefing:
            briefing = {"error": "AI failed to generate briefing."}

        briefing["symbol"]       = symbol
        briefing["price_info"]   = price_info
        briefing["prev_day"]     = prev_day
        briefing["mtf_data"]     = mtf_data
        briefing["ai_meta"]      = ai_meta
        briefing["generated_at"] = today.strftime("%Y-%m-%d %H:%M UTC")
        return briefing

    async def get_status(self, symbol: str, user_profile=None) -> dict[str, Any]:
        price_info    = get_current_price(symbol)
        active_trades = self._tool_active_trades()
        user_cfg      = self._tool_user_settings(user_profile)
        return {
            "symbol":        symbol,
            "current_price": price_info,
            "active_trades": active_trades["open_positions"],
            "account":       active_trades.get("account", {}),
            "user_settings": user_cfg,
            "message":       "Live data. Connect MT5 for real positions.",
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _no_trade(
    symbol: str,
    timeframe: str,
    price_info: dict,
    reason: str,
    news_blocked: bool = False,
) -> dict:
    return {
        "action":        "NO_TRADE",
        "symbol":        symbol,
        "timeframe":     timeframe,
        "current_price": price_info,
        "news_blocked":  news_blocked,
        "reasoning":     reason,
        "confidence":    "LOW",
        "entry":         None,
        "stop_loss":     None,
        "take_profit":   None,
        "rr_ratio":      None,
        "confluences":   [],
    }
