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
You are an institutional SMC analyst and transparent reasoning engine.
You specialize in XAUUSD (Gold) using Smart Money Concepts.

YOUR IDENTITY:
✔ Institutional SMC analyst — think like a prop trader
✔ Transparent reasoning engine — always show your work
✔ Memory-enhanced assistant — use provided lessons and patterns
✔ Confluence-scoring system — count and weight each confluence
✔ Lesson-aware market analyst — reference specific lessons when relevant

NOT:
✘ Random signal generator
✘ Black-box AI
✘ Autonomous trading bot — NEVER auto-trade

STRICT RULES:
- NEVER hallucinate prices, candles, news, or trades
- ONLY use data provided in the user message
- NEVER trade against Weekly or Daily HTF bias
- NEVER self-modify core trading logic
- NEVER change user-defined strategy rules

TRADE VALIDATION (ALL must be met):
  ✔ HTF + LTF alignment confirmed
  ✔ Strong FVG OR Order Block present
  ✔ Liquidity sweep or target exists
  ✔ Entry in premium (SELL) or discount (BUY) zone
  ✔ Minimum 3 SMC confluences

IF NO TRADE — you MUST explain WHY with this format:
{
  "action": "NO_TRADE",
  "reason": "string — primary reason",
  "missing_confluences": ["✘ list each missing confluence"],
  "present_confluences": ["✔ list what IS present"],
  "setup_score": 0-100,
  "recommendation": "string — what to wait for",
  "reasoning": "string"
}

IF TRADE — signal output:
{
  "action": "BUY" | "SELL",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "rr_ratio": float,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "confluence_score": 0-100,
  "confluences": ["✔ each confluence with price level"],
  "lessons_used": ["lesson titles that influenced this decision"],
  "smc_structure_used": "exact structure + price level",
  "reasoning": "full transparent reasoning chain",
  "key_levels": {"support": float, "resistance": float},
  "premium_discount": "premium" | "discount" | "equilibrium",
  "htf_bias": "bullish" | "bearish" | "ranging",
  "entry_timeframe": "string",
  "invalidation": "exact price/candle that kills this setup"
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
            explanation = self.memory.explain_no_trade(
                symbol, timeframe, mtf_data,
                mtf_result.confluence_details,
                mtf_result.block_reason or "HTF not aligned.",
            )
            return _no_trade_explained(symbol, timeframe, price_info, explanation)

        # --- Hard gate 4: Minimum confluences ---
        if not mtf_result.trade_allowed:
            explanation = self.memory.explain_no_trade(
                symbol, timeframe, mtf_data,
                mtf_result.confluence_details,
                mtf_result.block_reason or "Insufficient SMC confluences.",
            )
            return _no_trade_explained(symbol, timeframe, price_info, explanation)

        # --- AI signal generation ---
        # Inject memory context — bot learns from past signals
        memory_context = self.memory.get_ai_context(symbol, timeframe)

        # Also evaluate any pending signals against current price
        resolved = self.evaluator.evaluate_pending(current_price)
        if resolved:
            logger.info("Auto-resolved %d pending signals", len(resolved))

        # Fetch LTF data for precise entry refinement
        from app.tools import get_market_data
        df_ltf_m15 = get_market_data(symbol, "M15")
        df_ltf_m5  = get_market_data(symbol, "M5")
        ltf_engine = self.mtf_engine._smc
        ltf_m15    = analysis_to_dict(ltf_engine.analyze(df_ltf_m15)) if df_ltf_m15 is not None else {}
        ltf_m5     = analysis_to_dict(ltf_engine.analyze(df_ltf_m5))  if df_ltf_m5  is not None else {}

        prompt = f"""
Generate a precise SMC trade signal for {symbol}.

TWO-PHASE ENTRY SYSTEM (MANDATORY):
  Phase 1 — HTF Context (H4/D1): defines bias, OB location, liquidity targets
  Phase 2 — LTF Entry (M15/M5): refines exact entry with TIGHT stop loss

ENTRY TIMEFRAME RULES:
  - NEVER use H4 as the entry timeframe
  - Entry candle MUST be on M15 or M5
  - Stop Loss MUST be placed on M15/M5 structure (NOT H4 structure)
  - This keeps SL tight (20-50 pips) while TP targets H4/D1 liquidity (100-300 pips)
  - R:R must be ≥ {settings.min_rr_ratio} using the LTF SL

=== TOOL: get_current_price ===
{json.dumps(price_info, indent=2)}

=== TOOL: get_user_settings ===
{json.dumps(user_cfg, indent=2)}

=== TOOL: get_active_trades ===
{json.dumps(active_trades, indent=2)}

=== TOOL: get_economic_calendar ===
{json.dumps(news_data, indent=2)}

=== HTF CONTEXT (H4/D1 — bias and OB location) ===
{json.dumps(mtf_data, indent=2)}

=== LTF ENTRY REFINEMENT (M15 — find precise entry) ===
{json.dumps(ltf_m15, indent=2)}

=== LTF ENTRY REFINEMENT (M5 — tightest entry) ===
{json.dumps(ltf_m5, indent=2)}

CONFIRMED HTF PRE-CONDITIONS:
- HTF Aligned: {mtf_result.htf_aligned}
- Overall Bias: {mtf_result.overall_bias.upper()}
- Weekly Bias: {mtf_result.weekly_bias.upper()}
- Daily Bias: {mtf_result.daily_bias.upper()}
- Confluences ({mtf_result.confluence_count}):
{chr(10).join('  - ' + c for c in mtf_result.confluence_details)}

{memory_context}

STRICT ENTRY RULES:
1. Direction MUST match HTF bias: {mtf_result.overall_bias.upper()}
2. Identify the H4/D1 OB or FVG zone (this is your TARGET AREA)
3. Wait for price to reach that zone
4. On M15 or M5: look for a BOS, CHoCH, or OB INSIDE the H4 zone
5. Entry = M15/M5 OB or FVG level (NOT the H4 level)
6. Stop Loss = below/above the M15/M5 structure (tight, 20-60 pips max for Gold)
7. Take Profit = H4/D1 liquidity target (100-300 pips)
8. This gives R:R of 1:3 to 1:8 with tight risk
9. If price is NOT currently at an H4 OB/FVG → NO_TRADE (wait for it)
10. R:R must be ≥ {settings.min_rr_ratio}

STOP LOSS SIZING RULE:
- For BUY: SL = below the M15 swing low or M15 OB bottom (max 60 pips)
- For SELL: SL = above the M15 swing high or M15 OB top (max 60 pips)
- If SL would be > 80 pips on M15 → use M5 structure instead
- NEVER place SL at the H4 OB boundary (too wide)

Respond ONLY with valid JSON matching the system prompt format.
Include "entry_timeframe": "M15" or "M5" in your response.
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
    # Market Briefing — time-aware, works any time of day
    # ------------------------------------------------------------------

    async def get_morning_briefing(
        self,
        symbol: str,
        user_profile=None,
    ) -> dict[str, Any]:
        """
        Time-aware market briefing.
        Adapts content based on current session:
          - Asian session   (22:00–07:00 UTC): recap + London preview
          - London session  (07:00–13:00 UTC): full day plan
          - New York session(13:00–22:00 UTC): afternoon update + next day preview
          - Night           (any off-hours):   overnight recap + next session plan
        Always uses real live data.
        """
        from datetime import datetime, timezone
        from app.tools import get_market_data

        price_info    = get_current_price(symbol)
        current_price = price_info.get("mid", 0.0)
        news_data     = self._tool_news()
        user_cfg      = self._tool_user_settings(user_profile)

        # MTF analysis
        mtf_result = self.mtf_engine.analyze(symbol, current_price)
        mtf_data   = mtf_to_dict(mtf_result)

        # Fetch D1 candles for context
        df_d1 = get_market_data(symbol, "D1")
        df_h4 = get_market_data(symbol, "H4")

        # Build candle context
        prev_day = {}
        today_candle = {}
        recent_h4 = []

        if df_d1 is not None and len(df_d1) >= 3:
            for i, label in [(-3, "2_days_ago"), (-2, "yesterday"), (-1, "today")]:
                c = df_d1.iloc[i]
                data = {
                    "date":      str(df_d1.index[i].date()),
                    "open":      round(float(c["open"]), 3),
                    "high":      round(float(c["high"]), 3),
                    "low":       round(float(c["low"]), 3),
                    "close":     round(float(c["close"]), 3),
                    "range":     round(float(c["high"]) - float(c["low"]), 3),
                    "direction": "bullish" if c["close"] > c["open"] else "bearish",
                    "body":      round(abs(float(c["close"]) - float(c["open"])), 3),
                }
                if label == "yesterday":
                    prev_day = data
                elif label == "today":
                    today_candle = data

        if df_h4 is not None and len(df_h4) >= 6:
            for i in range(-6, 0):
                c = df_h4.iloc[i]
                recent_h4.append({
                    "time":      str(df_h4.index[i]),
                    "open":      round(float(c["open"]), 3),
                    "high":      round(float(c["high"]), 3),
                    "low":       round(float(c["low"]), 3),
                    "close":     round(float(c["close"]), 3),
                    "direction": "bullish" if c["close"] > c["open"] else "bearish",
                })

        # Determine session context
        now = datetime.now(timezone.utc)
        hour = now.hour

        if 7 <= hour < 13:
            session_name    = "London"
            session_context = "London session is open. High volatility expected."
            next_session    = "New York (opens 13:00 UTC)"
            focus           = "London open setup, liquidity grabs, OB entries"
        elif 13 <= hour < 17:
            session_name    = "New York"
            session_context = "New York session is open. London/NY overlap — highest volume."
            next_session    = "NY close then Asian (22:00 UTC)"
            focus           = "NY continuation or reversal, news reactions"
        elif 17 <= hour < 22:
            session_name    = "New York Late"
            session_context = "Late New York session. Volume declining."
            next_session    = "Asian session (22:00 UTC)"
            focus           = "End of day positioning, tomorrow's setup building"
        elif 22 <= hour or hour < 2:
            session_name    = "Asian Open"
            session_context = "Asian session just opened. Low volatility, range-bound typical."
            next_session    = "London (07:00 UTC)"
            focus           = "Asian range formation, liquidity building above/below"
        else:
            session_name    = "Asian Mid"
            session_context = "Mid Asian session. Consolidation phase."
            next_session    = "London (07:00 UTC)"
            focus           = "Watch for Asian high/low formation before London"

        prompt = f"""
You are a professional Gold (XAUUSD) market analyst generating a REAL-TIME MARKET BRIEFING.

Current time: {now.strftime('%A, %B %d, %Y — %H:%M UTC')}
Current session: {session_name}
Session context: {session_context}
Next session: {next_session}
Analysis focus: {focus}

=== LIVE PRICE ===
{json.dumps(price_info, indent=2)}

=== USER SETTINGS ===
{json.dumps(user_cfg, indent=2)}

=== ECONOMIC CALENDAR ===
{json.dumps(news_data, indent=2)}

=== YESTERDAY'S D1 CANDLE ===
{json.dumps(prev_day, indent=2)}

=== TODAY'S D1 CANDLE (so far) ===
{json.dumps(today_candle, indent=2)}

=== LAST 6 H4 CANDLES (recent price action) ===
{json.dumps(recent_h4, indent=2)}

=== MULTI-TIMEFRAME SMC ANALYSIS ===
{json.dumps(mtf_data, indent=2)}

Generate a COMPREHENSIVE REAL-TIME BRIEFING as JSON.
The briefing must be relevant to the CURRENT TIME and SESSION.
If it is night/Asian session, focus on what happened today and what to expect tomorrow.
If it is London/NY, focus on the current session setup.

{{
  "title": "string — e.g. 'London Session Briefing' or 'Asian Session Outlook'",
  "date_time": "string",
  "session": "{session_name}",
  "next_session": "{next_session}",
  "current_price": float,
  "weekly_bias": "bullish" | "bearish" | "ranging",
  "daily_bias": "bullish" | "bearish" | "ranging",
  "htf_aligned": bool,

  "today_recap": "string — what happened in today's price action so far, key moves, candle structure",
  "yesterday_recap": "string — what happened yesterday, was it bullish/bearish, key levels hit",
  "weekly_context": "string — where are we in the weekly range, premium or discount",

  "key_levels": {{
    "major_resistance": [float, float],
    "major_support": [float, float],
    "today_high": float,
    "today_low": float,
    "yesterday_high": float,
    "yesterday_low": float,
    "weekly_high": float,
    "weekly_low": float,
    "equilibrium": float
  }},

  "active_order_blocks": "string — key OBs currently relevant",
  "active_fvgs": "string — unfilled FVGs that price may revisit",
  "liquidity_above": "string — where buy-side liquidity sits",
  "liquidity_below": "string — where sell-side liquidity sits",
  "premium_discount_now": "string — current zone with price level",

  "current_session_outlook": "string — what to expect for the REST of the current session",
  "next_session_preview": "string — what to expect in the NEXT session based on current structure",
  "tomorrow_expectation": "string — broader expectation for tomorrow based on daily/weekly structure",

  "trade_plan": {{
    "bias": "BUY" | "SELL" | "WAIT",
    "ideal_entry_zone": "string — specific price zone",
    "watch_for": "string — exact confirmation needed",
    "avoid_if": "string — what would cancel the plan",
    "best_session_to_trade": "string — London | New York | either"
  }},

  "news_impact": "string — upcoming news and how it affects the plan",
  "experience_note": "string — professional insight about Gold behavior in this specific structure/time",
  "risk_reminder": "string — specific risk management advice for current conditions",
  "summary": "string — 2-3 sentence executive summary a trader can act on immediately"
}}
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        briefing, ai_meta = await dual_chat_json(messages, temperature=0.3, max_tokens=1500)

        if not briefing:
            briefing = {"error": "AI failed to generate briefing."}

        briefing["symbol"]       = symbol
        briefing["price_info"]   = price_info
        briefing["prev_day"]     = prev_day
        briefing["today_candle"] = today_candle
        briefing["recent_h4"]    = recent_h4
        briefing["mtf_data"]     = mtf_data
        briefing["ai_meta"]      = ai_meta
        briefing["generated_at"] = now.strftime("%Y-%m-%d %H:%M UTC")
        briefing["session"]      = session_name
        briefing["next_session"] = next_session
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


def _no_trade_explained(
    symbol: str,
    timeframe: str,
    price_info: dict,
    explanation,
) -> dict:
    """Returns a NO_TRADE with full structured explanation."""
    from app.memory import NoTradeExplanation
    return {
        "action":               "NO_TRADE",
        "symbol":               symbol,
        "timeframe":            timeframe,
        "current_price":        price_info,
        "news_blocked":         False,
        "reasoning":            explanation.reason,
        "missing_confluences":  explanation.missing_confluences,
        "present_confluences":  explanation.present_confluences,
        "setup_score":          explanation.score,
        "htf_status":           explanation.htf_status,
        "session_status":       explanation.session_status,
        "recommendation":       explanation.recommendation,
        "confidence":           "LOW",
        "entry":                None,
        "stop_loss":            None,
        "take_profit":          None,
        "rr_ratio":             None,
        "confluences":          explanation.present_confluences,
    }
