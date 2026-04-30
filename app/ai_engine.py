"""
Dual-AI Engine — Gemini + Groq working together.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Signal Request                                     │
  │       │                                             │
  │  ┌────┴────────────────────────┐                   │
  │  │                             │                   │
  │  ▼                             ▼                   │
  │  Gemini                      Groq                  │
  │  (deep analysis)             (quick signal)        │
  │  - Full MTF reasoning        - Fast JSON decision  │
  │  - Narrative + context       - Entry/SL/TP/RR      │
  │  - Higher token budget       - Low latency         │
  │       │                             │              │
  │       └──────────┬──────────────────┘              │
  │                  ▼                                  │
  │           Comparator                               │
  │           - Both agree  → HIGH confidence          │
  │           - One NO_TRADE → use the other           │
  │           - Both NO_TRADE → NO_TRADE               │
  │           - Conflict → pick higher RR + reasoning  │
  │                  │                                  │
  │                  ▼                                  │
  │           Final Signal                             │
  └─────────────────────────────────────────────────────┘

Fallback:
  - Gemini fails → Groq only
  - Groq fails   → Gemini only
  - Both fail    → NO_TRADE with error
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


# ---------------------------------------------------------------------------
# Per-provider prompts
# ---------------------------------------------------------------------------

GEMINI_SYSTEM = """
You are a senior SMC (Smart Money Concepts) analyst for XAUUSD (Gold).
Your role: DEEP ANALYSIS — thorough, structured, multi-timeframe reasoning.

Focus on:
- Full narrative of market structure across all timeframes
- Precise identification of OB, FVG, Liquidity, BOS, CHoCH
- Premium/Discount zone assessment
- Confluence scoring
- Risk factors and invalidation levels

Output MUST be valid JSON matching this exact schema:
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "rr_ratio": float,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "detailed multi-paragraph reasoning citing specific SMC structures",
  "confluences": ["list of specific confluences found"],
  "key_levels": {"support": float, "resistance": float},
  "premium_discount": "premium" | "discount" | "equilibrium",
  "htf_bias": "bullish" | "bearish" | "ranging",
  "entry_timeframe": "H1" | "M15" | "M5",
  "invalidation": "specific price level and condition that invalidates this trade",
  "analysis_depth": "DEEP"
}
""".strip()

GROQ_SYSTEM = """
You are a fast SMC (Smart Money Concepts) signal engine for XAUUSD (Gold).
Your role: QUICK DECISION — fast, precise, actionable signal.

Rules:
- Be concise and direct
- Only trade with confirmed HTF bias
- Entry at OB or FVG only
- R:R must be ≥ 2.0
- If setup is not clean → NO_TRADE immediately

Output MUST be valid JSON matching this exact schema:
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "rr_ratio": float,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "concise 1-2 sentence reasoning",
  "confluences": ["list of confluences"],
  "key_levels": {"support": float, "resistance": float},
  "premium_discount": "premium" | "discount" | "equilibrium",
  "htf_bias": "bullish" | "bearish" | "ranging",
  "entry_timeframe": "H1" | "M15" | "M5",
  "invalidation": "brief invalidation condition",
  "analysis_depth": "QUICK"
}
""".strip()


# ---------------------------------------------------------------------------
# Individual provider calls
# ---------------------------------------------------------------------------

async def _call_gemini(
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """
    Gemini — deep analysis, higher token budget, thorough reasoning.
    """
    if not settings.gemini_api_key:
        return {"error": "GEMINI_API_KEY not set", "provider": "gemini"}

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)

        full_prompt = (
            f"[SYSTEM]\n{GEMINI_SYSTEM}\n\n"
            f"[USER]\n{prompt}\n\n"
            "Respond ONLY with valid JSON. No markdown fences, no explanation outside JSON."
        )

        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        resp = await model.generate_content_async(full_prompt)
        raw = resp.text.strip()
        raw = _strip_markdown(raw)
        result = json.loads(raw)
        result["provider"] = "gemini"
        result["raw_response"] = raw
        logger.info("Gemini response: action=%s confidence=%s rr=%s",
                    result.get("action"), result.get("confidence"), result.get("rr_ratio"))
        return result

    except json.JSONDecodeError as e:
        logger.error("Gemini JSON parse error: %s | raw: %s", e, raw[:200] if 'raw' in dir() else "")
        return {"error": f"JSON parse error: {e}", "provider": "gemini", "action": "NO_TRADE"}
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return {"error": str(e), "provider": "gemini", "action": "NO_TRADE"}


async def _call_groq(
    prompt: str,
    max_tokens: int = 700,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Groq — fast signal, low latency, concise JSON output.
    """
    if not settings.groq_api_key:
        return {"error": "GROQ_API_KEY not set", "provider": "groq"}

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)

        resp = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        result["provider"] = "groq"
        result["raw_response"] = raw
        logger.info("Groq response: action=%s confidence=%s rr=%s",
                    result.get("action"), result.get("confidence"), result.get("rr_ratio"))
        return result

    except json.JSONDecodeError as e:
        logger.error("Groq JSON parse error: %s", e)
        return {"error": f"JSON parse error: {e}", "provider": "groq", "action": "NO_TRADE"}
    except Exception as e:
        logger.error("Groq call failed: %s", e)
        return {"error": str(e), "provider": "groq", "action": "NO_TRADE"}


async def _call_openai(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """OpenAI — paid fallback."""
    if not settings.openai_api_key:
        return {"error": "OPENAI_API_KEY not set", "provider": "openai"}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        result["provider"] = "openai"
        return result
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return {"error": str(e), "provider": "openai", "action": "NO_TRADE"}


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

@dataclass
class DualResult:
    final: dict                          # chosen best signal
    gemini: dict                         # raw gemini result
    groq: dict                           # raw groq result
    agreement: bool                      # both chose same action
    winner: str                          # "gemini" | "groq" | "both" | "fallback"
    comparison_notes: list[str] = field(default_factory=list)


def _is_valid_signal(sig: dict) -> bool:
    """Check if a signal has all required fields and is not an error."""
    if sig.get("error"):
        return False
    if sig.get("action") == "NO_TRADE":
        return False
    required = ["action", "entry", "stop_loss", "take_profit", "rr_ratio"]
    return all(sig.get(k) for k in required)


def _score_signal(sig: dict) -> float:
    """
    Score a signal 0-100 for comparison.
    Higher = better.
    """
    if not _is_valid_signal(sig):
        return 0.0

    score = 0.0
    # Confidence
    score += CONFIDENCE_RANK.get(sig.get("confidence", "LOW"), 0) * 20

    # R:R ratio (capped at 5)
    rr = float(sig.get("rr_ratio") or 0)
    score += min(rr, 5.0) * 8

    # Number of confluences
    confluences = sig.get("confluences", [])
    score += min(len(confluences), 5) * 4

    return round(score, 2)


def _compare(gemini: dict, groq: dict) -> DualResult:
    notes: list[str] = []

    gemini_valid = _is_valid_signal(gemini)
    groq_valid   = _is_valid_signal(groq)
    gemini_action = gemini.get("action", "NO_TRADE")
    groq_action   = groq.get("action", "NO_TRADE")

    # ── Both failed / errored ──────────────────────────────────────
    if not gemini_valid and not groq_valid:
        notes.append("Both providers returned NO_TRADE or error.")
        final = _merge_no_trade(gemini, groq)
        return DualResult(final=final, gemini=gemini, groq=groq,
                          agreement=True, winner="none", comparison_notes=notes)

    # ── Only one valid ─────────────────────────────────────────────
    if gemini_valid and not groq_valid:
        notes.append("Groq returned NO_TRADE. Using Gemini (deep analysis).")
        final = _enrich(gemini, groq, winner="gemini")
        return DualResult(final=final, gemini=gemini, groq=groq,
                          agreement=False, winner="gemini", comparison_notes=notes)

    if groq_valid and not gemini_valid:
        notes.append("Gemini returned NO_TRADE. Using Groq (quick signal).")
        final = _enrich(groq, gemini, winner="groq")
        return DualResult(final=final, gemini=gemini, groq=groq,
                          agreement=False, winner="groq", comparison_notes=notes)

    # ── Both valid — compare ───────────────────────────────────────
    agreement = (gemini_action == groq_action)
    gemini_score = _score_signal(gemini)
    groq_score   = _score_signal(groq)

    notes.append(f"Gemini: {gemini_action} | score={gemini_score} | conf={gemini.get('confidence')} | RR={gemini.get('rr_ratio')}")
    notes.append(f"Groq:   {groq_action}   | score={groq_score}   | conf={groq.get('confidence')} | RR={groq.get('rr_ratio')}")

    if agreement:
        notes.append(f"AGREEMENT: Both chose {gemini_action}. Boosting confidence.")
        # Merge: use Gemini's deep reasoning + Groq's precise levels
        final = _merge_agreed(gemini, groq)
        return DualResult(final=final, gemini=gemini, groq=groq,
                          agreement=True, winner="both", comparison_notes=notes)

    # Disagreement — pick higher score
    notes.append(f"DISAGREEMENT: Gemini={gemini_action} vs Groq={groq_action}.")
    if gemini_score >= groq_score:
        notes.append(f"Gemini wins (score {gemini_score} >= {groq_score}).")
        final = _enrich(gemini, groq, winner="gemini")
        winner = "gemini"
    else:
        notes.append(f"Groq wins (score {groq_score} > {gemini_score}).")
        final = _enrich(groq, gemini, winner="groq")
        winner = "groq"

    return DualResult(final=final, gemini=gemini, groq=groq,
                      agreement=False, winner=winner, comparison_notes=notes)


def _merge_agreed(gemini: dict, groq: dict) -> dict:
    """Both agree — use Gemini's deep reasoning, Groq's precise entry levels."""
    merged = dict(gemini)  # start with Gemini (deep analysis)

    # Use Groq's levels if they look more precise (more decimal places)
    for field in ["entry", "stop_loss", "take_profit"]:
        g_val = groq.get(field)
        if g_val:
            merged[field] = g_val

    # Boost confidence if both agree
    current_conf = CONFIDENCE_RANK.get(gemini.get("confidence", "LOW"), 1)
    if current_conf < 3:
        merged["confidence"] = "HIGH" if current_conf >= 2 else "MEDIUM"

    # Merge confluences
    g_conf = set(gemini.get("confluences", []))
    q_conf = set(groq.get("confluences", []))
    merged["confluences"] = list(g_conf | q_conf)

    merged["ai_agreement"] = True
    merged["providers_used"] = ["gemini", "groq"]
    merged["gemini_reasoning"] = gemini.get("reasoning", "")
    merged["groq_reasoning"]   = groq.get("reasoning", "")
    return merged


def _enrich(winner: dict, loser: dict, winner_name: str) -> dict:
    """One provider won — enrich with context from the other."""
    result = dict(winner)
    result["ai_agreement"] = False
    result["providers_used"] = [winner_name]
    result["other_provider_action"] = loser.get("action", "N/A")
    result["other_provider_reasoning"] = loser.get("reasoning", "")
    # Add any extra confluences from the losing provider
    w_conf = set(winner.get("confluences", []))
    l_conf = set(loser.get("confluences", []))
    result["confluences"] = list(w_conf | l_conf)
    return result


def _merge_no_trade(gemini: dict, groq: dict) -> dict:
    reason_g = gemini.get("reasoning") or gemini.get("error", "")
    reason_q = groq.get("reasoning")   or groq.get("error", "")
    return {
        "action":          "NO_TRADE",
        "confidence":      "LOW",
        "reasoning":       f"Gemini: {reason_g} | Groq: {reason_q}",
        "confluences":     [],
        "ai_agreement":    True,
        "providers_used":  [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def dual_signal(prompt: str, use_fallback: bool = True) -> DualResult:
    """
    Send prompt to both Gemini (deep) and Groq (quick) in parallel.
    Compare results and return the best signal.

    use_fallback: if both free providers fail, try OpenAI.
    """
    # Run both in parallel
    gemini_task = asyncio.create_task(_call_gemini(prompt, max_tokens=1500, temperature=0.3))
    groq_task   = asyncio.create_task(_call_groq(prompt,   max_tokens=700,  temperature=0.1))

    gemini_result, groq_result = await asyncio.gather(
        gemini_task, groq_task, return_exceptions=False
    )

    result = _compare(gemini_result, groq_result)

    # Fallback to OpenAI only if both failed AND fallback is enabled
    if (
        use_fallback
        and result.winner == "none"
        and settings.openai_api_key
        and "YOUR_KEY" not in settings.openai_api_key
    ):
        logger.warning("Both Gemini and Groq failed. Trying OpenAI fallback.")
        openai_result = await _call_openai(prompt)
        if _is_valid_signal(openai_result):
            openai_result["providers_used"] = ["openai_fallback"]
            result.final = openai_result
            result.winner = "openai_fallback"
            result.comparison_notes.append("OpenAI fallback used.")

    return result


async def dual_analyze(prompt: str) -> dict[str, Any]:
    """
    For /analyze — Gemini does the deep narrative, Groq adds a quick summary.
    Returns combined analysis.
    """
    # Gemini: full narrative (high tokens)
    gemini_task = asyncio.create_task(
        _call_gemini(prompt, max_tokens=2000, temperature=0.4)
    )
    # Groq: quick bullet summary
    groq_prompt = prompt + "\n\nProvide a SHORT 3-bullet summary of the key SMC points only."
    groq_task = asyncio.create_task(
        _call_groq(groq_prompt, max_tokens=400, temperature=0.2)
    )

    gemini_result, groq_result = await asyncio.gather(gemini_task, groq_task)

    return {
        "gemini_analysis": gemini_result.get("reasoning") or str(gemini_result),
        "groq_summary":    groq_result.get("reasoning")   or str(groq_result),
        "gemini_action":   gemini_result.get("action", "N/A"),
        "groq_action":     groq_result.get("action", "N/A"),
        "providers":       ["gemini", "groq"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown code fences from AI response."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single provider fallback (used by _chat in agent.py)
# ---------------------------------------------------------------------------

async def single_chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 800,
    json_mode: bool = False,
) -> str:
    """
    Single provider call — routes to AI_PROVIDER setting.
    Used for non-signal calls (analyze narrative, status, etc.)
    Falls back through providers if one fails.
    """
    providers_to_try = _get_provider_order()

    for provider in providers_to_try:
        try:
            result = await _call_provider(provider, messages, temperature, max_tokens, json_mode)
            if result and not result.get("error"):
                if json_mode:
                    return json.dumps(result)
                return result.get("reasoning") or json.dumps(result)
        except Exception as e:
            logger.warning("Provider %s failed: %s — trying next", provider, e)
            continue

    return json.dumps({"action": "NO_TRADE", "reasoning": "All AI providers failed."})


def _get_provider_order() -> list[str]:
    """Return providers in priority order based on AI_PROVIDER setting."""
    primary = settings.ai_provider.lower()
    all_providers = ["groq", "gemini", "openai"]
    order = [primary] + [p for p in all_providers if p != primary]
    # Filter to only configured ones
    configured = []
    for p in order:
        if p == "groq"   and settings.groq_api_key:
            configured.append(p)
        elif p == "gemini" and settings.gemini_api_key:
            configured.append(p)
        elif p == "openai" and settings.openai_api_key and "YOUR_KEY" not in settings.openai_api_key:
            configured.append(p)
    return configured or ["groq"]


async def _call_provider(
    provider: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> dict:
    prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
    if provider == "gemini":
        return await _call_gemini(prompt, max_tokens, temperature)
    elif provider == "groq":
        # Convert messages format for Groq
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)
        kwargs: dict = dict(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except Exception:
            return {"reasoning": raw}
    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        kwargs = dict(
            model=settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except Exception:
            return {"reasoning": raw}
    return {"error": f"Unknown provider: {provider}"}
