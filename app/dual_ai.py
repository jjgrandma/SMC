"""
Dual AI Engine — Gemini + Groq working together.

Strategy:
  - Gemini  → deep SMC analysis  (thorough, structured reasoning)
  - Groq    → quick signal idea  (fast, decisive JSON)
  - Comparator → picks the best result, uses the other as fallback

Modes:
  DUAL   → both run in parallel, best result chosen
  SINGLE → only one provider (fallback mode)

Comparison logic:
  1. Both return valid signals → compare confidence + RR → pick best
  2. One returns NO_TRADE, other has signal → use the signal (if valid)
  3. Both return NO_TRADE → NO_TRADE
  4. One fails/errors → use the other silently
  5. Both fail → return NO_TRADE with error message
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "": 0}


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class AIResponse:
    provider: str           # "gemini" | "groq" | "openai"
    raw: str = ""
    parsed: dict = field(default_factory=dict)
    success: bool = False
    error: str = ""
    latency_ms: int = 0
    is_json: bool = False


# ---------------------------------------------------------------------------
# Individual provider calls
# ---------------------------------------------------------------------------

async def _call_groq(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> AIResponse:
    start = time.monotonic()
    resp = AIResponse(provider="groq")
    try:
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
        result = await client.chat.completions.create(**kwargs)
        resp.raw = result.choices[0].message.content or ""
        resp.success = True
    except Exception as exc:
        resp.error = str(exc)
        logger.warning("Groq call failed: %s", exc)
    finally:
        resp.latency_ms = int((time.monotonic() - start) * 1000)
    return resp


async def _call_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> AIResponse:
    start = time.monotonic()
    resp = AIResponse(provider="gemini")
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        if json_mode:
            prompt += (
                "\n\n---\n"
                "IMPORTANT: Respond ONLY with a single valid JSON object. "
                "No markdown code blocks, no explanation, no extra text. "
                "Start your response with { and end with }"
            )
        result = await model.generate_content_async(prompt)
        resp.raw = result.text or ""
        resp.success = True
    except Exception as exc:
        resp.error = str(exc)
        logger.warning("Gemini call failed: %s", exc)
    finally:
        resp.latency_ms = int((time.monotonic() - start) * 1000)
    return resp


async def _call_openai(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> AIResponse:
    start = time.monotonic()
    resp = AIResponse(provider="openai")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        kwargs: dict = dict(
            model=settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        result = await client.chat.completions.create(**kwargs)
        resp.raw = result.choices[0].message.content or ""
        resp.success = True
    except Exception as exc:
        resp.error = str(exc)
        logger.warning("OpenAI call failed: %s", exc)
    finally:
        resp.latency_ms = int((time.monotonic() - start) * 1000)
    return resp


# ---------------------------------------------------------------------------
# JSON extractor (handles Gemini markdown wrapping)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Signal comparator
# ---------------------------------------------------------------------------

def _score_signal(sig: dict) -> float:
    """
    Score a signal for quality comparison.
    Higher = better.
    """
    if not sig or sig.get("action") == "NO_TRADE":
        return 0.0

    score = 0.0

    # Confidence
    conf = sig.get("confidence", "")
    score += CONFIDENCE_RANK.get(conf, 0) * 10

    # R:R ratio
    rr = sig.get("rr_ratio", 0)
    try:
        score += float(rr) * 5
    except (TypeError, ValueError):
        pass

    # Confluences count
    confluences = sig.get("confluences", [])
    score += len(confluences) * 3

    # Has all required fields
    for field_name in ("entry", "stop_loss", "take_profit"):
        val = sig.get(field_name)
        if val and float(val) > 0:
            score += 2

    # Premium/discount alignment
    action = sig.get("action", "")
    pd_zone = sig.get("premium_discount", "")
    if action == "BUY"  and pd_zone == "discount":
        score += 5
    if action == "SELL" and pd_zone == "premium":
        score += 5

    return score


def _compare_and_pick(
    gemini_resp: AIResponse,
    groq_resp: AIResponse,
) -> tuple[dict, dict]:
    """
    Returns (chosen_signal, comparison_meta).
    comparison_meta contains both results for transparency.
    """
    # Parse both
    gemini_sig = _extract_json(gemini_resp.raw) if gemini_resp.success else None
    groq_sig   = _extract_json(groq_resp.raw)   if groq_resp.success   else None

    if gemini_resp.success and gemini_sig:
        gemini_resp.parsed  = gemini_sig
        gemini_resp.is_json = True

    if groq_resp.success and groq_sig:
        groq_resp.parsed  = groq_sig
        groq_resp.is_json = True

    gemini_score = _score_signal(gemini_sig or {})
    groq_score   = _score_signal(groq_sig   or {})

    meta = {
        "gemini": {
            "success":    gemini_resp.success,
            "action":     (gemini_sig or {}).get("action", "FAILED"),
            "confidence": (gemini_sig or {}).get("confidence", ""),
            "rr_ratio":   (gemini_sig or {}).get("rr_ratio", 0),
            "score":      round(gemini_score, 1),
            "latency_ms": gemini_resp.latency_ms,
            "error":      gemini_resp.error,
        },
        "groq": {
            "success":    groq_resp.success,
            "action":     (groq_sig or {}).get("action", "FAILED"),
            "confidence": (groq_sig or {}).get("confidence", ""),
            "rr_ratio":   (groq_sig or {}).get("rr_ratio", 0),
            "score":      round(groq_score, 1),
            "latency_ms": groq_resp.latency_ms,
            "error":      groq_resp.error,
        },
    }

    # Decision logic
    # Case 1: both failed
    if not gemini_sig and not groq_sig:
        meta["chosen"] = "none"
        meta["reason"] = "Both providers failed"
        return {"action": "NO_TRADE", "reasoning": "Both AI providers failed to respond."}, meta

    # Case 2: only one succeeded
    if gemini_sig and not groq_sig:
        meta["chosen"] = "gemini"
        meta["reason"] = "Groq failed — using Gemini"
        return gemini_sig, meta

    if groq_sig and not gemini_sig:
        meta["chosen"] = "groq"
        meta["reason"] = "Gemini failed — using Groq"
        return groq_sig, meta

    # Case 3: both NO_TRADE
    if gemini_sig.get("action") == "NO_TRADE" and groq_sig.get("action") == "NO_TRADE":
        meta["chosen"] = "both_no_trade"
        meta["reason"] = "Both agree: NO_TRADE"
        # Merge reasoning
        combined = gemini_sig.copy()
        combined["reasoning"] = (
            f"[Gemini] {gemini_sig.get('reasoning','')}\n"
            f"[Groq] {groq_sig.get('reasoning','')}"
        )
        return combined, meta

    # Case 4: one NO_TRADE, one has signal → use the signal
    if gemini_sig.get("action") == "NO_TRADE" and groq_sig.get("action") != "NO_TRADE":
        meta["chosen"] = "groq"
        meta["reason"] = "Gemini says NO_TRADE, Groq has signal — using Groq"
        return groq_sig, meta

    if groq_sig.get("action") == "NO_TRADE" and gemini_sig.get("action") != "NO_TRADE":
        meta["chosen"] = "gemini"
        meta["reason"] = "Groq says NO_TRADE, Gemini has signal — using Gemini"
        return gemini_sig, meta

    # Case 5: both have signals — pick by score
    if gemini_score >= groq_score:
        meta["chosen"] = "gemini"
        meta["reason"] = f"Gemini scored higher ({gemini_score:.1f} vs {groq_score:.1f})"
        # Enrich with Groq's confluences if Gemini missed some
        chosen = gemini_sig.copy()
        groq_confs = groq_sig.get("confluences", [])
        gem_confs  = chosen.get("confluences", [])
        merged = list(dict.fromkeys(gem_confs + groq_confs))  # deduplicate, preserve order
        chosen["confluences"] = merged
        chosen["groq_note"] = groq_sig.get("reasoning", "")
        return chosen, meta
    else:
        meta["chosen"] = "groq"
        meta["reason"] = f"Groq scored higher ({groq_score:.1f} vs {gemini_score:.1f})"
        chosen = groq_sig.copy()
        gem_confs  = gemini_sig.get("confluences", [])
        groq_confs = chosen.get("confluences", [])
        merged = list(dict.fromkeys(groq_confs + gem_confs))
        chosen["confluences"] = merged
        chosen["gemini_note"] = gemini_sig.get("reasoning", "")
        return chosen, meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def dual_chat_json(
    messages: list[dict],
    temperature: float = 0.15,
    max_tokens: int = 800,
) -> tuple[dict, dict]:
    """
    Run Gemini (deep) + Groq (quick) in parallel for JSON signals.
    Returns (best_signal_dict, comparison_meta).
    """
    # Run both in parallel
    gemini_task = _call_gemini(messages, temperature, max_tokens, json_mode=True)
    groq_task   = _call_groq(messages, temperature, max_tokens, json_mode=True)

    gemini_resp, groq_resp = await asyncio.gather(gemini_task, groq_task)

    logger.info(
        "Dual AI: Gemini=%dms(%s) Groq=%dms(%s)",
        gemini_resp.latency_ms, "OK" if gemini_resp.success else "FAIL",
        groq_resp.latency_ms,   "OK" if groq_resp.success   else "FAIL",
    )

    return _compare_and_pick(gemini_resp, groq_resp)


async def dual_chat_narrative(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    """
    Run Gemini (deep analysis) + Groq (quick summary) in parallel for narrative.
    Returns (combined_narrative, meta).
    Gemini provides the main analysis, Groq adds a quick summary section.
    """
    # Groq gets a shorter, faster prompt
    groq_messages = messages.copy()
    # Append instruction to keep it brief
    groq_messages = groq_messages[:-1] + [{
        "role": "user",
        "content": groq_messages[-1]["content"] + "\n\nKeep your response under 200 words. Focus on key levels and bias only.",
    }]

    gemini_task = _call_gemini(messages, temperature, max_tokens, json_mode=False)
    groq_task   = _call_groq(groq_messages, temperature, 400, json_mode=False)

    gemini_resp, groq_resp = await asyncio.gather(gemini_task, groq_task)

    meta = {
        "gemini_latency_ms": gemini_resp.latency_ms,
        "groq_latency_ms":   groq_resp.latency_ms,
        "gemini_ok":         gemini_resp.success,
        "groq_ok":           groq_resp.success,
    }

    # Combine: Gemini = deep analysis, Groq = quick take
    parts = []

    if gemini_resp.success and gemini_resp.raw:
        parts.append("🔬 *Deep Analysis (Gemini)*\n" + gemini_resp.raw.strip())
    elif groq_resp.success and groq_resp.raw:
        parts.append("⚡ *Analysis (Groq)*\n" + groq_resp.raw.strip())

    if groq_resp.success and groq_resp.raw and gemini_resp.success:
        parts.append("\n⚡ *Quick Take (Groq)*\n" + groq_resp.raw.strip())

    if not parts:
        return "Both AI providers failed to respond.", meta

    return "\n\n".join(parts), meta


async def single_chat(
    messages: list[dict],
    provider: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
    json_mode: bool = False,
) -> str:
    """
    Single provider call — used as fallback.
    """
    if provider == "groq":
        resp = await _call_groq(messages, temperature, max_tokens, json_mode)
    elif provider == "gemini":
        resp = await _call_gemini(messages, temperature, max_tokens, json_mode)
    else:
        resp = await _call_openai(messages, temperature, max_tokens, json_mode)

    if resp.success:
        return resp.raw
    raise RuntimeError(f"{provider} failed: {resp.error}")
