"""
Self-Learning Memory System.

The bot learns from every signal it generates:
  - Tracks signal outcomes (win/loss/pending)
  - Extracts patterns from wins and losses
  - Builds a growing knowledge base
  - Injects relevant memory into every AI prompt

Storage: data/memory.json  (simple, persistent, no DB needed)

Learning cycle:
  1. Signal generated → saved to memory as PENDING
  2. Price moves → outcome evaluated (WIN/LOSS/BREAKEVEN)
  3. Pattern extracted → added to knowledge base
  4. Knowledge base injected into next AI prompt as context

Memory types:
  - signal_log     : every signal with outcome
  - patterns       : what worked, what didn't
  - session_notes  : observations per trading session
  - market_lessons : accumulated trading wisdom
  - stats          : win rate, avg RR, best TF, best session
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

MEMORY_FILE   = Path("data/memory.json")
LESSONS_FILE  = Path("data/lessons.json")
MAX_LOG_SIZE  = 500   # keep last 500 signals
MAX_LESSONS   = 100   # keep last 100 lessons


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    id: str
    timestamp: str
    symbol: str
    timeframe: str
    action: str                          # BUY | SELL | NO_TRADE
    entry: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confidence: str
    htf_bias: str
    premium_discount: str
    confluences: list[str]
    smc_structure: str
    reasoning: str
    outcome: Literal["PENDING", "WIN", "LOSS", "BREAKEVEN", "CANCELLED"] = "PENDING"
    outcome_price: float = 0.0
    outcome_pips: float = 0.0
    outcome_rr: float = 0.0
    outcome_note: str = ""
    session: str = ""                    # London | New York | Asian
    day_of_week: str = ""
    ai_provider_chosen: str = ""


@dataclass
class Lesson:
    id: str
    timestamp: str
    category: Literal[
        "PATTERN_WIN", "PATTERN_LOSS", "SESSION_NOTE",
        "STRUCTURE_NOTE", "RISK_NOTE", "GENERAL"
    ]
    title: str
    content: str
    symbol: str = "XAUUSD"
    timeframe: str = ""
    confidence: float = 0.0             # 0.0–1.0 how reliable this lesson is
    occurrences: int = 1


@dataclass
class MemoryStats:
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    pending: int = 0
    win_rate: float = 0.0
    avg_rr_won: float = 0.0
    avg_rr_lost: float = 0.0
    best_timeframe: str = ""
    best_session: str = ""
    worst_session: str = ""
    best_confluence: str = ""
    total_lessons: int = 0


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._signals: list[SignalRecord] = []
        self._lessons: list[Lesson] = []
        self._load()

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def log_signal(self, signal: dict) -> str:
        """Save a new signal as PENDING. Returns the signal ID."""
        now = datetime.now(timezone.utc)
        sig_id = f"{signal.get('symbol','X')}_{signal.get('timeframe','H1')}_{now.strftime('%Y%m%d_%H%M%S')}"

        hour = now.hour
        if 7 <= hour < 16:
            session = "London"
        elif 13 <= hour < 22:
            session = "NewYork"
        else:
            session = "Asian"

        record = SignalRecord(
            id=sig_id,
            timestamp=now.isoformat(),
            symbol=signal.get("symbol", "XAUUSD"),
            timeframe=signal.get("timeframe", "H1"),
            action=signal.get("action", "NO_TRADE"),
            entry=float(signal.get("entry") or 0),
            stop_loss=float(signal.get("stop_loss") or 0),
            take_profit=float(signal.get("take_profit") or 0),
            rr_ratio=float(signal.get("rr_ratio") or 0),
            confidence=signal.get("confidence", ""),
            htf_bias=signal.get("htf_bias", ""),
            premium_discount=signal.get("premium_discount", ""),
            confluences=signal.get("confluences", []),
            smc_structure=signal.get("smc_structure_used", signal.get("reasoning", "")[:200]),
            reasoning=signal.get("reasoning", "")[:500],
            session=session,
            day_of_week=now.strftime("%A"),
            ai_provider_chosen=signal.get("ai_comparison", {}).get("chosen", ""),
        )

        self._signals.append(record)
        if len(self._signals) > MAX_LOG_SIZE:
            self._signals = self._signals[-MAX_LOG_SIZE:]

        self._save()
        logger.info("Signal logged: %s %s %s", sig_id, record.action, record.symbol)
        return sig_id

    def update_outcome(
        self,
        sig_id: str,
        outcome: str,
        outcome_price: float = 0.0,
        note: str = "",
    ) -> bool:
        """Update a signal's outcome after the trade resolves."""
        for rec in self._signals:
            if rec.id == sig_id:
                rec.outcome = outcome
                rec.outcome_price = outcome_price
                rec.outcome_note = note

                if rec.entry > 0 and outcome_price > 0:
                    if rec.action == "BUY":
                        rec.outcome_pips = outcome_price - rec.entry
                    else:
                        rec.outcome_pips = rec.entry - outcome_price

                    if rec.stop_loss > 0 and rec.entry != rec.stop_loss:
                        sl_dist = abs(rec.entry - rec.stop_loss)
                        rec.outcome_rr = rec.outcome_pips / sl_dist

                self._save()
                self._extract_lesson(rec)
                return True
        return False

    # ------------------------------------------------------------------
    # Lesson management
    # ------------------------------------------------------------------

    def add_lesson(self, lesson: Lesson):
        # Check if similar lesson exists — increment occurrences
        for existing in self._lessons:
            if existing.title == lesson.title:
                existing.occurrences += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.timestamp = lesson.timestamp
                self._save()
                return

        self._lessons.append(lesson)
        if len(self._lessons) > MAX_LESSONS:
            # Remove lowest confidence lessons
            self._lessons.sort(key=lambda l: l.confidence, reverse=True)
            self._lessons = self._lessons[:MAX_LESSONS]
        self._save()

    def _extract_lesson(self, rec: SignalRecord):
        """Automatically extract a lesson from a completed signal."""
        now = datetime.now(timezone.utc).isoformat()

        if rec.outcome == "WIN":
            lesson = Lesson(
                id=f"lesson_{rec.id}",
                timestamp=now,
                category="PATTERN_WIN",
                title=f"WIN: {rec.action} on {rec.timeframe} in {rec.session} session",
                content=(
                    f"Signal: {rec.action} {rec.symbol} {rec.timeframe} | "
                    f"Session: {rec.session} | Day: {rec.day_of_week} | "
                    f"HTF Bias: {rec.htf_bias} | Zone: {rec.premium_discount} | "
                    f"Structure: {rec.smc_structure[:150]} | "
                    f"Confluences: {', '.join(rec.confluences[:3])} | "
                    f"R:R achieved: {rec.outcome_rr:.2f}"
                ),
                symbol=rec.symbol,
                timeframe=rec.timeframe,
                confidence=0.6,
            )
            self.add_lesson(lesson)

        elif rec.outcome == "LOSS":
            lesson = Lesson(
                id=f"lesson_{rec.id}",
                timestamp=now,
                category="PATTERN_LOSS",
                title=f"LOSS: {rec.action} on {rec.timeframe} in {rec.session} session",
                content=(
                    f"FAILED signal: {rec.action} {rec.symbol} {rec.timeframe} | "
                    f"Session: {rec.session} | Day: {rec.day_of_week} | "
                    f"HTF Bias: {rec.htf_bias} | Zone: {rec.premium_discount} | "
                    f"Structure used: {rec.smc_structure[:150]} | "
                    f"Note: {rec.outcome_note} | "
                    f"Avoid this setup when: similar conditions present"
                ),
                symbol=rec.symbol,
                timeframe=rec.timeframe,
                confidence=0.7,   # losses are more reliable lessons
            )
            self.add_lesson(lesson)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> MemoryStats:
        completed = [s for s in self._signals if s.outcome != "PENDING"]
        wins      = [s for s in completed if s.outcome == "WIN"]
        losses    = [s for s in completed if s.outcome == "LOSS"]
        pending   = [s for s in self._signals if s.outcome == "PENDING"]

        win_rate  = len(wins) / len(completed) * 100 if completed else 0.0
        avg_rr_w  = sum(s.outcome_rr for s in wins)   / len(wins)   if wins   else 0.0
        avg_rr_l  = sum(s.outcome_rr for s in losses) / len(losses) if losses else 0.0

        # Best timeframe by win rate
        tf_stats: dict[str, list] = {}
        for s in completed:
            tf_stats.setdefault(s.timeframe, []).append(s.outcome == "WIN")
        best_tf = max(tf_stats, key=lambda t: sum(tf_stats[t]) / len(tf_stats[t])) if tf_stats else ""

        # Best/worst session
        sess_stats: dict[str, list] = {}
        for s in completed:
            sess_stats.setdefault(s.session, []).append(s.outcome == "WIN")
        best_sess  = max(sess_stats, key=lambda t: sum(sess_stats[t]) / len(sess_stats[t])) if sess_stats else ""
        worst_sess = min(sess_stats, key=lambda t: sum(sess_stats[t]) / len(sess_stats[t])) if sess_stats else ""

        # Most common winning confluence
        conf_count: dict[str, int] = {}
        for s in wins:
            for c in s.confluences:
                conf_count[c] = conf_count.get(c, 0) + 1
        best_conf = max(conf_count, key=conf_count.get) if conf_count else ""

        return MemoryStats(
            total_signals=len(self._signals),
            wins=len(wins),
            losses=len(losses),
            breakevens=len([s for s in completed if s.outcome == "BREAKEVEN"]),
            pending=len(pending),
            win_rate=round(win_rate, 1),
            avg_rr_won=round(avg_rr_w, 2),
            avg_rr_lost=round(avg_rr_l, 2),
            best_timeframe=best_tf,
            best_session=best_sess,
            worst_session=worst_sess,
            best_confluence=best_conf,
            total_lessons=len(self._lessons),
        )

    # ------------------------------------------------------------------
    # Memory context for AI prompt injection
    # ------------------------------------------------------------------

    def get_ai_context(self, symbol: str, timeframe: str, action: str = "") -> str:
        """
        Returns a compact memory context string to inject into AI prompts.
        Includes relevant lessons, recent outcomes, and stats.
        """
        stats = self.get_stats()
        lines = ["=== MEMORY & LEARNED PATTERNS ==="]

        # Stats summary
        if stats.total_signals > 0:
            lines.append(
                f"Performance: {stats.total_signals} signals | "
                f"Win rate: {stats.win_rate}% | "
                f"Avg RR won: {stats.avg_rr_won} | "
                f"Best TF: {stats.best_timeframe} | "
                f"Best session: {stats.best_session}"
            )
            if stats.worst_session:
                lines.append(f"⚠️ Worst session: {stats.worst_session} — be cautious")

        # Relevant lessons (filter by symbol + timeframe + action)
        relevant = [
            l for l in self._lessons
            if (l.symbol == symbol or l.symbol == "XAUUSD")
            and (not l.timeframe or l.timeframe == timeframe)
            and l.confidence >= 0.5
        ]
        # Sort by confidence + occurrences
        relevant.sort(key=lambda l: l.confidence * l.occurrences, reverse=True)

        win_lessons  = [l for l in relevant if l.category == "PATTERN_WIN"][:3]
        loss_lessons = [l for l in relevant if l.category == "PATTERN_LOSS"][:3]
        other        = [l for l in relevant if l.category not in ("PATTERN_WIN", "PATTERN_LOSS")][:2]

        if win_lessons:
            lines.append("\n✅ PATTERNS THAT WORKED:")
            for l in win_lessons:
                lines.append(f"  [{l.occurrences}x] {l.title}: {l.content[:200]}")

        if loss_lessons:
            lines.append("\n❌ PATTERNS TO AVOID:")
            for l in loss_lessons:
                lines.append(f"  [{l.occurrences}x] {l.title}: {l.content[:200]}")

        if other:
            lines.append("\n📝 OTHER NOTES:")
            for l in other:
                lines.append(f"  {l.title}: {l.content[:150]}")

        # Recent outcomes (last 5 completed)
        recent = [s for s in reversed(self._signals) if s.outcome != "PENDING"][:5]
        if recent:
            lines.append("\n📊 RECENT OUTCOMES:")
            for s in recent:
                emoji = "✅" if s.outcome == "WIN" else "❌" if s.outcome == "LOSS" else "➖"
                lines.append(
                    f"  {emoji} {s.action} {s.timeframe} {s.session} "
                    f"RR:{s.outcome_rr:.1f} — {s.outcome_note[:80]}"
                )

        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def get_recent_signals(self, limit: int = 10) -> list[SignalRecord]:
        return list(reversed(self._signals[-limit:]))

    def get_pending_signals(self) -> list[SignalRecord]:
        return [s for s in self._signals if s.outcome == "PENDING"]

    def get_lessons(self) -> list[Lesson]:
        return sorted(self._lessons, key=lambda l: l.confidence * l.occurrences, reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                self._signals = [SignalRecord(**s) for s in data.get("signals", [])]
            except Exception as exc:
                logger.error("Failed to load memory: %s", exc)
                self._signals = []

        if LESSONS_FILE.exists():
            try:
                data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
                self._lessons = [Lesson(**l) for l in data.get("lessons", [])]
            except Exception as exc:
                logger.error("Failed to load lessons: %s", exc)
                self._lessons = []

    def _save(self):
        try:
            MEMORY_FILE.write_text(
                json.dumps({"signals": [asdict(s) for s in self._signals]}, indent=2),
                encoding="utf-8",
            )
            LESSONS_FILE.write_text(
                json.dumps({"lessons": [asdict(l) for l in self._lessons]}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save memory: %s", exc)


# ---------------------------------------------------------------------------
# Outcome evaluator — checks pending signals against current price
# ---------------------------------------------------------------------------

class OutcomeEvaluator:
    """
    Periodically checks pending signals against live price.
    Marks them WIN/LOSS/BREAKEVEN automatically.
    """

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def evaluate_pending(self, current_price: float) -> list[dict]:
        """
        Check all pending signals against current price.
        Returns list of resolved outcomes.
        """
        resolved = []
        for sig in self.memory.get_pending_signals():
            if sig.action == "NO_TRADE" or sig.entry == 0:
                continue

            result = self._check_outcome(sig, current_price)
            if result:
                self.memory.update_outcome(
                    sig.id,
                    result["outcome"],
                    result["price"],
                    result["note"],
                )
                resolved.append({
                    "id":      sig.id,
                    "action":  sig.action,
                    "symbol":  sig.symbol,
                    "tf":      sig.timeframe,
                    "outcome": result["outcome"],
                    "note":    result["note"],
                })
        return resolved

    def _check_outcome(self, sig: SignalRecord, price: float) -> dict | None:
        if sig.action == "BUY":
            if price >= sig.take_profit:
                return {"outcome": "WIN",  "price": price, "note": f"TP hit at {price}"}
            if price <= sig.stop_loss:
                return {"outcome": "LOSS", "price": price, "note": f"SL hit at {price}"}
        elif sig.action == "SELL":
            if price <= sig.take_profit:
                return {"outcome": "WIN",  "price": price, "note": f"TP hit at {price}"}
            if price >= sig.stop_loss:
                return {"outcome": "LOSS", "price": price, "note": f"SL hit at {price}"}
        return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_memory: MemoryStore | None = None


def get_memory() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore()
    return _memory
