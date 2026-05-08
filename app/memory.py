"""
Adaptive Intelligence Memory System.

Implements:
  1. NO-TRADE explanation engine — explains WHY no trade with missing confluences
  2. Lesson usage tracking — tracks times_used, usage_score, confluence_weight
  3. Adaptive lesson prioritization — high-importance + frequently-used lessons first
  4. Confluence scoring — scores each confluence by historical win rate
  5. Institutional SMC analyst behavior — transparent reasoning, never black-box

Core rules:
  - NEVER self-modify core trading logic
  - NEVER change user-defined strategy rules
  - NEVER auto-trade
  - ALWAYS explain reasoning transparently
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

MEMORY_FILE  = Path("data/memory.json")
LESSONS_FILE = Path("data/lessons.json")
MAX_SIGNALS  = 500
MAX_LESSONS  = 150


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    id: str
    timestamp: str
    symbol: str
    timeframe: str
    action: str
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
    session: str = ""
    day_of_week: str = ""
    ai_provider_chosen: str = ""


@dataclass
class Lesson:
    """
    Enhanced lesson with usage tracking and adaptive scoring.
    """
    id: str
    timestamp: str
    category: Literal[
        "PATTERN_WIN", "PATTERN_LOSS", "SESSION_NOTE",
        "STRUCTURE_NOTE", "RISK_NOTE", "GENERAL",
        "SupplyDemand", "Liquidity", "MarketStructure",
        "FibonacciLevel", "OrderBlock", "FVG", "Momentum",
    ]
    title: str
    content: str
    symbol: str = "XAUUSD"
    timeframe: str = ""
    # Reliability
    confidence: float = 0.5          # 0.0–1.0
    importance: Literal["Critical", "High", "Medium", "Low"] = "Medium"
    # Usage tracking
    times_used: int = 0              # how many times injected into AI prompt
    times_contributed: int = 0       # how many times it was in a winning signal
    usage_score: float = 0.0         # 0–100 composite score
    confluence_weight: float = 0.1   # weight in confluence scoring (0.0–1.0)
    # History
    occurrences: int = 1             # how many times this pattern was seen
    last_used: str = ""


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
    top_lessons: list = field(default_factory=list)


@dataclass
class NoTradeExplanation:
    """Structured explanation for why no trade was taken."""
    reason: str
    missing_confluences: list[str]
    present_confluences: list[str]
    htf_status: str
    session_status: str
    recommendation: str
    score: int   # 0–100, how close to a valid setup


# ---------------------------------------------------------------------------
# Confluence scorer
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """
    Scores each confluence by historical win rate.
    Learns which confluences actually lead to wins.
    """

    def __init__(self):
        self._scores: dict[str, dict] = {}   # confluence → {wins, total}

    def record(self, confluences: list[str], outcome: str):
        for c in confluences:
            key = c[:80]
            if key not in self._scores:
                self._scores[key] = {"wins": 0, "total": 0}
            self._scores[key]["total"] += 1
            if outcome == "WIN":
                self._scores[key]["wins"] += 1

    def score(self, confluence: str) -> float:
        """Return win rate for this confluence (0.0–1.0)."""
        key = confluence[:80]
        d   = self._scores.get(key, {})
        t   = d.get("total", 0)
        w   = d.get("wins", 0)
        return round(w / t, 2) if t >= 3 else 0.5   # default 50% if not enough data

    def top_confluences(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N confluences by win rate (min 3 occurrences)."""
        qualified = [
            (k, v["wins"] / v["total"])
            for k, v in self._scores.items()
            if v["total"] >= 3
        ]
        return sorted(qualified, key=lambda x: -x[1])[:n]

    def to_dict(self) -> dict:
        return self._scores

    def from_dict(self, data: dict):
        self._scores = data


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._signals: list[SignalRecord] = []
        self._lessons: list[Lesson] = []
        self._scorer  = ConfluenceScorer()
        self._load()

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def log_signal(self, signal: dict) -> str:
        now    = datetime.now(timezone.utc)
        sig_id = f"{signal.get('symbol','X')}_{signal.get('timeframe','H1')}_{now.strftime('%Y%m%d_%H%M%S')}"
        hour   = now.hour
        session = "London" if 7 <= hour < 16 else "NewYork" if 13 <= hour < 22 else "Asian"

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
            smc_structure=signal.get("smc_structure_used", signal.get("reasoning", ""))[:200],
            reasoning=signal.get("reasoning", "")[:500],
            session=session,
            day_of_week=now.strftime("%A"),
            ai_provider_chosen=signal.get("ai_comparison", {}).get("chosen", ""),
        )
        self._signals.append(record)
        if len(self._signals) > MAX_SIGNALS:
            self._signals = self._signals[-MAX_SIGNALS:]
        self._save()
        return sig_id

    def update_outcome(self, sig_id: str, outcome: str, price: float = 0.0, note: str = "") -> bool:
        for rec in self._signals:
            if rec.id == sig_id:
                rec.outcome       = outcome
                rec.outcome_price = price
                rec.outcome_note  = note
                if rec.entry > 0 and price > 0:
                    pips = (price - rec.entry) if rec.action == "BUY" else (rec.entry - price)
                    rec.outcome_pips = pips
                    sl_dist = abs(rec.entry - rec.stop_loss)
                    if sl_dist > 0:
                        rec.outcome_rr = round(pips / sl_dist, 2)
                self._save()
                self._extract_lesson(rec)
                self._scorer.record(rec.confluences, outcome)
                self._update_lesson_scores(rec.confluences, outcome)
                return True
        return False

    # ------------------------------------------------------------------
    # Lesson management
    # ------------------------------------------------------------------

    def add_lesson(self, lesson: Lesson):
        for existing in self._lessons:
            if existing.title == lesson.title:
                existing.occurrences += 1
                existing.confidence   = min(1.0, existing.confidence + 0.05)
                existing.timestamp    = lesson.timestamp
                existing.usage_score  = self._calc_usage_score(existing)
                self._save()
                return
        self._lessons.append(lesson)
        self._trim_lessons()
        self._save()

    def record_lesson_used(self, lesson_id: str, contributed_to_win: bool = False):
        """Call this when a lesson was injected into an AI prompt."""
        for l in self._lessons:
            if l.id == lesson_id:
                l.times_used += 1
                l.last_used   = datetime.now(timezone.utc).isoformat()
                if contributed_to_win:
                    l.times_contributed += 1
                l.usage_score = self._calc_usage_score(l)
                self._save()
                return

    def _calc_usage_score(self, lesson: Lesson) -> float:
        """
        Composite score 0–100:
          - importance weight
          - confidence
          - contribution rate (wins / times_used)
          - occurrences (how often this pattern appears)
        """
        importance_w = {"Critical": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.2}.get(lesson.importance, 0.5)
        contrib_rate = (lesson.times_contributed / lesson.times_used) if lesson.times_used > 0 else 0.5
        occ_score    = min(1.0, lesson.occurrences / 20)
        score = (
            importance_w   * 30 +
            lesson.confidence * 25 +
            contrib_rate   * 30 +
            occ_score      * 15
        )
        return round(score, 1)

    def _update_lesson_scores(self, confluences: list[str], outcome: str):
        """Update confluence_weight for lessons whose content matches winning confluences."""
        for lesson in self._lessons:
            for conf in confluences:
                if conf[:50].lower() in lesson.content.lower() or conf[:50].lower() in lesson.title.lower():
                    if outcome == "WIN":
                        lesson.confluence_weight = min(1.0, lesson.confluence_weight + 0.02)
                        lesson.times_contributed += 1
                    else:
                        lesson.confluence_weight = max(0.0, lesson.confluence_weight - 0.01)
                    lesson.usage_score = self._calc_usage_score(lesson)
        self._save()

    def _extract_lesson(self, rec: SignalRecord):
        now = datetime.now(timezone.utc).isoformat()
        if rec.outcome == "WIN":
            lesson = Lesson(
                id=f"lesson_{rec.id}",
                timestamp=now,
                category="PATTERN_WIN",
                title=f"WIN: {rec.action} {rec.timeframe} {rec.session}",
                content=(
                    f"{rec.action} {rec.symbol} {rec.timeframe} | "
                    f"Session:{rec.session} Day:{rec.day_of_week} | "
                    f"HTF:{rec.htf_bias} Zone:{rec.premium_discount} | "
                    f"Structure:{rec.smc_structure[:120]} | "
                    f"Confluences:{', '.join(rec.confluences[:4])} | "
                    f"RR:{rec.outcome_rr:.2f}"
                ),
                symbol=rec.symbol,
                timeframe=rec.timeframe,
                importance="High",
                confidence=0.65,
                confluence_weight=0.15,
            )
            self.add_lesson(lesson)
        elif rec.outcome == "LOSS":
            lesson = Lesson(
                id=f"lesson_{rec.id}",
                timestamp=now,
                category="PATTERN_LOSS",
                title=f"LOSS: {rec.action} {rec.timeframe} {rec.session}",
                content=(
                    f"FAILED: {rec.action} {rec.symbol} {rec.timeframe} | "
                    f"Session:{rec.session} Day:{rec.day_of_week} | "
                    f"HTF:{rec.htf_bias} Zone:{rec.premium_discount} | "
                    f"Structure:{rec.smc_structure[:120]} | "
                    f"Note:{rec.outcome_note} | "
                    f"AVOID when similar conditions"
                ),
                symbol=rec.symbol,
                timeframe=rec.timeframe,
                importance="High",
                confidence=0.75,
                confluence_weight=0.0,
            )
            self.add_lesson(lesson)

    def _trim_lessons(self):
        if len(self._lessons) > MAX_LESSONS:
            self._lessons.sort(key=lambda l: l.usage_score, reverse=True)
            self._lessons = self._lessons[:MAX_LESSONS]

    # ------------------------------------------------------------------
    # NO-TRADE explanation
    # ------------------------------------------------------------------

    def explain_no_trade(
        self,
        symbol: str,
        timeframe: str,
        mtf_data: dict,
        present_confluences: list[str],
        block_reason: str,
    ) -> NoTradeExplanation:
        """
        Generate a structured explanation for why no trade was taken.
        Shows exactly what's missing vs what's present.
        """
        required = [
            "HTF bias aligned (W1 + D1 agree)",
            "Break of Structure (BOS) confirmed",
            "Unmitigated Order Block present",
            "Fair Value Gap (FVG) present",
            "Liquidity sweep or target identified",
            "Entry in Premium/Discount zone",
            "Minimum 3 SMC confluences",
        ]

        present_lower = [c.lower() for c in present_confluences]

        def is_present(req: str) -> bool:
            keywords = req.lower().split()[:3]
            return any(all(kw in c for kw in keywords) for c in present_lower)

        missing  = [r for r in required if not is_present(r)]
        present  = [r for r in required if is_present(r)]

        # Score how close we are (0–100)
        score = int(len(present) / len(required) * 100)

        # HTF status
        htf_aligned = mtf_data.get("htf_aligned", False)
        w_bias      = mtf_data.get("weekly_bias", "ranging")
        d_bias      = mtf_data.get("daily_bias", "ranging")
        htf_status  = (
            f"✅ W1:{w_bias.upper()} D1:{d_bias.upper()} — ALIGNED"
            if htf_aligned
            else f"❌ W1:{w_bias.upper()} vs D1:{d_bias.upper()} — NOT ALIGNED"
        )

        # Session status
        now     = datetime.now(timezone.utc)
        hour    = now.hour
        session = "London" if 7 <= hour < 16 else "NewYork" if 13 <= hour < 22 else "Asian"
        stats   = self.get_stats()
        worst   = stats.worst_session
        sess_ok = session != worst
        sess_status = (
            f"✅ {session} session (good)"
            if sess_ok
            else f"⚠️ {session} session (historically weak for this setup)"
        )

        # Recommendation
        if score >= 70:
            rec = "Setup is close. Wait for BOS confirmation or OB retest."
        elif score >= 50:
            rec = "Partial setup. Monitor for confluence development."
        else:
            rec = "No clear setup. Stay out and wait for better conditions."

        return NoTradeExplanation(
            reason=block_reason,
            missing_confluences=missing,
            present_confluences=present,
            htf_status=htf_status,
            session_status=sess_status,
            recommendation=rec,
            score=score,
        )

    # ------------------------------------------------------------------
    # AI context injection
    # ------------------------------------------------------------------

    def get_ai_context(self, symbol: str, timeframe: str, action: str = "") -> str:
        """
        Returns adaptive memory context for AI prompt injection.
        Prioritizes: Critical > High importance, high usage_score, high confluence_weight.
        """
        stats = self.get_stats()
        lines = ["=== ADAPTIVE MEMORY & LEARNED PATTERNS ==="]

        # Performance summary
        if stats.total_signals > 0:
            lines.append(
                f"Bot performance: {stats.total_signals} signals | "
                f"Win rate: {stats.win_rate}% | "
                f"Best TF: {stats.best_timeframe} | "
                f"Best session: {stats.best_session}"
            )
            if stats.worst_session:
                lines.append(f"⚠️ Worst session: {stats.worst_session} — extra caution required")

        # Top confluence scores
        top_confs = self._scorer.top_confluences(5)
        if top_confs:
            lines.append("\n🏆 HIGHEST WIN-RATE CONFLUENCES:")
            for conf, wr in top_confs:
                lines.append(f"  [{wr*100:.0f}% WR] {conf}")

        # Adaptive lessons — sorted by importance + usage_score
        relevant = [
            l for l in self._lessons
            if (l.symbol == symbol or l.symbol in ("XAUUSD", "XAUUSDm"))
            and (not l.timeframe or l.timeframe == timeframe)
            and l.confidence >= 0.4
        ]

        importance_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        relevant.sort(
            key=lambda l: (importance_order.get(l.importance, 2), -l.usage_score),
        )

        wins   = [l for l in relevant if l.category == "PATTERN_WIN"][:4]
        losses = [l for l in relevant if l.category == "PATTERN_LOSS"][:4]
        other  = [l for l in relevant if l.category not in ("PATTERN_WIN", "PATTERN_LOSS")][:3]

        if wins:
            lines.append("\n✅ PATTERNS THAT WORK (use these):")
            for l in wins:
                lines.append(
                    f"  [{l.importance}] [{l.times_used}x used] "
                    f"[score:{l.usage_score:.0f}] [weight:{l.confluence_weight:.2f}] "
                    f"{l.title}: {l.content[:180]}"
                )

        if losses:
            lines.append("\n❌ PATTERNS TO AVOID (these failed):")
            for l in losses:
                lines.append(
                    f"  [{l.importance}] [{l.occurrences}x seen] "
                    f"{l.title}: {l.content[:180]}"
                )

        if other:
            lines.append("\n📝 ADDITIONAL CONTEXT:")
            for l in other:
                lines.append(f"  [{l.importance}] {l.title}: {l.content[:150]}")

        # Recent outcomes
        recent = [s for s in reversed(self._signals) if s.outcome != "PENDING"][:5]
        if recent:
            lines.append("\n📊 RECENT TRADE OUTCOMES:")
            for s in recent:
                emoji = "✅" if s.outcome == "WIN" else "❌" if s.outcome == "LOSS" else "➖"
                lines.append(
                    f"  {emoji} {s.action} {s.timeframe} {s.session} "
                    f"RR:{s.outcome_rr:.1f} — {s.outcome_note[:80]}"
                )

        lines.append("\n=== ADAPTIVE INTELLIGENCE RULES ===")
        lines.append("- Prioritize HIGH/CRITICAL importance lessons")
        lines.append("- Favour confluences with proven win rates above")

        # Day of week awareness
        day_analysis = self.get_day_analysis()
        today        = day_analysis["today"]
        rating       = day_analysis["today_rating"]
        today_stats  = day_analysis["today_stats"]
        if today_stats.get("total", 0) >= 3:
            wr = today_stats.get("win_rate", 0)
            lines.append(
                f"- TODAY IS {today.upper()} — Historical WR: {wr}% "
                f"[{rating}] — {day_analysis['today_rec']}"
            )
            if rating == "AVOID":
                lines.append(f"  ⚠️ CAUTION: {today} is historically a weak trading day. Be extra selective.")

        lines.append("- NEVER self-modify core trading logic")
        lines.append("- NEVER auto-trade — signal only")
        lines.append("- ALWAYS explain reasoning transparently")
        lines.append("=== END MEMORY ===")

        # Track usage
        for l in relevant[:10]:
            l.times_used += 1
            l.last_used   = datetime.now(timezone.utc).isoformat()
            l.usage_score = self._calc_usage_score(l)
        self._save()

        return "\n".join(lines)

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

        tf_stats: dict[str, list] = {}
        for s in completed:
            tf_stats.setdefault(s.timeframe, []).append(s.outcome == "WIN")
        best_tf = max(tf_stats, key=lambda t: sum(tf_stats[t]) / len(tf_stats[t])) if tf_stats else ""

        sess_stats: dict[str, list] = {}
        for s in completed:
            sess_stats.setdefault(s.session, []).append(s.outcome == "WIN")
        best_sess  = max(sess_stats, key=lambda t: sum(sess_stats[t]) / len(sess_stats[t])) if sess_stats else ""
        worst_sess = min(sess_stats, key=lambda t: sum(sess_stats[t]) / len(sess_stats[t])) if sess_stats else ""

        conf_count: dict[str, int] = {}
        for s in wins:
            for c in s.confluences:
                conf_count[c] = conf_count.get(c, 0) + 1
        best_conf = max(conf_count, key=conf_count.get) if conf_count else ""

        top_lessons = sorted(self._lessons, key=lambda l: l.usage_score, reverse=True)[:5]

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
            top_lessons=[
                {"title": l.title, "score": l.usage_score,
                 "importance": l.importance, "times_used": l.times_used}
                for l in top_lessons
            ],
        )

    def get_day_analysis(self) -> dict:
        """
        Analyze historical performance by day of week.
        Returns which days are best/worst for trading based on real history.
        """
        from datetime import datetime, timezone

        completed = [s for s in self._signals if s.outcome not in ("PENDING", "CANCELLED")]
        today     = datetime.now(timezone.utc).strftime("%A")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_stats: dict[str, dict] = {}

        for day in day_order:
            trades = [s for s in completed if s.day_of_week == day]
            wins   = [s for s in trades if s.outcome == "WIN"]
            losses = [s for s in trades if s.outcome == "LOSS"]
            total  = len(wins) + len(losses)
            wr     = round(len(wins) / total * 100, 1) if total else 0.0

            # Pip stats
            win_pips  = sum(s.outcome_pips for s in wins  if s.outcome_pips > 0)
            loss_pips = sum(s.outcome_pips for s in losses if s.outcome_pips < 0)
            net_pips  = round(win_pips + loss_pips, 1)

            # Best session on this day
            sess_wins: dict[str, int] = {}
            for s in wins:
                sess_wins[s.session] = sess_wins.get(s.session, 0) + 1
            best_sess = max(sess_wins, key=sess_wins.get) if sess_wins else "N/A"

            day_stats[day] = {
                "total":     total,
                "wins":      len(wins),
                "losses":    len(losses),
                "win_rate":  wr,
                "net_pips":  net_pips,
                "best_session": best_sess,
                "is_today":  day == today,
            }

        # Rank days
        ranked = sorted(
            [(d, v) for d, v in day_stats.items() if v["total"] >= 2],
            key=lambda x: (-x[1]["win_rate"], -x[1]["net_pips"]),
        )

        best_day  = ranked[0][0]  if ranked else "Not enough data"
        worst_day = ranked[-1][0] if ranked else "Not enough data"

        # Today's recommendation
        today_stats = day_stats.get(today, {})
        today_wr    = today_stats.get("win_rate", 0)
        today_total = today_stats.get("total", 0)

        if today_total < 3:
            today_rec = "Not enough history for today. Trade with normal caution."
            today_rating = "NEUTRAL"
        elif today_wr >= 65:
            today_rec = f"Strong day historically ({today_wr}% WR). Good conditions to trade."
            today_rating = "GOOD"
        elif today_wr >= 50:
            today_rec = f"Average day ({today_wr}% WR). Trade selectively, wait for A+ setups only."
            today_rating = "NEUTRAL"
        else:
            today_rec = f"Weak day historically ({today_wr}% WR). Consider reducing size or skipping."
            today_rating = "AVOID"

        return {
            "today":        today,
            "today_stats":  today_stats,
            "today_rating": today_rating,
            "today_rec":    today_rec,
            "day_stats":    day_stats,
            "best_day":     best_day,
            "worst_day":    worst_day,
            "ranked":       [(d, v["win_rate"], v["total"]) for d, v in ranked],
            "total_completed": len(completed),
        }
        return [s for s in self._signals if s.outcome == "PENDING"]

    def get_lessons(self) -> list[Lesson]:
        return sorted(self._lessons, key=lambda l: l.usage_score, reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                self._signals = [SignalRecord(**s) for s in data.get("signals", [])]
                self._scorer.from_dict(data.get("confluence_scores", {}))
            except Exception as exc:
                logger.error("Failed to load memory: %s", exc)
                self._signals = []

        if LESSONS_FILE.exists():
            try:
                data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
                raw  = data.get("lessons", [])
                self._lessons = []
                for item in raw:
                    # Handle old lesson format (missing new fields)
                    item.setdefault("importance", "Medium")
                    item.setdefault("times_used", 0)
                    item.setdefault("times_contributed", 0)
                    item.setdefault("usage_score", 0.0)
                    item.setdefault("confluence_weight", 0.1)
                    item.setdefault("last_used", "")
                    self._lessons.append(Lesson(**item))
            except Exception as exc:
                logger.error("Failed to load lessons: %s", exc)
                self._lessons = []

    def _save(self):
        try:
            MEMORY_FILE.write_text(
                json.dumps({
                    "signals": [asdict(s) for s in self._signals],
                    "confluence_scores": self._scorer.to_dict(),
                }, indent=2),
                encoding="utf-8",
            )
            LESSONS_FILE.write_text(
                json.dumps({"lessons": [asdict(l) for l in self._lessons]}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save memory: %s", exc)


# ---------------------------------------------------------------------------
# Outcome evaluator
# ---------------------------------------------------------------------------

class OutcomeEvaluator:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def evaluate_pending(self, current_price: float) -> list[dict]:
        resolved = []
        for sig in self.memory.get_pending_signals():
            if sig.action == "NO_TRADE" or sig.entry == 0:
                continue
            result = self._check(sig, current_price)
            if result:
                self.memory.update_outcome(sig.id, result["outcome"], result["price"], result["note"])
                resolved.append({"id": sig.id, "outcome": result["outcome"]})
        return resolved

    def _check(self, sig: SignalRecord, price: float) -> dict | None:
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
