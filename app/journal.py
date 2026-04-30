"""
Trade Journal — persistent history of every signal given.
Tracks: signal details, outcome, P&L in PIPS, session, timeframe.
Storage: data/journal.json

Pip convention for XAUUSD:
  1 pip = $0.10 price move (industry standard for Gold)
  Entry 4600 → TP 4650 = 50 points = 500 pips
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

from app.pip_utils import price_to_pips, get_pip_size, format_pips

logger = logging.getLogger(__name__)
JOURNAL_FILE = Path("data/journal.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    id: str
    timestamp: str                  # ISO UTC when signal was given
    symbol: str
    timeframe: str
    action: str                     # BUY | SELL
    entry: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confidence: str                 # HIGH | MEDIUM | LOW
    htf_bias: str
    premium_discount: str
    session: str                    # London | NewYork | Asian
    day_of_week: str
    confluences: list[str] = field(default_factory=list)
    reasoning: str = ""
    # Pip distances at signal time
    sl_pips: float = 0.0            # SL distance in pips
    tp_pips: float = 0.0            # TP distance in pips
    # Outcome fields
    outcome: Literal["PENDING", "WIN", "LOSS", "BREAKEVEN", "CANCELLED"] = "PENDING"
    outcome_price: float = 0.0
    outcome_time: str = ""
    outcome_pips: float = 0.0       # pips gained/lost (+ = profit, - = loss)
    outcome_points: float = 0.0     # raw price difference
    outcome_rr_achieved: float = 0.0
    outcome_note: str = ""
    # Source
    ai_provider: str = ""
    mode: str = "AUTO"              # AUTO | MANUAL


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------

@dataclass
class PerformanceStats:
    # Totals
    total: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    pending: int = 0
    cancelled: int = 0
    # Rates
    win_rate: float = 0.0
    loss_rate: float = 0.0
    # RR
    avg_rr_target: float = 0.0
    avg_rr_achieved: float = 0.0
    best_rr: float = 0.0
    worst_rr: float = 0.0
    # Pip P&L
    total_pips: float = 0.0         # net pips (wins - losses)
    pips_won: float = 0.0           # total pips from winning trades
    pips_lost: float = 0.0          # total pips from losing trades
    avg_pips_win: float = 0.0       # average pips per win
    avg_pips_loss: float = 0.0      # average pips per loss
    best_trade_pips: float = 0.0    # best single trade in pips
    worst_trade_pips: float = 0.0   # worst single trade in pips
    # Streaks
    current_streak: int = 0
    current_streak_type: str = ""   # WIN | LOSS
    best_win_streak: int = 0
    worst_loss_streak: int = 0
    # Breakdown by timeframe
    by_timeframe: dict = field(default_factory=dict)
    # Breakdown by session
    by_session: dict = field(default_factory=dict)
    # Breakdown by confidence
    by_confidence: dict = field(default_factory=dict)
    # Breakdown by day
    by_day: dict = field(default_factory=dict)
    # Breakdown by direction
    by_direction: dict = field(default_factory=dict)
    # Recent form (last 10)
    last_10: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Journal store
# ---------------------------------------------------------------------------

class TradeJournal:
    def __init__(self):
        JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[TradeRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, signal: dict) -> str:
        """Log a new signal. Returns the trade ID."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 7 <= hour < 16:
            session = "London"
        elif 13 <= hour < 22:
            session = "NewYork"
        else:
            session = "Asian"

        trade_id = (
            f"{signal.get('symbol','X')}_"
            f"{signal.get('timeframe','H1')}_"
            f"{now.strftime('%Y%m%d_%H%M%S')}"
        )

        sym    = signal.get("symbol", "XAUUSDm")
        entry  = float(signal.get("entry") or 0)
        sl     = float(signal.get("stop_loss") or 0)
        tp     = float(signal.get("take_profit") or 0)

        rec = TradeRecord(
            id=trade_id,
            timestamp=now.isoformat(),
            symbol=sym,
            timeframe=signal.get("timeframe", "H1"),
            action=signal.get("action", ""),
            entry=entry,
            stop_loss=sl,
            take_profit=tp,
            rr_ratio=float(signal.get("rr_ratio") or 0),
            confidence=signal.get("confidence", ""),
            htf_bias=signal.get("htf_bias", ""),
            premium_discount=signal.get("premium_discount", ""),
            session=session,
            day_of_week=now.strftime("%A"),
            confluences=signal.get("confluences", []),
            reasoning=str(signal.get("reasoning", ""))[:300],
            ai_provider=signal.get("ai_comparison", {}).get("chosen", ""),
            mode=signal.get("mode", "AUTO"),
            # Pre-calculate pip distances
            sl_pips=price_to_pips(abs(entry - sl), sym) if entry and sl else 0.0,
            tp_pips=price_to_pips(abs(tp - entry), sym) if entry and tp else 0.0,
        )
        self._records.append(rec)
        self._save()
        return trade_id

    def set_outcome(
        self,
        trade_id: str,
        outcome: str,
        price: float = 0.0,
        note: str = "",
    ) -> TradeRecord | None:
        for rec in self._records:
            if rec.id == trade_id:
                rec.outcome       = outcome
                rec.outcome_price = price
                rec.outcome_time  = datetime.now(timezone.utc).isoformat()
                rec.outcome_note  = note

                if rec.entry > 0 and price > 0:
                    # Raw price difference
                    if rec.action == "BUY":
                        rec.outcome_points = round(price - rec.entry, 3)
                    else:
                        rec.outcome_points = round(rec.entry - price, 3)

                    # Convert to pips
                    rec.outcome_pips = price_to_pips(rec.outcome_points, rec.symbol)

                    # R:R achieved
                    sl_dist = abs(rec.entry - rec.stop_loss)
                    if sl_dist > 0:
                        rec.outcome_rr_achieved = round(
                            abs(rec.outcome_points) / sl_dist, 2
                        ) * (1 if rec.outcome_points >= 0 else -1)

                self._save()
                return rec
        return None

    def auto_evaluate(self, current_price: float) -> list[TradeRecord]:
        """Auto-mark pending signals as WIN/LOSS based on current price."""
        resolved = []
        for rec in self._records:
            if rec.outcome != "PENDING" or rec.entry == 0:
                continue
            if rec.action == "BUY":
                if current_price >= rec.take_profit:
                    self.set_outcome(rec.id, "WIN", current_price, "TP hit (auto)")
                    resolved.append(rec)
                elif current_price <= rec.stop_loss:
                    self.set_outcome(rec.id, "LOSS", current_price, "SL hit (auto)")
                    resolved.append(rec)
            elif rec.action == "SELL":
                if current_price <= rec.take_profit:
                    self.set_outcome(rec.id, "WIN", current_price, "TP hit (auto)")
                    resolved.append(rec)
                elif current_price >= rec.stop_loss:
                    self.set_outcome(rec.id, "LOSS", current_price, "SL hit (auto)")
                    resolved.append(rec)
        return resolved

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[TradeRecord]:
        return list(reversed(self._records))

    def get_pending(self) -> list[TradeRecord]:
        return [r for r in self._records if r.outcome == "PENDING"]

    def get_completed(self) -> list[TradeRecord]:
        return [r for r in self._records if r.outcome not in ("PENDING", "CANCELLED")]

    def get_by_id(self, trade_id: str) -> TradeRecord | None:
        for r in self._records:
            if r.id == trade_id:
                return r
        return None

    def get_recent(self, days: int = 30) -> list[TradeRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            r for r in reversed(self._records)
            if datetime.fromisoformat(r.timestamp) >= cutoff
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, days: int | None = None) -> PerformanceStats:
        records = self.get_recent(days) if days else list(self._records)
        completed = [r for r in records if r.outcome not in ("PENDING", "CANCELLED")]
        wins      = [r for r in completed if r.outcome == "WIN"]
        losses    = [r for r in completed if r.outcome == "LOSS"]
        bes       = [r for r in completed if r.outcome == "BREAKEVEN"]
        pending   = [r for r in records   if r.outcome == "PENDING"]
        cancelled = [r for r in records   if r.outcome == "CANCELLED"]

        total_decided = len(wins) + len(losses) + len(bes)
        win_rate  = round(len(wins)   / total_decided * 100, 1) if total_decided else 0.0
        loss_rate = round(len(losses) / total_decided * 100, 1) if total_decided else 0.0

        avg_rr_t = round(sum(r.rr_ratio for r in completed) / len(completed), 2) if completed else 0.0
        avg_rr_a = round(sum(r.outcome_rr_achieved for r in wins) / len(wins), 2) if wins else 0.0
        best_rr  = round(max((r.outcome_rr_achieved for r in wins), default=0.0), 2)
        worst_rr = round(min((r.outcome_rr_achieved for r in losses), default=0.0), 2)

        # Pip P&L
        pips_won   = sum(r.outcome_pips for r in wins   if r.outcome_pips > 0)
        pips_lost  = sum(r.outcome_pips for r in losses if r.outcome_pips < 0)
        total_pips = round(pips_won + pips_lost, 1)
        avg_pw     = round(pips_won  / len(wins)   if wins   else 0.0, 1)
        avg_pl     = round(pips_lost / len(losses) if losses else 0.0, 1)
        best_pip   = round(max((r.outcome_pips for r in wins),   default=0.0), 1)
        worst_pip  = round(min((r.outcome_pips for r in losses), default=0.0), 1)

        # Streaks
        cur_streak = 0
        cur_type   = ""
        best_ws    = 0
        worst_ls   = 0
        tmp_w = tmp_l = 0
        for r in completed:
            if r.outcome == "WIN":
                tmp_w += 1
                tmp_l  = 0
                best_ws = max(best_ws, tmp_w)
            elif r.outcome == "LOSS":
                tmp_l += 1
                tmp_w  = 0
                worst_ls = max(worst_ls, tmp_l)

        if completed:
            last = completed[-1]
            if last.outcome == "WIN":
                cur_streak = tmp_w
                cur_type   = "WIN"
            elif last.outcome == "LOSS":
                cur_streak = tmp_l
                cur_type   = "LOSS"

        # Breakdowns
        def breakdown(items: list[TradeRecord], key_fn) -> dict:
            result: dict[str, dict] = {}
            for r in items:
                k = key_fn(r)
                if k not in result:
                    result[k] = {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
                result[k]["total"] += 1
                if r.outcome == "WIN":
                    result[k]["wins"] += 1
                elif r.outcome == "LOSS":
                    result[k]["losses"] += 1
            for k in result:
                t = result[k]["wins"] + result[k]["losses"]
                result[k]["win_rate"] = round(result[k]["wins"] / t * 100, 1) if t else 0.0
            return result

        last_10 = [
            {
                "action":  r.action,
                "outcome": r.outcome,
                "rr":      r.outcome_rr_achieved,
                "pips":    r.outcome_pips,
            }
            for r in completed[-10:]
        ]

        return PerformanceStats(
            total=len(records),
            wins=len(wins),
            losses=len(losses),
            breakevens=len(bes),
            pending=len(pending),
            cancelled=len(cancelled),
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_rr_target=avg_rr_t,
            avg_rr_achieved=avg_rr_a,
            best_rr=best_rr,
            worst_rr=worst_rr,
            total_pips=total_pips,
            pips_won=round(pips_won, 1),
            pips_lost=round(pips_lost, 1),
            avg_pips_win=avg_pw,
            avg_pips_loss=avg_pl,
            best_trade_pips=best_pip,
            worst_trade_pips=worst_pip,
            current_streak=cur_streak,
            current_streak_type=cur_type,
            best_win_streak=best_ws,
            worst_loss_streak=worst_ls,
            by_timeframe=breakdown(completed, lambda r: r.timeframe),
            by_session=breakdown(completed, lambda r: r.session),
            by_confidence=breakdown(completed, lambda r: r.confidence),
            by_day=breakdown(completed, lambda r: r.day_of_week),
            by_direction=breakdown(completed, lambda r: r.action),
            last_10=last_10,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not JOURNAL_FILE.exists():
            return
        try:
            data = json.loads(JOURNAL_FILE.read_text(encoding="utf-8"))
            self._records = [TradeRecord(**r) for r in data.get("trades", [])]
            logger.info("Journal loaded: %d records", len(self._records))
        except Exception as exc:
            logger.error("Failed to load journal: %s", exc)
            self._records = []

    def _save(self):
        try:
            JOURNAL_FILE.write_text(
                json.dumps({"trades": [asdict(r) for r in self._records]}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save journal: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_journal: TradeJournal | None = None


def get_journal() -> TradeJournal:
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal
