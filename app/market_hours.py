"""
Market Hours Module.

Gold (XAUUSD) trading hours:
  - Opens: Sunday 22:00 UTC (Sydney open)
  - Closes: Friday 21:00 UTC (NY close)
  - Closed: Saturday all day + Sunday until 22:00 UTC

Sessions:
  - Sydney/Asian:  22:00–08:00 UTC
  - London:        07:00–16:00 UTC
  - New York:      13:00–22:00 UTC
  - London/NY overlap: 13:00–16:00 UTC (highest volume)

Also tracks:
  - Daily close (21:00–22:00 UTC) — low liquidity, avoid trading
  - End of week (Friday 18:00+ UTC) — reduce exposure
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Literal


@dataclass
class MarketStatus:
    is_open: bool
    session: str                    # Sydney | Asian | London | NewYork | Overlap | Closed
    next_open: str                  # when market opens next
    next_close: str                 # when current session closes
    day_of_week: str
    hour_utc: int
    warning: str                    # any caution message
    trade_quality: Literal["OPTIMAL", "GOOD", "CAUTION", "AVOID", "CLOSED"]
    reason: str                     # explanation


def get_market_status(dt: datetime | None = None) -> MarketStatus:
    """
    Returns current Gold market status.
    """
    now     = dt or datetime.now(timezone.utc)
    weekday = now.weekday()   # 0=Monday, 6=Sunday
    hour    = now.hour
    minute  = now.minute
    day     = now.strftime("%A")

    # ── Saturday — market closed ──────────────────────────────────────
    if weekday == 5:
        # Next open: Sunday 22:00 UTC
        next_open_dt = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now.hour >= 22:
            next_open_dt += timedelta(days=1)
        else:
            next_open_dt += timedelta(days=1)   # next Sunday
        return MarketStatus(
            is_open=False,
            session="Closed",
            next_open="Sunday 22:00 UTC",
            next_close="N/A",
            day_of_week=day,
            hour_utc=hour,
            warning="Market is closed. No trading until Sunday 22:00 UTC.",
            trade_quality="CLOSED",
            reason="Saturday — Gold market closed.",
        )

    # ── Sunday before 22:00 — market closed ──────────────────────────
    if weekday == 6 and hour < 22:
        mins_to_open = (22 - hour) * 60 - minute
        return MarketStatus(
            is_open=False,
            session="Pre-Open",
            next_open=f"Today 22:00 UTC ({mins_to_open} min)",
            next_close="N/A",
            day_of_week=day,
            hour_utc=hour,
            warning=f"Market opens in {mins_to_open} minutes.",
            trade_quality="CLOSED",
            reason="Sunday pre-open — market not yet active.",
        )

    # ── Friday after 21:00 — market closing ──────────────────────────
    if weekday == 4 and hour >= 21:
        return MarketStatus(
            is_open=False,
            session="Closing",
            next_open="Sunday 22:00 UTC",
            next_close="Now",
            day_of_week=day,
            hour_utc=hour,
            warning="Market closing. Avoid new positions.",
            trade_quality="CLOSED",
            reason="Friday close — Gold market closing for the weekend.",
        )

    # ── Friday 18:00–21:00 — end of week caution ─────────────────────
    if weekday == 4 and hour >= 18:
        return MarketStatus(
            is_open=True,
            session="NewYork",
            next_open="N/A",
            next_close="Friday 21:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="End of week. Reduce position size. Close trades before 21:00 UTC.",
            trade_quality="CAUTION",
            reason="Friday late session — weekend gap risk.",
        )

    # ── Daily close gap (21:00–22:00 UTC Mon–Thu) ────────────────────
    if weekday < 4 and hour == 21:
        return MarketStatus(
            is_open=True,
            session="DailyClose",
            next_open="N/A",
            next_close="22:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="Daily close period (21:00–22:00 UTC). Low liquidity. Avoid entries.",
            trade_quality="AVOID",
            reason="Daily candle close — spread widens, liquidity drops.",
        )

    # ── Determine active session ──────────────────────────────────────
    # London/NY overlap (best quality)
    if 13 <= hour < 16:
        return MarketStatus(
            is_open=True,
            session="Overlap",
            next_open="N/A",
            next_close="16:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="",
            trade_quality="OPTIMAL",
            reason="London/NY overlap — highest volume and liquidity.",
        )

    # London session
    if 7 <= hour < 16:
        quality = "OPTIMAL" if 8 <= hour < 12 else "GOOD"
        return MarketStatus(
            is_open=True,
            session="London",
            next_open="N/A",
            next_close="16:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="" if quality == "OPTIMAL" else "Late London session — volume declining.",
            trade_quality=quality,
            reason="London session — strong institutional activity.",
        )

    # New York session
    if 13 <= hour < 21:
        quality = "OPTIMAL" if 13 <= hour < 17 else "GOOD"
        return MarketStatus(
            is_open=True,
            session="NewYork",
            next_open="N/A",
            next_close="21:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="" if quality == "OPTIMAL" else "Late NY session — volume declining.",
            trade_quality=quality,
            reason="New York session — strong USD-driven moves.",
        )

    # Asian/Sydney session
    if 22 <= hour or hour < 7:
        quality = "CAUTION" if weekday in (0, 4) else "GOOD"  # Monday open / Friday close
        return MarketStatus(
            is_open=True,
            session="Asian",
            next_open="N/A",
            next_close="07:00 UTC",
            day_of_week=day,
            hour_utc=hour,
            warning="Asian session — lower volatility. Range-bound typical for Gold.",
            trade_quality=quality,
            reason="Asian session — accumulation phase, wait for London.",
        )

    # Fallback
    return MarketStatus(
        is_open=True,
        session="Unknown",
        next_open="N/A",
        next_close="N/A",
        day_of_week=day,
        hour_utc=hour,
        warning="",
        trade_quality="GOOD",
        reason="Market open.",
    )


def is_trading_allowed(dt: datetime | None = None) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Blocks signals when market is closed or quality is CLOSED/AVOID.
    """
    status = get_market_status(dt)
    if status.trade_quality in ("CLOSED", "AVOID"):
        return False, status.reason
    return True, ""


def is_weekend() -> bool:
    now = datetime.now(timezone.utc)
    return now.weekday() >= 5 or (now.weekday() == 4 and now.hour >= 21)


def is_saturday() -> bool:
    return datetime.now(timezone.utc).weekday() == 5


def format_market_status(status: MarketStatus) -> str:
    """Format market status for Telegram display."""
    quality_emoji = {
        "OPTIMAL": "🟢",
        "GOOD":    "🟡",
        "CAUTION": "🟠",
        "AVOID":   "🔴",
        "CLOSED":  "⛔",
    }.get(status.trade_quality, "⚪")

    session_emoji = {
        "London":   "🇬🇧",
        "NewYork":  "🇺🇸",
        "Asian":    "🌏",
        "Overlap":  "🔥",
        "Closed":   "⛔",
        "Pre-Open": "⏳",
        "Closing":  "🔔",
        "DailyClose": "⏸",
    }.get(status.session, "📍")

    lines = [
        f"{quality_emoji} *{status.trade_quality}* — {session_emoji} {status.session}",
        f"📅 {status.day_of_week}  🕐 {status.hour_utc:02d}:00 UTC",
    ]

    if not status.is_open:
        lines.append(f"⛔ Market closed — next open: `{status.next_open}`")
    else:
        lines.append(f"✅ Market open — closes: `{status.next_close}`")

    if status.warning:
        lines.append(f"⚠️ _{status.warning}_")

    lines.append(f"_{status.reason}_")
    return "\n".join(lines)
