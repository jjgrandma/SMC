"""
Multi-Timeframe (MTF) Analysis Module.

Implements the full cascade:  1W → 1D → 4H → 1H → 15M → 5M → 1M

Responsibilities:
  - Build HTF bias (Weekly + Daily)
  - Cascade bias down to LTF entry timeframe
  - Calculate Premium / Discount zones
  - Count SMC confluences (hard gate: minimum 3 required)
  - Return get_active_trades() and get_user_settings() tool data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from app.config import get_settings
from app.smc_engine import analysis_to_dict, SMCAnalysis
from app.tools import get_market_data

logger = logging.getLogger(__name__)
settings = get_settings()

# Engine selection
if settings.use_external_smc:
    from app.smc_adapter import SMCEngineAdapter as _Engine
else:
    from app.smc_engine import SMCEngine as _Engine  # type: ignore

# ---------------------------------------------------------------------------
# Timeframe cascade definition
# ---------------------------------------------------------------------------

HTF_TIMEFRAMES  = ["W1", "D1"]          # bias definition
MTF_TIMEFRAMES  = ["H4", "H1"]          # confirmation
LTF_TIMEFRAMES  = ["M15", "M5", "M1"]   # entry refinement

ALL_TIMEFRAMES  = HTF_TIMEFRAMES + MTF_TIMEFRAMES + LTF_TIMEFRAMES


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TimeframeBias:
    timeframe: str
    trend: str                              # bullish | bearish | ranging
    last_bos_direction: str | None
    last_choch_direction: str | None
    active_fvg_count: int
    active_ob_count: int
    liquidity_count: int
    premium_discount: str                   # premium | discount | equilibrium
    key_high: float
    key_low: float
    equilibrium: float
    smc_data: dict = field(default_factory=dict)


@dataclass
class MTFResult:
    symbol: str
    weekly_bias: str
    daily_bias: str
    htf_aligned: bool                       # W1 + D1 agree
    overall_bias: str                       # final directional bias
    timeframes: dict[str, TimeframeBias]    # all TF results
    confluence_count: int
    confluence_details: list[str]
    trade_allowed: bool                     # False if < 3 confluences
    block_reason: str | None
    current_price: float


# ---------------------------------------------------------------------------
# Premium / Discount calculator
# ---------------------------------------------------------------------------

def _premium_discount(
    current_price: float,
    key_high: float,
    key_low: float,
) -> tuple[str, float]:
    """
    Returns (zone, equilibrium).
    Premium  = price above 50% of the range (sell zone)
    Discount = price below 50% of the range (buy zone)
    """
    if key_high <= key_low:
        return "equilibrium", current_price
    eq = (key_high + key_low) / 2
    threshold = (key_high - key_low) * 0.1   # 10% buffer around EQ
    if current_price > eq + threshold:
        return "premium", eq
    if current_price < eq - threshold:
        return "discount", eq
    return "equilibrium", eq


# ---------------------------------------------------------------------------
# Confluence counter
# ---------------------------------------------------------------------------

def _count_confluences(
    bias: TimeframeBias,
    overall_bias: str,
    current_price: float,
) -> list[str]:
    """
    Returns a list of confluence reasons found.
    Minimum 3 required to allow a trade.
    """
    found: list[str] = []

    # 1. Trend alignment
    if bias.trend == overall_bias:
        found.append(f"Trend aligned ({bias.timeframe}: {bias.trend})")

    # 2. BOS in direction
    if bias.last_bos_direction == overall_bias:
        found.append(f"BOS confirmed {overall_bias} on {bias.timeframe}")

    # 3. CHoCH confirmation
    if bias.last_choch_direction == overall_bias:
        found.append(f"CHoCH confirmed {overall_bias} on {bias.timeframe}")

    # 4. Active FVG present
    if bias.active_fvg_count > 0:
        found.append(f"Active FVG present on {bias.timeframe} ({bias.active_fvg_count})")

    # 5. Active Order Block present
    if bias.active_ob_count > 0:
        found.append(f"Active OB present on {bias.timeframe} ({bias.active_ob_count})")

    # 6. Liquidity target exists
    if bias.liquidity_count > 0:
        found.append(f"Liquidity target on {bias.timeframe} ({bias.liquidity_count} zones)")

    # 7. Premium/Discount alignment
    if overall_bias == "bullish" and bias.premium_discount == "discount":
        found.append(f"Price in DISCOUNT zone on {bias.timeframe} — optimal BUY area")
    elif overall_bias == "bearish" and bias.premium_discount == "premium":
        found.append(f"Price in PREMIUM zone on {bias.timeframe} — optimal SELL area")

    return found


# ---------------------------------------------------------------------------
# Core MTF engine
# ---------------------------------------------------------------------------

class MTFEngine:
    MIN_CONFLUENCES = 3

    def __init__(self):
        self._smc = _Engine()

    def analyze(self, symbol: str, current_price: float) -> MTFResult:
        timeframes_data: dict[str, TimeframeBias] = {}

        # Analyze each timeframe
        for tf in ALL_TIMEFRAMES:
            try:
                df = get_market_data(symbol, tf)
                if df is None or len(df) < 20:
                    logger.warning("Insufficient data for %s %s", symbol, tf)
                    continue
                analysis = self._smc.analyze(df)
                bias = self._build_bias(tf, analysis, current_price)
                timeframes_data[tf] = bias
            except Exception as exc:
                logger.error("MTF analysis failed for %s %s: %s", symbol, tf, exc)

        # Determine HTF bias
        weekly_bias = timeframes_data.get("W1", None)
        daily_bias  = timeframes_data.get("D1", None)

        w_trend = weekly_bias.trend if weekly_bias else "ranging"
        d_trend = daily_bias.trend  if daily_bias  else "ranging"

        # HTF alignment check
        htf_aligned = (
            w_trend == d_trend
            and w_trend in ("bullish", "bearish")
        )
        overall_bias = w_trend if htf_aligned else (
            d_trend if d_trend != "ranging" else "ranging"
        )

        # Count confluences across H1 + H4 (entry timeframes)
        all_confluences: list[str] = []
        for tf in ["H4", "H1", "M15"]:
            if tf in timeframes_data:
                found = _count_confluences(
                    timeframes_data[tf], overall_bias, current_price
                )
                all_confluences.extend(found)

        # Deduplicate
        seen: set[str] = set()
        unique_confluences: list[str] = []
        for c in all_confluences:
            if c not in seen:
                seen.add(c)
                unique_confluences.append(c)

        confluence_count = len(unique_confluences)
        trade_allowed = (
            htf_aligned
            and overall_bias != "ranging"
            and confluence_count >= self.MIN_CONFLUENCES
        )

        block_reason: str | None = None
        if not htf_aligned:
            block_reason = "HTF not aligned — W1 and D1 disagree. Stay out."
        elif overall_bias == "ranging":
            block_reason = "No clear higher timeframe bias. Stay out."
        elif confluence_count < self.MIN_CONFLUENCES:
            block_reason = (
                f"Only {confluence_count} SMC confluence(s) found. "
                f"Minimum {self.MIN_CONFLUENCES} required."
            )

        return MTFResult(
            symbol=symbol,
            weekly_bias=w_trend,
            daily_bias=d_trend,
            htf_aligned=htf_aligned,
            overall_bias=overall_bias,
            timeframes=timeframes_data,
            confluence_count=confluence_count,
            confluence_details=unique_confluences,
            trade_allowed=trade_allowed,
            block_reason=block_reason,
            current_price=current_price,
        )

    def _build_bias(
        self,
        tf: str,
        analysis: SMCAnalysis,
        current_price: float,
    ) -> TimeframeBias:
        d = analysis_to_dict(analysis)

        # Key high/low from swing points
        swings = d.get("swing_points", [])
        highs = [s["price"] for s in swings if s["kind"] in ("HH", "LH", "HIGH")]
        lows  = [s["price"] for s in swings if s["kind"] in ("HL", "LL", "LOW")]
        key_high = max(highs) if highs else current_price
        key_low  = min(lows)  if lows  else current_price

        pd_zone, eq = _premium_discount(current_price, key_high, key_low)

        last_bos   = d.get("last_bos")
        last_choch = d.get("last_choch")

        return TimeframeBias(
            timeframe=tf,
            trend=analysis.trend,
            last_bos_direction=last_bos["direction"]   if last_bos   else None,
            last_choch_direction=last_choch["direction"] if last_choch else None,
            active_fvg_count=len(d.get("fvg", [])),
            active_ob_count=len(d.get("order_blocks", [])),
            liquidity_count=len(d.get("liquidity_zones", [])),
            premium_discount=pd_zone,
            key_high=round(key_high, 2),
            key_low=round(key_low, 2),
            equilibrium=round(eq, 2),
            smc_data=d,
        )


# ---------------------------------------------------------------------------
# Tool functions called by the agent
# ---------------------------------------------------------------------------

def get_active_trades(trader=None) -> dict[str, Any]:
    """
    Returns open positions and exposure.
    Connects to MT5 if enabled, otherwise returns empty state.
    """
    if trader is not None and settings.mt5_enabled:
        try:
            positions = trader.get_open_positions()
            account   = trader.get_account_info()
            return {
                "open_positions": positions,
                "position_count": len(positions),
                "account": account,
                "has_buy":  any(p.get("type") == 0 for p in positions),
                "has_sell": any(p.get("type") == 1 for p in positions),
            }
        except Exception as exc:
            logger.error("get_active_trades failed: %s", exc)

    return {
        "open_positions": [],
        "position_count": 0,
        "account": {},
        "has_buy": False,
        "has_sell": False,
    }


def get_user_settings(user_profile=None) -> dict[str, Any]:
    """
    Returns user trading settings for the AI agent context.
    """
    if user_profile:
        return {
            "account_balance":  user_profile.account_balance,
            "risk_percent":     user_profile.risk_percent,
            "max_risk_amount":  round(user_profile.account_balance * user_profile.risk_percent / 100, 2),
            "trading_mode":     "auto" if settings.auto_trade else "manual",
            "min_confidence":   user_profile.min_confidence,
            "symbol":           user_profile.symbol,
            "timeframe":        user_profile.timeframe,
            "alerts_enabled":   user_profile.alerts_enabled,
            "min_rr_ratio":     settings.min_rr_ratio,
        }
    return {
        "account_balance":  settings.default_account_balance,
        "risk_percent":     settings.max_risk_percent,
        "max_risk_amount":  round(settings.default_account_balance * settings.max_risk_percent / 100, 2),
        "trading_mode":     "auto" if settings.auto_trade else "manual",
        "min_confidence":   "MEDIUM",
        "symbol":           settings.symbol,
        "timeframe":        "H1",
        "alerts_enabled":   True,
        "min_rr_ratio":     settings.min_rr_ratio,
    }


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def mtf_to_dict(result: MTFResult) -> dict:
    tfs = {}
    for tf, bias in result.timeframes.items():
        tfs[tf] = {
            "trend":               bias.trend,
            "bos_direction":       bias.last_bos_direction,
            "choch_direction":     bias.last_choch_direction,
            "active_fvg":          bias.active_fvg_count,
            "active_ob":           bias.active_ob_count,
            "liquidity_zones":     bias.liquidity_count,
            "premium_discount":    bias.premium_discount,
            "key_high":            bias.key_high,
            "key_low":             bias.key_low,
            "equilibrium":         bias.equilibrium,
        }
    return {
        "symbol":              result.symbol,
        "current_price":       result.current_price,
        "weekly_bias":         result.weekly_bias,
        "daily_bias":          result.daily_bias,
        "htf_aligned":         result.htf_aligned,
        "overall_bias":        result.overall_bias,
        "confluence_count":    result.confluence_count,
        "confluence_details":  result.confluence_details,
        "trade_allowed":       result.trade_allowed,
        "block_reason":        result.block_reason,
        "timeframes":          tfs,
    }
