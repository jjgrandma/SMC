"""
Risk Management Module.
Calculates position sizes, validates R:R, enforces trade limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RiskParams:
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    account_balance: float
    risk_percent: float | None = None
    is_swing: bool = False  # True = allow wider SL for swing trades


@dataclass
class RiskResult:
    approved: bool
    lot_size: float
    risk_amount: float
    rr_ratio: float
    pip_risk: float
    rejection_reason: str | None = None


class RiskManager:
    # XAUUSD: 1 pip = $0.10, pip value per lot = $10
    PIP_SIZE = 0.10          # 1 pip for Gold
    MIN_LOT  = 0.01
    MAX_LOT  = 100.0
    LOT_STEP = 0.01
    # SL limits in PRICE POINTS (not pips) for Gold
    # M15/M5 entry: SL should be 10-30 pts (100-300 pips)
    # H4 entry: SL would be 50-150 pts (500-1500 pips) — too wide
    MAX_SL_POINTS       = 35.0   # max 35 pts SL for intraday entries (~350 pips)
    MAX_SL_POINTS_SWING = 150.0  # max 150 pts SL for swing trades

    def validate_and_size(self, params: RiskParams) -> RiskResult:
        risk_pct = params.risk_percent or settings.max_risk_percent

        # --- R:R check ---
        pip_risk   = abs(params.entry - params.stop_loss)
        pip_reward = abs(params.take_profit - params.entry)

        if pip_risk == 0:
            return RiskResult(
                approved=False, lot_size=0, risk_amount=0,
                rr_ratio=0, pip_risk=0,
                rejection_reason="Stop loss equals entry price.",
            )

        rr_ratio = round(pip_reward / pip_risk, 2)
        if rr_ratio < settings.min_rr_ratio:
            return RiskResult(
                approved=False, lot_size=0, risk_amount=0,
                rr_ratio=rr_ratio, pip_risk=pip_risk,
                rejection_reason=f"R:R {rr_ratio} is below minimum {settings.min_rr_ratio}.",
            )

        # --- SL too wide check (prevents H4-level SL on LTF entries) ---
        from app.pip_utils import price_to_pips
        sl_pips   = price_to_pips(pip_risk, params.symbol)
        is_swing  = getattr(params, "is_swing", False)
        max_pts   = self.MAX_SL_POINTS_SWING if is_swing else self.MAX_SL_POINTS

        if pip_risk > max_pts:
            return RiskResult(
                approved=False, lot_size=0, risk_amount=0,
                rr_ratio=rr_ratio, pip_risk=pip_risk,
                rejection_reason=(
                    f"SL too wide: {pip_risk:.1f} pts / {sl_pips:.0f} pips "
                    f"(max {max_pts} pts). "
                    f"Use M15/M5 structure for a tighter entry."
                ),
            )

        # --- Direction sanity ---
        if params.direction == "BUY":
            if params.stop_loss >= params.entry:
                return RiskResult(
                    approved=False, lot_size=0, risk_amount=0,
                    rr_ratio=rr_ratio, pip_risk=pip_risk,
                    rejection_reason="BUY: stop loss must be below entry.",
                )
            if params.take_profit <= params.entry:
                return RiskResult(
                    approved=False, lot_size=0, risk_amount=0,
                    rr_ratio=rr_ratio, pip_risk=pip_risk,
                    rejection_reason="BUY: take profit must be above entry.",
                )
        elif params.direction == "SELL":
            if params.stop_loss <= params.entry:
                return RiskResult(
                    approved=False, lot_size=0, risk_amount=0,
                    rr_ratio=rr_ratio, pip_risk=pip_risk,
                    rejection_reason="SELL: stop loss must be above entry.",
                )
            if params.take_profit >= params.entry:
                return RiskResult(
                    approved=False, lot_size=0, risk_amount=0,
                    rr_ratio=rr_ratio, pip_risk=pip_risk,
                    rejection_reason="SELL: take profit must be below entry.",
                )

        # --- Position sizing ---
        risk_amount  = params.account_balance * (risk_pct / 100)

        # Convert price distance to pips for sizing
        from app.pip_utils import get_pip_size, calc_pip_value
        pip_size      = get_pip_size(params.symbol)
        pips_at_risk  = pip_risk / pip_size if pip_size > 0 else pip_risk / 0.01
        pip_val       = calc_pip_value(params.symbol, 1.0)   # $ per pip per lot
        lot_size      = risk_amount / (pips_at_risk * pip_val) if pips_at_risk > 0 else 0
        lot_size      = self._round_lot(lot_size)

        if lot_size < self.MIN_LOT:
            lot_size = self.MIN_LOT
        if lot_size > self.MAX_LOT:
            lot_size = self.MAX_LOT

        logger.info(
            "Risk approved: %s %s | lots=%.2f | risk=$%.2f | R:R=%.2f",
            params.direction, params.symbol, lot_size, risk_amount, rr_ratio,
        )

        return RiskResult(
            approved=True,
            lot_size=lot_size,
            risk_amount=round(risk_amount, 2),
            rr_ratio=rr_ratio,
            pip_risk=round(pip_risk, 2),
        )

    def _round_lot(self, lot: float) -> float:
        steps = round(lot / self.LOT_STEP)
        return round(steps * self.LOT_STEP, 2)
