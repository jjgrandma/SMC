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


@dataclass
class RiskResult:
    approved: bool
    lot_size: float
    risk_amount: float
    rr_ratio: float
    pip_risk: float
    rejection_reason: str | None = None


class RiskManager:
    # XAUUSD: 1 lot = 100 oz, pip value ≈ $1 per 0.01 move per lot
    # Simplified: pip_value_per_lot = 1.0 USD for XAUUSD (0.01 price move)
    PIP_VALUE_PER_LOT = 1.0
    PIP_SIZE = 0.01
    MIN_LOT = 0.01
    MAX_LOT = 100.0
    LOT_STEP = 0.01

    def validate_and_size(self, params: RiskParams) -> RiskResult:
        risk_pct = params.risk_percent or settings.max_risk_percent

        # --- R:R check ---
        pip_risk = abs(params.entry - params.stop_loss)
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
