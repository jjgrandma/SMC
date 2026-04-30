"""
Smart Money Concepts (SMC) Engine.
Analyzes OHLCV data for:
  - Market Structure (HH, HL, LH, LL)
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Fair Value Gap (FVG)
  - Order Block (OB)
  - Liquidity Zones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["HH", "HL", "LH", "LL", "HIGH", "LOW"]


@dataclass
class BOS:
    index: int
    price: float
    direction: Literal["bullish", "bearish"]
    broken_swing: SwingPoint


@dataclass
class CHoCH:
    index: int
    price: float
    direction: Literal["bullish", "bearish"]


@dataclass
class FVG:
    start_index: int
    end_index: int
    top: float
    bottom: float
    direction: Literal["bullish", "bearish"]
    filled: bool = False


@dataclass
class OrderBlock:
    index: int
    top: float
    bottom: float
    direction: Literal["bullish", "bearish"]
    mitigated: bool = False


@dataclass
class LiquidityZone:
    price: float
    kind: Literal["BSL", "SSL"]   # Buy-Side / Sell-Side Liquidity
    index: int


@dataclass
class SMCAnalysis:
    swing_points: list[SwingPoint] = field(default_factory=list)
    bos_list: list[BOS] = field(default_factory=list)
    choch_list: list[CHoCH] = field(default_factory=list)
    fvg_list: list[FVG] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_zones: list[LiquidityZone] = field(default_factory=list)
    trend: Literal["bullish", "bearish", "ranging"] = "ranging"
    last_bos: BOS | None = None
    last_choch: CHoCH | None = None


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SMCEngine:
    """
    Stateless SMC analysis engine.
    All methods accept a pandas DataFrame with columns:
    open, high, low, close (indexed by time).
    """

    SWING_LOOKBACK: int = 5   # candles each side to confirm swing

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> SMCAnalysis:
        if df is None or len(df) < self.SWING_LOOKBACK * 2 + 1:
            return SMCAnalysis()

        result = SMCAnalysis()
        result.swing_points = self._detect_swings(df)
        result.trend = self._determine_trend(result.swing_points)
        result.bos_list = self._detect_bos(df, result.swing_points)
        result.choch_list = self._detect_choch(df, result.swing_points, result.trend)
        result.fvg_list = self._detect_fvg(df)
        result.order_blocks = self._detect_order_blocks(df, result.bos_list)
        result.liquidity_zones = self._detect_liquidity(df, result.swing_points)

        result.last_bos = result.bos_list[-1] if result.bos_list else None
        result.last_choch = result.choch_list[-1] if result.choch_list else None
        return result

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def _detect_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        lb = self.SWING_LOOKBACK
        swings: list[SwingPoint] = []

        for i in range(lb, n - lb):
            # Swing High
            if highs[i] == max(highs[i - lb: i + lb + 1]):
                swings.append(SwingPoint(index=i, price=highs[i], kind="HIGH"))
            # Swing Low
            if lows[i] == min(lows[i - lb: i + lb + 1]):
                swings.append(SwingPoint(index=i, price=lows[i], kind="LOW"))

        swings.sort(key=lambda s: s.index)
        return self._label_swings(swings)

    def _label_swings(self, swings: list[SwingPoint]) -> list[SwingPoint]:
        highs = [s for s in swings if s.kind == "HIGH"]
        lows = [s for s in swings if s.kind == "LOW"]

        for i in range(1, len(highs)):
            highs[i].kind = "HH" if highs[i].price > highs[i - 1].price else "LH"

        for i in range(1, len(lows)):
            lows[i].kind = "HL" if lows[i].price > lows[i - 1].price else "LL"

        return sorted(swings, key=lambda s: s.index)

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def _determine_trend(self, swings: list[SwingPoint]) -> Literal["bullish", "bearish", "ranging"]:
        highs = [s for s in swings if s.kind in ("HH", "LH")]
        lows = [s for s in swings if s.kind in ("HL", "LL")]

        if not highs or not lows:
            return "ranging"

        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        hh_count = sum(1 for s in recent_highs if s.kind == "HH")
        hl_count = sum(1 for s in recent_lows if s.kind == "HL")
        lh_count = sum(1 for s in recent_highs if s.kind == "LH")
        ll_count = sum(1 for s in recent_lows if s.kind == "LL")

        if hh_count >= 2 and hl_count >= 2:
            return "bullish"
        if lh_count >= 2 and ll_count >= 2:
            return "bearish"
        return "ranging"

    # ------------------------------------------------------------------
    # Break of Structure
    # ------------------------------------------------------------------

    def _detect_bos(self, df: pd.DataFrame, swings: list[SwingPoint]) -> list[BOS]:
        closes = df["close"].values
        bos_list: list[BOS] = []

        swing_highs = [s for s in swings if s.kind in ("HH", "LH", "HIGH")]
        swing_lows = [s for s in swings if s.kind in ("HL", "LL", "LOW")]

        for i in range(1, len(closes)):
            # Bullish BOS: close breaks above a prior swing high
            for sh in swing_highs:
                if sh.index < i and closes[i] > sh.price:
                    if not any(b.index == i and b.direction == "bullish" for b in bos_list):
                        bos_list.append(BOS(
                            index=i,
                            price=sh.price,
                            direction="bullish",
                            broken_swing=sh,
                        ))
                        break

            # Bearish BOS: close breaks below a prior swing low
            for sl in swing_lows:
                if sl.index < i and closes[i] < sl.price:
                    if not any(b.index == i and b.direction == "bearish" for b in bos_list):
                        bos_list.append(BOS(
                            index=i,
                            price=sl.price,
                            direction="bearish",
                            broken_swing=sl,
                        ))
                        break

        return bos_list

    # ------------------------------------------------------------------
    # Change of Character
    # ------------------------------------------------------------------

    def _detect_choch(
        self,
        df: pd.DataFrame,
        swings: list[SwingPoint],
        trend: str,
    ) -> list[CHoCH]:
        closes = df["close"].values
        choch_list: list[CHoCH] = []

        if trend == "bullish":
            # CHoCH = close breaks below a HL (higher low) in uptrend
            targets = [s for s in swings if s.kind == "HL"]
            for i in range(1, len(closes)):
                for t in targets:
                    if t.index < i and closes[i] < t.price:
                        if not any(c.index == i for c in choch_list):
                            choch_list.append(CHoCH(index=i, price=t.price, direction="bearish"))
                        break

        elif trend == "bearish":
            # CHoCH = close breaks above a LH (lower high) in downtrend
            targets = [s for s in swings if s.kind == "LH"]
            for i in range(1, len(closes)):
                for t in targets:
                    if t.index < i and closes[i] > t.price:
                        if not any(c.index == i for c in choch_list):
                            choch_list.append(CHoCH(index=i, price=t.price, direction="bullish"))
                        break

        return choch_list

    # ------------------------------------------------------------------
    # Fair Value Gap
    # ------------------------------------------------------------------

    def _detect_fvg(self, df: pd.DataFrame) -> list[FVG]:
        fvg_list: list[FVG] = []
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        for i in range(1, n - 1):
            # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
            if lows[i + 1] > highs[i - 1]:
                fvg_list.append(FVG(
                    start_index=i - 1,
                    end_index=i + 1,
                    top=lows[i + 1],
                    bottom=highs[i - 1],
                    direction="bullish",
                ))

            # Bearish FVG: gap between candle[i-1] low and candle[i+1] high
            if highs[i + 1] < lows[i - 1]:
                fvg_list.append(FVG(
                    start_index=i - 1,
                    end_index=i + 1,
                    top=lows[i - 1],
                    bottom=highs[i + 1],
                    direction="bearish",
                ))

        # Mark filled FVGs
        closes = df["close"].values
        for fvg in fvg_list:
            for j in range(fvg.end_index + 1, n):
                if fvg.direction == "bullish" and closes[j] < fvg.bottom:
                    fvg.filled = True
                    break
                if fvg.direction == "bearish" and closes[j] > fvg.top:
                    fvg.filled = True
                    break

        return fvg_list

    # ------------------------------------------------------------------
    # Order Blocks
    # ------------------------------------------------------------------

    def _detect_order_blocks(self, df: pd.DataFrame, bos_list: list[BOS]) -> list[OrderBlock]:
        obs: list[OrderBlock] = []
        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        for bos in bos_list:
            # Look back for the last opposing candle before the BOS
            search_start = max(0, bos.index - 20)
            if bos.direction == "bullish":
                # Last bearish candle before bullish BOS
                for i in range(bos.index - 1, search_start, -1):
                    if closes[i] < opens[i]:  # bearish candle
                        obs.append(OrderBlock(
                            index=i,
                            top=highs[i],
                            bottom=lows[i],
                            direction="bullish",
                        ))
                        break
            else:
                # Last bullish candle before bearish BOS
                for i in range(bos.index - 1, search_start, -1):
                    if closes[i] > opens[i]:  # bullish candle
                        obs.append(OrderBlock(
                            index=i,
                            top=highs[i],
                            bottom=lows[i],
                            direction="bearish",
                        ))
                        break

        # Mark mitigated OBs (price returned into the zone)
        closes_arr = df["close"].values
        for ob in obs:
            for j in range(ob.index + 1, len(closes_arr)):
                if ob.bottom <= closes_arr[j] <= ob.top:
                    ob.mitigated = True
                    break

        return obs

    # ------------------------------------------------------------------
    # Liquidity Zones
    # ------------------------------------------------------------------

    def _detect_liquidity(self, df: pd.DataFrame, swings: list[SwingPoint]) -> list[LiquidityZone]:
        zones: list[LiquidityZone] = []
        # Buy-Side Liquidity = above swing highs (equal highs)
        # Sell-Side Liquidity = below swing lows (equal lows)
        highs = [s for s in swings if s.kind in ("HH", "LH", "HIGH")]
        lows = [s for s in swings if s.kind in ("HL", "LL", "LOW")]

        seen_prices: set[float] = set()
        for sh in highs[-10:]:
            rounded = round(sh.price, 1)
            if rounded not in seen_prices:
                zones.append(LiquidityZone(price=sh.price, kind="BSL", index=sh.index))
                seen_prices.add(rounded)

        for sl in lows[-10:]:
            rounded = round(sl.price, 1)
            if rounded not in seen_prices:
                zones.append(LiquidityZone(price=sl.price, kind="SSL", index=sl.index))
                seen_prices.add(rounded)

        return zones


# ---------------------------------------------------------------------------
# Convenience serializer
# ---------------------------------------------------------------------------

def analysis_to_dict(analysis: SMCAnalysis) -> dict:
    def swing_d(s: SwingPoint):
        return {"index": s.index, "price": s.price, "kind": s.kind}

    def bos_d(b: BOS):
        return {"index": b.index, "price": b.price, "direction": b.direction}

    def choch_d(c: CHoCH):
        return {"index": c.index, "price": c.price, "direction": c.direction}

    def fvg_d(f: FVG):
        return {
            "start": f.start_index, "end": f.end_index,
            "top": f.top, "bottom": f.bottom,
            "direction": f.direction, "filled": f.filled,
        }

    def ob_d(o: OrderBlock):
        return {
            "index": o.index, "top": o.top, "bottom": o.bottom,
            "direction": o.direction, "mitigated": o.mitigated,
        }

    def liq_d(lz: LiquidityZone):
        return {"price": lz.price, "kind": lz.kind, "index": lz.index}

    return {
        "trend": analysis.trend,
        "swing_points": [swing_d(s) for s in analysis.swing_points[-20:]],
        "bos": [bos_d(b) for b in analysis.bos_list[-5:]],
        "choch": [choch_d(c) for c in analysis.choch_list[-5:]],
        "fvg": [fvg_d(f) for f in analysis.fvg_list if not f.filled][-10:],
        "order_blocks": [ob_d(o) for o in analysis.order_blocks if not o.mitigated][-5:],
        "liquidity_zones": [liq_d(lz) for lz in analysis.liquidity_zones],
        "last_bos": bos_d(analysis.last_bos) if analysis.last_bos else None,
        "last_choch": choch_d(analysis.last_choch) if analysis.last_choch else None,
    }
