"""
SMC Adapter — wraps `smartmoneyconcepts` v0.0.27
(github.com/joshyattridge/smart-money-concepts)

Verified column names from the installed library:
  swing_highs_lows → ['HighLow', 'Level']
  bos_choch        → ['BOS', 'CHOCH', 'Level', 'BrokenIndex']
  fvg              → ['FVG', 'Top', 'Bottom', 'MitigatedIndex']
  ob               → ['OB', 'Top', 'Bottom', 'OBVolume', 'MitigatedIndex', 'Percentage']
  liquidity        → ['Liquidity', 'Level', 'End', 'Swept']

Drop-in replacement for SMCEngine — same interface, better logic.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from app.smc_engine import (
    SMCAnalysis,
    SwingPoint,
    BOS,
    CHoCH,
    FVG,
    OrderBlock,
    LiquidityZone,
)

logger = logging.getLogger(__name__)

try:
    # Fix Windows cp1252 encoding crash caused by the star emoji
    # in the library's __init__.py print statement
    import sys as _sys
    import io as _io
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        _sys.stdout = _io.TextIOWrapper(
            _sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    from smartmoneyconcepts import smc as _lib
    SMC_LIB_AVAILABLE = True
except ImportError:
    _lib = None  # type: ignore
    SMC_LIB_AVAILABLE = False
    logger.warning("smartmoneyconcepts not installed — falling back to built-in engine.")
except Exception as _e:
    _lib = None  # type: ignore
    SMC_LIB_AVAILABLE = False
    logger.warning("smartmoneyconcepts failed to load (%s) — falling back.", _e)


class SMCEngineAdapter:
    """
    Wraps joshyattridge/smart-money-concepts and returns SMCAnalysis.
    Falls back to built-in SMCEngine if library is unavailable.
    """

    SWING_LENGTH: int = 5

    def analyze(self, df: pd.DataFrame) -> SMCAnalysis:
        if df is None or len(df) < self.SWING_LENGTH * 2 + 2:
            return SMCAnalysis()

        if not SMC_LIB_AVAILABLE:
            from app.smc_engine import SMCEngine
            return SMCEngine().analyze(df)

        df = self._normalize(df)
        result = SMCAnalysis()

        swing_df = self._safe_call(
            _lib.swing_highs_lows, df, swing_length=self.SWING_LENGTH
        )

        result.swing_points = self._parse_swings(df, swing_df)
        result.trend = self._determine_trend(result.swing_points)
        result.bos_list, result.choch_list = self._parse_bos_choch(df, swing_df)
        result.fvg_list = self._parse_fvg(df)
        result.order_blocks = self._parse_ob(df, swing_df)
        result.liquidity_zones = self._parse_liquidity(df, swing_df)

        result.last_bos = result.bos_list[-1] if result.bos_list else None
        result.last_choch = result.choch_list[-1] if result.choch_list else None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.set_index("time")
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _safe_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.error("%s failed: %s", fn.__name__, exc)
            return pd.DataFrame()

    def _idx(self, df: pd.DataFrame, label) -> int:
        try:
            return df.index.get_loc(label)
        except Exception:
            return 0

    def _is_mitigated(self, val) -> bool:
        """
        MitigatedIndex = 0.0  → NOT mitigated (library uses 0 as sentinel)
        MitigatedIndex = NaN  → NOT mitigated
        MitigatedIndex > 0    → mitigated at that candle index
        """
        if val is None:
            return False
        try:
            f = float(val)
            return not (pd.isna(f) or f == 0.0)
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Swing points  →  ['HighLow', 'Level']
    # ------------------------------------------------------------------

    def _parse_swings(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> list[SwingPoint]:
        if swing_df is None or swing_df.empty:
            return []
        swings: list[SwingPoint] = []
        for ts, row in swing_df.dropna(subset=["HighLow"]).iterrows():
            val = row["HighLow"]
            idx = self._idx(df, ts)
            kind: Literal["HIGH", "LOW"] = "HIGH" if val == 1 else "LOW"
            price = float(row["Level"])
            swings.append(SwingPoint(index=idx, price=price, kind=kind))
        swings.sort(key=lambda s: s.index)
        return self._label_swings(swings)

    def _label_swings(self, swings: list[SwingPoint]) -> list[SwingPoint]:
        highs = [s for s in swings if s.kind in ("HIGH", "HH", "LH")]
        lows  = [s for s in swings if s.kind in ("LOW",  "HL", "LL")]
        for i in range(1, len(highs)):
            highs[i].kind = "HH" if highs[i].price > highs[i - 1].price else "LH"
        for i in range(1, len(lows)):
            lows[i].kind  = "HL" if lows[i].price  > lows[i - 1].price  else "LL"
        return sorted(swings, key=lambda s: s.index)

    def _determine_trend(
        self, swings: list[SwingPoint]
    ) -> Literal["bullish", "bearish", "ranging"]:
        highs = [s for s in swings if s.kind in ("HH", "LH")]
        lows  = [s for s in swings if s.kind in ("HL", "LL")]
        if not highs or not lows:
            return "ranging"
        hh = sum(1 for s in highs[-3:] if s.kind == "HH")
        hl = sum(1 for s in lows[-3:]  if s.kind == "HL")
        lh = sum(1 for s in highs[-3:] if s.kind == "LH")
        ll = sum(1 for s in lows[-3:]  if s.kind == "LL")
        if hh >= 2 and hl >= 2:
            return "bullish"
        if lh >= 2 and ll >= 2:
            return "bearish"
        return "ranging"

    # ------------------------------------------------------------------
    # BOS / CHoCH  →  ['BOS', 'CHOCH', 'Level', 'BrokenIndex']
    # ------------------------------------------------------------------

    def _parse_bos_choch(
        self, df: pd.DataFrame, swing_df: pd.DataFrame
    ) -> tuple[list[BOS], list[CHoCH]]:
        bos_list:   list[BOS]   = []
        choch_list: list[CHoCH] = []

        bc_df = self._safe_call(_lib.bos_choch, df, swing_df, close_break=True)
        if bc_df is None or bc_df.empty:
            return bos_list, choch_list

        dummy = SwingPoint(index=0, price=0.0, kind="HIGH")

        for ts, row in bc_df.iterrows():
            idx   = self._idx(df, ts)
            level = float(row.get("Level") or 0)

            bos_val = row.get("BOS")
            if not pd.isna(bos_val) and bos_val != 0:
                direction: Literal["bullish", "bearish"] = (
                    "bullish" if bos_val == 1 else "bearish"
                )
                bos_list.append(BOS(
                    index=idx, price=level,
                    direction=direction, broken_swing=dummy,
                ))

            choch_val = row.get("CHOCH")   # ← correct column name
            if not pd.isna(choch_val) and choch_val != 0:
                direction = "bullish" if choch_val == 1 else "bearish"
                choch_list.append(CHoCH(index=idx, price=level, direction=direction))

        return bos_list, choch_list

    # ------------------------------------------------------------------
    # FVG  →  ['FVG', 'Top', 'Bottom', 'MitigatedIndex']
    # ------------------------------------------------------------------

    def _parse_fvg(self, df: pd.DataFrame) -> list[FVG]:
        fvg_df = self._safe_call(_lib.fvg, df, join_consecutive=False)
        if fvg_df is None or fvg_df.empty:
            return []

        result: list[FVG] = []
        for ts, row in fvg_df.dropna(subset=["FVG"]).iterrows():
            val = row["FVG"]
            if val == 0:
                continue
            idx = self._idx(df, ts)
            direction: Literal["bullish", "bearish"] = (
                "bullish" if val == 1 else "bearish"
            )
            mit = row.get("MitigatedIndex")
            filled = self._is_mitigated(mit)
            result.append(FVG(
                start_index=max(0, idx - 1),
                end_index=idx + 1,
                top=float(row.get("Top") or 0),
                bottom=float(row.get("Bottom") or 0),
                direction=direction,
                filled=filled,
            ))
        return result

    # ------------------------------------------------------------------
    # Order Blocks  →  ['OB', 'Top', 'Bottom', 'OBVolume', 'MitigatedIndex', 'Percentage']
    # ------------------------------------------------------------------

    def _parse_ob(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> list[OrderBlock]:
        ob_df = self._safe_call(_lib.ob, df, swing_df)
        if ob_df is None or ob_df.empty:
            return []

        result: list[OrderBlock] = []
        for ts, row in ob_df.dropna(subset=["OB"]).iterrows():
            val = row["OB"]
            if val == 0:
                continue
            idx = self._idx(df, ts)
            direction: Literal["bullish", "bearish"] = (
                "bullish" if val == 1 else "bearish"
            )
            mit = row.get("MitigatedIndex")
            mitigated = self._is_mitigated(mit)
            result.append(OrderBlock(
                index=idx,
                top=float(row.get("Top") or 0),
                bottom=float(row.get("Bottom") or 0),
                direction=direction,
                mitigated=mitigated,
            ))
        return result

    # ------------------------------------------------------------------
    # Liquidity  →  ['Liquidity', 'Level', 'End', 'Swept']
    # ------------------------------------------------------------------

    def _parse_liquidity(
        self, df: pd.DataFrame, swing_df: pd.DataFrame
    ) -> list[LiquidityZone]:
        liq_df = self._safe_call(_lib.liquidity, df, swing_df, range_percent=0.01)
        if liq_df is None or liq_df.empty:
            return []

        result: list[LiquidityZone] = []
        for ts, row in liq_df.dropna(subset=["Liquidity"]).iterrows():
            val = row["Liquidity"]
            if val == 0:
                continue
            swept = bool(row.get("Swept", False))
            if swept:
                continue   # skip already-swept zones
            idx  = self._idx(df, ts)
            kind: Literal["BSL", "SSL"] = "BSL" if val == 1 else "SSL"
            result.append(LiquidityZone(
                price=float(row.get("Level") or 0),
                kind=kind,
                index=idx,
            ))
        return result
