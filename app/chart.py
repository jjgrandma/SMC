"""
Chart Generator — creates professional candlestick charts for Telegram.

Generates:
  - Multi-timeframe overview (D1 + H4 + H1)
  - Single timeframe with SMC overlays (OBs, FVGs, key levels)
  - Signal chart (entry, SL, TP marked)

Returns: BytesIO image buffer ready to send via Telegram
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Use non-interactive backend — no display needed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import mplfinance as mpf


# ---------------------------------------------------------------------------
# Color scheme — dark professional theme
# ---------------------------------------------------------------------------

COLORS = {
    "bg":         "#0d1117",
    "bg2":        "#161b22",
    "grid":       "#21262d",
    "text":       "#e6edf3",
    "text_dim":   "#8b949e",
    "bull":       "#26a641",
    "bear":       "#f85149",
    "bull_body":  "#26a641",
    "bear_body":  "#f85149",
    "wick":       "#8b949e",
    "ob_bull":    "#1f6feb",
    "ob_bear":    "#da3633",
    "fvg_bull":   "#238636",
    "fvg_bear":   "#b91c1c",
    "entry":      "#f0e68c",
    "sl":         "#ff4444",
    "tp":         "#44ff88",
    "eq":         "#8b949e",
    "price_line": "#58a6ff",
    "ma50":       "#f0883e",
    "ma200":      "#bc8cff",
}

MC = mpf.make_marketcolors(
    up=COLORS["bull"],
    down=COLORS["bear"],
    edge={"up": COLORS["bull"], "down": COLORS["bear"]},
    wick={"up": COLORS["wick"], "down": COLORS["wick"]},
    volume={"up": COLORS["bull"], "down": COLORS["bear"]},
)

STYLE = mpf.make_mpf_style(
    marketcolors=MC,
    facecolor=COLORS["bg"],
    edgecolor=COLORS["bg2"],
    figcolor=COLORS["bg"],
    gridcolor=COLORS["grid"],
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
    rc={
        "axes.labelcolor":  COLORS["text"],
        "axes.titlecolor":  COLORS["text"],
        "xtick.color":      COLORS["text_dim"],
        "ytick.color":      COLORS["text_dim"],
        "text.color":       COLORS["text"],
        "font.size":        9,
        "font.family":      "monospace",
    },
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prep_df(df: pd.DataFrame, n_candles: int = 80) -> pd.DataFrame:
    """Prepare DataFrame for mplfinance — needs DatetimeIndex + OHLCV columns."""
    df = df.copy().tail(n_candles)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "Date"
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Chart 1 — Multi-timeframe overview (3 panels)
# ---------------------------------------------------------------------------

def chart_mtf(
    df_d1: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    symbol: str,
    current_price: float,
    mtf_data: dict | None = None,
) -> io.BytesIO:
    """
    3-panel chart: D1 (top) | H4 (middle) | H1 (bottom)
    Shows recent price action across timeframes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), facecolor=COLORS["bg"])
    fig.subplots_adjust(hspace=0.35, top=0.93, bottom=0.05, left=0.02, right=0.95)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"{symbol}  •  Multi-Timeframe Overview  •  {now_str}\n"
        f"Price: {current_price}",
        color=COLORS["text"], fontsize=11, fontweight="bold", y=0.97,
    )

    configs = [
        (df_d1, "D1 — Daily",   50, axes[0]),
        (df_h4, "H4 — 4 Hour",  60, axes[1]),
        (df_h1, "H1 — 1 Hour",  80, axes[2]),
    ]

    for df_raw, title, n, ax in configs:
        if df_raw is None or len(df_raw) < 5:
            ax.text(0.5, 0.5, f"No data for {title}", transform=ax.transAxes,
                    ha="center", color=COLORS["text_dim"])
            ax.set_facecolor(COLORS["bg"])
            continue

        df = _prep_df(df_raw, n)
        try:
            mpf.plot(
                df, type="candle", style=STYLE,
                ax=ax, volume=False, show_nontrading=False,
            )
        except Exception as e:
            logger.warning("Chart panel failed for %s: %s", title, e)
            continue

        # Current price line
        ax.axhline(current_price, color=COLORS["price_line"],
                   linewidth=0.8, linestyle="--", alpha=0.7)

        # Title
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
        chg = last_close - prev_close
        chg_pct = (chg / prev_close * 100) if prev_close else 0
        chg_color = COLORS["bull"] if chg >= 0 else COLORS["bear"]
        chg_sign  = "+" if chg >= 0 else ""

        ax.set_title(
            f"{title}  |  {last_close:.2f}  "
            f"({chg_sign}{chg:.2f} / {chg_sign}{chg_pct:.2f}%)",
            color=COLORS["text"], fontsize=9, pad=4, loc="left",
        )
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text_dim"], labelsize=7)

        # Add SMC bias label if available
        if mtf_data:
            tf_key = title.split(" ")[0]
            tfs = mtf_data.get("timeframes", {})
            if tf_key in tfs:
                trend = tfs[tf_key].get("trend", "")
                pd_zone = tfs[tf_key].get("premium_discount", "")
                label_color = COLORS["bull"] if trend == "bullish" else COLORS["bear"] if trend == "bearish" else COLORS["text_dim"]
                ax.text(
                    0.01, 0.97,
                    f"{trend.upper()} | {pd_zone.upper()}",
                    transform=ax.transAxes, va="top", ha="left",
                    color=label_color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["bg2"], alpha=0.8),
                )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Chart 2 — Signal chart (entry, SL, TP zones)
# ---------------------------------------------------------------------------

def chart_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal: dict,
    smc_data: dict | None = None,
) -> io.BytesIO:
    """
    Single timeframe chart with signal levels marked:
    - Entry zone (yellow)
    - Stop Loss (red)
    - Take Profit (green)
    - Key SMC levels
    """
    action     = signal.get("action", "")
    entry      = float(signal.get("entry") or 0)
    sl         = float(signal.get("stop_loss") or 0)
    tp         = float(signal.get("take_profit") or 0)
    tp2        = float(signal.get("tp2") or 0)
    confidence = signal.get("confidence", "")
    price_now  = signal.get("current_price", {}).get("mid", entry)

    df_plot = _prep_df(df, 60)
    if len(df_plot) < 5:
        return _empty_chart(f"Insufficient data for {symbol} {timeframe}")

    # Build add-plots for horizontal lines
    hlines_prices = []
    hlines_colors = []

    if entry:
        hlines_prices.append(entry)
        hlines_colors.append(COLORS["entry"])
    if sl:
        hlines_prices.append(sl)
        hlines_colors.append(COLORS["sl"])
    if tp:
        hlines_prices.append(tp)
        hlines_colors.append(COLORS["tp"])
    if tp2:
        hlines_prices.append(tp2)
        hlines_colors.append(COLORS["tp"])
    if price_now:
        hlines_prices.append(float(price_now))
        hlines_colors.append(COLORS["price_line"])

    fig, axes = mpf.plot(
        df_plot, type="candle", style=STYLE,
        figsize=(12, 7),
        hlines=dict(hlines=hlines_prices, colors=hlines_colors,
                    linewidths=[1.2] * len(hlines_prices),
                    linestyle=["--"] * len(hlines_prices)),
        returnfig=True,
        volume=False,
        show_nontrading=False,
    )

    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])

    # Shade SL-Entry zone (risk zone)
    if entry and sl:
        ax.axhspan(
            min(entry, sl), max(entry, sl),
            alpha=0.12, color=COLORS["sl"], zorder=0,
        )

    # Shade Entry-TP zone (reward zone)
    if entry and tp:
        ax.axhspan(
            min(entry, tp), max(entry, tp),
            alpha=0.10, color=COLORS["tp"], zorder=0,
        )

    # Labels on the right side
    y_range = df_plot["high"].max() - df_plot["low"].min()
    offset  = y_range * 0.003

    label_items = []
    if entry:
        label_items.append((entry + offset, f"ENTRY  {entry:.2f}", COLORS["entry"]))
    if sl:
        label_items.append((sl + offset, f"SL  {sl:.2f}", COLORS["sl"]))
    if tp:
        label_items.append((tp + offset, f"TP1  {tp:.2f}", COLORS["tp"]))
    if tp2:
        label_items.append((tp2 + offset, f"TP2  {tp2:.2f}", COLORS["tp"]))
    if price_now:
        label_items.append((float(price_now) + offset, f"NOW  {float(price_now):.2f}", COLORS["price_line"]))

    for y_pos, label, color in label_items:
        ax.text(
            len(df_plot) - 1, y_pos, label,
            color=color, fontsize=8, fontweight="bold",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["bg2"], alpha=0.85),
        )

    # Title
    action_sym = "▲ BUY" if action == "BUY" else "▼ SELL" if action == "SELL" else "— NO TRADE"
    action_col = COLORS["bull"] if action == "BUY" else COLORS["bear"] if action == "SELL" else COLORS["text_dim"]
    rr = signal.get("rr_ratio", "?")
    conf_str = f"  [{confidence}]" if confidence else ""

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"{symbol} {timeframe}  •  {action_sym}{conf_str}  •  R:R 1:{rr}  •  {now_str}",
        color=action_col, fontsize=11, fontweight="bold", y=0.99,
    )
    fig.patch.set_facecolor(COLORS["bg"])

    # Confluences annotation
    confluences = signal.get("confluences", [])
    if confluences:
        conf_text = "\n".join([f"• {c[:50]}" for c in confluences[:4]])
        ax.text(
            0.01, 0.02, conf_text,
            transform=ax.transAxes, va="bottom", ha="left",
            color=COLORS["text_dim"], fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg2"], alpha=0.85),
        )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Chart 3 — Briefing chart (D1 + H4 with annotations)
# ---------------------------------------------------------------------------

def chart_briefing(
    df_d1: pd.DataFrame,
    df_h4: pd.DataFrame,
    symbol: str,
    current_price: float,
    briefing: dict,
) -> io.BytesIO:
    """
    2-panel briefing chart with key level annotations.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor=COLORS["bg"])
    fig.subplots_adjust(hspace=0.3, top=0.93, bottom=0.05, left=0.02, right=0.95)

    session   = briefing.get("session", "")
    next_sess = briefing.get("next_session", "")
    w_bias    = briefing.get("weekly_bias", "ranging").upper()
    d_bias    = briefing.get("daily_bias", "ranging").upper()
    now_str   = briefing.get("generated_at", "")

    bias_color = COLORS["bull"] if w_bias == "BULLISH" else COLORS["bear"] if w_bias == "BEARISH" else COLORS["text_dim"]

    fig.suptitle(
        f"{symbol}  •  {session} Session  •  {now_str}\n"
        f"Weekly: {w_bias}  |  Daily: {d_bias}  |  Price: {current_price}",
        color=bias_color, fontsize=11, fontweight="bold", y=0.97,
    )

    configs = [
        (df_d1, "D1 — Daily Context", 40, axes[0]),
        (df_h4, "H4 — Session Detail", 60, axes[1]),
    ]

    kl = briefing.get("key_levels", {})

    for df_raw, title, n, ax in configs:
        if df_raw is None or len(df_raw) < 5:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", color=COLORS["text_dim"])
            ax.set_facecolor(COLORS["bg"])
            continue

        df = _prep_df(df_raw, n)
        try:
            mpf.plot(df, type="candle", style=STYLE, ax=ax,
                     volume=False, show_nontrading=False)
        except Exception as e:
            logger.warning("Briefing chart panel failed: %s", e)
            continue

        ax.set_facecolor(COLORS["bg"])
        ax.set_title(title, color=COLORS["text"], fontsize=9, pad=4, loc="left")

        # Current price
        ax.axhline(current_price, color=COLORS["price_line"],
                   linewidth=1.0, linestyle="--", alpha=0.8,
                   label=f"Now: {current_price}")

        # Key levels from briefing
        level_map = [
            ("today_high",     COLORS["bull"],    "T.High"),
            ("today_low",      COLORS["bear"],    "T.Low"),
            ("yesterday_high", COLORS["bull"],    "Y.High"),
            ("yesterday_low",  COLORS["bear"],    "Y.Low"),
            ("weekly_high",    "#f0883e",         "W.High"),
            ("weekly_low",     "#f0883e",         "W.Low"),
            ("equilibrium",    COLORS["eq"],      "EQ 50%"),
        ]

        y_range = df["high"].max() - df["low"].min()
        offset  = y_range * 0.003

        for key, color, label in level_map:
            val = kl.get(key)
            if val and float(val) > 0:
                fval = float(val)
                if df["low"].min() * 0.99 <= fval <= df["high"].max() * 1.01:
                    ax.axhline(fval, color=color, linewidth=0.7,
                               linestyle=":", alpha=0.7)
                    ax.text(
                        len(df) * 0.02, fval + offset,
                        f"{label} {fval:.1f}",
                        color=color, fontsize=7, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor=COLORS["bg2"], alpha=0.7),
                    )

        # Resistance zones
        for res in (kl.get("major_resistance") or []):
            if res and float(res) > 0:
                fres = float(res)
                if df["low"].min() * 0.99 <= fres <= df["high"].max() * 1.01:
                    ax.axhspan(fres - y_range * 0.003, fres + y_range * 0.003,
                               alpha=0.15, color=COLORS["bear"])

        # Support zones
        for sup in (kl.get("major_support") or []):
            if sup and float(sup) > 0:
                fsup = float(sup)
                if df["low"].min() * 0.99 <= fsup <= df["high"].max() * 1.01:
                    ax.axhspan(fsup - y_range * 0.003, fsup + y_range * 0.003,
                               alpha=0.15, color=COLORS["bull"])

        ax.tick_params(colors=COLORS["text_dim"], labelsize=7)

    # Next session annotation
    axes[1].text(
        0.99, 0.03,
        f"Next: {next_sess}",
        transform=axes[1].transAxes, va="bottom", ha="right",
        color=COLORS["text_dim"], fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg2"], alpha=0.85),
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_chart(message: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLORS["bg"])
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", color=COLORS["text_dim"], fontsize=12)
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf
