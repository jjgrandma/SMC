"""
XAUUSD Live Analysis Script
============================
Standalone script — runs independently from the trading bot.
Does NOT affect any bot functionality.

Features:
  - Live XAUUSD prices via yfinance (or MT5 if connected)
  - RSI(14), SMA(20), SMA(50)
  - Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)
  - Smart Money order block detection
  - Liquidity grab detection
  - Interactive Plotly chart (candlestick + indicators + signals)
  - BUY/SELL alerts based on Fib bounce + RSI + order block confluence
  - Saves chart as HTML (interactive) + PNG + CSV
  - Runs continuously, updates every 60 seconds

Usage:
  python xauusd_analysis.py
  python xauusd_analysis.py --timeframe H1 --candles 100
  python xauusd_analysis.py --mt5  (use MT5 if available)

Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL      = os.environ.get("SYMBOL", "XAUUSDm")
TIMEFRAME   = "H1"
N_CANDLES   = 100
UPDATE_SECS = 60
OUTPUT_DIR  = "charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

def fetch_data(symbol: str, timeframe: str, n: int, use_mt5: bool = False) -> pd.DataFrame:
    """Fetch OHLCV data. Tries MT5 first if requested, falls back to yfinance."""

    if use_mt5:
        try:
            import MetaTrader5 as mt5
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            if mt5.initialize():
                rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), 0, n)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df.set_index("time", inplace=True)
                    df.rename(columns={"tick_volume": "volume"}, inplace=True)
                    print(f"[MT5] {len(df)} candles loaded for {symbol} {timeframe}")
                    return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            print(f"[MT5] Failed: {e} — falling back to yfinance")

    # yfinance fallback
    import yfinance as yf
    sym_map = {"XAUUSDm": "GC=F", "XAUUSD": "GC=F"}
    yf_sym  = sym_map.get(symbol, symbol)
    tf_map  = {
        "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
        "H1": "1h", "H4": "4h", "D1": "1d",
    }
    period_map = {
        "M1": "1d", "M5": "5d", "M15": "5d", "M30": "10d",
        "H1": "60d", "H4": "60d", "D1": "1y",
    }
    df = yf.download(
        yf_sym,
        period=period_map.get(timeframe, "60d"),
        interval=tf_map.get(timeframe, "1h"),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna().tail(n)
    print(f"[yfinance] {len(df)} candles loaded for {yf_sym} {timeframe}")
    return df


def get_live_price(symbol: str) -> dict:
    """Get current bid/ask."""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {"bid": tick.bid, "ask": tick.ask, "mid": (tick.bid + tick.ask) / 2, "source": "MT5"}
    except Exception:
        pass
    try:
        import yfinance as yf
        sym_map = {"XAUUSDm": "GC=F", "XAUUSD": "GC=F"}
        data = yf.download(sym_map.get(symbol, symbol), period="1d", interval="1m", progress=False, auto_adjust=True)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0].lower() for c in data.columns]
            mid = float(data["close"].iloc[-1])
            return {"bid": mid - 0.15, "ask": mid + 0.15, "mid": mid, "source": "yfinance"}
    except Exception:
        pass
    return {"bid": 0, "ask": 0, "mid": 0, "source": "error"}


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, SMA, Bollinger Bands."""
    import ta

    df = df.copy()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # SMAs
    df["sma20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    return df


def fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Calculate Fibonacci retracement levels from recent swing high/low."""
    recent = df.tail(lookback)
    high   = float(recent["high"].max())
    low    = float(recent["low"].min())
    diff   = high - low

    levels = {
        "high":  high,
        "low":   low,
        "0.0":   high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5":   high - 0.5   * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0":   low,
    }
    return levels


def detect_order_blocks(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """
    Detect Smart Money order blocks.
    An order block is the last opposing candle before a strong move.
    Bullish OB: last bearish candle before a strong bullish move.
    Bearish OB: last bullish candle before a strong bearish move.
    """
    obs = []
    data = df.tail(lookback).reset_index()
    closes = data["close"].values
    opens  = data["open"].values
    highs  = data["high"].values
    lows   = data["low"].values
    atr    = float(df["atr"].iloc[-1]) if "atr" in df.columns else 1.0

    for i in range(2, len(data) - 1):
        move = abs(closes[i] - opens[i])
        # Strong move = body > 1.5x ATR
        if move < atr * 1.5:
            continue

        if closes[i] > opens[i]:  # bullish strong candle
            # Look back for last bearish candle = bullish OB
            for j in range(i - 1, max(0, i - 5), -1):
                if closes[j] < opens[j]:
                    obs.append({
                        "type":    "bullish",
                        "top":     highs[j],
                        "bottom":  lows[j],
                        "index":   j,
                        "time":    data.iloc[j].get("time", data.index[j]),
                    })
                    break
        else:  # bearish strong candle
            for j in range(i - 1, max(0, i - 5), -1):
                if closes[j] > opens[j]:
                    obs.append({
                        "type":    "bearish",
                        "top":     highs[j],
                        "bottom":  lows[j],
                        "index":   j,
                        "time":    data.iloc[j].get("time", data.index[j]),
                    })
                    break

    return obs


def detect_liquidity_grabs(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """
    Detect liquidity grabs (stop hunts).
    Price wicks above/below a swing high/low then reverses.
    """
    grabs = []
    data  = df.tail(lookback)
    highs = data["high"].values
    lows  = data["low"].values
    closes = data["close"].values

    for i in range(2, len(data) - 1):
        prev_high = max(highs[:i])
        prev_low  = min(lows[:i])

        # Bullish grab: wick below prev low, close above it
        if lows[i] < prev_low and closes[i] > prev_low:
            grabs.append({
                "type":  "bullish_grab",
                "price": lows[i],
                "index": i,
                "time":  data.index[i],
            })

        # Bearish grab: wick above prev high, close below it
        if highs[i] > prev_high and closes[i] < prev_high:
            grabs.append({
                "type":  "bearish_grab",
                "price": highs[i],
                "index": i,
                "time":  data.index[i],
            })

    return grabs


def generate_signals(df: pd.DataFrame, fib: dict, obs: list, grabs: list) -> list[dict]:
    """
    Generate BUY/SELL signals based on confluence:
    - Fib level bounce + RSI condition + Order Block proximity
    """
    signals = []
    if len(df) < 3:
        return signals

    last      = df.iloc[-1]
    prev      = df.iloc[-2]
    price     = float(last["close"])
    rsi       = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50
    atr       = float(last["atr"]) if not pd.isna(last["atr"]) else 1.0
    tolerance = atr * 0.5

    # Check each Fib level
    for level_name, level_price in fib.items():
        if level_name in ("high", "low"):
            continue
        if abs(price - level_price) > tolerance:
            continue

        # BUY signal: price at Fib support + RSI oversold + near bullish OB
        if price <= level_price + tolerance and rsi < 45:
            ob_nearby = any(
                ob["type"] == "bullish" and ob["bottom"] <= price <= ob["top"] + atr
                for ob in obs
            )
            grab_nearby = any(g["type"] == "bullish_grab" for g in grabs[-3:])
            confidence = "HIGH" if (ob_nearby and grab_nearby) else "MEDIUM" if ob_nearby else "LOW"
            signals.append({
                "type":       "BUY",
                "price":      price,
                "fib_level":  level_name,
                "rsi":        round(rsi, 1),
                "confidence": confidence,
                "reason":     f"Fib {level_name} bounce | RSI {rsi:.0f} | {'OB+Grab' if ob_nearby and grab_nearby else 'OB' if ob_nearby else 'Fib only'}",
                "sl":         round(level_price - atr * 1.5, 2),
                "tp":         round(price + atr * 3, 2),
            })

        # SELL signal: price at Fib resistance + RSI overbought + near bearish OB
        if price >= level_price - tolerance and rsi > 55:
            ob_nearby = any(
                ob["type"] == "bearish" and ob["bottom"] - atr <= price <= ob["top"]
                for ob in obs
            )
            grab_nearby = any(g["type"] == "bearish_grab" for g in grabs[-3:])
            confidence = "HIGH" if (ob_nearby and grab_nearby) else "MEDIUM" if ob_nearby else "LOW"
            signals.append({
                "type":       "SELL",
                "price":      price,
                "fib_level":  level_name,
                "rsi":        round(rsi, 1),
                "confidence": confidence,
                "reason":     f"Fib {level_name} resistance | RSI {rsi:.0f} | {'OB+Grab' if ob_nearby and grab_nearby else 'OB' if ob_nearby else 'Fib only'}",
                "sl":         round(level_price + atr * 1.5, 2),
                "tp":         round(price - atr * 3, 2),
            })

    return signals


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fib: dict,
    obs: list,
    grabs: list,
    signals: list,
    live_price: dict,
) -> go.Figure:
    """Build interactive Plotly chart with all overlays."""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[
            f"{symbol} {timeframe} — SMC Analysis",
            "RSI (14)",
            "MACD",
        ],
    )

    # ── Candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#26a641",
        decreasing_line_color="#f85149",
        increasing_fillcolor="#26a641",
        decreasing_fillcolor="#f85149",
    ), row=1, col=1)

    # ── SMAs ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sma20"],
        name="SMA 20", line=dict(color="#f0883e", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sma50"],
        name="SMA 50", line=dict(color="#bc8cff", width=1.2),
    ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"],
        name="BB Upper", line=dict(color="#58a6ff", width=0.8, dash="dot"),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"],
        name="BB Lower", line=dict(color="#58a6ff", width=0.8, dash="dot"),
        fill="tonexty", fillcolor="rgba(88,166,255,0.05)",
        showlegend=False,
    ), row=1, col=1)

    # ── Fibonacci levels ──────────────────────────────────────────────
    fib_colors = {
        "0.236": "#ffd700", "0.382": "#ffa500",
        "0.5":   "#ff6b6b", "0.618": "#ff4444",
        "0.786": "#cc0000",
    }
    for level_name, level_price in fib.items():
        if level_name in ("high", "low"):
            continue
        color = fib_colors.get(level_name, "#888888")
        fig.add_hline(
            y=level_price, line_dash="dash",
            line_color=color, line_width=0.8,
            annotation_text=f"Fib {level_name} ({level_price:.2f})",
            annotation_position="right",
            annotation_font_color=color,
            annotation_font_size=9,
            row=1, col=1,
        )

    # ── Order Blocks ──────────────────────────────────────────────────
    for ob in obs[-5:]:
        color = "rgba(38,166,65,0.15)" if ob["type"] == "bullish" else "rgba(248,81,73,0.15)"
        border = "#26a641" if ob["type"] == "bullish" else "#f85149"
        fig.add_hrect(
            y0=ob["bottom"], y1=ob["top"],
            fillcolor=color,
            line_color=border, line_width=0.5,
            annotation_text=f"{'Bull' if ob['type']=='bullish' else 'Bear'} OB",
            annotation_font_size=8,
            annotation_font_color=border,
            row=1, col=1,
        )

    # ── Liquidity grabs ───────────────────────────────────────────────
    for grab in grabs[-5:]:
        color = "#26a641" if "bullish" in grab["type"] else "#f85149"
        symbol_marker = "triangle-up" if "bullish" in grab["type"] else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[grab["time"]],
            y=[grab["price"]],
            mode="markers",
            marker=dict(symbol=symbol_marker, size=10, color=color),
            name=f"Liq Grab",
            showlegend=False,
        ), row=1, col=1)

    # ── Signals ───────────────────────────────────────────────────────
    for sig in signals:
        color  = "#26a641" if sig["type"] == "BUY" else "#f85149"
        marker = "triangle-up" if sig["type"] == "BUY" else "triangle-down"
        size   = {"HIGH": 18, "MEDIUM": 14, "LOW": 10}.get(sig["confidence"], 12)
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[sig["price"]],
            mode="markers+text",
            marker=dict(symbol=marker, size=size, color=color,
                        line=dict(color="white", width=1)),
            text=[f"{sig['type']} {sig['confidence']}"],
            textposition="top center" if sig["type"] == "BUY" else "bottom center",
            textfont=dict(color=color, size=9),
            name=f"{sig['type']} Signal",
            showlegend=False,
        ), row=1, col=1)

    # ── Live price line ───────────────────────────────────────────────
    if live_price.get("mid"):
        fig.add_hline(
            y=live_price["mid"],
            line_color="#58a6ff", line_width=1.5, line_dash="solid",
            annotation_text=f"  Live: {live_price['mid']:.2f}",
            annotation_font_color="#58a6ff",
            annotation_font_size=10,
            row=1, col=1,
        )

    # ── RSI ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        name="RSI", line=dict(color="#f0883e", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a641", line_width=0.8, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="#888888", line_width=0.5, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)", row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,65,0.05)", row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────
    colors = ["#26a641" if v >= 0 else "#f85149" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["macd_hist"],
        name="MACD Hist", marker_color=colors, opacity=0.7,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        name="MACD", line=dict(color="#58a6ff", width=1.2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"],
        name="Signal", line=dict(color="#f0883e", width=1.2),
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.update_layout(
        title=dict(
            text=f"{symbol} {timeframe} — SMC + Fibonacci Analysis  |  {now_str}",
            font=dict(color="#e6edf3", size=14),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="monospace"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="#161b22", bordercolor="#21262d",
            font=dict(color="#e6edf3", size=9),
        ),
        height=900,
        margin=dict(l=60, r=120, t=60, b=40),
    )
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#21262d", zerolinecolor="#21262d",
            showgrid=True, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#21262d", zerolinecolor="#21262d",
            showgrid=True, row=i, col=1,
        )

    return fig


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def print_signals(signals: list, fib: dict, live_price: dict):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  XAUUSD Analysis  |  {now}")
    print(f"  Live Price: {live_price.get('mid', 'N/A'):.2f}  (source: {live_price.get('source','')})")
    print(f"{'='*60}")

    print("\n  Fibonacci Levels:")
    for k, v in fib.items():
        if k not in ("high", "low"):
            print(f"    Fib {k:5s}: {v:.2f}")

    if signals:
        print(f"\n  {'='*40}")
        print(f"  SIGNALS ({len(signals)} found):")
        for sig in signals:
            icon = "▲ BUY " if sig["type"] == "BUY" else "▼ SELL"
            print(f"  {icon} @ {sig['price']:.2f}  [{sig['confidence']}]")
            print(f"         {sig['reason']}")
            print(f"         SL: {sig['sl']:.2f}  TP: {sig['tp']:.2f}")
    else:
        print("\n  No signals at current price.")

    print(f"{'='*60}\n")


def run(symbol: str, timeframe: str, n_candles: int, use_mt5: bool, continuous: bool):
    iteration = 0

    def handle_exit(sig, frame):
        print("\n\nStopping analysis. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    while True:
        iteration += 1
        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Update #{iteration} — fetching data...")

        try:
            # Fetch data
            df = fetch_data(symbol, timeframe, n_candles, use_mt5)
            if df is None or len(df) < 20:
                print("Insufficient data. Retrying in 30s...")
                time.sleep(30)
                continue

            # Add indicators
            df = add_indicators(df)

            # Analysis
            fib    = fibonacci_levels(df)
            obs    = detect_order_blocks(df)
            grabs  = detect_liquidity_grabs(df)
            price  = get_live_price(symbol)
            sigs   = generate_signals(df, fib, obs, grabs)

            # Print to console
            print_signals(sigs, fib, price)
            print(f"  Order Blocks detected: {len(obs)}")
            print(f"  Liquidity Grabs:       {len(grabs)}")

            # Build chart
            fig = build_chart(df, symbol, timeframe, fib, obs, grabs, sigs, price)

            # Save HTML (interactive)
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
            html_path = f"{OUTPUT_DIR}/{symbol}_{timeframe}_{ts}.html"
            fig.write_html(html_path)
            print(f"  Chart saved: {html_path}")

            # Save PNG
            try:
                png_path = f"{OUTPUT_DIR}/{symbol}_{timeframe}_latest.png"
                # Convert index to string to avoid Plotly serialization issues
                df_plot = df.copy()
                df_plot.index = df_plot.index.astype(str)
                fig_png = build_chart(df_plot, symbol, timeframe, fib, obs, grabs, sigs, price)
                fig_png.write_image(png_path, width=1400, height=900, scale=1.5)
                print(f"  PNG saved:   {png_path}")
            except Exception as e:
                print(f"  PNG skipped: {e}")

            # Save CSV
            csv_path = f"{OUTPUT_DIR}/{symbol}_{timeframe}_data.csv"
            df.to_csv(csv_path)
            print(f"  CSV saved:   {csv_path}")

            if not continuous:
                print("\nSingle run complete.")
                break

            print(f"\n  Next update in {UPDATE_SECS}s... (Ctrl+C to stop)")
            time.sleep(UPDATE_SECS)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD SMC + Fibonacci Analysis")
    parser.add_argument("--symbol",    default=SYMBOL,    help="Symbol (default: XAUUSDm)")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe: M5 M15 H1 H4 D1")
    parser.add_argument("--candles",   default=N_CANDLES, type=int, help="Number of candles")
    parser.add_argument("--mt5",       action="store_true", help="Use MT5 for data")
    parser.add_argument("--once",      action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════╗
║   XAUUSD SMC + Fibonacci Analysis        ║
║   Symbol:    {args.symbol:<28} ║
║   Timeframe: {args.timeframe:<28} ║
║   Candles:   {args.candles:<28} ║
║   Data:      {'MT5' if args.mt5 else 'yfinance':<28} ║
╚══════════════════════════════════════════╝
""")

    run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_candles=args.candles,
        use_mt5=args.mt5,
        continuous=not args.once,
    )
