"""
Pip calculation utilities.

XAUUSD pip conventions:
  - 1 pip  = $0.10  (4th decimal for most brokers, but Gold uses 2 decimals)
  - 1 point = $0.01 (smallest price move on Gold)
  - Most traders refer to Gold moves in "pips" where 1 pip = $0.10

Broker conventions vary. We support both:
  - Standard: 1 pip = 0.10  (10 points)  ← most common for Gold
  - Points:   1 pip = 0.01  (1 point)    ← some brokers

We default to: 1 pip = 0.10 for XAUUSD (industry standard)

Examples:
  Entry 4600, TP 4650 → 50 points → 500 pips
  Entry 4600, SL 4580 → 20 points → 200 pips
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Symbol pip definitions
# ---------------------------------------------------------------------------

# pip_size = the price movement that equals 1 pip
SYMBOL_PIP_SIZE: dict[str, float] = {
    "XAUUSD":  0.10,   # Gold: 1 pip = $0.10
    "XAUUSDm": 0.10,   # Exness micro Gold
    "XAUUSD.": 0.10,
    "EURUSD":  0.0001,
    "GBPUSD":  0.0001,
    "USDJPY":  0.01,
    "USDCHF":  0.0001,
    "AUDUSD":  0.0001,
    "USDCAD":  0.0001,
    "NZDUSD":  0.0001,
    "XAGUSD":  0.001,  # Silver
}

DEFAULT_PIP_SIZE = 0.10  # fallback for unknown symbols


def get_pip_size(symbol: str) -> float:
    """Return the pip size for a given symbol."""
    # Try exact match first
    if symbol in SYMBOL_PIP_SIZE:
        return SYMBOL_PIP_SIZE[symbol]
    # Try prefix match (handles broker suffixes like XAUUSDm, EURUSD.r etc.)
    for key, size in SYMBOL_PIP_SIZE.items():
        if symbol.upper().startswith(key.upper()):
            return size
    return DEFAULT_PIP_SIZE


def price_to_pips(price_diff: float, symbol: str) -> float:
    """Convert a price difference to pips."""
    pip_size = get_pip_size(symbol)
    if pip_size == 0:
        return 0.0
    return round(price_diff / pip_size, 1)


def pips_to_price(pips: float, symbol: str) -> float:
    """Convert pips to price difference."""
    return round(pips * get_pip_size(symbol), 5)


def calc_pip_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    Calculate pip value in USD per lot.
    XAUUSD: 1 pip ($0.10 move) × 100 oz per lot = $10 per pip per lot
    """
    pip_values: dict[str, float] = {
        "XAUUSD":  10.0,   # $10 per pip per standard lot
        "XAUUSDm": 10.0,
        "EURUSD":  10.0,
        "GBPUSD":  10.0,
        "USDJPY":  9.09,   # approximate
        "USDCHF":  10.0,
        "AUDUSD":  10.0,
        "USDCAD":  7.69,   # approximate
        "NZDUSD":  10.0,
    }
    base_value = pip_values.get(symbol, 10.0)
    return round(base_value * lot_size, 2)


def format_pips(pips: float, symbol: str = "XAUUSD") -> str:
    """Format pips for display."""
    sign = "+" if pips > 0 else ""
    return f"{sign}{pips:.1f} pips"
