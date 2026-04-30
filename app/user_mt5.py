"""
Per-User MT5 Connection Manager.

Each Telegram user can connect their own MT5 account.
Credentials are stored in their user profile (encrypted in memory,
stored in data/user_profiles.json — keep this file private).

IMPORTANT SECURITY NOTE:
  MT5 passwords are stored in plaintext in user_profiles.json.
  This file must NEVER be committed to git (it's in .gitignore).
  For production, consider encrypting credentials at rest.

Usage flow:
  1. User sends /mt5setup
  2. Bot asks for login, password, server (via conversation)
  3. Bot tests connection
  4. If success → saves credentials to profile
  5. User can now use /mt5status, /mt5connect etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.user_profile import UserProfile, get_profile_store

logger = logging.getLogger(__name__)

# Cache of active MT5 connections per user
# key: user_id → MT5Trader instance
_user_traders: dict[int, Any] = {}


@dataclass
class UserMT5Status:
    user_id: int
    connected: bool
    account_login: int = 0
    account_name: str = ""
    broker: str = ""
    server: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0
    leverage: int = 0
    trade_mode: str = ""
    error: str = ""


def get_user_trader(user_id: int):
    """Get or create an MT5Trader for a specific user."""
    if user_id in _user_traders:
        return _user_traders[user_id]

    from app.trader import MT5Trader
    trader = MT5Trader()
    _user_traders[user_id] = trader
    return trader


def connect_user_mt5(profile: UserProfile) -> UserMT5Status:
    """
    Connect a user's MT5 account using their stored credentials.
    Password is decrypted in memory only — never logged or returned.
    """
    if not profile.mt5_login or not profile.mt5_password or not profile.mt5_server:
        return UserMT5Status(
            user_id=profile.user_id,
            connected=False,
            error="MT5 credentials not set. Use /mt5setup to configure.",
        )

    try:
        import MetaTrader5 as mt5
    except ImportError:
        return UserMT5Status(
            user_id=profile.user_id,
            connected=False,
            error="MetaTrader5 library not available on this server. Run locally on Windows.",
        )

    # Decrypt password in memory — NEVER log this value
    from app.crypto import decrypt_password
    plain_password = decrypt_password(profile.mt5_password)
    if not plain_password:
        return UserMT5Status(
            user_id=profile.user_id,
            connected=False,
            error="Failed to decrypt MT5 password. Use /mt5setup to re-enter credentials.",
        )

    # Initialize with user's credentials
    connected = mt5.initialize(
        login=profile.mt5_login,
        password=plain_password,   # used here, never stored or logged
        server=profile.mt5_server,
    )

    # Immediately clear from local variable
    plain_password = None

    if not connected:
        err = mt5.last_error()
        return UserMT5Status(
            user_id=profile.user_id,
            connected=False,
            error=f"MT5 connection failed: {err}",
        )

    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        return UserMT5Status(
            user_id=profile.user_id,
            connected=False,
            error=f"Cannot get account info: {mt5.last_error()}",
        )

    trade_modes = {
        mt5.ACCOUNT_TRADE_MODE_DEMO:    "DEMO",
        mt5.ACCOUNT_TRADE_MODE_REAL:    "REAL",
        mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
    }

    # Update profile with real balance
    store = get_profile_store()
    store.set_mt5_connected(profile.user_id, True)
    p = store.get(profile.user_id)
    p.account_balance = account.balance
    store.update(p)

    # Cache the trader
    from app.trader import MT5Trader
    trader = MT5Trader()
    trader._connected = True
    _user_traders[profile.user_id] = trader

    logger.info(
        "User %d connected MT5: account=%d balance=%.2f mode=%s",
        profile.user_id, account.login, account.balance,
        trade_modes.get(account.trade_mode, "?"),
    )

    return UserMT5Status(
        user_id=profile.user_id,
        connected=True,
        account_login=account.login,
        account_name=account.name,
        broker=account.company,
        server=account.server,
        balance=account.balance,
        equity=account.equity,
        margin_free=account.margin_free,
        leverage=account.leverage,
        trade_mode=trade_modes.get(account.trade_mode, "UNKNOWN"),
    )


def disconnect_user_mt5(user_id: int):
    """Disconnect a user's MT5 session."""
    if user_id in _user_traders:
        try:
            _user_traders[user_id].disconnect()
        except Exception:
            pass
        del _user_traders[user_id]

    store = get_profile_store()
    store.set_mt5_connected(user_id, False)


def get_user_mt5_status(user_id: int) -> UserMT5Status:
    """Get current MT5 status for a user."""
    profile = get_profile_store().get(user_id)

    if not profile.mt5_login:
        return UserMT5Status(
            user_id=user_id,
            connected=False,
            error="No MT5 credentials. Use /mt5setup",
        )

    if user_id not in _user_traders or not profile.mt5_connected:
        return UserMT5Status(
            user_id=user_id,
            connected=False,
            account_login=profile.mt5_login,
            server=profile.mt5_server,
            error="Not connected. Use /mt5connect",
        )

    # Try to get live account info
    try:
        trader = _user_traders[user_id]
        account = trader.get_account_info()
        if "error" not in account:
            trade_modes = {"0": "DEMO", "1": "REAL", "2": "CONTEST"}
            return UserMT5Status(
                user_id=user_id,
                connected=True,
                account_login=account.get("login", profile.mt5_login),
                account_name=account.get("name", ""),
                broker=account.get("company", ""),
                server=account.get("server", profile.mt5_server),
                balance=account.get("balance", 0),
                equity=account.get("equity", 0),
                margin_free=account.get("margin_free", 0),
                leverage=account.get("leverage", 0),
                trade_mode=account.get("trade_mode_name", ""),
            )
    except Exception as exc:
        logger.error("get_user_mt5_status failed for user %d: %s", user_id, exc)

    return UserMT5Status(
        user_id=user_id,
        connected=False,
        error="Connection lost. Use /mt5connect to reconnect.",
    )
