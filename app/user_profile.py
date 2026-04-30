"""
User Profile Store.
Persists each Telegram user's account settings to a local JSON file.
Stores: account_balance, risk_percent, symbol preferences.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_FILE = Path("data/user_profiles.json")


@dataclass
class UserProfile:
    user_id: int
    username: str = ""
    account_balance: float = 10000.0
    risk_percent: float = 1.0        # % of balance risked per trade
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    alerts_enabled: bool = True
    min_confidence: str = "MEDIUM"   # LOW | MEDIUM | HIGH
    # Per-user MT5 credentials (optional)
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_connected: bool = False      # last known connection state


class ProfileStore:
    def __init__(self):
        PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[int, UserProfile] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, user_id: int) -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
            self._save()
        return self._profiles[user_id]

    def update(self, profile: UserProfile) -> None:
        self._profiles[profile.user_id] = profile
        self._save()

    def set_balance(self, user_id: int, balance: float) -> UserProfile:
        p = self.get(user_id)
        p.account_balance = balance
        self.update(p)
        return p

    def set_risk(self, user_id: int, risk_percent: float) -> UserProfile:
        p = self.get(user_id)
        p.risk_percent = risk_percent
        self.update(p)
        return p

    def set_alerts(self, user_id: int, enabled: bool) -> UserProfile:
        p = self.get(user_id)
        p.alerts_enabled = enabled
        self.update(p)
        return p

    def set_timeframe(self, user_id: int, timeframe: str) -> UserProfile:
        p = self.get(user_id)
        p.timeframe = timeframe
        self.update(p)
        return p

    def set_min_confidence(self, user_id: int, level: str) -> UserProfile:
        p = self.get(user_id)
        p.min_confidence = level.upper()
        self.update(p)
        return p

    def set_mt5_credentials(
        self,
        user_id: int,
        login: int,
        password: str,
        server: str,
    ) -> UserProfile:
        from app.crypto import encrypt_password
        p = self.get(user_id)
        p.mt5_login    = login
        p.mt5_password = encrypt_password(password)   # stored encrypted
        p.mt5_server   = server
        p.mt5_connected = False
        self.update(p)
        return p

    def set_mt5_connected(self, user_id: int, connected: bool) -> UserProfile:
        p = self.get(user_id)
        p.mt5_connected = connected
        self.update(p)
        return p

    def clear_mt5_credentials(self, user_id: int) -> UserProfile:
        p = self.get(user_id)
        p.mt5_login     = 0
        p.mt5_password  = ""
        p.mt5_server    = ""
        p.mt5_connected = False
        self.update(p)
        return p

    def all_alert_subscribers(self) -> list[UserProfile]:
        """Return all users who have alerts enabled."""
        return [p for p in self._profiles.values() if p.alerts_enabled]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not PROFILES_FILE.exists():
            return
        try:
            raw = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
            for uid_str, data in raw.items():
                uid = int(uid_str)
                self._profiles[uid] = UserProfile(**data)
        except Exception as exc:
            logger.error("Failed to load profiles: %s", exc)

    def _save(self) -> None:
        try:
            data = {str(uid): asdict(p) for uid, p in self._profiles.items()}
            PROFILES_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to save profiles: %s", exc)


# Singleton
_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store
