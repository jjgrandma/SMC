"""
Credential encryption for MT5 passwords.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
The encryption key is stored in data/secret.key — never committed to git.

Security model:
  - Passwords are encrypted before saving to user_profiles.json
  - The key file (data/secret.key) must be kept private
  - Even with access to user_profiles.json, passwords cannot be read
    without the key file
  - The admin cannot see passwords — only the encrypted ciphertext
  - Passwords are only decrypted in memory at connection time
  - They are never logged, printed, or returned to any API response
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KEY_FILE = Path("data/secret.key")


def _get_or_create_key() -> bytes:
    """
    Load or generate the encryption key.
    Key is stored in data/secret.key — never committed to git.
    """
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    # Generate new key
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    # Restrict file permissions on Unix
    try:
        import os
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass
    logger.info("Generated new encryption key at %s", KEY_FILE)
    return key


def encrypt_password(plaintext: str) -> str:
    """
    Encrypt a password. Returns base64-encoded ciphertext string.
    Returns empty string if plaintext is empty.
    """
    if not plaintext:
        return ""
    try:
        from cryptography.fernet import Fernet
        key = _get_or_create_key()
        f   = Fernet(key)
        return f.encrypt(plaintext.encode()).decode()
    except Exception as exc:
        logger.error("Encryption failed: %s", exc)
        return ""


def decrypt_password(ciphertext: str) -> str:
    """
    Decrypt a password. Returns plaintext string.
    Returns empty string if decryption fails.
    NEVER log or print the return value.
    """
    if not ciphertext:
        return ""
    try:
        from cryptography.fernet import Fernet
        key = _get_or_create_key()
        f   = Fernet(key)
        return f.decrypt(ciphertext.encode()).decode()
    except Exception as exc:
        logger.error("Decryption failed (wrong key or corrupted data): %s", exc)
        return ""


def mask_password(password: str) -> str:
    """Return a masked version for display — never show real password."""
    if not password:
        return "(not set)"
    return "•" * min(len(password), 8)


def is_encrypted(value: str) -> bool:
    """Check if a value looks like a Fernet-encrypted string."""
    return value.startswith("gAAAAA") and len(value) > 50
