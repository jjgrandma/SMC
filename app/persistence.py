"""
Persistence layer — handles data storage across Railway redeploys.

Strategy:
  1. Primary: local files in data/ (works locally, lost on Railway redeploy)
  2. Backup: environment variables (survives redeploys, limited size)
  3. Future: database (Supabase/MongoDB for production)

On Railway:
  - Set JOURNAL_BACKUP=true in environment variables
  - The system will save/load data from env vars as base64 JSON
  - This survives redeploys but has a ~32KB limit per variable

For production use Railway Volume (mount at /app/data).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def save_to_env_backup(key: str, data: dict) -> bool:
    """
    Save data as base64 JSON to an environment variable.
    Used as backup when file storage is not persistent.
    Note: This only works if you can set env vars dynamically
    (not possible on Railway without the API).
    """
    try:
        encoded = base64.b64encode(
            json.dumps(data, separators=(",", ":")).encode()
        ).decode()
        os.environ[key] = encoded
        return True
    except Exception as exc:
        logger.error("Failed to save backup %s: %s", key, exc)
        return False


def load_from_env_backup(key: str) -> dict | None:
    """Load data from environment variable backup."""
    encoded = os.environ.get(key, "")
    if not encoded:
        return None
    try:
        return json.loads(base64.b64decode(encoded).decode())
    except Exception as exc:
        logger.error("Failed to load backup %s: %s", key, exc)
        return None


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_json_file(filepath: Path, env_backup_key: str | None = None) -> dict:
    """
    Read a JSON file. Falls back to env var backup if file missing.
    """
    ensure_data_dir()

    if filepath.exists():
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read %s: %s", filepath, exc)

    # Try env var backup
    if env_backup_key:
        data = load_from_env_backup(env_backup_key)
        if data:
            logger.info("Restored %s from env backup", filepath.name)
            # Write back to file
            try:
                filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
            return data

    return {}


def write_json_file(filepath: Path, data: dict, env_backup_key: str | None = None):
    """
    Write a JSON file. Also saves to env var backup if key provided.
    """
    ensure_data_dir()
    try:
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to write %s: %s", filepath, exc)

    if env_backup_key:
        save_to_env_backup(env_backup_key, data)


def get_storage_info() -> dict:
    """Return info about current storage state."""
    files = {}
    for f in DATA_DIR.glob("*.json"):
        try:
            size = f.stat().st_size
            files[f.name] = f"{size / 1024:.1f} KB"
        except Exception:
            files[f.name] = "unknown"

    return {
        "data_dir":   str(DATA_DIR.absolute()),
        "files":      files,
        "persistent": DATA_DIR.exists(),
        "railway_volume": os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "not mounted"),
    }
