"""
Keep-alive mechanism for Railway deployment.
Prevents the service from sleeping by periodically pinging the health endpoint.
"""

import asyncio
import logging
from datetime import datetime

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KeepAlive:
    """
    Pings the API health endpoint every 10 minutes to keep Railway awake.
    Railway free tier sleeps after 5 minutes of inactivity — this prevents that.
    """

    def __init__(self, interval_minutes: int = 10):
        self.interval = interval_minutes * 60
        self._running = False
        self.api_url = settings.api_base_url.rstrip("/")

    async def start(self):
        """Start the keep-alive loop."""
        self._running = True
        logger.info(
            "Keep-alive started — pinging %s/health every %d minutes",
            self.api_url,
            self.interval // 60,
        )

        while self._running:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{self.api_url}/health")
                    if resp.status_code == 200:
                        logger.debug("Keep-alive ping successful: %s", resp.json())
                    else:
                        logger.warning("Keep-alive ping failed: HTTP %d", resp.status_code)
            except Exception as exc:
                logger.warning("Keep-alive ping error: %s", exc)

            await asyncio.sleep(self.interval)

    def stop(self):
        """Stop the keep-alive loop."""
        self._running = False
        logger.info("Keep-alive stopped.")
