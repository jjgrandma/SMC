"""Launch the FastAPI backend."""
import os
import uvicorn
from app.config import get_settings

settings = get_settings()

# Railway injects PORT — must use it
# Also respect API_PORT set by start.sh
port = int(os.environ.get("PORT", os.environ.get("API_PORT", settings.api_port)))

uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=port,
    reload=False,
    log_level="info",
)
