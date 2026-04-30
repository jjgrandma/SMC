"""Launch the FastAPI backend."""
import os
import uvicorn
from app.config import get_settings

settings = get_settings()

# Railway injects PORT env variable — must use it
port = int(os.environ.get("PORT", settings.api_port))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
