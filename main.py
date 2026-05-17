"""UniMind Memory API entrypoint."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .api.routes import router
    from .config import HOST, PORT, validate_config
except ImportError:  # pragma: no cover
    from api.routes import router  # type: ignore
    from config import HOST, PORT, validate_config  # type: ignore

validate_config()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

app = FastAPI(
    title="UniMind Memory API",
    description="FastAPI memory backend with SQLite/Supabase and semantic search.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# New concise API.
app.include_router(router)

# Backward-compatible aliases used by the Flutter app.
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "success": True,
        "message": "UniMind Memory API is running",
        "docs": "/docs",
        "health": "/health",
        "legacy_health": "/api/v1/health",
        "version": "2.1.0",
    }


if __name__ == "__main__":
    # Detect production vs development mode.
    is_production = os.getenv("RENDER", "") != "" or os.getenv("RAILWAY_ENVIRONMENT", "") != ""
    reload_mode = not is_production

    print("=" * 50)
    print("  UniMind Memory API Starting")
    print(f"  URL: http://{HOST}:{PORT}")
    print(f"  Docs: http://{HOST}:{PORT}/docs")
    print(f"  Mode: {'production' if is_production else 'development'}")
    print("=" * 50)
    uvicorn.run(
        "unimind_memory.main:app" if __package__ else "main:app",
        host=HOST,
        port=PORT,
        reload=reload_mode,
    )
