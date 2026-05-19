"""UniMind Memory API entrypoint."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .api.routes import router
    from .api.memory import get_memory_manager, set_memory_manager
    from .config import HOST, PORT, LOG_LEVEL, validate_config
except ImportError:  # pragma: no cover
    from api.routes import router  # type: ignore
    from api.memory import get_memory_manager, set_memory_manager  # type: ignore
    from config import HOST, PORT, LOG_LEVEL, validate_config  # type: ignore

validate_config()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("unimind.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize MemoryManager at startup to validate connectivity."""
    start = time.time()
    logger.info("=" * 55)
    logger.info("UniMind Memory API starting up...")
    logger.info("=" * 55)

    try:
        from .memory.memory_manager import MemoryManager
    except ImportError:
        from memory.memory_manager import MemoryManager  # type: ignore

    try:
        manager = MemoryManager()
        set_memory_manager(manager)
        elapsed = round(time.time() - start, 2)

        if manager._using_supabase:
            # Validate Supabase connectivity with a real query
            try:
                health = manager.storage.health_check()
                if health.get("connected"):
                    logger.info(
                        "Supabase connected: %d memories, %d messages (%.3fs latency)",
                        health.get("memory_count", 0),
                        health.get("message_count", 0),
                        health.get("latency_seconds", 0),
                    )
                else:
                    logger.error("Supabase health check FAILED: %s", health.get("error"))
            except Exception as exc:
                logger.error("Supabase health check exception: %s", exc)
        else:
            logger.info("Using local SQLite + FAISS storage")

        logger.info(
            "Embedding provider: %s", manager.embedding_service.provider,
        )
        logger.info(
            "LLM available: %s", manager.llm_service.available,
        )
        logger.info(
            "MemoryManager ready in %.2fs (supabase=%s)",
            elapsed, manager._using_supabase,
        )
    except Exception as exc:
        logger.error("MemoryManager initialization FAILED: %s", exc, exc_info=True)
        logger.warning("Backend will attempt lazy-init on first request")

    logger.info("=" * 55)
    logger.info("UniMind Memory API is READY")
    logger.info("=" * 55)

    yield  # App runs

    logger.info("UniMind Memory API shutting down...")


app = FastAPI(
    title="UniMind Memory API",
    description="FastAPI memory backend with Supabase PostgreSQL and semantic search.",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.time() - start) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response

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
        "version": "2.2.0",
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
