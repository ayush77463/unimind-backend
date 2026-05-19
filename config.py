"""Configuration for the UniMind memory backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.absolute()
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
UNIMIND_EMBED_CACHE_SIZE = int(os.getenv("UNIMIND_EMBED_CACHE_SIZE", "1024"))
UNIMIND_DECAY_TTL_SECONDS = float(os.getenv("UNIMIND_DECAY_TTL_SECONDS", "21600"))
UNIMIND_VECTOR_BACKEND = os.getenv("UNIMIND_VECTOR_BACKEND", "chroma").strip().lower()
UNIMIND_ENABLE_LOCAL_MODELS = os.getenv("UNIMIND_ENABLE_LOCAL_MODELS", "0") == "1"
UNIMIND_EMBEDDING_PROVIDER = os.getenv("UNIMIND_EMBEDDING_PROVIDER", "auto").strip().lower()
UNIMIND_ALLOW_MODEL_DOWNLOADS = os.getenv("UNIMIND_ALLOW_MODEL_DOWNLOADS", "0") == "1"
UNIMIND_SENTENCE_MODEL = os.getenv(
    "UNIMIND_SENTENCE_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
UNIMIND_SENTIMENT_MODEL = os.getenv(
    "UNIMIND_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

# ── Storage paths ────────────────────────────────────────────
# Render persistent disk mounts to /opt/render/project/src/storage
# when configured. Fall back to local ./storage for development.
_render_disk = os.getenv("RENDER_DISK_PATH")
if _render_disk:
    STORAGE_DIR = Path(_render_disk)
else:
    STORAGE_DIR = Path(os.getenv("UNIMIND_STORAGE_DIR", BASE_DIR / "storage"))

SQLITE_DB_PATH = Path(os.getenv("UNIMIND_DB_PATH", STORAGE_DIR / "memory.db"))
FAISS_DIR = Path(os.getenv("UNIMIND_FAISS_DIR", STORAGE_DIR / "faiss"))
FAISS_INDEX_PATH = FAISS_DIR / "memory.index"
FAISS_IDS_PATH = FAISS_DIR / "memory_ids.json"
CHROMA_DIR = Path(os.getenv("UNIMIND_CHROMA_DIR", STORAGE_DIR / "chroma"))

LEGACY_SHORT_TERM_FILE = STORAGE_DIR / "short_term_memory.json"
LEGACY_EPISODIC_FILE = STORAGE_DIR / "episodic_memory.json"
LEGACY_SEMANTIC_FILE = STORAGE_DIR / "semantic_memory.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/text-embedding-004",
)

# ── Supabase / PostgreSQL cloud persistence ──────────────────
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
SUPABASE_DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
SUPABASE_POOL_MIN = int(os.getenv("SUPABASE_POOL_MIN", "1"))
SUPABASE_POOL_MAX = int(os.getenv("SUPABASE_POOL_MAX", "3"))
SUPABASE_CONNECT_RETRIES = int(os.getenv("SUPABASE_CONNECT_RETRIES", "3"))
SUPABASE_STATEMENT_TIMEOUT_MS = int(os.getenv("SUPABASE_STATEMENT_TIMEOUT_MS", "30000"))

SHORT_TERM_MAX = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "20"))
EPISODIC_MAX = int(os.getenv("EPISODIC_MEMORY_MAX", "100"))
SEMANTIC_MAX = int(os.getenv("SEMANTIC_MEMORY_MAX", "500"))
HASH_EMBEDDING_DIM = int(os.getenv("HASH_EMBEDDING_DIM", "384"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def is_placeholder_key(api_key: str | None) -> bool:
    key = (api_key or "").strip()
    return (
        not key
        or key == "your_gemini_api_key_here"
        or key == "your_api_key_here"
        or "PASTE_YOUR" in key
    )


def gemini_enabled() -> bool:
    return (
        os.getenv("UNIMIND_DISABLE_GEMINI") != "1"
        and not is_placeholder_key(GEMINI_API_KEY)
    )


def supabase_db_enabled() -> bool:
    """Return True when Supabase PostgreSQL credentials are configured."""
    if os.getenv("UNIMIND_FORCE_LOCAL_STORAGE") == "1":
        return False
    return bool(SUPABASE_DB_HOST.strip() and SUPABASE_DB_PASSWORD.strip())


def validate_config() -> bool:
    """Print a structured startup summary without making API keys mandatory."""
    logger = logging.getLogger("unimind.config")
    print("=" * 55)
    print("  UniMind Memory Backend v2.2.0")
    print("=" * 55)
    if supabase_db_enabled():
        print(f"  Storage:    Supabase PostgreSQL")
        print(f"  DB Host:    {SUPABASE_DB_HOST}")
        print(f"  DB Port:    {SUPABASE_DB_PORT}")
        logger.info("Storage backend: Supabase PostgreSQL at %s", SUPABASE_DB_HOST)
    else:
        print(f"  Storage:    local SQLite at {SQLITE_DB_PATH}")
        print(f"  FAISS:      {FAISS_INDEX_PATH}")
        logger.info("Storage backend: local SQLite at %s", SQLITE_DB_PATH)
    if gemini_enabled():
        print(f"  Gemini:     enabled ({GEMINI_LLM_MODEL})")
        print(f"  Embeddings: {GEMINI_EMBEDDING_MODEL}")
        logger.info("Gemini configured: model=%s embeddings=%s", GEMINI_LLM_MODEL, GEMINI_EMBEDDING_MODEL)
        print("=" * 55)
        return True

    print("  Gemini:     DISABLED (no API key)")
    print("  Embeddings: local_hash (fallback)")
    logger.warning(
        "Gemini is not configured; memory endpoints will use local "
        "deterministic embeddings and /chat will return 503."
    )
    print("=" * 55)
    return False
