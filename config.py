"""Configuration for the UniMind memory backend."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.absolute()
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

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

SHORT_TERM_MAX = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "20"))
EPISODIC_MAX = int(os.getenv("EPISODIC_MEMORY_MAX", "100"))
SEMANTIC_MAX = int(os.getenv("SEMANTIC_MEMORY_MAX", "500"))
HASH_EMBEDDING_DIM = int(os.getenv("HASH_EMBEDDING_DIM", "384"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


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
    return bool(SUPABASE_DB_HOST.strip() and SUPABASE_DB_PASSWORD.strip())


def validate_config() -> bool:
    """Print a small startup summary without making API keys mandatory."""
    if supabase_db_enabled():
        print("Storage backend: Supabase PostgreSQL at", SUPABASE_DB_HOST)
    else:
        print("Storage backend: local SQLite at", SQLITE_DB_PATH)
        print("UniMind memory storage:", STORAGE_DIR)
        print("FAISS index:", FAISS_INDEX_PATH)
    if gemini_enabled():
        print("Gemini is configured for LLM calls and embeddings")
        return True

    print(
        "Gemini is not configured; memory endpoints will use local "
        "deterministic embeddings and /chat will return 503."
    )
    return False
