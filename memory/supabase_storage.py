"""PostgreSQL persistence for UniMind — Supabase cloud backend.

Drop-in replacement for the local SQLite ``MemoryStorage``.  Every public
method matches the same signature so that ``MemoryManager`` can swap backends
without changing any business logic.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]

try:
    from ..config import (
        SUPABASE_DB_HOST,
        SUPABASE_DB_NAME,
        SUPABASE_DB_PASSWORD,
        SUPABASE_DB_PORT,
        SUPABASE_DB_USER,
    )
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        SUPABASE_DB_HOST,
        SUPABASE_DB_NAME,
        SUPABASE_DB_PASSWORD,
        SUPABASE_DB_PORT,
        SUPABASE_DB_USER,
    )

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    except json.JSONDecodeError:
        return {}


class SupabaseStorage:
    """Cloud-persistent storage backed by Supabase PostgreSQL."""

    def __init__(self) -> None:
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required for Supabase storage. "
                "Install it with: pip install psycopg2-binary"
            )
        self._dsn = {
            "host": SUPABASE_DB_HOST,
            "port": SUPABASE_DB_PORT,
            "dbname": SUPABASE_DB_NAME,
            "user": SUPABASE_DB_USER,
            "password": SUPABASE_DB_PASSWORD,
            "sslmode": "require",
            "connect_timeout": 10,
        }
        self._init_schema()
        logger.info("SupabaseStorage connected to %s", SUPABASE_DB_HOST)

    @contextmanager
    def _connect(self):
        conn = psycopg2.connect(**self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user_created
                    ON messages(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_user_session_created
                    ON messages(user_id, session_id, created_at);

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_norm TEXT NOT NULL,
                    summary TEXT,
                    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'manual',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding REAL[],
                    embedding_dim INTEGER,
                    embedding_provider TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_type
                    ON memories(user_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_user_norm
                    ON memories(user_id, memory_type, content_norm);

                CREATE TABLE IF NOT EXISTS migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
            """)
            # Cosine similarity function for vector search
            cur.execute("""
                CREATE OR REPLACE FUNCTION cosine_similarity(a REAL[], b REAL[])
                RETURNS DOUBLE PRECISION
                LANGUAGE sql IMMUTABLE STRICT
                AS $$
                    SELECT CASE
                        WHEN array_length(a, 1) IS NULL
                          OR array_length(b, 1) IS NULL
                          OR array_length(a, 1) != array_length(b, 1)
                        THEN 0.0
                        ELSE (
                            SELECT COALESCE(
                                sum(a_val * b_val) / NULLIF(
                                    sqrt(sum(a_val * a_val)) * sqrt(sum(b_val * b_val)),
                                    0
                                ),
                                0.0
                            )
                            FROM unnest(a, b) AS t(a_val, b_val)
                        )
                    END
                $$;
            """)

    # ── Messages ──────────────────────────────────────────────────

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        message_id = f"msg_{uuid.uuid4().hex}"
        row = {
            "id": message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at or _utc_now(),
        }
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO messages
                    (id, user_id, session_id, role, content, metadata_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["id"], row["user_id"], row["session_id"],
                    row["role"], row["content"],
                    _json_dumps(row["metadata"]), row["created_at"],
                ),
            )
        return row

    def get_recent_messages(
        self,
        user_id: str,
        last_n: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        last_n = max(1, int(last_n))
        where = "user_id = %s"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = %s"
            params.append(session_id)
        with self._connect() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"""
                SELECT * FROM messages
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                [*params, last_n],
            )
            rows = cur.fetchall()
        return [self._message_from_row(r) for r in reversed(rows)]

    def get_message_count(self, user_id: str, session_id: str | None = None) -> int:
        where = "user_id = %s"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = %s"
            params.append(session_id)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params)
            return int(cur.fetchone()[0])

    def clear_messages(self, user_id: str, session_id: str | None = None) -> None:
        where = "user_id = %s"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = %s"
            params.append(session_id)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM messages WHERE {where}", params)

    # ── Memories ──────────────────────────────────────────────────

    def upsert_memory(
        self,
        *,
        memory_id: str,
        user_id: str,
        memory_type: str,
        content: str,
        summary: str | None,
        importance: float,
        source: str,
        metadata: dict[str, Any] | None,
        embedding: np.ndarray | None,
        embedding_provider: str | None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> bool:
        now = updated_at or _utc_now()
        existing = self.get_memory(memory_id)
        created = created_at or (existing["created_at"] if existing else now)
        emb_list: list[float] | None = None
        emb_dim: int | None = None
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            emb_list = vec.tolist()
            emb_dim = int(vec.shape[0])

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (
                    id, user_id, memory_type, content, content_norm, summary,
                    importance, source, metadata_json, embedding, embedding_dim,
                    embedding_provider, created_at, updated_at, last_accessed_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    content_norm = EXCLUDED.content_norm,
                    summary = EXCLUDED.summary,
                    importance = EXCLUDED.importance,
                    source = EXCLUDED.source,
                    metadata_json = EXCLUDED.metadata_json,
                    embedding = EXCLUDED.embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_provider = EXCLUDED.embedding_provider,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    memory_id, user_id, memory_type, content,
                    _normalize_text(content), summary,
                    float(max(0.0, min(1.0, importance))), source,
                    _json_dumps(metadata), emb_list, emb_dim,
                    embedding_provider, created, now,
                    existing.get("last_accessed_at") if existing else None,
                ),
            )
        return existing is not None

    def update_embedding(
        self, memory_id: str, embedding: np.ndarray, embedding_provider: str,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET embedding = %s, embedding_dim = %s,
                    embedding_provider = %s, updated_at = %s
                WHERE id = %s
                """,
                (vec.tolist(), int(vec.shape[0]), embedding_provider, _utc_now(), memory_id),
            )

    def update_importance(self, memory_id: str, importance: float) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET importance = %s WHERE id = %s",
                (float(max(0.0, min(1.0, importance))), memory_id),
            )

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET metadata_json = %s, updated_at = %s WHERE id = %s",
                (_json_dumps(metadata), _utc_now(), memory_id),
            )

    def get_memory(
        self, memory_id: str, include_embedding: bool = False,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
            row = cur.fetchone()
        return self._memory_from_row(row, include_embedding) if row else None

    def find_duplicate_memory(
        self, user_id: str, memory_type: str, content: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM memories
                WHERE user_id = %s AND memory_type = %s AND content_norm = %s
                LIMIT 1
                """,
                (user_id, memory_type, _normalize_text(content)),
            )
            row = cur.fetchone()
        return self._memory_from_row(row) if row else None

    def list_memories(
        self,
        *,
        user_id: str | None = None,
        memory_types: Iterable[str] | None = None,
        include_embedding: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        types = list(memory_types or [])
        if types:
            placeholders = ",".join(["%s"] * len(types))
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(types)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        limit_sql = ""
        if limit:
            limit_sql = "LIMIT %s OFFSET %s"
            params.append(limit)
            params.append(max(0, int(offset)))

        with self._connect() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"""
                SELECT * FROM memories
                {where}
                ORDER BY updated_at DESC
                {limit_sql}
                """,
                params,
            )
            rows = cur.fetchall()
        return [self._memory_from_row(r, include_embedding) for r in rows]

    def mark_memories_accessed(self, memory_ids: Iterable[str]) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.cursor()
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE memories SET last_accessed_at = %s WHERE id = %s",
                [(now, mid) for mid in ids],
            )

    def delete_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))

    def clear_memories(
        self, user_id: str, memory_types: Iterable[str] | None = None,
    ) -> None:
        types = list(memory_types or [])
        with self._connect() as conn:
            cur = conn.cursor()
            if types:
                placeholders = ",".join(["%s"] * len(types))
                cur.execute(
                    f"DELETE FROM memories WHERE user_id = %s AND memory_type IN ({placeholders})",
                    [user_id, *types],
                )
            else:
                cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))

    def clear_user(self, user_id: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))

    def has_migration(self, name: str) -> bool:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM migrations WHERE name = %s", (name,))
            return cur.fetchone() is not None

    def mark_migration(self, name: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO migrations(name, applied_at)
                VALUES (%s, %s)
                ON CONFLICT(name) DO UPDATE SET applied_at = EXCLUDED.applied_at
                """,
                (name, _utc_now()),
            )

    # ── Vector search (used by SupabaseRetriever) ─────────────────

    def vector_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Return (memory_id, cosine_similarity) pairs."""
        vec = np.asarray(query_embedding, dtype=np.float32).tolist()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, cosine_similarity(embedding, %s::REAL[]) AS sim
                FROM memories
                WHERE user_id = %s AND embedding IS NOT NULL
                ORDER BY sim DESC
                LIMIT %s
                """,
                (vec, user_id, top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    # ── Row mappers ───────────────────────────────────────────────

    def _message_from_row(self, row: dict) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "metadata": _json_loads(row["metadata_json"]),
            "timestamp": row["created_at"],
            "created_at": row["created_at"],
        }

    def _memory_from_row(
        self, row: dict, include_embedding: bool = False,
    ) -> dict[str, Any]:
        memory = {
            "id": row["id"],
            "user_id": row["user_id"],
            "memory_type": row["memory_type"],
            "content": row["content"],
            "fact": row["content"],
            "summary": row["summary"] or row["content"],
            "importance": float(row["importance"]),
            "source": row["source"],
            "metadata": _json_loads(row["metadata_json"]),
            "embedding_dim": row["embedding_dim"],
            "embedding_provider": row["embedding_provider"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed_at": row["last_accessed_at"],
            "timestamp": row["created_at"],
            "date_added": row["created_at"][:10] if row["created_at"] else "",
        }
        category = memory["metadata"].get("category")
        if category:
            memory["category"] = category
        elif memory["memory_type"] == "preference":
            memory["category"] = "preference"
        else:
            memory["category"] = memory["memory_type"]

        if include_embedding and row["embedding"] is not None:
            memory["embedding"] = np.asarray(row["embedding"], dtype=np.float32)
        return memory
