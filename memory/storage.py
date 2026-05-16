"""SQLite persistence for UniMind messages and long-term memories."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
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


class MemoryStorage:
    """Small SQLite repository used by the custom memory engine.

    SQLite stores durable metadata and embeddings. FAISS only stores the vector
    search structure, so the index can always be rebuilt from this database.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

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
                    importance REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'manual',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB,
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
                """
            )

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
            "created_at": created_at or utc_now(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages
                    (id, user_id, session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["user_id"],
                    row["session_id"],
                    row["role"],
                    row["content"],
                    _json_dumps(row["metadata"]),
                    row["created_at"],
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
        where = "user_id = ?"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [*params, last_n],
            ).fetchall()
        return [self._message_from_row(row) for row in reversed(rows)]

    def get_message_count(self, user_id: str, session_id: str | None = None) -> int:
        where = "user_id = ?"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)
        with self._connect() as conn:
            return int(
                conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE {where}",
                    params,
                ).fetchone()[0]
            )

    def clear_messages(self, user_id: str, session_id: str | None = None) -> None:
        where = "user_id = ?"
        params: list[Any] = [user_id]
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM messages WHERE {where}", params)

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
        now = updated_at or utc_now()
        existing = self.get_memory(memory_id)
        created = created_at or (existing["created_at"] if existing else now)
        vector_blob = None
        vector_dim = None
        if embedding is not None:
            vector = np.asarray(embedding, dtype=np.float32)
            vector_blob = vector.tobytes()
            vector_dim = int(vector.shape[0])

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, memory_type, content, content_norm, summary,
                    importance, source, metadata_json, embedding, embedding_dim,
                    embedding_provider, created_at, updated_at, last_accessed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    memory_type = excluded.memory_type,
                    content = excluded.content,
                    content_norm = excluded.content_norm,
                    summary = excluded.summary,
                    importance = excluded.importance,
                    source = excluded.source,
                    metadata_json = excluded.metadata_json,
                    embedding = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    embedding_provider = excluded.embedding_provider,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    user_id,
                    memory_type,
                    content,
                    normalize_text(content),
                    summary,
                    float(max(0.0, min(1.0, importance))),
                    source,
                    _json_dumps(metadata),
                    vector_blob,
                    vector_dim,
                    embedding_provider,
                    created,
                    now,
                    existing.get("last_accessed_at") if existing else None,
                ),
            )
        return existing is not None

    def update_embedding(
        self,
        memory_id: str,
        embedding: np.ndarray,
        embedding_provider: str,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET embedding = ?, embedding_dim = ?, embedding_provider = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    vector.tobytes(),
                    int(vector.shape[0]),
                    embedding_provider,
                    utc_now(),
                    memory_id,
                ),
            )

    def update_importance(self, memory_id: str, importance: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (float(max(0.0, min(1.0, importance))), memory_id),
            )

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (_json_dumps(metadata), utc_now(), memory_id),
            )

    def get_memory(
        self,
        memory_id: str,
        include_embedding: bool = False,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._memory_from_row(row, include_embedding) if row else None

    def find_duplicate_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND memory_type = ? AND content_norm = ?
                LIMIT 1
                """,
                (user_id, memory_type, normalize_text(content)),
            ).fetchone()
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
            clauses.append("user_id = ?")
            params.append(user_id)
        types = list(memory_types or [])
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(types)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        limit_sql = "LIMIT ? OFFSET ?" if limit else ""
        if limit:
            params.append(limit)
            params.append(max(0, int(offset)))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                {where}
                ORDER BY updated_at DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
        return [self._memory_from_row(row, include_embedding) for row in rows]

    def mark_memories_accessed(self, memory_ids: Iterable[str]) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        now = utc_now()
        with self._connect() as conn:
            conn.executemany(
                "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
                [(now, memory_id) for memory_id in ids],
            )

    def delete_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def clear_memories(
        self,
        user_id: str,
        memory_types: Iterable[str] | None = None,
    ) -> None:
        types = list(memory_types or [])
        with self._connect() as conn:
            if types:
                placeholders = ",".join("?" for _ in types)
                conn.execute(
                    f"""
                    DELETE FROM memories
                    WHERE user_id = ? AND memory_type IN ({placeholders})
                    """,
                    [user_id, *types],
                )
            else:
                conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))

    def clear_user(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))

    def has_migration(self, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM migrations WHERE name = ?",
                (name,),
            ).fetchone()
        return row is not None

    def mark_migration(self, name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO migrations(name, applied_at)
                VALUES (?, ?)
                """,
                (name, utc_now()),
            )

    def _message_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
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
        self,
        row: sqlite3.Row,
        include_embedding: bool = False,
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
            "date_added": row["created_at"][:10],
        }
        category = memory["metadata"].get("category")
        if category:
            memory["category"] = category
        elif memory["memory_type"] == "preference":
            memory["category"] = "preference"
        else:
            memory["category"] = memory["memory_type"]

        if include_embedding and row["embedding"] is not None:
            memory["embedding"] = np.frombuffer(row["embedding"], dtype=np.float32)
        return memory
