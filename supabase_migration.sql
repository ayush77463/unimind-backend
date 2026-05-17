-- UniMind Memory: Supabase Migration Script
-- Run this in your Supabase SQL Editor to manually set up the database schema,
-- or let the backend automatically create it on first run.

-- Messages (Short-term memory)
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

-- Memories (Long-term semantic & episodic)
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

-- Migrations tracking
CREATE TABLE IF NOT EXISTS migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Cosine similarity function for pgvector-less array retrieval
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
