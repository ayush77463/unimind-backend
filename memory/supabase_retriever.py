"""PostgreSQL vector retrieval — replaces local FAISS for cloud deployments.

Uses the ``cosine_similarity`` SQL function created by ``SupabaseStorage``
and applies the same hybrid ranking logic as the local ``MemoryRetriever``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from .embedding_service import EmbeddingService
from .supabase_storage import SupabaseStorage

logger = logging.getLogger(__name__)


class SupabaseRetriever:
    """Embedding retrieval over PostgreSQL with explainable hybrid ranking."""

    def __init__(
        self,
        storage: SupabaseStorage,
        embedding_service: EmbeddingService,
    ) -> None:
        self.storage = storage
        self.embedding_service = embedding_service
        self.last_debug: dict = {}
        self.last_warnings: list[str] = []

    # No-ops — PostgreSQL handles index management inline.
    def rebuild(self) -> None:
        pass

    def add_or_update_memory(self, memory_id: str) -> None:
        pass  # embedding is stored inline via upsert_memory

    def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: Iterable[str] | None = None,
        debug: bool = False,
        min_relevance: float = 0.08,
        min_evidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        top_k = max(1, min(20, int(top_k)))
        self.last_warnings = []
        self.last_debug = {
            "query": query,
            "top_k": top_k,
            "selected": [],
            "dropped": [],
        }

        try:
            query_embedding = self.embedding_service.embed(
                query, task_type="retrieval_query",
            ).vector
        except Exception as exc:
            logger.exception("Query embedding failed")
            self.last_warnings.append(f"query_embedding_failed: {exc}")
            self.last_debug["candidate_count"] = 0
            return []

        try:
            raw_hits = self.storage.vector_search(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=max(top_k * 10, 50),
            )
        except Exception as exc:
            logger.warning("PostgreSQL vector search failed: %s", exc)
            self.last_warnings.append(f"vector_search_failed: {exc}")
            self.last_debug["candidate_count"] = 0
            return []

        if not raw_hits:
            self.last_debug["candidate_count"] = 0
            return []

        allowed_types = set(memory_types or [])
        results: list[dict[str, Any]] = []
        dropped: list[dict] = []
        seen_norms: set[str] = set()
        query_tokens = self._tokens(query)

        for memory_id, raw_score in raw_hits:
            memory = self.storage.get_memory(memory_id)
            if memory is None or memory["user_id"] != user_id:
                continue
            if allowed_types and memory["memory_type"] not in allowed_types:
                continue

            semantic_score = max(0.0, min(1.0, float(raw_score)))
            keyword_score = self._keyword_score(query_tokens, memory)
            recency_score = self._recency_score(
                memory.get("updated_at") or memory.get("created_at")
            )
            importance_score = max(0.0, min(1.0, float(memory["importance"])))
            final_score = (
                0.55 * semantic_score
                + 0.20 * importance_score
                + 0.15 * recency_score
                + 0.10 * keyword_score
            )
            evidence_score = max(semantic_score, keyword_score)
            if evidence_score < 0.01:
                final_score *= 0.35
            norm = self._normalize_text(memory["content"])
            if norm in seen_norms:
                dropped.append(self._drop_debug(memory, final_score, evidence_score, "duplicate"))
                continue
            if final_score < min_relevance:
                dropped.append(self._drop_debug(memory, final_score, evidence_score, "below_relevance_threshold"))
                continue
            if evidence_score < min_evidence:
                dropped.append(self._drop_debug(memory, final_score, evidence_score, "low_query_evidence"))
                continue

            seen_norms.add(norm)
            memory.update({
                "semantic_score": round(semantic_score, 4),
                "importance_score": round(importance_score, 4),
                "recency_score": round(recency_score, 4),
                "keyword_score": round(keyword_score, 4),
                "relevance_score": round(final_score, 4),
            })
            if debug:
                memory["selection_reason"] = self._selection_reason(
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    importance_score=importance_score,
                    recency_score=recency_score,
                )
            results.append(memory)

        results.sort(key=lambda item: item["relevance_score"], reverse=True)
        selected = results[:top_k]
        self.storage.mark_memories_accessed(item["id"] for item in selected)
        self.last_debug.update({
            "candidate_count": len(results) + len(dropped),
            "selected_count": len(selected),
            "selected": [
                {
                    "id": item["id"],
                    "memory_type": item["memory_type"],
                    "category": item.get("category"),
                    "relevance_score": item.get("relevance_score"),
                    "reason": item.get("selection_reason")
                    or self._selection_reason(
                        semantic_score=float(item.get("semantic_score", 0.0)),
                        keyword_score=float(item.get("keyword_score", 0.0)),
                        importance_score=float(item.get("importance_score", 0.0)),
                        recency_score=float(item.get("recency_score", 0.0)),
                    ),
                }
                for item in selected
            ],
            "dropped": dropped[:20],
            "warnings": self.last_warnings,
        })
        return selected

    # ── Scoring helpers (same logic as local MemoryRetriever) ─────

    def _recency_score(self, iso_timestamp: str | None) -> float:
        if not iso_timestamp:
            return 0.5
        try:
            created = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.5
        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
            / 86400.0,
        )
        return float(0.5 ** (age_days / 30.0))

    def _keyword_score(self, query_tokens: set[str], memory: dict) -> float:
        if not query_tokens:
            return 0.0
        text = " ".join([
            str(memory.get("content", "")),
            str(memory.get("summary", "")),
            str(memory.get("category", "")),
            str(memory.get("memory_type", "")),
        ])
        memory_tokens = self._tokens(text)
        if not memory_tokens:
            return 0.0
        overlap = query_tokens & memory_tokens
        return min(1.0, len(overlap) / max(1, len(query_tokens)))

    def _tokens(self, text: str) -> set[str]:
        stop = {
            "about", "and", "are", "for", "from", "how", "that", "the",
            "this", "user", "what", "which", "with", "you",
        }
        tokens: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) <= 2 or token in stop:
                continue
            tokens.add(token)
            if token.endswith("s") and len(token) > 4:
                tokens.add(token[:-1])
        return tokens

    def _normalize_text(self, value: str) -> str:
        return " ".join((value or "").lower().split())

    def _drop_debug(self, memory: dict, final_score: float, evidence_score: float, reason: str) -> dict:
        return {
            "id": memory.get("id"),
            "memory_type": memory.get("memory_type"),
            "relevance_score": round(final_score, 4),
            "evidence_score": round(evidence_score, 4),
            "reason": reason,
        }

    def _selection_reason(self, *, semantic_score: float, keyword_score: float,
                          importance_score: float, recency_score: float) -> str:
        reasons: list[str] = []
        if semantic_score >= 0.35:
            reasons.append("semantic match")
        if keyword_score >= 0.4:
            reasons.append("keyword overlap")
        if importance_score >= 0.8:
            reasons.append("high importance")
        if recency_score >= 0.7:
            reasons.append("recent")
        return ", ".join(reasons) or "best available ranked memory"
