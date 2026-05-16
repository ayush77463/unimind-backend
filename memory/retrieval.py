"""FAISS retrieval plus explainable memory ranking."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

try:
    from ..config import FAISS_IDS_PATH, FAISS_INDEX_PATH
except ImportError:  # pragma: no cover
    from config import FAISS_IDS_PATH, FAISS_INDEX_PATH  # type: ignore

from .embedding_service import EmbeddingService
from .storage import MemoryStorage


logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Embedding retrieval over FAISS with explainable hybrid ranking."""

    def __init__(
        self,
        storage: MemoryStorage,
        embedding_service: EmbeddingService,
        index_path: str | Path = FAISS_INDEX_PATH,
        ids_path: str | Path = FAISS_IDS_PATH,
    ):
        self.storage = storage
        self.embedding_service = embedding_service
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index: faiss.IndexFlatIP | None = None
        self.memory_ids: list[str] = []
        self.last_debug: dict = {}
        self.last_warnings: list[str] = []
        self._load_or_rebuild()

    def _load_or_rebuild(self) -> None:
        if not self.index_path.exists() or not self.ids_path.exists():
            self.rebuild()
            return
        try:
            index = faiss.read_index(str(self.index_path))
            ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
            if not isinstance(ids, list) or len(ids) != index.ntotal:
                raise ValueError("FAISS id map does not match index size")
            self.index = index
            self.memory_ids = [str(item) for item in ids]
        except Exception as exc:
            logger.warning("FAISS index load failed; rebuilding: %s", exc)
            self.rebuild()

    def rebuild(self) -> None:
        records = self.storage.list_memories(include_embedding=True)
        if not records:
            self.index = None
            self.memory_ids = []
            self._delete_index_files()
            return

        try:
            probe = self.embedding_service.embed(
                "UniMind embedding dimension probe",
                task_type="retrieval_document",
            )
        except Exception as exc:
            logger.exception("Embedding probe failed during FAISS rebuild")
            self.last_warnings = [f"embedding_probe_failed: {exc}"]
            self.index = None
            self.memory_ids = []
            return
        target_dim = int(probe.vector.shape[0])
        vectors: list[np.ndarray] = []
        ids: list[str] = []

        for record in records:
            vector = record.get("embedding")
            provider = record.get("embedding_provider")
            if (
                vector is None
                or int(record.get("embedding_dim") or 0) != target_dim
                or provider != probe.provider
            ):
                try:
                    embedded = self.embedding_service.embed(
                        record["content"],
                        task_type="retrieval_document",
                    )
                    vector = embedded.vector
                    self.storage.update_embedding(
                        record["id"],
                        vector,
                        embedded.provider,
                    )
                except Exception as exc:
                    logger.warning("Skipping memory %s during rebuild: %s", record["id"], exc)
                    continue
            vectors.append(np.asarray(vector, dtype=np.float32))
            ids.append(record["id"])

        if not vectors:
            self.index = None
            self.memory_ids = []
            self._delete_index_files()
            return

        matrix = np.vstack(vectors).astype(np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self.index = index
        self.memory_ids = ids
        self._persist()

    def add_or_update_memory(self, memory_id: str) -> None:
        record = self.storage.get_memory(memory_id, include_embedding=True)
        if record is None:
            self.rebuild()
            return

        vector = record.get("embedding")
        if vector is None:
            try:
                embedded = self.embedding_service.embed(
                    record["content"],
                    task_type="retrieval_document",
                )
                vector = embedded.vector
                self.storage.update_embedding(memory_id, vector, embedded.provider)
            except Exception as exc:
                logger.warning("Could not embed memory %s; rebuilding skipped: %s", memory_id, exc)
                self.last_warnings = [f"embedding_failed: {exc}"]
                return

        vector = np.asarray(vector, dtype=np.float32)
        if (
            self.index is None
            or memory_id in self.memory_ids
            or self.index.d != int(vector.shape[0])
        ):
            self.rebuild()
            return

        self.index.add(vector.reshape(1, -1))
        self.memory_ids.append(memory_id)
        self._persist()

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
    ) -> list[dict]:
        top_k = max(1, min(20, int(top_k)))
        self.last_warnings = []
        self.last_debug = {
            "query": query,
            "top_k": top_k,
            "selected": [],
            "dropped": [],
        }
        if self.index is None or self.index.ntotal == 0:
            self.last_debug["candidate_count"] = 0
            return []

        try:
            query_embedding = self.embedding_service.embed(
                query,
                task_type="retrieval_query",
            ).vector
        except Exception as exc:
            logger.exception("Query embedding failed")
            self.last_warnings.append(f"query_embedding_failed: {exc}")
            self.last_debug["candidate_count"] = 0
            return []
        if query_embedding.shape[0] != self.index.d:
            self.rebuild()
            if self.index is None or query_embedding.shape[0] != self.index.d:
                self.last_warnings.append("embedding_dimension_mismatch")
                return []

        candidate_count = (
            self.index.ntotal
            if self.index.ntotal <= 10000
            else min(self.index.ntotal, max(top_k * 50, 100))
        )
        try:
            scores, positions = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                candidate_count,
            )
        except Exception as exc:
            logger.warning("FAISS search failed; rebuilding once: %s", exc)
            self.last_warnings.append("vector_search_failed_rebuilt")
            self.rebuild()
            if self.index is None or self.index.ntotal == 0:
                self.last_debug["candidate_count"] = 0
                return []
            candidate_count = min(self.index.ntotal, candidate_count)
            try:
                scores, positions = self.index.search(
                    query_embedding.reshape(1, -1).astype(np.float32),
                    candidate_count,
                )
            except Exception as retry_exc:
                logger.exception("FAISS search failed after rebuild")
                self.last_warnings.append(f"vector_search_failed: {retry_exc}")
                self.last_debug["candidate_count"] = 0
                return []

        allowed_types = set(memory_types or [])
        results: list[dict] = []
        dropped: list[dict] = []
        seen_norms: set[str] = set()
        query_tokens = self._tokens(query)
        for raw_score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(self.memory_ids):
                continue
            memory_id = self.memory_ids[int(position)]
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
                dropped.append(
                    self._drop_debug(memory, final_score, evidence_score, "duplicate")
                )
                continue
            if final_score < min_relevance:
                dropped.append(
                    self._drop_debug(memory, final_score, evidence_score, "below_relevance_threshold")
                )
                continue
            if evidence_score < min_evidence:
                dropped.append(
                    self._drop_debug(memory, final_score, evidence_score, "low_query_evidence")
                )
                continue
            seen_norms.add(norm)
            memory.update(
                {
                    "semantic_score": round(semantic_score, 4),
                    "importance_score": round(importance_score, 4),
                    "recency_score": round(recency_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "relevance_score": round(final_score, 4),
                }
            )
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
        self.last_debug.update(
            {
                "candidate_count": len(results) + len(dropped),
                "selected_count": len(selected),
                "selected": [
                    {
                        "id": item["id"],
                        "memory_type": item["memory_type"],
                        "category": item.get("category"),
                        "semantic_score": item.get("semantic_score"),
                        "keyword_score": item.get("keyword_score"),
                        "importance_score": item.get("importance_score"),
                        "recency_score": item.get("recency_score"),
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
            }
        )
        if debug:
            logger.debug("Memory retrieval debug: %s", self.last_debug)
        return selected

    def _persist(self) -> None:
        if self.index is None:
            self._delete_index_files()
            return
        try:
            faiss.write_index(self.index, str(self.index_path))
            self.ids_path.write_text(
                json.dumps(self.memory_ids, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("FAISS index persist failed: %s", exc)
            self.last_warnings.append(f"vector_store_persist_failed: {exc}")

    def _delete_index_files(self) -> None:
        for path in (self.index_path, self.ids_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

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
        text = " ".join(
            [
                str(memory.get("content", "")),
                str(memory.get("summary", "")),
                str(memory.get("category", "")),
                str(memory.get("memory_type", "")),
            ]
        )
        memory_tokens = self._tokens(text)
        if not memory_tokens:
            return 0.0
        overlap = query_tokens & memory_tokens
        return min(1.0, len(overlap) / max(1, len(query_tokens)))

    def _tokens(self, text: str) -> set[str]:
        stop = {
            "about",
            "and",
            "are",
            "for",
            "from",
            "how",
            "that",
            "the",
            "this",
            "user",
            "what",
            "which",
            "with",
            "you",
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

    def _drop_debug(
        self,
        memory: dict,
        final_score: float,
        evidence_score: float,
        reason: str,
    ) -> dict:
        return {
            "id": memory.get("id"),
            "memory_type": memory.get("memory_type"),
            "category": memory.get("category"),
            "relevance_score": round(final_score, 4),
            "evidence_score": round(evidence_score, 4),
            "reason": reason,
        }

    def _selection_reason(
        self,
        *,
        semantic_score: float,
        keyword_score: float,
        importance_score: float,
        recency_score: float,
    ) -> str:
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
