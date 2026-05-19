"""Persistent Chroma vector mirror for UniMind memory retrieval."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from ..config import CHROMA_DIR
except ImportError:  # pragma: no cover
    from config import CHROMA_DIR  # type: ignore

from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """Chroma KNN retrieval with storage-backed hydration and fallback search."""

    def __init__(
        self,
        *,
        storage,
        embedding_service: EmbeddingService,
        fallback_retriever=None,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = "unimind_memories",
    ) -> None:
        self.storage = storage
        self.embedding_service = embedding_service
        self.fallback_retriever = fallback_retriever
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.last_debug: dict[str, Any] = {}
        self.last_warnings: list[str] = []
        self._client = None
        self._collection = None
        self._available = False
        self._setup()

    @property
    def available(self) -> bool:
        return self._available and self._collection is not None

    def _setup(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info("Chroma vector store ready at %s", self.persist_dir)
        except Exception as exc:
            self._available = False
            self.last_warnings = [f"chroma_unavailable: {exc}"]
            logger.warning("Chroma vector store unavailable; using fallback retriever: %s", exc)

    def rebuild(self) -> None:
        if not self.available:
            self._fallback_rebuild()
            return

        start = time.time()
        self._clear_collection()
        records = self.storage.list_memories(include_embedding=True)
        if not records:
            self._fallback_rebuild()
            return

        try:
            probe = self.embedding_service.embed(
                "UniMind Chroma embedding dimension probe",
                task_type="retrieval_document",
            )
            target_dim = int(probe.vector.shape[0])
            target_provider = probe.provider
        except Exception as exc:
            self.last_warnings = [f"chroma_embedding_probe_failed: {exc}"]
            logger.warning("Chroma rebuild skipped because embedding probe failed: %s", exc)
            self._fallback_rebuild()
            return

        batch_ids: list[str] = []
        batch_embeddings: list[list[float]] = []
        batch_documents: list[str] = []
        batch_metadatas: list[dict[str, Any]] = []

        for record in records:
            vector = record.get("embedding")
            provider = record.get("embedding_provider")
            dim = int(record.get("embedding_dim") or 0)
            if vector is None or dim != target_dim or provider != target_provider:
                try:
                    embedded = self.embedding_service.embed(
                        record["content"],
                        task_type="retrieval_document",
                    )
                    vector = embedded.vector
                    provider = embedded.provider
                    dim = int(vector.shape[0])
                    self.storage.update_embedding(record["id"], vector, provider)
                except Exception as exc:
                    logger.warning("Skipping Chroma sync for memory %s: %s", record["id"], exc)
                    continue
            if provider != target_provider or dim != target_dim:
                continue
            batch_ids.append(record["id"])
            batch_embeddings.append(np.asarray(vector, dtype=np.float32).tolist())
            batch_documents.append(str(record.get("content") or ""))
            batch_metadatas.append(self._metadata_for_record(record, provider, dim))
            if len(batch_ids) >= 128:
                self._upsert_batch(batch_ids, batch_embeddings, batch_documents, batch_metadatas)
                batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

        if batch_ids:
            self._upsert_batch(batch_ids, batch_embeddings, batch_documents, batch_metadatas)

        self._fallback_rebuild()
        logger.info(
            "Chroma rebuild complete records=%d latency_ms=%.1f",
            len(records),
            (time.time() - start) * 1000,
        )

    def add_or_update_memory(self, memory_id: str) -> None:
        if not self.available:
            self._fallback_add_or_update(memory_id)
            return
        record = self.storage.get_memory(memory_id, include_embedding=True)
        if record is None:
            self.delete_memory(memory_id)
            return

        vector = record.get("embedding")
        provider = record.get("embedding_provider")
        expected_provider = self.embedding_service.provider
        provider_stale = (
            provider
            and expected_provider != "gemini"
            and expected_provider != "sentence_transformer:lazy"
            and provider != expected_provider
        )
        if vector is None or provider_stale:
            try:
                embedded = self.embedding_service.embed(
                    record["content"],
                    task_type="retrieval_document",
                )
                vector = embedded.vector
                provider = embedded.provider
                self.storage.update_embedding(memory_id, vector, provider)
            except Exception as exc:
                self.last_warnings = [f"chroma_embedding_failed: {exc}"]
                logger.warning("Chroma sync skipped for memory %s: %s", memory_id, exc)
                self._fallback_add_or_update(memory_id)
                return

        dim = int(np.asarray(vector, dtype=np.float32).shape[0])
        try:
            self._collection.upsert(
                ids=[memory_id],
                embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                documents=[str(record.get("content") or "")],
                metadatas=[self._metadata_for_record(record, provider, dim)],
            )
        except Exception as exc:
            logger.warning("Chroma upsert failed for %s; rebuilding once: %s", memory_id, exc)
            self.last_warnings.append(f"chroma_upsert_failed: {exc}")
            self.rebuild()
        finally:
            self._fallback_add_or_update(memory_id)

    def delete_memory(self, memory_id: str) -> None:
        if self.available:
            try:
                self._collection.delete(ids=[memory_id])
            except Exception as exc:
                logger.debug("Chroma delete ignored for %s: %s", memory_id, exc)

    def clear_user(self, user_id: str) -> None:
        if self.available:
            try:
                self._collection.delete(where={"user_id": user_id})
            except Exception as exc:
                logger.debug("Chroma user clear ignored for %s: %s", user_id, exc)
        self._fallback_rebuild()

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
            "vector_backend": "chroma",
            "selected": [],
            "dropped": [],
        }

        if not self.available:
            return self._fallback_search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                memory_types=memory_types,
                debug=debug,
                min_relevance=min_relevance,
                min_evidence=min_evidence,
                reason="chroma_unavailable",
            )

        try:
            if self._collection.count() == 0:
                self.rebuild()
        except Exception:
            pass

        try:
            query_embedding = self.embedding_service.embed(
                query,
                task_type="retrieval_query",
            )
            query_vector = query_embedding.vector
            query_provider = query_embedding.provider
        except Exception as exc:
            logger.exception("Chroma query embedding failed")
            self.last_warnings.append(f"query_embedding_failed: {exc}")
            self.last_debug["candidate_count"] = 0
            return []

        raw = self._query_chroma(
            query_vector=query_vector,
            user_id=user_id,
            top_k=max(top_k * 10, 50),
        )
        if raw is None:
            self.rebuild()
            raw = self._query_chroma(
                query_vector=query_vector,
                user_id=user_id,
                top_k=max(top_k * 10, 50),
            )
        if raw is None:
            return self._fallback_search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                memory_types=memory_types,
                debug=debug,
                min_relevance=min_relevance,
                min_evidence=min_evidence,
                reason="chroma_query_failed",
            )

        ids = [str(item) for item in (raw.get("ids") or [[]])[0]]
        distances = [float(item) for item in (raw.get("distances") or [[]])[0]]
        if not ids:
            self.last_debug["candidate_count"] = 0
            return []

        try:
            hydrated = self.storage.get_memories_by_ids(ids)
        except Exception as exc:
            self.last_warnings.append(f"batch_hydration_failed: {exc}")
            hydrated = {}

        allowed_types = set(memory_types or [])
        query_tokens = self._tokens(query)
        results: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        seen_norms: set[str] = set()
        provider_mismatches = 0

        for memory_id, distance in zip(ids, distances):
            memory = hydrated.get(memory_id) or self.storage.get_memory(memory_id)
            if memory is None or memory.get("user_id") != user_id:
                dropped.append(self._drop_debug({"id": memory_id}, 0.0, 0.0, "not_found"))
                continue
            if allowed_types and memory["memory_type"] not in allowed_types:
                continue
            mem_provider = memory.get("embedding_provider") or ""
            mem_dim = int(memory.get("embedding_dim") or 0)
            if mem_provider != query_provider or mem_dim != int(query_vector.shape[0]):
                provider_mismatches += 1
                dropped.append(self._drop_debug(memory, 0.0, 0.0, "embedding_provider_mismatch"))
                self.add_or_update_memory(memory_id)
                continue

            semantic_score = max(0.0, min(1.0, 1.0 - float(distance)))
            keyword_score = self._keyword_score(query_tokens, memory)
            recency_score = self._recency_score(memory.get("updated_at") or memory.get("created_at"))
            importance_score = max(0.0, min(1.0, float(memory["importance"])))
            final_score = (
                0.58 * semantic_score
                + 0.20 * importance_score
                + 0.12 * recency_score
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
                dropped.append(
                    self._drop_debug(memory, final_score, evidence_score, "below_relevance_threshold")
                )
                continue
            if evidence_score < min_evidence:
                dropped.append(self._drop_debug(memory, final_score, evidence_score, "low_query_evidence"))
                continue

            seen_norms.add(norm)
            memory.update(
                {
                    "semantic_score": round(semantic_score, 4),
                    "importance_score": round(importance_score, 4),
                    "recency_score": round(recency_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "relevance_score": round(final_score, 4),
                    "vector_backend": "chroma",
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
        if provider_mismatches:
            self.last_warnings.append(f"provider_mismatch_reindexed:{provider_mismatches}")

        self.last_debug.update(
            {
                "candidate_count": len(results) + len(dropped),
                "selected_count": len(selected),
                "query_provider": query_provider,
                "provider_mismatches": provider_mismatches,
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
        return selected

    def _query_chroma(
        self,
        *,
        query_vector: np.ndarray,
        user_id: str,
        top_k: int,
    ) -> dict[str, Any] | None:
        try:
            return self._collection.query(
                query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
                n_results=top_k,
                where={"user_id": user_id},
                include=["distances", "metadatas", "documents"],
            )
        except Exception as exc:
            self.last_warnings.append(f"chroma_query_failed: {exc}")
            logger.warning("Chroma query failed: %s", exc)
            return None

    def _upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def _metadata_for_record(
        self,
        record: dict[str, Any],
        provider: str | None,
        dim: int,
    ) -> dict[str, Any]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return {
            "user_id": str(record.get("user_id") or ""),
            "memory_type": str(record.get("memory_type") or ""),
            "category": str(record.get("ai_category") or record.get("category") or "General"),
            "memory_category": str(record.get("memory_category") or metadata.get("category") or ""),
            "source": str(record.get("source") or ""),
            "embedding_provider": str(provider or ""),
            "embedding_dim": int(dim),
            "updated_at": str(record.get("updated_at") or ""),
        }

    def _clear_collection(self) -> None:
        try:
            existing = self._collection.get()
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        except Exception as exc:
            logger.debug("Chroma collection clear ignored: %s", exc)

    def _fallback_search(self, **kwargs) -> list[dict[str, Any]]:
        reason = kwargs.pop("reason", "fallback")
        if self.fallback_retriever is None:
            self.last_warnings.append(reason)
            self.last_debug["candidate_count"] = 0
            return []
        results = self.fallback_retriever.search(**kwargs)
        self.last_warnings.extend(getattr(self.fallback_retriever, "last_warnings", []))
        self.last_warnings.append(reason)
        self.last_debug = dict(getattr(self.fallback_retriever, "last_debug", {}))
        self.last_debug["vector_backend"] = f"fallback:{self.fallback_name}"
        self.last_debug["fallback_reason"] = reason
        return results

    def _fallback_rebuild(self) -> None:
        if self.fallback_retriever is not None:
            self.fallback_retriever.rebuild()

    def _fallback_add_or_update(self, memory_id: str) -> None:
        if self.fallback_retriever is not None:
            self.fallback_retriever.add_or_update_memory(memory_id)

    @property
    def fallback_name(self) -> str:
        return self.fallback_retriever.__class__.__name__ if self.fallback_retriever else "none"

    def diagnostics(self) -> dict[str, Any]:
        count = 0
        if self.available:
            try:
                count = int(self._collection.count())
            except Exception:
                count = 0
        return {
            "status": "ready" if self.available else "degraded",
            "persist_dir": str(self.persist_dir),
            "collection": self.collection_name,
            "count": count,
            "fallback": self.fallback_name,
            "warnings": self.last_warnings[-5:],
        }

    def close(self) -> None:
        try:
            system = getattr(self._client, "_system", None)
            stop = getattr(system, "stop", None)
            if callable(stop):
                stop()
        except Exception as exc:
            logger.debug("Chroma client close ignored: %s", exc)
        finally:
            self._collection = None
            self._client = None
            self._available = False

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

    def _keyword_score(self, query_tokens: set[str], memory: dict[str, Any]) -> float:
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
        memory: dict[str, Any],
        final_score: float,
        evidence_score: float,
        reason: str,
    ) -> dict[str, Any]:
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
