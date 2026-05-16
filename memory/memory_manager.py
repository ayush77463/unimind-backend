"""Custom long-term memory engine for UniMind.

This module intentionally avoids LangChain, LangGraph, agents, and graph
pipelines. The architecture is plain Python: validate input, store records,
embed text, retrieve with FAISS, rank memories, and build an explainable prompt.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from ..config import (
        EPISODIC_MAX,
        FAISS_IDS_PATH,
        FAISS_INDEX_PATH,
        LEGACY_EPISODIC_FILE,
        LEGACY_SEMANTIC_FILE,
        LEGACY_SHORT_TERM_FILE,
        SEMANTIC_MAX,
        SHORT_TERM_MAX,
        SQLITE_DB_PATH,
        STORAGE_DIR,
    )
    from ..services.llm_service import LLMService, LLMUnavailableError
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        EPISODIC_MAX,
        FAISS_IDS_PATH,
        FAISS_INDEX_PATH,
        LEGACY_EPISODIC_FILE,
        LEGACY_SEMANTIC_FILE,
        LEGACY_SHORT_TERM_FILE,
        SEMANTIC_MAX,
        SHORT_TERM_MAX,
        SQLITE_DB_PATH,
        STORAGE_DIR,
    )
    from services.llm_service import LLMService, LLMUnavailableError  # type: ignore

from .embedding_service import EmbeddingService
from .fact_extractor import ExtractedMemory, FactExtractor
from .retrieval import MemoryRetriever
from .scoring import decayed_importance, importance_for_category
from .storage import MemoryStorage, utc_now
from .summarizer import Summarizer

MEMORY_TYPES = {"episodic", "semantic", "preference"}
logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates storage, extraction, embedding, retrieval, and chat."""

    def __init__(
        self,
        *,
        storage_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        faiss_index_path: str | Path | None = None,
        faiss_ids_path: str | Path | None = None,
        run_migrations: bool = True,
        llm_service: LLMService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        base_dir = Path(storage_dir) if storage_dir else STORAGE_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        default_faiss_dir = base_dir / "faiss"
        default_faiss_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path) if db_path else (
            base_dir / "memory.db" if storage_dir else Path(SQLITE_DB_PATH)
        )
        self.storage = MemoryStorage(self.db_path)
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = MemoryRetriever(
            storage=self.storage,
            embedding_service=self.embedding_service,
            index_path=faiss_index_path
            or (default_faiss_dir / "memory.index" if storage_dir else FAISS_INDEX_PATH),
            ids_path=faiss_ids_path
            or (default_faiss_dir / "memory_ids.json" if storage_dir else FAISS_IDS_PATH),
        )
        self.fact_extractor = FactExtractor()
        self.summarizer = Summarizer()
        self.llm_service = llm_service or LLMService()

        self.short_term = _ShortTermAdapter(self)
        self.semantic = _SemanticAdapter(self)
        self.episodic = _EpisodicAdapter(self)

        if run_migrations:
            self._migrate_legacy_json(base_dir)

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.storage.add_message(
            user_id=user_id,
            role=role,
            content=content,
            session_id=session_id,
            metadata=metadata,
        )

    def add_fact(self, user_id: str, fact: str, category: str = "general") -> str:
        category = category.lower()
        memory_type = "preference" if category == "preference" else "semantic"
        memory_id, _ = self._store_memory_result(
            user_id=user_id,
            content=fact,
            memory_type=memory_type,
            importance=importance_for_category(category, memory_type),
            source="manual",
            metadata={"category": category},
        )
        return memory_id

    def store_memory(
        self,
        *,
        user_id: str,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.6,
        source: str = "manual",
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        memory_id, _ = self._store_memory_result(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            source=source,
            summary=summary,
            metadata=metadata,
            memory_id=memory_id,
            created_at=created_at,
        )
        return memory_id

    def retrieve_memories(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: Iterable[str] | None = None,
        debug: bool = False,
        min_evidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        self.apply_memory_decay(user_id=user_id)
        return self.retriever.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_types=memory_types,
            debug=debug,
            min_evidence=min_evidence,
        )

    def build_context(self, user_id: str, query: str, top_k: int = 5) -> str:
        return self.build_context_payload(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )["context"]

    def build_context_payload(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_chars: int = 2000,
        debug: bool = False,
    ) -> dict[str, Any]:
        # Contextual prompting injects only the most relevant long-term memory
        # so the Flutter chat prompt remains compact and predictable.
        max_chars = max(500, min(6000, int(max_chars)))
        facts = self.retrieve_memories(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_types=["semantic", "preference"],
            debug=debug,
            min_evidence=0.04,
        )
        fact_debug = dict(self.retriever.last_debug)
        fact_warnings = list(self.retriever.last_warnings)
        episodes = self.retrieve_memories(
            user_id=user_id,
            query=query,
            top_k=3,
            memory_types=["episodic"],
            debug=debug,
            min_evidence=0.04 if query.lower() != "general" else 0.0,
        )
        episode_debug = dict(self.retriever.last_debug)
        episode_warnings = list(self.retriever.last_warnings)
        parts: list[str] = []
        used_chars = 0

        if facts:
            fact_lines = []
            for memory in facts:
                label = memory.get("category") or memory.get("memory_type")
                line = f"  - ({label}) {memory['content']}"
                if used_chars + len(line) + 28 > max_chars:
                    break
                fact_lines.append(line)
                used_chars += len(line) + 1
            if fact_lines:
                parts.append("KNOWN FACTS ABOUT USER:\n" + "\n".join(fact_lines))
                used_chars += len("KNOWN FACTS ABOUT USER:\n")

        if episodes:
            episode_lines = []
            for episode in episodes:
                line = f"  - [{episode.get('updated_at', '')[:16].replace('T', ' ')}] {episode.get('summary', '')}"
                if used_chars + len(line) + 32 > max_chars:
                    break
                episode_lines.append(line)
                used_chars += len(line) + 1
            if episode_lines:
                parts.append("PAST CONVERSATION HISTORY:\n" + "\n".join(episode_lines))
                used_chars += len("PAST CONVERSATION HISTORY:\n")

        context = "\n\n".join(part for part in parts if part).strip()
        payload = {
            "success": True,
            "user_id": user_id,
            "query": query,
            "context": context[:max_chars],
            "context_length": min(len(context), max_chars),
            "warnings": [*fact_warnings, *episode_warnings],
        }
        if debug:
            payload["debug"] = {
                "facts": fact_debug,
                "episodes": episode_debug,
                "context_budget": {
                    "max_chars": max_chars,
                    "used_chars": min(len(context), max_chars),
                    "truncated": len(context) > max_chars,
                },
            }
        return payload

    def add_exchange(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        session_id: str | None = None,
        tags: list[str] | None = None,
        ai_enrich: bool = True,
    ) -> dict[str, Any]:
        exchange = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        for message in exchange:
            if self._is_memory_worthy_message(message):
                self.add_message(
                    user_id=user_id,
                    role=message["role"],
                    content=message["content"],
                    session_id=session_id,
                )

        facts_added = self._extract_and_store(
            user_id=user_id,
            messages=exchange,
            use_llm=ai_enrich,
        )
        episode_id = self._upsert_episode_from_messages(
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            use_llm=False,
        )
        return {
            "success": True,
            "episode_id": episode_id,
            "facts_added": facts_added,
            "short_term_messages": self.storage.get_message_count(user_id),
        }

    def save_episode(
        self,
        user_id: str,
        tags: list[str] | None = None,
        ai_enrich: bool = True,
        session_id: str | None = None,
        clear_short_term: bool = True,
    ) -> str:
        messages = self.storage.get_recent_messages(
            user_id=user_id,
            last_n=max(SHORT_TERM_MAX, 200),
            session_id=session_id,
        )
        eligible = [message for message in messages if self._is_memory_worthy_message(message)]
        if not eligible:
            return "no_messages_to_save"
        if not any(m["role"] == "user" for m in eligible) or not any(
            m["role"] == "assistant" for m in eligible
        ):
            if clear_short_term:
                self.storage.clear_messages(user_id, session_id=session_id)
            return "no_complete_exchange_to_save"

        if ai_enrich:
            self._extract_and_store(user_id=user_id, messages=eligible, use_llm=True)

        episode_id = self._upsert_episode_from_messages(
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            use_llm=ai_enrich,
        )
        if clear_short_term:
            self.storage.clear_messages(user_id, session_id=session_id)
        return episode_id

    def chat_with_memory(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.add_message(user_id, "user", user_message, session_id=session_id)
        facts_added = self._extract_and_store(
            user_id=user_id,
            messages=[{"role": "user", "content": user_message}],
            use_llm=True,
        )
        context = self.build_context(user_id=user_id, query=user_message)
        recent = self.storage.get_recent_messages(
            user_id=user_id,
            last_n=8,
            session_id=session_id,
        )
        prompt = self._build_chat_prompt(
            user_message=user_message,
            context=context,
            recent_messages=recent,
        )
        response = self.llm_service.generate_response(prompt)
        self.add_message(user_id, "assistant", response, session_id=session_id)
        episode_id = self._upsert_episode_from_messages(
            user_id=user_id,
            session_id=session_id,
            tags=["backend-chat"],
            use_llm=False,
        )
        return {
            "response": response,
            "memory_used": {
                "facts_added": facts_added,
                "episode_id": episode_id,
                "short_term_messages": self.storage.get_message_count(
                    user_id,
                    session_id=session_id,
                ),
                "memory_context_preview": context[:500] if context else "",
            },
        }

    def stream_chat_with_memory(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
    ):
        self.add_message(user_id, "user", user_message, session_id=session_id)
        facts_added = self._extract_and_store(
            user_id=user_id,
            messages=[{"role": "user", "content": user_message}],
            use_llm=True,
        )
        context = self.build_context(user_id=user_id, query=user_message)
        recent = self.storage.get_recent_messages(
            user_id=user_id,
            last_n=8,
            session_id=session_id,
        )
        prompt = self._build_chat_prompt(
            user_message=user_message,
            context=context,
            recent_messages=recent,
        )

        full_response: list[str] = []
        stream = getattr(self.llm_service, "generate_response_stream", None)
        try:
            chunks = stream(prompt) if callable(stream) else self._smooth_chunks(
                self.llm_service.generate_response(prompt)
            )
            for chunk in chunks:
                text = str(chunk)
                if not text:
                    continue
                full_response.append(text)
                yield {
                    "type": "delta",
                    "delta": text,
                }
        except GeneratorExit:
            logger.info("Client cancelled backend chat stream for user %s", user_id)
            raise

        final_text = "".join(full_response).strip()
        if final_text:
            self.add_message(user_id, "assistant", final_text, session_id=session_id)
            episode_id = self._upsert_episode_from_messages(
                user_id=user_id,
                session_id=session_id,
                tags=["backend-chat"],
                use_llm=False,
            )
        else:
            episode_id = ""

        yield {
            "type": "done",
            "done": True,
            "memory_used": {
                "facts_added": facts_added,
                "episode_id": episode_id,
                "short_term_messages": self.storage.get_message_count(
                    user_id,
                    session_id=session_id,
                ),
                "memory_context_preview": context[:500] if context else "",
            },
        }

    def list_all_memories(
        self,
        *,
        user_id: str,
        query: str = "",
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.apply_memory_decay(user_id=user_id)
        category = (category or "").strip().lower() or None
        if query.strip():
            candidates = self.retrieve_memories(
                user_id=user_id,
                query=query,
                top_k=min(20, max(1, limit + offset)),
                debug=False,
            )
        else:
            candidates = self.storage.list_memories(
                user_id=user_id,
                limit=max(1, min(100, limit)),
                offset=max(0, offset),
            )
        if category:
            candidates = [
                memory
                for memory in candidates
                if self._memory_bucket(memory) == category
                or str(memory.get("category", "")).lower() == category
            ]
        if query.strip():
            candidates = candidates[max(0, offset) : max(0, offset) + limit]
        return [self._memory_card(memory) for memory in candidates[:limit]]

    def delete_memory(self, memory_id: str) -> bool:
        existing = self.storage.get_memory(memory_id)
        if existing is None:
            return False
        self.storage.delete_memory(memory_id)
        self.retriever.rebuild()
        return True

    def pin_memory(self, memory_id: str, pinned: bool) -> dict[str, Any] | None:
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            return None
        metadata = dict(memory.get("metadata") or {})
        metadata["pinned"] = bool(pinned)
        self.storage.update_metadata(memory_id, metadata)
        updated = self.storage.get_memory(memory_id)
        return self._memory_card(updated) if updated else None

    def clear_all(self, user_id: str) -> None:
        self.storage.clear_user(user_id)
        self.retriever.rebuild()

    def clear_short_term(self, user_id: str) -> None:
        self.storage.clear_messages(user_id)

    def get_status(self, user_id: str) -> dict[str, Any]:
        self.apply_memory_decay(user_id=user_id)
        facts = self.semantic.get_all_facts(user_id)
        episodes = self.episodic.get_all_summaries(user_id)
        return {
            "user_id": user_id,
            "short_term_messages": self.storage.get_message_count(user_id),
            "total_facts": len(facts),
            "total_episodes": len(episodes),
            "recent_episode_summaries": episodes[-5:],
        }

    def _store_memory_result(
        self,
        *,
        user_id: str,
        content: str,
        memory_type: str,
        importance: float,
        source: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> tuple[str, bool]:
        memory_type = memory_type.lower().strip()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory_type: {memory_type}")
        clean_content = " ".join((content or "").split())
        if len(clean_content) < 3:
            raise ValueError("Memory content is too short")

        existing = None
        if memory_id:
            existing = self.storage.get_memory(memory_id)
        else:
            existing = self.storage.find_duplicate_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=clean_content,
            )
        if existing and not memory_id:
            return existing["id"], False

        final_id = memory_id or f"mem_{uuid.uuid4().hex}"
        try:
            embedded = self.embedding_service.embed(
                clean_content,
                task_type="retrieval_document",
            )
        except Exception as exc:
            logger.exception("Memory embedding failed")
            raise RuntimeError(f"Memory embedding failed: {exc}") from exc
        updated_at = utc_now()
        if created_at and existing is None:
            updated_at = created_at
        replaced = self.storage.upsert_memory(
            memory_id=final_id,
            user_id=user_id,
            memory_type=memory_type,
            content=clean_content,
            summary=summary or clean_content,
            importance=importance,
            source=source,
            metadata=metadata or {},
            embedding=embedded.vector,
            embedding_provider=embedded.provider,
            created_at=created_at,
            updated_at=updated_at,
        )
        self.retriever.add_or_update_memory(final_id)
        return final_id, not replaced

    def _memory_card(self, memory: dict[str, Any]) -> dict[str, Any]:
        metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
        return {
            "id": memory["id"],
            "user_id": memory.get("user_id"),
            "content": memory.get("content", ""),
            "summary": memory.get("summary") or memory.get("content", ""),
            "memory_type": memory.get("memory_type"),
            "category": memory.get("category") or metadata.get("category") or "general",
            "bucket": self._memory_bucket(memory),
            "importance": memory.get("importance", 0.0),
            "pinned": bool(metadata.get("pinned")),
            "source": memory.get("source"),
            "created_at": memory.get("created_at"),
            "updated_at": memory.get("updated_at"),
            "last_accessed_at": memory.get("last_accessed_at"),
            "relevance_score": memory.get("relevance_score"),
        }

    def _memory_bucket(self, memory: dict[str, Any]) -> str:
        memory_type = str(memory.get("memory_type", "")).lower()
        category = str(memory.get("category", "")).lower()
        if memory_type == "episodic":
            return "conversations"
        if category == "preference" or memory_type == "preference":
            return "preferences"
        if category in {"goal", "goals"}:
            return "goals"
        if category in {"personal", "identity", "location", "home", "study", "work"}:
            return "personal_facts"
        return "semantic"

    def _smooth_chunks(self, text: str, chunk_size: int = 28):
        words = text.split(" ")
        buffer: list[str] = []
        size = 0
        for word in words:
            buffer.append(word)
            size += len(word) + 1
            if size >= chunk_size:
                yield " ".join(buffer) + " "
                buffer = []
                size = 0
        if buffer:
            yield " ".join(buffer)

    def _extract_and_store(
        self,
        *,
        user_id: str,
        messages: list[dict[str, Any]],
        use_llm: bool,
    ) -> int:
        extracted: list[ExtractedMemory] = self.fact_extractor.extract(messages)
        if use_llm:
            try:
                for item in self.llm_service.extract_memories(messages):
                    category = item["category"]
                    extracted.append(
                        ExtractedMemory(
                            content=item["content"],
                            memory_type="preference"
                            if category == "preference"
                            else "semantic",
                            category=category,
                            importance=item["importance"],
                            source="llm_extractor",
                        )
                    )
            except Exception as exc:
                logger.warning("LLM extraction failed; continuing with local memories: %s", exc)

        extracted.extend(self._extract_recurring_topics(user_id=user_id, messages=messages))

        stored = 0
        for memory in extracted:
            try:
                _, created = self._store_memory_result(
                    user_id=user_id,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    importance=memory.importance,
                    source=memory.source,
                    metadata={"category": memory.category},
                )
                if created:
                    stored += 1
            except Exception as exc:
                logger.warning("Skipping extracted memory because storage failed: %s", exc)
        return stored

    def _upsert_episode_from_messages(
        self,
        *,
        user_id: str,
        session_id: str | None,
        tags: list[str],
        use_llm: bool,
    ) -> str:
        messages = self.storage.get_recent_messages(
            user_id=user_id,
            last_n=max(SHORT_TERM_MAX, 200),
            session_id=session_id,
        )
        eligible = [message for message in messages if self._is_memory_worthy_message(message)]
        if not eligible:
            return ""

        summary = None
        if use_llm:
            try:
                summary = self.llm_service.summarize(eligible)
            except Exception as exc:
                logger.warning("LLM summarization failed; using deterministic summary: %s", exc)
        summary = summary or self.summarizer.summarize(eligible)
        if self.summarizer.is_trivial(summary):
            return "trivial_conversation"

        episode_id = self._episode_id_for_session(session_id) or f"ep_{uuid.uuid4().hex}"
        # Episodic memory is the conversation timeline: a short summary plus a
        # small inspectable slice of recent messages.
        self._store_memory_result(
            user_id=user_id,
            memory_id=episode_id,
            content=summary,
            summary=summary,
            memory_type="episodic",
            importance=importance_for_category("episodic", "episodic"),
            source="conversation_summary",
            metadata={
                "tags": tags,
                "message_count": len(eligible),
                "conversation": [
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "timestamp": message.get("timestamp") or message.get("created_at"),
                    }
                    for message in eligible[-12:]
                ],
            },
        )
        return episode_id

    def _build_chat_prompt(
        self,
        *,
        user_message: str,
        context: str,
        recent_messages: list[dict[str, Any]],
    ) -> str:
        # The prompt never exposes implementation details; memories are used as
        # personalization hints for the current answer.
        history = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in recent_messages[-8:]
        )
        return f"""
You are UniMind, a helpful personalized AI assistant.
Use the memory context naturally. Do not mention internal memory systems.

MEMORY CONTEXT:
{context or "No relevant memories yet."}

RECENT CONVERSATION:
{history}

CURRENT USER MESSAGE:
{user_message}

Answer clearly and helpfully:
""".strip()

    def _migrate_legacy_json(self, base_dir: Path) -> None:
        self._migrate_short_term(base_dir / "short_term_memory.json")
        self._migrate_semantic(base_dir / "semantic_memory.json")
        self._migrate_episodic(base_dir / "episodic_memory.json")

        if base_dir == STORAGE_DIR:
            self._migrate_short_term(LEGACY_SHORT_TERM_FILE)
            self._migrate_semantic(LEGACY_SEMANTIC_FILE)
            self._migrate_episodic(LEGACY_EPISODIC_FILE)

    def _migrate_short_term(self, path: Path) -> None:
        name = f"legacy_short_term:{path.resolve()}"
        if self.storage.has_migration(name) or not path.exists():
            return
        data = self._load_json(path)
        if isinstance(data, dict):
            for user_id, messages in data.items():
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role")
                    content = str(message.get("content", "")).strip()
                    if role in {"user", "assistant"} and content:
                        self.storage.add_message(
                            user_id=str(user_id),
                            role=role,
                            content=content,
                            created_at=message.get("timestamp"),
                        )
        self.storage.mark_migration(name)

    def _migrate_semantic(self, path: Path) -> None:
        name = f"legacy_semantic:{path.resolve()}"
        if self.storage.has_migration(name) or not path.exists():
            return
        data = self._load_json(path)
        if isinstance(data, dict):
            for user_id, memories in data.items():
                if not isinstance(memories, list):
                    continue
                for item in memories:
                    if not isinstance(item, dict):
                        continue
                    content = str(item.get("fact") or item.get("content") or "").strip()
                    if not content:
                        continue
                    category = str(item.get("category", "general")).lower()
                    memory_type = "preference" if category == "preference" else "semantic"
                    self._store_memory_result(
                        user_id=str(user_id),
                        memory_id=item.get("id"),
                        content=content,
                        memory_type=memory_type,
                        importance=float(item.get("importance", 0.65) or 0.65),
                        source="legacy_semantic_json",
                        metadata={"category": category, "legacy": True},
                        created_at=item.get("timestamp"),
                    )
        self.storage.mark_migration(name)

    def _migrate_episodic(self, path: Path) -> None:
        name = f"legacy_episodic:{path.resolve()}"
        if self.storage.has_migration(name) or not path.exists():
            return
        data = self._load_json(path)
        if isinstance(data, dict):
            for user_id, episodes in data.items():
                if not isinstance(episodes, list):
                    continue
                for episode in episodes:
                    if not isinstance(episode, dict):
                        continue
                    summary = str(episode.get("summary", "")).strip()
                    if not summary:
                        continue
                    self._store_memory_result(
                        user_id=str(user_id),
                        memory_id=episode.get("id"),
                        content=summary,
                        summary=summary,
                        memory_type="episodic",
                        importance=0.55,
                        source="legacy_episodic_json",
                        metadata={
                            "tags": episode.get("tags") or [],
                            "message_count": episode.get("message_count", 0),
                            "conversation": episode.get("conversation") or [],
                            "legacy": True,
                        },
                        created_at=episode.get("timestamp"),
                    )
        self.storage.mark_migration(name)

    def _load_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not migrate %s: %s", path, exc)
            return None

    def apply_memory_decay(self, user_id: str | None = None) -> int:
        changed = 0
        now = datetime.now(timezone.utc)
        for memory in self.storage.list_memories(user_id=user_id):
            new_importance = decayed_importance(memory, now=now)
            if abs(new_importance - float(memory["importance"])) >= 0.01:
                self.storage.update_importance(memory["id"], new_importance)
                changed += 1
        return changed

    def _extract_recurring_topics(
        self,
        *,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> list[ExtractedMemory]:
        recent = self.storage.get_recent_messages(user_id=user_id, last_n=80)
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for message in [*recent, *messages]:
            key = (str(message.get("role", "")), str(message.get("content", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            combined.append(message)
        topics = self.fact_extractor.extract_topics(combined)
        counts = Counter(topic.lower() for topic in topics)
        memories: list[ExtractedMemory] = []
        for topic, count in counts.items():
            if count < 2:
                continue
            label = topic[:1].upper() + topic[1:]
            memories.append(
                ExtractedMemory(
                    content=f"User has recurring interest in {label}",
                    memory_type="semantic",
                    category="recurring_topic",
                    importance=importance_for_category("recurring_topic"),
                    source="topic_counter",
                )
            )
        return memories

    def _episode_id_for_session(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id.strip())
        return f"session_{safe[:80]}" if safe else None

    def _is_memory_worthy_message(self, message: dict[str, Any]) -> bool:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            return False
        lower = content.lower()
        blocked = [
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "error getting response",
            "i encountered an error",
            "please check your gemini api key",
        ]
        return not any(fragment in lower for fragment in blocked)


class _ShortTermAdapter:
    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def add_message(self, user_id: str, role: str, content: str) -> None:
        self.manager.add_message(user_id, role, content)

    def get_recent_messages(self, user_id: str, last_n: int = 10) -> list[dict[str, Any]]:
        return self.manager.storage.get_recent_messages(user_id, last_n=last_n)

    def get_full_history(self, user_id: str) -> list[dict[str, Any]]:
        return self.manager.storage.get_recent_messages(user_id, last_n=1000)

    def get_message_count(self, user_id: str) -> int:
        return self.manager.storage.get_message_count(user_id)

    def clear(self, user_id: str) -> None:
        self.manager.storage.clear_messages(user_id)


class _SemanticAdapter:
    """Semantic memory stores durable user facts for long-term personalization."""

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def add_fact(self, user_id: str, fact: str, category: str = "general") -> str:
        return self.manager.add_fact(user_id, fact, category)

    def search_relevant_facts(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.manager.retrieve_memories(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_types=["semantic", "preference"],
        )

    def get_all_facts(self, user_id: str) -> list[dict[str, Any]]:
        self.manager.apply_memory_decay(user_id=user_id)
        return self.manager.storage.list_memories(
            user_id=user_id,
            memory_types=["semantic", "preference"],
        )

    def clear_all_facts(self, user_id: str) -> None:
        self.manager.storage.clear_memories(
            user_id,
            memory_types=["semantic", "preference"],
        )
        self.manager.retriever.rebuild()

    def get_total_count(self) -> int:
        return len(
            self.manager.storage.list_memories(
                memory_types=["semantic", "preference"],
            )
        )


class _EpisodicAdapter:
    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def add_episode(
        self,
        user_id: str,
        summary: str,
        conversation: list[dict[str, Any]],
        tags: list[str] | None = None,
        episode_id: str | None = None,
    ) -> str:
        memory_id = episode_id or f"ep_{uuid.uuid4().hex}"
        return self.manager.store_memory(
            user_id=user_id,
            memory_id=memory_id,
            content=summary,
            summary=summary,
            memory_type="episodic",
            importance=0.55,
            source="manual_episode",
            metadata={
                "tags": tags or [],
                "conversation": conversation,
                "message_count": len(conversation),
            },
        )

    def get_recent_episodes(self, user_id: str, last_n: int = 5) -> list[dict[str, Any]]:
        self.manager.apply_memory_decay(user_id=user_id)
        episodes = self.manager.storage.list_memories(
            user_id=user_id,
            memory_types=["episodic"],
        )
        episodes.sort(key=lambda item: item["updated_at"])
        return [self._episode_shape(item) for item in episodes[-last_n:]]

    def get_all_summaries(self, user_id: str) -> list[str]:
        return [
            f"[{episode['date']}] {episode['summary']}"
            for episode in self.get_recent_episodes(user_id, last_n=EPISODIC_MAX)
        ]

    def get_all_episodes(self, user_id: str) -> list[dict[str, Any]]:
        return self.get_recent_episodes(user_id, last_n=EPISODIC_MAX)

    def clear(self, user_id: str) -> None:
        self.manager.storage.clear_memories(user_id, memory_types=["episodic"])
        self.manager.retriever.rebuild()

    def _episode_shape(self, item: dict[str, Any]) -> dict[str, Any]:
        metadata = item.get("metadata", {})
        return {
            "id": item["id"],
            "summary": item["summary"],
            "conversation": metadata.get("conversation") or [],
            "tags": metadata.get("tags") or [],
            "timestamp": item["created_at"],
            "updated_at": item["updated_at"],
            "date": item["updated_at"][:16].replace("T", " "),
            "message_count": metadata.get("message_count", 0),
        }


__all__ = ["LLMUnavailableError", "MemoryManager"]
