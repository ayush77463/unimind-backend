"""Compatibility wrapper for semantic memories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_manager import MemoryManager


class SemanticMemory:
    def __init__(
        self,
        gemini_api_key: str | None = None,
        storage_file: str | Path | None = None,
        enable_embeddings: bool = True,
    ):
        storage_dir = Path(storage_file).parent if storage_file else None
        self.manager = MemoryManager(storage_dir=storage_dir, run_migrations=False)

    def add_fact(self, user_id: str, fact: str, category: str = "general") -> str:
        return self.manager.add_fact(user_id, fact, category)

    def search_relevant_facts(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.manager.semantic.search_relevant_facts(user_id, query, top_k)

    def get_all_facts(self, user_id: str) -> list[dict[str, Any]]:
        return self.manager.semantic.get_all_facts(user_id)

    def delete_fact(self, fact_id: str) -> None:
        self.manager.storage.delete_memory(fact_id)
        self.manager.retriever.rebuild()

    def clear_all_facts(self, user_id: str) -> None:
        self.manager.semantic.clear_all_facts(user_id)

    def get_total_count(self) -> int:
        return self.manager.semantic.get_total_count()
