"""Compatibility wrapper for episodic memories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_manager import MemoryManager


class EpisodicMemory:
    def __init__(
        self,
        max_episodes: int = 100,
        storage_file: str | Path | None = None,
    ):
        storage_dir = Path(storage_file).parent if storage_file else None
        self.manager = MemoryManager(storage_dir=storage_dir, run_migrations=False)
        self.max_episodes = max_episodes

    def add_episode(
        self,
        user_id: str,
        summary: str,
        conversation: list[dict[str, Any]],
        tags: list[str] | None = None,
        episode_id: str | None = None,
    ) -> str:
        return self.manager.episodic.add_episode(
            user_id=user_id,
            summary=summary,
            conversation=conversation,
            tags=tags or [],
            episode_id=episode_id,
        )

    def get_recent_episodes(self, user_id: str, last_n: int = 5) -> list[dict[str, Any]]:
        return self.manager.episodic.get_recent_episodes(user_id, last_n)

    def get_all_summaries(self, user_id: str) -> list[str]:
        return self.manager.episodic.get_all_summaries(user_id)

    def get_all_episodes(self, user_id: str) -> list[dict[str, Any]]:
        return self.manager.episodic.get_all_episodes(user_id)

    def clear(self, user_id: str) -> None:
        self.manager.episodic.clear(user_id)

    def get_all_users(self) -> list[str]:
        users = {
            item["user_id"]
            for item in self.manager.storage.list_memories(memory_types=["episodic"])
        }
        return sorted(users)
