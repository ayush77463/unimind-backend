"""Compatibility wrapper for recent conversation messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_manager import MemoryManager


class ShortTermMemory:
    def __init__(
        self,
        max_messages: int = 20,
        storage_file: str | Path | None = None,
    ):
        storage_dir = Path(storage_file).parent if storage_file else None
        self.manager = MemoryManager(storage_dir=storage_dir, run_migrations=False)
        self.max_messages = max_messages

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

    def get_all_users(self) -> list[str]:
        rows = self.manager.storage.list_memories()
        users = {row["user_id"] for row in rows}
        return sorted(users)
