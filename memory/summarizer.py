"""Conversation summarization with a deterministic fallback."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


class Summarizer:
    def summarize(self, messages: Iterable[dict]) -> str:
        # Episodic memory stores a compact record of a conversation, not every
        # token. The recent message slice is kept separately for inspection.
        messages = list(messages)
        user_messages = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        if not user_messages:
            return "General conversation"

        first = self._clip(user_messages[0], 120)
        if len(user_messages) == 1:
            return f"Conversation about: {first}"

        latest = self._clip(user_messages[-1], 80)
        if latest == first:
            return f"Conversation about: {first}"

        if len(messages) >= 8:
            topics = self._topics(user_messages)
            if topics:
                topic_text = ", ".join(topics[:3])
                return f"Long conversation covering {topic_text}; latest topic: {latest}"

        return f"Conversation about: {first}; latest topic: {latest}"

    def is_trivial(self, summary: str) -> bool:
        return summary.strip().upper() == "TRIVIAL"

    def _clip(self, text: str, limit: int) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return f"{clean[: max(0, limit - 3)].rstrip()}..."

    def _topics(self, messages: list[str]) -> list[str]:
        stop = {
            "about",
            "answer",
            "correctly",
            "explain",
            "hello",
            "help",
            "please",
            "tell",
            "that",
            "this",
            "what",
            "which",
            "with",
            "you",
        }
        words: list[str] = []
        for message in messages:
            words.extend(
                word
                for word in re.findall(r"[a-z0-9]{4,}", message.lower())
                if word not in stop
            )
        return [word for word, _ in Counter(words).most_common(4)]
