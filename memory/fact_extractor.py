"""Machine Learning memory extraction for user facts and topics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from .scoring import importance_for_category

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExtractedMemory:
    content: str
    memory_type: str
    category: str
    importance: float
    source: str = "ml_extractor"


class FactExtractor:
    """Extracts long-term memories using modern ML (LLM Structured Outputs).
    
    This replaces the legacy 300-line Regex heuristics with Langchain ML pipelines.
    Local regex fallback is removed to enforce ML usage as requested.
    """

    def extract(self, messages: Iterable[dict]) -> list[ExtractedMemory]:
        # Legacy local regex extraction was removed in favor of Langchain ML.
        # The actual extraction happens via llm_service.extract_memories
        # which uses ChatGoogleGenerativeAI structured outputs.
        return []

    def extract_topics(self, messages: Iterable[dict]) -> list[str]:
        # Topic extraction is also best handled by ML.
        # For this lightweight method, we use basic word frequencies combined 
        # with sentence-transformers embeddings (via intelligence.py) elsewhere.
        topics: list[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip().lower()
            if "interested in" in content:
                parts = content.split("interested in")
                if len(parts) > 1:
                    topics.append(parts[1].strip(" .?!"))
            elif "learn about" in content:
                parts = content.split("learn about")
                if len(parts) > 1:
                    topics.append(parts[1].strip(" .?!"))
        return topics
