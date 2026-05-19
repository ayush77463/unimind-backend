"""Minimal Langchain LLM client used by the backend /chat demo endpoint."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, AsyncGenerator

from pydantic import BaseModel, Field

try:
    from ..config import GEMINI_API_KEY, GEMINI_LLM_MODEL, gemini_enabled
except ImportError:  # pragma: no cover
    from config import GEMINI_API_KEY, GEMINI_LLM_MODEL, gemini_enabled  # type: ignore


class LLMUnavailableError(RuntimeError):
    """Raised when backend /chat is requested without a configured LLM."""


logger = logging.getLogger(__name__)


class ExtractedMemoryItem(BaseModel):
    content: str = Field(description="One concise sentence representing a durable fact about the user.")
    category: str = Field(description="Must be one of: personal, preference, skill, goal, location, home, study, work, recurring_topic, general")
    importance: float = Field(description="Importance score from 0.0 to 1.0", ge=0.0, le=1.0)


class MemoryExtractionList(BaseModel):
    memories: list[ExtractedMemoryItem] = Field(default_factory=list, description="List of durable facts extracted.")


class LLMService:
    def __init__(self, api_key: str | None = None, model_name: str = GEMINI_LLM_MODEL):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        self.model_name = model_name
        self._model = None
        self.timeout_seconds = float(os.getenv("UNIMIND_API_TIMEOUT_SECONDS", "20"))
        self._setup()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _setup(self) -> None:
        if os.getenv("UNIMIND_DISABLE_GEMINI") == "1" or not gemini_enabled():
            return
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            self._model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.7,
                max_retries=2,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Langchain LLM setup failed: %s", exc)
            self._model = None

    def generate_response(self, prompt: str) -> str:
        if self._model is None:
            raise LLMUnavailableError("Backend Gemini LLM is not configured")
        
        try:
            response = self._model.invoke(prompt)
            text = str(response.content).strip()
            if not text:
                raise RuntimeError("LLM returned an empty response")
            return text
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}")

    async def agenerate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        if self._model is None:
            raise LLMUnavailableError("Backend Gemini LLM is not configured")
        
        try:
            async for chunk in self._model.astream(prompt):
                if chunk.content:
                    yield str(chunk.content)
        except Exception as exc:
            logger.warning("Async streaming failed, falling back to sync. Error: %s", exc)
            fallback = self.generate_response(prompt)
            for part in self._smooth_chunks(fallback):
                yield part

    def extract_memories(self, messages: Iterable[dict]) -> list[dict]:
        if self._model is None:
            return []
        conversation = "\n".join(
            f"{message.get('role', '').upper()}: {message.get('content', '')}"
            for message in messages
        )
        prompt = f"""
Extract durable memories about the user from this conversation.
Only extract clear facts about the user's life, preferences, skills, goals, locations, or identity.

Conversation:
{conversation}
"""
        try:
            structured_llm = self._model.with_structured_output(MemoryExtractionList)
            result = structured_llm.invoke(prompt)
            
            valid = []
            for item in result.memories:
                content = item.content.strip()
                if len(content) < 8:
                    continue
                valid.append({
                    "content": content,
                    "category": item.category,
                    "importance": item.importance,
                })
            return valid
        except Exception as exc:
            logger.warning("Langchain LLM memory extraction failed; using local extraction only: %s", exc)
            return []

    def summarize(self, messages: Iterable[dict]) -> str | None:
        if self._model is None:
            return None
        conversation = "\n".join(
            f"{message.get('role', '').upper()}: {message.get('content', '')}"
            for message in list(messages)[-10:]
        )
        prompt = f"""
Summarize this conversation in one concise sentence.
If it is only trivial greeting or no meaningful exchange, return TRIVIAL.

Conversation:
{conversation}

Summary:
"""
        try:
            return self.generate_response(prompt).strip()
        except Exception as exc:
            logger.warning("LLM summarization failed; using local summary: %s", exc)
            return None

    def _smooth_chunks(self, text: str, chunk_size: int = 28):
        words = text.split(" ")
        current: list[str] = []
        current_len = 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                yield " ".join(current) + " "
                current = []
                current_len = 0
        if current:
            yield " ".join(current)
