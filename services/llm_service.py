"""Minimal Gemini LLM client used by the backend /chat demo endpoint."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Iterable

try:
    from ..config import GEMINI_API_KEY, GEMINI_LLM_MODEL, gemini_enabled
except ImportError:  # pragma: no cover
    from config import GEMINI_API_KEY, GEMINI_LLM_MODEL, gemini_enabled  # type: ignore


class LLMUnavailableError(RuntimeError):
    """Raised when backend /chat is requested without a configured LLM."""


logger = logging.getLogger(__name__)


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
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            logger.warning("Gemini LLM setup failed: %s", exc)
            self._model = None

    def generate_response(self, prompt: str) -> str:
        if self._model is None:
            raise LLMUnavailableError("Backend Gemini LLM is not configured")
        response = self._run_with_timeout(
            lambda: self._model.generate_content(prompt),
            "Gemini response timed out",
        )
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text

    def generate_response_stream(self, prompt: str):
        if self._model is None:
            raise LLMUnavailableError("Backend Gemini LLM is not configured")
        try:
            response = self._model.generate_content(prompt, stream=True)
            emitted = False
            for chunk in response:
                text = getattr(chunk, "text", "") or ""
                if text:
                    emitted = True
                    yield text
            if not emitted:
                fallback = self.generate_response(prompt)
                for part in self._smooth_chunks(fallback):
                    yield part
        except TypeError:
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
Return only valid JSON as an array. Each item must have:
- content: one sentence
- category: personal, preference, skill, goal, location, home, study, work,
  recurring_topic, or general
- importance: number from 0.0 to 1.0

Return [] if there are no durable memories.

Conversation:
{conversation}
"""
        try:
            text = self.generate_response(prompt)
            decoded = json.loads(self._strip_json_fence(text))
            if not isinstance(decoded, list):
                return []
            valid = []
            for item in decoded[:5]:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if len(content) < 8:
                    continue
                category = str(item.get("category", "general")).strip().lower()
                if category not in {
                    "personal",
                    "preference",
                    "skill",
                    "goal",
                    "location",
                    "home",
                    "study",
                    "work",
                    "recurring_topic",
                    "general",
                }:
                    category = "general"
                try:
                    importance = float(item.get("importance", 0.6))
                except (TypeError, ValueError):
                    importance = 0.6
                valid.append(
                    {
                        "content": content,
                        "category": category,
                        "importance": max(0.0, min(1.0, importance)),
                    }
                )
            return valid
        except Exception as exc:
            logger.warning("LLM memory extraction failed; using local extraction only: %s", exc)
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

    def _strip_json_fence(self, value: str) -> str:
        clean = value.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean = "\n".join(lines).strip()
        return clean

    def _run_with_timeout(self, func, timeout_message: str):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"{timeout_message} after {self.timeout_seconds:.1f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

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
