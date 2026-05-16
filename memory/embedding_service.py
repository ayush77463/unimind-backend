"""Embedding generation without LangChain abstractions."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import numpy as np

try:
    from ..config import (
        GEMINI_API_KEY,
        GEMINI_EMBEDDING_MODEL,
        HASH_EMBEDDING_DIM,
        gemini_enabled,
    )
except ImportError:  # pragma: no cover - supports running from backend folder
    from config import (  # type: ignore
        GEMINI_API_KEY,
        GEMINI_EMBEDDING_MODEL,
        HASH_EMBEDDING_DIM,
        gemini_enabled,
    )


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    provider: str


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embeds text with Gemini when possible and a local fallback otherwise."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        fallback_dim: int = HASH_EMBEDDING_DIM,
    ):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self._gemini = None
        self._gemini_failed = False
        self._cache: dict[tuple[str, str], EmbeddingResult] = {}
        self.timeout_seconds = float(os.getenv("UNIMIND_API_TIMEOUT_SECONDS", "12"))
        self._setup_gemini()

    @property
    def provider(self) -> str:
        return "gemini" if self._gemini is not None and not self._gemini_failed else "local_hash"

    def _setup_gemini(self) -> None:
        if os.getenv("UNIMIND_DISABLE_GEMINI") == "1" or not gemini_enabled():
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._gemini = genai
        except Exception as exc:
            logger.warning("Gemini embedding setup failed; using local embeddings: %s", exc)
            self._gemini = None

    def embed(self, text: str, task_type: str = "retrieval_document") -> EmbeddingResult:
        clean_text = " ".join((text or "").split())
        if not clean_text:
            clean_text = "empty"
        cache_key = (task_type, clean_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._gemini is not None and not self._gemini_failed:
            try:
                result = self._embed_with_gemini(clean_text, task_type)
                self._cache[cache_key] = result
                return result
            except Exception as exc:
                self._gemini_failed = True
                logger.warning("Gemini embedding failed; using local embeddings: %s", exc)

        result = EmbeddingResult(
            vector=self._local_hash_embedding(clean_text),
            provider="local_hash",
        )
        self._cache[cache_key] = result
        return result

    def _embed_with_gemini(self, text: str, task_type: str) -> EmbeddingResult:
        assert self._gemini is not None
        try:
            response = self._run_with_timeout(
                lambda: self._gemini.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type,
                ),
                "Gemini embedding timed out",
            )
        except TypeError:
            response = self._run_with_timeout(
                lambda: self._gemini.embed_content(
                    model=self.model_name,
                    content=text,
                ),
                "Gemini embedding timed out",
            )
        vector = response.get("embedding")
        if not vector:
            raise RuntimeError("Gemini returned an empty embedding")
        return EmbeddingResult(
            vector=self._normalize(np.asarray(vector, dtype=np.float32)),
            provider=f"gemini:{self.model_name}",
        )

    def _local_hash_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dim, dtype=np.float32)
        tokens = self._tokens(text)
        if not tokens:
            tokens = ["empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + math.log1p(len(token)) / 4.0
            vector[index] += sign * weight

        return self._normalize(vector)

    def _tokens(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        tokens: list[str] = []
        for token in raw_tokens:
            tokens.append(token)
            if len(token) > 3 and token.endswith("s"):
                tokens.append(token[:-1])
            if len(token) > 5:
                tokens.append(token[:5])
        return tokens

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return (vector / norm).astype(np.float32)

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
