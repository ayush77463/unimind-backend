"""Embedding generation for Gemini, optional local transformers, and fallback hashes."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import numpy as np

try:
    from ..config import (
        GEMINI_API_KEY,
        GEMINI_EMBEDDING_MODEL,
        HASH_EMBEDDING_DIM,
        UNIMIND_ALLOW_MODEL_DOWNLOADS,
        UNIMIND_EMBED_CACHE_SIZE,
        UNIMIND_EMBEDDING_PROVIDER,
        UNIMIND_ENABLE_LOCAL_MODELS,
        UNIMIND_SENTENCE_MODEL,
        gemini_enabled,
    )
except ImportError:  # pragma: no cover - supports running from backend folder
    from config import (  # type: ignore
        GEMINI_API_KEY,
        GEMINI_EMBEDDING_MODEL,
        HASH_EMBEDDING_DIM,
        UNIMIND_ALLOW_MODEL_DOWNLOADS,
        UNIMIND_EMBED_CACHE_SIZE,
        UNIMIND_EMBEDDING_PROVIDER,
        UNIMIND_ENABLE_LOCAL_MODELS,
        UNIMIND_SENTENCE_MODEL,
        gemini_enabled,
    )


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    provider: str


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embeds text with Gemini, optional SentenceTransformers, and a hash fallback."""

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
        self._sentence_model = None
        self._sentence_model_failed = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self.embedding_provider_preference = UNIMIND_EMBEDDING_PROVIDER
        self.enable_local_models = bool(UNIMIND_ENABLE_LOCAL_MODELS)
        self.allow_model_downloads = bool(UNIMIND_ALLOW_MODEL_DOWNLOADS)
        self.sentence_model_name = UNIMIND_SENTENCE_MODEL
        self.cache_size = max(0, int(UNIMIND_EMBED_CACHE_SIZE))
        self._cache: OrderedDict[tuple[str, str], EmbeddingResult] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._sentence_lock = threading.Lock()
        self.timeout_seconds = float(os.getenv("UNIMIND_API_TIMEOUT_SECONDS", "12"))
        self._setup_gemini()

    @property
    def provider(self) -> str:
        if self._should_use_gemini():
            return "gemini"
        if self._sentence_model is not None:
            return f"sentence_transformer:{self.sentence_model_name}"
        if self._should_use_sentence_transformer():
            return "sentence_transformer:lazy"
        return "local_hash"

    @property
    def local_models_enabled(self) -> bool:
        return self.enable_local_models

    def _setup_gemini(self) -> None:
        if (
            self.embedding_provider_preference == "sentence_transformer"
            or os.getenv("UNIMIND_DISABLE_GEMINI") == "1"
            or not gemini_enabled()
        ):
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._gemini = genai
        except Exception as exc:
            logger.warning("Gemini embedding setup failed; using local embeddings: %s", exc)
            self._gemini = None

    def _should_use_gemini(self) -> bool:
        return (
            self.embedding_provider_preference in {"auto", "gemini"}
            and self._gemini is not None
            and self._consecutive_failures < self._max_consecutive_failures
        )

    def _should_use_sentence_transformer(self) -> bool:
        return (
            self.enable_local_models
            and self.embedding_provider_preference in {"auto", "sentence_transformer"}
            and not self._sentence_model_failed
        )

    def embed(self, text: str, task_type: str = "retrieval_document") -> EmbeddingResult:
        start = time.time()
        clean_text = " ".join((text or "").split())
        if not clean_text:
            clean_text = "empty"
        cache_key = (task_type, clean_text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(
                "Embedding cache hit provider=%s task=%s chars=%d latency_ms=%.1f",
                cached.provider, task_type, len(clean_text), (time.time() - start) * 1000,
            )
            return cached

        if self._should_use_gemini():
            try:
                result = self._embed_with_gemini(clean_text, task_type)
                self._consecutive_failures = 0  # Reset on success
                self._cache_set(cache_key, result)
                logger.info(
                    "Embedding generated provider=%s task=%s chars=%d latency_ms=%.1f",
                    result.provider, task_type, len(clean_text), (time.time() - start) * 1000,
                )
                return result
            except Exception as exc:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.error(
                        "Gemini embedding failed %d times consecutively; using local fallback until restart: %s",
                        self._consecutive_failures, exc,
                    )
                else:
                    logger.warning(
                        "Gemini embedding failed (attempt %d/%d); using local fallback for this call: %s",
                        self._consecutive_failures, self._max_consecutive_failures, exc,
                    )

        if self._should_use_sentence_transformer():
            try:
                result = self._embed_with_sentence_transformer(clean_text)
                self._cache_set(cache_key, result)
                logger.info(
                    "Embedding generated provider=%s task=%s chars=%d latency_ms=%.1f",
                    result.provider, task_type, len(clean_text), (time.time() - start) * 1000,
                )
                return result
            except Exception as exc:
                self._sentence_model_failed = True
                logger.warning(
                    "SentenceTransformer embedding unavailable; using local fallback: %s",
                    exc,
                )

        result = EmbeddingResult(
            vector=self._local_hash_embedding(clean_text),
            provider="local_hash",
        )
        self._cache_set(cache_key, result)
        logger.debug(
            "Embedding generated provider=%s task=%s chars=%d latency_ms=%.1f",
            result.provider, task_type, len(clean_text), (time.time() - start) * 1000,
        )
        return result

    def _cache_get(self, key: tuple[str, str]) -> EmbeddingResult | None:
        if self.cache_size <= 0:
            return None
        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

    def _cache_set(self, key: tuple[str, str], result: EmbeddingResult) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def reset_failures(self) -> None:
        """Reset the failure counter to re-enable Gemini embeddings."""
        self._consecutive_failures = 0
        logger.info("Embedding failure counter reset; Gemini embeddings re-enabled")

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

    def _embed_with_sentence_transformer(self, text: str) -> EmbeddingResult:
        model = self._get_sentence_model()
        vector = self._run_with_timeout(
            lambda: model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            "SentenceTransformer embedding timed out",
        )
        return EmbeddingResult(
            vector=self._normalize(np.asarray(vector, dtype=np.float32)),
            provider=f"sentence_transformer:{self.sentence_model_name}",
        )

    def _get_sentence_model(self):
        if self._sentence_model is not None:
            return self._sentence_model
        with self._sentence_lock:
            if self._sentence_model is not None:
                return self._sentence_model
            if not self.allow_model_downloads:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install requirements-ai.txt"
                ) from exc

            kwargs = {}
            if not self.allow_model_downloads:
                kwargs["local_files_only"] = True
            try:
                self._sentence_model = SentenceTransformer(
                    self.sentence_model_name,
                    **kwargs,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Installed sentence-transformers does not support offline lazy loading"
                ) from exc
            return self._sentence_model

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

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_preference": self.embedding_provider_preference,
            "local_models_enabled": self.enable_local_models,
            "allow_model_downloads": self.allow_model_downloads,
            "sentence_model": self.sentence_model_name,
            "sentence_model_loaded": self._sentence_model is not None,
            "sentence_model_failed": self._sentence_model_failed,
            "cache_size": self.cache_size,
            "cache_entries": len(self._cache),
        }
