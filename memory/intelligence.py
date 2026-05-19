"""ML-style NLP intelligence for UniMind memories.

The pipeline uses semantic topic prototypes, cosine similarity, and a small
logistic scoring model. Transformer sentiment is available behind env gates;
deterministic fallback logic keeps Render and tests stable without downloads.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .embedding_service import EmbeddingService
from .scoring import clamp_score, importance_label_for_score

try:
    from ..config import (
        UNIMIND_ALLOW_MODEL_DOWNLOADS,
        UNIMIND_ENABLE_LOCAL_MODELS,
        UNIMIND_SENTIMENT_MODEL,
    )
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        UNIMIND_ALLOW_MODEL_DOWNLOADS,
        UNIMIND_ENABLE_LOCAL_MODELS,
        UNIMIND_SENTIMENT_MODEL,
    )

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    cosine_similarity = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _build_topic_keywords(prototypes: dict[str, list[str]]) -> dict[str, set[str]]:
    return {
        topic: set(re.findall(r"[a-z0-9]+", " ".join(examples).lower()))
        for topic, examples in prototypes.items()
    }


@dataclass(frozen=True)
class MemoryIntelligenceResult:
    importance_score: float
    importance_label: str
    category: str
    sentiment: str
    signals: dict[str, float | int | str] = field(default_factory=dict)


class MemoryIntelligenceAnalyzer:
    """Semantic topic, sentiment, and importance enrichment for memories."""

    categories = (
        "AI",
        "Programming",
        "Flutter",
        "Education",
        "Projects",
        "Productivity",
        "Personal",
        "Emotional",
        "Career",
    )

    _generic_phrases = {
        "hi",
        "hii",
        "hello",
        "hey",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "yes",
        "no",
        "cool",
    }

    _topic_prototypes: dict[str, list[str]] = {
        "AI": [
            "artificial intelligence machine learning deep learning neural networks",
            "LLM embeddings transformer semantic retrieval vector search Gemini assistant",
            "NLP generative AI prompt engineering memory intelligence",
        ],
        "Programming": [
            "software engineering backend API database server deployment Python",
            "code debugging tests architecture repository GitHub FastAPI Supabase",
            "frontend backend SDK TypeScript system integration",
        ],
        "Flutter": [
            "Flutter Dart widget Material mobile application provider state UI",
            "Android APK performance scrolling rendering chat interface",
        ],
        "Education": [
            "college university school exam assignment course study learning",
            "student academic research lecture classroom education",
        ],
        "Projects": [
            "project app prototype product feature roadmap architecture build",
            "implementation dashboard pipeline system UniMind release",
        ],
        "Productivity": [
            "task todo schedule deadline habit focus planning workflow notes",
            "time management reminder productivity routine prioritization",
        ],
        "Personal": [
            "name hometown family location home personal identity living from",
            "favorite preference likes dislikes user profile",
        ],
        "Emotional": [
            "happy sad anxious stressed confused excited frustrated feeling mood",
            "emotion motivation worry pressure overwhelmed calm",
        ],
        "Career": [
            "career job internship interview resume portfolio engineer developer work",
            "professional skills hiring workplace engineering role",
        ],
    }

    _topic_keywords: dict[str, set[str]] = _build_topic_keywords(_topic_prototypes)

    _sentiment_keywords: dict[str, set[str]] = {
        "Excited": {"amazing", "awesome", "excited", "fantastic", "love", "thrilled", "wow"},
        "Motivated": {"aim", "build", "determined", "goal", "motivated", "plan", "ready", "want"},
        "Happy": {"glad", "great", "happy", "joy", "nice", "pleased"},
        "Stressed": {"anxious", "deadline", "overwhelmed", "pressure", "stress", "stressed", "worried"},
        "Confused": {"confused", "doubt", "lost", "stuck", "unclear", "understand"},
        "Frustrated": {"annoyed", "broken", "failed", "frustrated", "hate", "issue", "problem"},
    }

    _legacy_topic_bias = {
        "goal": "Projects",
        "preference": "Personal",
        "personal": "Personal",
        "identity": "Personal",
        "location": "Personal",
        "home": "Personal",
        "study": "Education",
        "skill": "Career",
        "work": "Career",
        "recurring_topic": "AI",
        "episodic": "Projects",
        "conversation": "Projects",
    }

    _important_terms = {
        "always",
        "career",
        "deadline",
        "engineer",
        "engineering",
        "important",
        "love",
        "need",
        "project",
        "remember",
        "want",
        "work",
    }

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        *,
        enable_transformers: bool | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService(api_key="")
        self.enable_transformers = (
            bool(UNIMIND_ENABLE_LOCAL_MODELS)
            if enable_transformers is None
            else bool(enable_transformers)
        )
        self.allow_model_downloads = bool(UNIMIND_ALLOW_MODEL_DOWNLOADS)
        self.sentiment_model_name = UNIMIND_SENTIMENT_MODEL
        self._sentiment_pipeline = None
        self._sentiment_failed = False
        self._sentiment_lock = threading.Lock()
        self._prototype_matrix: np.ndarray | None = None
        self._prototype_labels: list[str] = []

    def analyze(
        self,
        text: str,
        *,
        memory_type: str = "semantic",
        base_importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        prior_memories: Iterable[dict[str, Any]] | None = None,
    ) -> MemoryIntelligenceResult:
        clean_text = " ".join((text or "").split())
        tokens = self._tokens(clean_text)
        legacy_category = self._legacy_category(memory_type, metadata)

        if self._is_generic(clean_text, tokens):
            return MemoryIntelligenceResult(
                importance_score=0.2,
                importance_label="Low",
                category="General",
                sentiment="Neutral",
                signals={
                    "topic_confidence": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_polarity": 0.0,
                    "legacy_category": legacy_category,
                    "intelligence_provider": self._provider_name(),
                },
            )

        topic, topic_confidence = self._classify_topic(clean_text, tokens, legacy_category)
        sentiment, sentiment_confidence, sentiment_polarity = self._classify_sentiment(clean_text, tokens)
        repetition_score = self._repetition_score(tokens, prior_memories or [])
        specificity_score = self._specificity_score(clean_text, tokens)
        technical_score = self._technical_score(tokens)
        source_score = self._source_score(metadata or {})
        important_term_score = min(1.0, len(tokens & self._important_terms) / 4.0)
        base_score = clamp_score(base_importance)

        features = {
            "base_importance": base_score,
            "topic_confidence": topic_confidence,
            "specificity": specificity_score,
            "repetition": repetition_score,
            "technical": technical_score,
            "source": source_score,
            "sentiment_confidence": sentiment_confidence,
            "important_terms": important_term_score,
            "episodic": 1.0 if memory_type == "episodic" else 0.0,
        }
        importance_score = self._predict_importance(features)

        category_floors = {
            "goal": 0.86,
            "personal": 0.82,
            "identity": 0.82,
            "location": 0.74,
            "home": 0.74,
            "study": 0.62,
            "work": 0.62,
            "skill": 0.6,
            "preference": 0.58,
            "recurring_topic": 0.62,
            "episodic": 0.5,
            "conversation": 0.5,
        }
        importance_score = max(importance_score, category_floors.get(legacy_category, 0.0))
        if base_importance >= 0.75:
            importance_score = max(importance_score, base_score)
        if topic in {"AI", "Flutter", "Programming", "Projects"} and technical_score >= 0.4:
            importance_score += 0.06
        if sentiment in {"Excited", "Motivated"} and topic != "General":
            importance_score += 0.04

        importance_score = round(clamp_score(importance_score), 4)
        signals: dict[str, float | int | str] = {
            "topic_confidence": round(topic_confidence, 3),
            "sentiment_confidence": round(sentiment_confidence, 3),
            "sentiment_polarity": round(sentiment_polarity, 3),
            "technical_score": round(technical_score, 3),
            "specificity_score": round(specificity_score, 3),
            "repetition_score": round(repetition_score, 3),
            "legacy_category": legacy_category,
            "intelligence_provider": self._provider_name(),
        }
        for key, value in features.items():
            signals[f"importance_{key}"] = round(value, 3)

        return MemoryIntelligenceResult(
            importance_score=importance_score,
            importance_label=importance_label_for_score(importance_score),
            category=topic,
            sentiment=sentiment,
            signals=signals,
        )

    def _classify_topic(
        self,
        text: str,
        tokens: set[str],
        legacy_category: str,
    ) -> tuple[str, float]:
        semantic_scores = self._semantic_topic_scores(text)
        lexical_scores = {
            topic: min(1.0, len(tokens & keywords) / 4.0)
            for topic, keywords in self._topic_keywords.items()
        }
        scores: dict[str, float] = {}
        for topic in self.categories:
            scores[topic] = 0.72 * semantic_scores.get(topic, 0.0) + 0.28 * lexical_scores.get(topic, 0.0)

        biased_topic = self._legacy_topic_bias.get(legacy_category)
        if biased_topic:
            scores[biased_topic] = scores.get(biased_topic, 0.0) + 0.08
        if "flutter" in tokens:
            scores["Flutter"] = scores.get("Flutter", 0.0) + 0.25
        if {"ai", "llm", "embedding", "retrieval", "semantic", "vector"} & tokens:
            scores["AI"] = scores.get("AI", 0.0) + 0.18

        topic, score = max(scores.items(), key=lambda item: item[1])
        confidence = clamp_score(score)
        if confidence < 0.16:
            return "General", confidence
        return topic, confidence

    def _semantic_topic_scores(self, text: str) -> dict[str, float]:
        try:
            matrix, labels = self._topic_prototype_matrix()
            query = self.embedding_service.embed(text, task_type="retrieval_query").vector.reshape(1, -1)
            if query.shape[1] != matrix.shape[1]:
                return {}
            if cosine_similarity is not None:
                similarities = cosine_similarity(query, matrix)[0]
            else:
                similarities = np.dot(query, matrix.T)[0]
        except Exception as exc:
            logger.debug("Semantic topic scoring unavailable: %s", exc)
            return {}

        scores: dict[str, float] = {}
        for label, score in zip(labels, similarities):
            normalized = (float(score) + 1.0) / 2.0
            scores[label] = max(scores.get(label, 0.0), normalized)
        return scores

    def _topic_prototype_matrix(self) -> tuple[np.ndarray, list[str]]:
        if self._prototype_matrix is not None:
            return self._prototype_matrix, self._prototype_labels

        vectors: list[np.ndarray] = []
        labels: list[str] = []
        for topic, prototypes in self._topic_prototypes.items():
            for prototype in prototypes:
                embedded = self.embedding_service.embed(
                    prototype,
                    task_type="retrieval_document",
                )
                vectors.append(embedded.vector)
                labels.append(topic)
        self._prototype_matrix = np.vstack(vectors).astype(np.float32)
        self._prototype_labels = labels
        return self._prototype_matrix, self._prototype_labels

    def _classify_sentiment(self, text: str, tokens: set[str]) -> tuple[str, float, float]:
        if self.enable_transformers and not self._sentiment_failed:
            try:
                result = self._transformer_sentiment(text)
                if result is not None:
                    return result
            except Exception as exc:
                self._sentiment_failed = True
                logger.warning("Transformer sentiment unavailable; using fallback: %s", exc)
        return self._fallback_sentiment(tokens)

    def _transformer_sentiment(self, text: str) -> tuple[str, float, float] | None:
        pipeline = self._get_sentiment_pipeline()
        output = pipeline(text[:512])
        item = output[0] if isinstance(output, list) and output else {}
        raw_label = str(item.get("label", "")).upper()
        confidence = clamp_score(float(item.get("score", 0.0) or 0.0))
        if not raw_label:
            return None

        tokens = self._tokens(text)
        fallback_label, fallback_confidence, fallback_polarity = self._fallback_sentiment(tokens)
        if raw_label == "POSITIVE":
            if fallback_label in {"Excited", "Motivated", "Happy"}:
                return fallback_label, max(confidence, fallback_confidence), confidence
            return "Happy", confidence, confidence
        if raw_label == "NEGATIVE":
            if fallback_label in {"Stressed", "Confused", "Frustrated"}:
                return fallback_label, max(confidence, fallback_confidence), -confidence
            return "Frustrated", confidence, -confidence
        return fallback_label, fallback_confidence, fallback_polarity

    def _get_sentiment_pipeline(self):
        if self._sentiment_pipeline is not None:
            return self._sentiment_pipeline
        with self._sentiment_lock:
            if self._sentiment_pipeline is not None:
                return self._sentiment_pipeline
            if not self.allow_model_downloads:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("transformers is not installed; install requirements-ai.txt") from exc

            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                tokenizer=self.sentiment_model_name,
                local_files_only=not self.allow_model_downloads,
            )
            return self._sentiment_pipeline

    def _fallback_sentiment(self, tokens: set[str]) -> tuple[str, float, float]:
        scores: Counter[str] = Counter()
        for label, keywords in self._sentiment_keywords.items():
            overlap = tokens & keywords
            if overlap:
                scores[label] += len(overlap)
        if not scores:
            return "Neutral", 0.0, 0.0
        label, raw_score = scores.most_common(1)[0]
        confidence = min(1.0, raw_score / 3.0)
        polarity = confidence if label in {"Excited", "Motivated", "Happy"} else -confidence
        return label, confidence, polarity

    def _predict_importance(self, features: dict[str, float]) -> float:
        z = (
            -1.1
            + 1.65 * features["base_importance"]
            + 1.05 * features["topic_confidence"]
            + 0.95 * features["specificity"]
            + 0.75 * features["technical"]
            + 0.55 * features["repetition"]
            + 0.5 * features["source"]
            + 0.35 * features["sentiment_confidence"]
            + 0.4 * features["important_terms"]
            + 0.25 * features["episodic"]
        )
        return float(1.0 / (1.0 + math.exp(-z)))

    def _specificity_score(self, text: str, tokens: set[str]) -> float:
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / 140.0)
        token_score = min(1.0, len(tokens) / 18.0)
        has_entity = bool(re.search(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text))
        entity_bonus = 0.12 if has_entity else 0.0
        return clamp_score(0.55 * token_score + 0.35 * length_score + entity_bonus)

    def _technical_score(self, tokens: set[str]) -> float:
        technical_terms = (
            self._topic_keywords["AI"]
            | self._topic_keywords["Programming"]
            | self._topic_keywords["Flutter"]
        )
        return min(1.0, len(tokens & technical_terms) / 5.0)

    def _source_score(self, metadata: dict[str, Any]) -> float:
        source = str(metadata.get("source") or metadata.get("upload_source") or "").lower()
        category = str(metadata.get("category") or "").lower()
        if source in {"document_upload", "image_upload"}:
            return 0.65
        if category in {"goal", "personal", "identity", "preference"}:
            return 0.55
        return 0.25 if source else 0.0

    def _repetition_score(
        self,
        tokens: set[str],
        prior_memories: Iterable[dict[str, Any]],
    ) -> float:
        if not tokens:
            return 0.0
        matches = 0
        for memory in prior_memories:
            memory_tokens = self._tokens(
                " ".join(
                    [
                        str(memory.get("content", "")),
                        str(memory.get("summary", "")),
                        str(memory.get("category", "")),
                        str(memory.get("ai_category", "")),
                    ]
                )
            )
            if len(tokens & memory_tokens) >= 2:
                matches += 1
        return min(1.0, matches / 4.0)

    def _legacy_category(
        self,
        memory_type: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        metadata = metadata or {}
        value = (
            metadata.get("memory_category")
            or metadata.get("legacy_category")
            or metadata.get("category")
            or memory_type
            or "general"
        )
        return str(value).strip().lower() or "general"

    def _is_generic(self, text: str, tokens: set[str]) -> bool:
        lowered = text.strip().lower()
        if lowered in self._generic_phrases:
            return True
        return len(tokens) <= 2 and len(lowered) <= 18

    def _tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            if len(token) <= 1:
                continue
            tokens.add(token)
            if token.endswith("s") and len(token) > 4:
                tokens.add(token[:-1])
        if "artificial" in tokens and "intelligence" in tokens:
            tokens.add("ai")
        if "machine" in tokens and "learning" in tokens:
            tokens.add("ml")
        return tokens

    def _provider_name(self) -> str:
        providers = ["semantic_prototypes", "sklearn_cosine", "logistic_importance"]
        if self.enable_transformers and not self._sentiment_failed:
            providers.append("transformer_sentiment_lazy")
        else:
            providers.append("deterministic_sentiment")
        return "+".join(providers)
