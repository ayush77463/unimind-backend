"""Deterministic memory extraction for explicit user facts and topics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .scoring import importance_for_category


@dataclass(frozen=True)
class ExtractedMemory:
    content: str
    memory_type: str
    category: str
    importance: float
    source: str = "local_extractor"


class FactExtractor:
    """Extracts simple long-term memories without depending on an LLM."""

    _topic_stopwords = {
        "about",
        "answer",
        "answers",
        "correctly",
        "developed",
        "else",
        "hey",
        "hii",
        "me",
        "model",
        "myself",
        "please",
        "something",
        "that",
        "the",
        "this",
        "u",
        "you",
        "yourself",
    }

    def extract(self, messages: Iterable[dict]) -> list[ExtractedMemory]:
        memories: list[ExtractedMemory] = []
        seen: set[str] = set()

        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
                for memory in self._extract_from_sentence(sentence):
                    normalized = memory.content.lower()
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    memories.append(memory)

        return memories

    def _extract_from_sentence(self, raw_sentence: str) -> list[ExtractedMemory]:
        sentence = raw_sentence.strip()
        if not sentence or self._is_question_like(sentence):
            return []

        lowered = sentence.lower()
        if "remember that " in lowered:
            idx = lowered.find("remember that ")
            sentence = sentence[idx + len("remember that ") :].strip()
            lowered = sentence.lower()

        candidates: list[tuple[str, str]] = []

        name_match = re.search(
            r"\bmy name is\s+([a-zA-Z][a-zA-Z .'-]{1,80})",
            sentence,
            flags=re.IGNORECASE,
        )
        if name_match:
            candidates.append(
                ("personal", f"User's name is {self._clean(name_match.group(1))}")
            )

        favorite_match = re.search(
            r"\bmy favou?rite\s+([a-zA-Z ]{2,40})\s+is\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if favorite_match:
            subject = self._clean(favorite_match.group(1)).lower()
            value = self._clean(favorite_match.group(2))
            candidates.append(
                ("preference", f"User's favorite {subject} is {value}")
            )

        from_match = re.search(
            r"\bi(?:'m| am)\s+from\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if from_match:
            candidates.append(("personal", f"User is from {self._clean(from_match.group(1))}"))

        enrolled_match = re.search(
            r"\bi(?:'m| am)?\s*(?:currently\s+)?"
            r"(?:enrolled in|studying|learning|taking)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if enrolled_match:
            value = self._clean(enrolled_match.group(1))
            verb = "enrolled in" if "enrolled" in lowered else "studying"
            candidates.append(("study", f"User is {verb} {value}"))

        current_location_match = re.search(
            r"\bi(?:'m| am)?\s*(?:currently\s+)?(?:in|at|living in|staying in)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if current_location_match and not re.search(r"\binterested in\b", lowered):
            candidates.append(
                (
                    "location",
                    f"User is currently in {self._clean(current_location_match.group(1))}",
                )
            )

        hometown_match = re.search(
            r"\b(?:my\s+)?(?:hometown|home town|native place)\s+is\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if hometown_match:
            candidates.append(("home", f"User's hometown is {self._clean(hometown_match.group(1))}"))

        going_home_match = re.search(
            r"\bgoing to\s+(?:my\s+)?(?:hometown|home town|home)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if going_home_match:
            candidates.append(("home", f"User's hometown is {self._clean(going_home_match.group(1))}"))

        work_match = re.search(
            r"\bi\s+(?:work as|work at|am working as|am working at)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if work_match:
            candidates.append(("work", f"User works as {self._clean(work_match.group(1))}"))

        skill_match = re.search(
            r"\bi\s+(?:know|can|work with|use|have learned)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if skill_match:
            candidates.append(("skill", f"User knows {self._clean(skill_match.group(1))}"))

        goal_match = re.search(
            r"\bi\s+(?:want to|plan to|aim to|hope to|need to)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if goal_match:
            candidates.append(("goal", f"User wants to {self._clean(goal_match.group(1))}"))

        like_match = re.search(
            r"\bi\s+(love|like|prefer|enjoy)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if like_match:
            verb = like_match.group(1).lower()
            fact_verb = "prefers" if verb == "prefer" else f"{verb}s"
            candidates.append(
                (
                    "preference",
                    f"User {fact_verb} {self._clean(like_match.group(2))}",
                )
            )

        dislike_match = re.search(
            r"\bi\s+(hate|dislike)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if dislike_match:
            candidates.append(
                ("preference", f"User dislikes {self._clean(dislike_match.group(2))}")
            )

        identity_match = re.search(
            r"\bi(?:'m| am)\s+(a|an)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if identity_match and not candidates:
            article = identity_match.group(1).lower()
            value = self._clean(identity_match.group(2))
            candidates.append(("personal", f"User is {article} {value}"))

        interested_match = re.search(
            r"\bi(?:'m| am)?\s*(?:interested in|curious about)\s+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if interested_match:
            candidates.append(
                ("recurring_topic", f"User is interested in {self._clean(interested_match.group(1))}")
            )

        return [
            ExtractedMemory(
                content=fact,
                memory_type="preference" if category == "preference" else "semantic",
                category=category,
                importance=self._importance(category),
            )
            for category, fact in candidates
            if len(fact) >= 8
        ]

    def extract_topics(self, messages: Iterable[dict]) -> list[str]:
        topics: list[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", ""))
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
                topic = self._topic_from_sentence(sentence)
                if topic:
                    topics.append(topic)
        return topics

    def _clean(self, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")
        value = re.sub(
            r"\s+(?:please\s+)?(?:remember|save|store)(?:\s+it|\s+this)?$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return value.strip(" .,!?:;\"'")

    def _is_question_like(self, sentence: str) -> bool:
        lowered = sentence.lower()
        if "?" in sentence:
            return True
        return bool(
            re.search(r"\b(what|which|who|when|where|why|how)\b", lowered)
            and "remember that" not in lowered
        )

    def _importance(self, category: str) -> float:
        return importance_for_category(category)

    def _topic_from_sentence(self, raw_sentence: str) -> str | None:
        sentence = self._clean(raw_sentence)
        if not sentence:
            return None

        patterns = [
            r"\b(?:tell me about|explain|teach me|help me with)\s+(.+)$",
            r"\bwhat (?:is|are)\s+(.+)$",
            r"\bhow (?:do|does|to|can i)\s+(.+)$",
            r"\bi(?:'m| am)?\s*(?:interested in|curious about|learning about)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not match:
                continue
            topic = self._clean(match.group(1))
            topic = re.sub(r"\b(?:please|now|today|correctly)\b", "", topic, flags=re.IGNORECASE)
            topic = re.sub(r"\s+", " ", topic).strip()
            if not self._is_valid_topic(topic):
                return None
            return topic[:80]
        return None

    def _is_valid_topic(self, topic: str) -> bool:
        lowered = topic.lower()
        tokens = re.findall(r"[a-z0-9]+", lowered)
        if len(topic) < 4 or not tokens:
            return False
        useful_tokens = [token for token in tokens if token not in self._topic_stopwords]
        return bool(useful_tokens)
