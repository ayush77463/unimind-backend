"""Simple scoring rules for long-term personalization memories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DEFAULT_IMPORTANCE = {
    "goal": 0.9,
    "personal": 0.9,
    "identity": 0.9,
    "location": 0.82,
    "home": 0.82,
    "study": 0.76,
    "work": 0.76,
    "preference": 0.7,
    "skill": 0.66,
    "recurring_topic": 0.62,
    "general": 0.35,
    "episodic": 0.45,
}

DECAY_FLOORS = {
    "goal": 0.82,
    "personal": 0.82,
    "identity": 0.82,
    "location": 0.68,
    "home": 0.68,
    "study": 0.45,
    "work": 0.45,
    "preference": 0.42,
    "skill": 0.38,
    "recurring_topic": 0.28,
    "general": 0.15,
    "episodic": 0.2,
}

DECAY_HALF_LIFE_DAYS = {
    "goal": 3650.0,
    "personal": 3650.0,
    "identity": 3650.0,
    "location": 720.0,
    "home": 720.0,
    "study": 300.0,
    "work": 300.0,
    "preference": 180.0,
    "skill": 240.0,
    "recurring_topic": 90.0,
    "general": 60.0,
    "episodic": 120.0,
}

DECAY_GRACE_DAYS = 30.0


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def importance_label_for_score(value: float) -> str:
    score = clamp_score(value)
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def importance_for_category(category: str, memory_type: str = "semantic") -> float:
    key = (category or memory_type or "general").strip().lower()
    if memory_type == "preference":
        key = "preference"
    if memory_type == "episodic" and key == "general":
        key = "episodic"
    return DEFAULT_IMPORTANCE.get(key, DEFAULT_IMPORTANCE["general"])


def memory_category(memory: dict[str, Any]) -> str:
    metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    category = memory.get("category") or metadata.get("category") or memory.get("memory_type")
    return str(category or "general").strip().lower()


def decayed_importance(
    memory: dict[str, Any],
    *,
    now: datetime | None = None,
) -> float:
    """Lower stale low-value memories while preserving durable user facts."""
    current = clamp_score(float(memory.get("importance", 0.5) or 0.5))
    category = memory_category(memory)

    if category in {"goal", "personal", "identity"} or current >= 0.95:
        return current

    reference = memory.get("last_accessed_at") or memory.get("updated_at") or memory.get("created_at")
    age_days = _age_days(reference, now=now)
    if age_days <= DECAY_GRACE_DAYS:
        return current

    half_life = DECAY_HALF_LIFE_DAYS.get(category, DECAY_HALF_LIFE_DAYS["general"])
    effective_age = age_days - DECAY_GRACE_DAYS
    decayed = current * (0.5 ** (effective_age / half_life))
    floor = DECAY_FLOORS.get(category, DECAY_FLOORS["general"])
    return round(max(floor, decayed), 4)


def _age_days(value: str | None, *, now: datetime | None = None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0)
