"""Pydantic schemas for the UniMind memory API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MemoryType = Literal["episodic", "semantic", "preference"]


class RequestModel(BaseModel):
    @field_validator(
        "user_id",
        "message",
        "content",
        "fact",
        "query",
        "user_message",
        "assistant_message",
        check_fields=False,
    )
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("role", check_fields=False)
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"user", "assistant"}:
            raise ValueError("Role must be 'user' or 'assistant'")
        return value

    @field_validator("category", check_fields=False)
    @classmethod
    def normalize_category(cls, value: str) -> str:
        value = value.strip().lower()
        return value or "general"


class ChatRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, max_length=120)


class ChatResponse(BaseModel):
    user_id: str
    response: str
    memory_used: dict[str, Any]
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MemoryStoreRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=10000)
    memory_type: MemoryType = "semantic"
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    source: str = Field(default="manual", min_length=1, max_length=80)
    summary: str | None = Field(default=None, max_length=10000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStoreResponse(BaseModel):
    success: bool
    memory_id: str
    message: str


class MemoryRetrieveResponse(BaseModel):
    success: bool
    user_id: str
    query: str
    memories: list[dict[str, Any]]
    total_found: int
    warnings: list[str] | None = None
    debug: dict[str, Any] | None = None


class AddMessageRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., min_length=1, max_length=20)
    content: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, max_length=120)


class AddExchangeRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    user_message: str = Field(..., min_length=1, max_length=10000)
    assistant_message: str = Field(..., min_length=1, max_length=20000)
    session_id: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=20)
    ai_enrich: bool = True


class SaveEpisodeRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)
    ai_enrich: bool = True
    session_id: str | None = Field(default=None, max_length=120)
    clear_short_term: bool = True


class AddFactRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    fact: str = Field(..., min_length=1, max_length=2000)
    category: str = Field(default="general", min_length=1, max_length=40)


class FactResponse(BaseModel):
    success: bool
    fact_id: str
    message: str


class SearchMemoryRequest(RequestModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    debug: bool = False


class SearchResponse(BaseModel):
    user_id: str
    query: str
    results: list[dict[str, Any]]
    total_found: int
    success: bool = True
    warnings: list[str] | None = None
    debug: dict[str, Any] | None = None


class MemoryStatusResponse(BaseModel):
    user_id: str
    short_term_messages: int
    total_facts: int
    total_episodes: int
    recent_episode_summaries: list[str]


class PinMemoryRequest(RequestModel):
    memory_id: str = Field(..., min_length=1, max_length=160)
    pinned: bool = True
