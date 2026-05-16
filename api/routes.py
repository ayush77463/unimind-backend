"""Compatibility router that combines chat and memory routes."""

from __future__ import annotations

from fastapi import APIRouter

from .chat import router as chat_router
from .memory import get_memory_manager, memory_manager, router as memory_router, set_memory_manager

router = APIRouter()
router.include_router(chat_router)
router.include_router(memory_router)

__all__ = [
    "get_memory_manager",
    "memory_manager",
    "router",
    "set_memory_manager",
]
