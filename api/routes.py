"""Compatibility router that combines chat, memory, and document routes."""

from __future__ import annotations

from fastapi import APIRouter

from .chat import router as chat_router
from .document import router as document_router
from .memory import get_memory_manager, memory_manager, router as memory_router, set_memory_manager

router = APIRouter()
router.include_router(chat_router)
router.include_router(memory_router)
router.include_router(document_router)

__all__ = [
    "get_memory_manager",
    "memory_manager",
    "router",
    "set_memory_manager",
]
