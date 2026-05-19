"""Memory API routes for the custom UniMind memory engine."""

from __future__ import annotations

import logging
import threading
import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from ..memory.memory_manager import MEMORY_TYPES, MemoryManager
    from ..models.schemas import (
        AddExchangeRequest,
        AddFactRequest,
        AddMessageRequest,
        FactResponse,
        MemoryAnalyticsResponse,
        MemoryRetrieveResponse,
        MemoryStatusResponse,
        MemoryStoreRequest,
        MemoryStoreResponse,
        PinMemoryRequest,
        SaveEpisodeRequest,
        SearchMemoryRequest,
        SearchResponse,
    )
except ImportError:  # pragma: no cover
    from memory.memory_manager import MEMORY_TYPES, MemoryManager  # type: ignore
    from models.schemas import (  # type: ignore
        AddExchangeRequest,
        AddFactRequest,
        AddMessageRequest,
        FactResponse,
        MemoryAnalyticsResponse,
        MemoryRetrieveResponse,
        MemoryStatusResponse,
        MemoryStoreRequest,
        MemoryStoreResponse,
        PinMemoryRequest,
        SaveEpisodeRequest,
        SearchMemoryRequest,
        SearchResponse,
    )

router = APIRouter(tags=["Memory"])
memory_manager: MemoryManager | None = None
_memory_manager_lock = threading.Lock()
logger = logging.getLogger("unimind.api")


def set_memory_manager(manager: MemoryManager) -> None:
    global memory_manager
    with _memory_manager_lock:
        memory_manager = manager
    logger.info("MemoryManager set via startup (supabase=%s)", manager._using_supabase)


def get_memory_manager() -> MemoryManager:
    global memory_manager
    if memory_manager is None:
        with _memory_manager_lock:
            if memory_manager is None:
                logger.warning("Lazy-initializing MemoryManager (startup init may have failed)...")
                start = time.time()
                memory_manager = MemoryManager()
                elapsed = round(time.time() - start, 2)
                logger.info(
                    "MemoryManager lazy-init complete in %.2fs (supabase=%s)",
                    elapsed, memory_manager._using_supabase,
                )
    return memory_manager


def _error(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message},
    )


@router.get("/health")
async def health_check():
    start = time.time()
    try:
        manager = get_memory_manager()
        response = {
            "success": True,
            "status": "healthy",
            "service": "UniMind Memory API",
            "version": "2.2.0",
            "storage": "supabase_postgresql" if manager._using_supabase else "sqlite",
            "vector_store": manager.retriever.__class__.__name__,
            "vector_backend": getattr(manager, "vector_backend", "fallback"),
            "embedding_provider": manager.embedding_service.provider,
            "embedding": manager.embedding_service.diagnostics(),
            "local_models_enabled": manager.embedding_service.local_models_enabled,
            "llm_available": manager.llm_service.available,
            "memory_types": ["episodic", "semantic", "preference"],
            "ai_pipeline": {
                "topic_classifier": "semantic_prototypes+sklearn_cosine",
                "importance_model": "fixed_logistic_features",
                "sentiment": "transformer_lazy_or_deterministic_fallback",
                "analytics": "pandas_numpy",
            },
            "latency_ms": 0,
        }
        diagnostics = getattr(manager.retriever, "diagnostics", None)
        if callable(diagnostics):
            response["chroma"] = diagnostics()
        # Real DB health check
        if manager._using_supabase:
            try:
                db_health = manager.storage.health_check()
                response["database"] = db_health
                if not db_health.get("connected"):
                    response["status"] = "degraded"
                    response["warning"] = "Database connectivity issue"
            except Exception as db_exc:
                response["database"] = {"connected": False, "error": str(db_exc)}
                response["status"] = "degraded"
        response["latency_ms"] = round((time.time() - start) * 1000, 2)
        return response
    except Exception as exc:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error("Health check failed: %s", exc)
        return {
            "success": True,
            "status": "starting",
            "service": "UniMind Memory API",
            "version": "2.2.0",
            "note": f"Initializing: {exc}",
            "latency_ms": elapsed_ms,
        }


@router.post("/memory/store", response_model=MemoryStoreResponse)
def store_memory(request: MemoryStoreRequest):
    logger.info("POST /memory/store user=%s type=%s", request.user_id, request.memory_type)
    try:
        manager = get_memory_manager()
        memory_id = manager.store_memory(
            user_id=request.user_id,
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            source=request.source,
            summary=request.summary,
            metadata=request.metadata,
        )
        logger.info("Memory stored: id=%s user=%s", memory_id, request.user_id)
        return MemoryStoreResponse(
            success=True,
            memory_id=memory_id,
            message="Memory stored",
        )
    except Exception as exc:
        logger.error("Memory store failed for user=%s: %s", request.user_id, exc)
        return _error(f"Memory was not stored: {exc}")


@router.get(
    "/memory/retrieve",
    response_model=MemoryRetrieveResponse,
    response_model_exclude_none=True,
)
def retrieve_memory(
    user_id: Annotated[str, Query(min_length=1, max_length=100)],
    query: Annotated[str, Query(min_length=1, max_length=2000)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    memory_type: str | None = Query(default=None),
    debug: bool = Query(default=False),
):
    try:
        memory_types = _parse_memory_types(memory_type)
        manager = get_memory_manager()
        memories = manager.retrieve_memories(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_types=memory_types,
            debug=debug,
        )
        return MemoryRetrieveResponse(
            success=True,
            user_id=user_id,
            query=query,
            memories=memories,
            total_found=len(memories),
            warnings=manager.retriever.last_warnings or None,
            debug=manager.retriever.last_debug if debug else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _error(f"Memory retrieval failed: {exc}")


@router.post("/memory/message")
def add_message_to_memory(request: AddMessageRequest):
    try:
        manager = get_memory_manager()
        manager.add_message(
            user_id=request.user_id,
            role=request.role,
            content=request.content,
            session_id=request.session_id,
        )
        return {
            "success": True,
            "message": "Message added to memory",
            "total_messages": manager.storage.get_message_count(request.user_id),
        }
    except Exception as exc:
        return _error(f"Memory message was not stored: {exc}")


@router.post("/memory/exchange")
def add_exchange_to_memory(request: AddExchangeRequest):
    logger.info("POST /memory/exchange user=%s session=%s", request.user_id, request.session_id)
    start = time.time()
    try:
        result = get_memory_manager().add_exchange(
            user_id=request.user_id,
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            session_id=request.session_id,
            tags=request.tags,
            ai_enrich=request.ai_enrich,
        )
        elapsed = round(time.time() - start, 2)
        logger.info(
            "Exchange stored: user=%s facts=%s episode=%s (%.2fs)",
            request.user_id, result.get("facts_added"), result.get("episode_id"), elapsed,
        )
        return result
    except Exception as exc:
        logger.error("Exchange store failed for user=%s: %s", request.user_id, exc, exc_info=True)
        return _error(f"Exchange was not stored: {exc}")


@router.get("/memory/short-term/{user_id}")
def get_short_term_memory(user_id: str, last_n: int = Query(default=10, ge=1, le=100)):
    manager = get_memory_manager()
    messages = manager.storage.get_recent_messages(user_id=user_id, last_n=last_n)
    count = manager.storage.get_message_count(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "messages": messages,
        "total_in_memory": count,
        "count": count,
    }


@router.post("/memory/episode/save")
def save_episode(request: SaveEpisodeRequest):
    try:
        episode_id = get_memory_manager().save_episode(
            user_id=request.user_id,
            tags=request.tags,
            ai_enrich=request.ai_enrich,
            session_id=request.session_id,
            clear_short_term=request.clear_short_term,
        )
        return {
            "success": True,
            "episode_id": episode_id,
            "message": "Episode saved",
        }
    except Exception as exc:
        return _error(f"Episode was not saved: {exc}")


@router.get("/memory/episodes/{user_id}")
def get_episodes(user_id: str, last_n: int = Query(default=5, ge=1, le=100)):
    episodes = get_memory_manager().episodic.get_recent_episodes(user_id, last_n)
    return {
        "success": True,
        "user_id": user_id,
        "episodes": episodes,
        "total_returned": len(episodes),
    }


@router.post("/memory/fact", response_model=FactResponse)
def add_fact(request: AddFactRequest):
    try:
        fact_id = get_memory_manager().add_fact(
            user_id=request.user_id,
            fact=request.fact,
            category=request.category,
        )
        return FactResponse(
            success=True,
            fact_id=fact_id,
            message="Fact stored in semantic memory",
        )
    except Exception as exc:
        return _error(f"Fact was not stored: {exc}")


@router.post(
    "/memory/search",
    response_model=SearchResponse,
    response_model_exclude_none=True,
)
def search_semantic_memory(request: SearchMemoryRequest):
    try:
        manager = get_memory_manager()
        results = manager.retrieve_memories(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            memory_types=["semantic", "preference"],
            debug=request.debug,
        )
        return SearchResponse(
            user_id=request.user_id,
            query=request.query,
            results=results,
            total_found=len(results),
            warnings=manager.retriever.last_warnings or None,
            debug=manager.retriever.last_debug if request.debug else None,
        )
    except Exception as exc:
        return _error(f"Search failed: {exc}")


@router.get("/memory/facts/{user_id}")
def get_all_facts(user_id: str):
    facts = get_memory_manager().semantic.get_all_facts(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "facts": facts,
        "total_facts": len(facts),
    }


@router.get("/memory/all/{user_id}")
def get_all_memories(
    user_id: str,
    query: str = Query(default="", max_length=2000),
    category: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    manager = get_memory_manager()
    memories = manager.list_all_memories(
        user_id=user_id,
        query=query,
        category=category,
        limit=limit,
        offset=offset,
    )
    return {
        "success": True,
        "user_id": user_id,
        "query": query,
        "category": category,
        "memories": memories,
        "total_returned": len(memories),
        "limit": limit,
        "offset": offset,
    }


@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    deleted = get_memory_manager().delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"success": True, "memory_id": memory_id, "message": "Memory deleted"}


@router.post("/memory/pin")
def pin_memory(request: PinMemoryRequest):
    memory = get_memory_manager().pin_memory(
        memory_id=request.memory_id,
        pinned=request.pinned,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "success": True,
        "memory_id": request.memory_id,
        "pinned": request.pinned,
        "memory": memory,
    }


@router.get("/memory/status/{user_id}", response_model=MemoryStatusResponse)
def get_memory_status(user_id: str):
    return MemoryStatusResponse(**get_memory_manager().get_status(user_id))


@router.get(
    "/memory/analytics/{user_id}",
    response_model=MemoryAnalyticsResponse,
)
def get_memory_analytics(user_id: str):
    return MemoryAnalyticsResponse(**get_memory_manager().get_analytics(user_id))


@router.get("/memory/context/{user_id}")
def get_memory_context(
    user_id: str,
    query: str = Query("general", min_length=1, max_length=2000),
    debug: bool = Query(default=False),
    max_chars: int = Query(default=2000, ge=500, le=6000),
    top_k: int = Query(default=5, ge=1, le=20),
):
    start = time.time()
    try:
        payload = get_memory_manager().build_context_payload(
            user_id=user_id,
            query=query,
            top_k=top_k,
            max_chars=max_chars,
            debug=debug,
        )
        elapsed = round(time.time() - start, 3)
        logger.info(
            "Context built: user=%s query='%.40s' length=%d (%.3fs)",
            user_id, query, payload.get("context_length", 0), elapsed,
        )
        if not payload.get("warnings"):
            payload.pop("warnings", None)
        if not debug:
            payload.pop("debug", None)
        return payload
    except Exception as exc:
        elapsed = round(time.time() - start, 3)
        logger.error("Context build failed for user=%s: %s (%.3fs)", user_id, exc, elapsed)
        payload = {
            "success": True,
            "user_id": user_id,
            "query": query,
            "context": "",
            "context_length": 0,
        }
        if debug:
            payload["warnings"] = [f"context_build_failed: {exc}"]
        return payload


@router.delete("/memory/clear/{user_id}")
def clear_all_memory(user_id: str):
    get_memory_manager().clear_all(user_id)
    return {"success": True, "message": f"All memory cleared for user: {user_id}"}


@router.delete("/memory/short-term/{user_id}")
def clear_short_term(user_id: str):
    get_memory_manager().clear_short_term(user_id)
    return {"success": True, "message": "Short-term memory cleared"}


def _parse_memory_types(memory_type: str | None) -> list[str] | None:
    if not memory_type:
        return None
    values = [item.strip().lower() for item in memory_type.split(",") if item.strip()]
    invalid = [item for item in values if item not in MEMORY_TYPES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported memory_type: {', '.join(invalid)}",
        )
    return values or None
