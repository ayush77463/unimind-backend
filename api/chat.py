"""Chat route for backend-demo conversations with memory."""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from ..memory.memory_manager import LLMUnavailableError
    from ..models.schemas import ChatRequest, ChatResponse
    from .memory import get_memory_manager
except ImportError:  # pragma: no cover
    from memory.memory_manager import LLMUnavailableError  # type: ignore
    from models.schemas import ChatRequest, ChatResponse  # type: ignore
    from api.memory import get_memory_manager  # type: ignore

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_memory(request: ChatRequest):
    try:
        result = get_memory_manager().chat_with_memory(
            user_id=request.user_id,
            user_message=request.message,
            session_id=request.session_id,
        )
        return ChatResponse(
            user_id=request.user_id,
            response=result["response"],
            memory_used=result["memory_used"],
        )
    except LLMUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": str(exc),
                "message": (
                    "Backend /chat requires GEMINI_API_KEY. "
                    "Memory storage and retrieval endpoints are still available."
                ),
            },
        )
    except TimeoutError as exc:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": str(exc),
                "message": "Backend LLM request timed out. Memory endpoints remain available.",
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Chat error: {exc}"},
        )


@router.post("/chat/stream")
async def stream_chat_with_memory(request: ChatRequest):
    async def events():
        try:
            async for item in get_memory_manager().stream_chat_with_memory(
                user_id=request.user_id,
                user_message=request.message,
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except LLMUnavailableError as exc:
            yield _sse_error(str(exc), "llm_unavailable")
        except TimeoutError as exc:
            yield _sse_error(str(exc), "timeout")
        except Exception as exc:
            yield _sse_error(f"Chat stream error: {exc}", "stream_error")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_error(message: str, code: str) -> str:
    return "data: " + json.dumps(
        {
            "type": "error",
            "error": message,
            "code": code,
        },
        ensure_ascii=False,
    ) + "\n\n"
