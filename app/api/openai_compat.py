"""OpenAI-compatible endpoints for Open WebUI integration.

Provides ``/v1/models`` and ``/v1/chat/completions`` so that Open WebUI
(or any OpenAI-compatible client) can use our RAG backend as a drop-in
LLM provider.

* One model per Milvus collection — ``GET /v1/models`` enumerates every
  document collection (excluding internal ``_``-prefixed ones) and exposes
  each as ``rag/{collection_name}``.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.repositories.milvus._client import get_client
from app.services.internal.generate import (
    build_context_block,
    build_messages as build_rag_messages,
    generate,
    generate_stream,
)
from app.services.public.search import search_documents

router = APIRouter(tags=["OpenAI Compatible"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_PREFIX = "rag/"
_INTERNAL_COLLECTION_PREFIX = "_"


# ---------------------------------------------------------------------------
# Schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[list[str] | str] = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "cs431-rag"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collection_from_model(model_id: str) -> str | None:
    """Extract collection name from a model id like ``rag-my_docs``.

    Returns ``None`` if the model id doesn't start with the prefix.
    """
    if model_id.startswith(_MODEL_PREFIX):
        return model_id[len(_MODEL_PREFIX) :]
    return None


def _list_document_collections() -> list[str]:
    """Return all Milvus collections that are user-document collections.

    Internal collections (prefixed with ``_``) are filtered out.
    """
    client = get_client()
    all_cols: list[str] = client.list_collections()
    return sorted(c for c in all_cols if not c.startswith(_INTERNAL_COLLECTION_PREFIX))


def _trim_openai_history(
    messages: list[ChatMessage],
    max_turns: int,
) -> list[dict[str, str]]:
    """Extract the last *max_turns* (user, assistant) pairs from the
    Open WebUI message array, excluding the final user message (which
    becomes the RAG query) and any system messages.
    """
    # Separate system, history, and the last user message
    non_system = [m for m in messages if m.role != "system"]
    if not non_system:
        return []
    # Everything except the last message is history
    history_msgs = non_system[:-1]
    relevant = [m for m in history_msgs if m.role in ("user", "assistant")]
    trimmed = relevant[-(max_turns * 2) :]
    return [{"role": m.role, "content": m.content} for m in trimmed]


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _build_non_streaming_response(
    completion_id: str,
    model: str,
    content: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _build_streaming_chunk(
    completion_id: str,
    model: str,
    content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build a single SSE ``data:`` line in OpenAI streaming format."""
    delta: dict[str, str] = {}
    if content is not None:
        delta["content"] = content
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List available RAG models",
    description=(
        "Returns one model per Milvus document collection. "
        "Each model ID has the format ``rag-{collection_name}``."
    ),
)
async def list_models() -> ModelListResponse:
    loop = asyncio.get_running_loop()
    collections = await loop.run_in_executor(None, _list_document_collections)
    models = [ModelObject(id=f"{_MODEL_PREFIX}{col}") for col in collections]
    return ModelListResponse(data=models)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@router.post(
    "/v1/chat/completions",
    summary="Chat completion with RAG",
    description=(
        "OpenAI-compatible chat completion endpoint. The model ID encodes "
        "the Milvus collection to query (e.g. ``rag-my_docs``). "
        "Supports both streaming (SSE) and non-streaming responses."
    ),
)
async def chat_completions(request: ChatCompletionRequest):
    # 1. Resolve collection from model ID
    collection_name = _collection_from_model(request.model)
    if collection_name is None:
        # Return an OpenAI-style error
        return _openai_error(
            f"Unknown model '{request.model}'. Expected format: rag-{{collection_name}}",
            code="model_not_found",
            status_code=404,
        )

    # 2. Extract the latest user message as the RAG query
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        return _openai_error(
            "No user message found in the request.",
            code="invalid_request_error",
            status_code=400,
        )
    user_query = user_messages[-1].content

    # 3. Retrieve relevant documents
    try:
        search_results = await search_documents(
            query=user_query,
            collection_name=collection_name,
            search_type=settings.GENERATION_SEARCH_TYPE,
            top_k=settings.GENERATION_RAG_TOP_K,
            rerank=settings.OPENWEBUI_RERANKING_ENABLED,
        )
    except Exception as exc:
        logger.error(f"Search failed for collection '{collection_name}': {exc}")
        return _openai_error(
            f"Search failed: {exc}",
            code="internal_error",
            status_code=500,
        )

    # 4. Build sources and prompt
    sources: list[dict[str, Any]] = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "text": r.text,
            "score": r.score,
        }
        for r in search_results
    ]

    history = _trim_openai_history(request.messages, settings.GENERATION_HISTORY_TURNS)
    context_block = build_context_block(sources)
    llm_messages = build_rag_messages(
        user_query=user_query,
        context_block=context_block,
        history=history,
    )

    completion_id = _make_completion_id()

    # 5. Generate response
    if request.stream:
        return StreamingResponse(
            _stream_response(completion_id, request.model, llm_messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    try:
        answer = await generate(llm_messages)
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        return _openai_error(
            f"Generation failed: {exc}",
            code="internal_error",
            status_code=500,
        )

    return _build_non_streaming_response(completion_id, request.model, answer)


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------


async def _stream_response(
    completion_id: str,
    model: str,
    llm_messages: list[dict[str, str]],
):
    """Async generator that yields OpenAI-format SSE chunks."""
    try:
        queue = await generate_stream(llm_messages)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield _build_streaming_chunk(completion_id, model, content=token)

        # Final chunk with finish_reason
        yield _build_streaming_chunk(completion_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error(f"Streaming generation error: {exc}")
        # Send an error chunk and terminate
        error_chunk = {
            "error": {
                "message": str(exc),
                "type": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _openai_error(
    message: str,
    code: str = "internal_error",
    status_code: int = 500,
) -> dict[str, Any]:
    """Return an OpenAI-style error response dict.

    We use a plain dict + JSONResponse status code rather than raising
    because Open WebUI expects a specific JSON shape for errors.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": code,
                "param": None,
                "code": code,
            }
        },
    )
