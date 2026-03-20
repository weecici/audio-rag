"""OpenAI-compatible endpoints for Open WebUI integration.

Provides ``/models`` and ``/chat/completions`` so that Open WebUI
(or any OpenAI-compatible client) can use our RAG backend as a drop-in
LLM provider.

* One model per Milvus collection — ``GET /models`` enumerates every
  document collection (excluding internal ``_``-prefixed ones) and exposes
  each as ``RAG_KB/{collection_name}``.
"""

from fastapi import APIRouter
from app.schemas import (
    ChatCompletionRequest,
    ModelListResponse,
)
from app.services.public.openai_compat import (
    list_models as list_models_svc,
    chat_completions as chat_completions_svc,
)

router = APIRouter(tags=["OpenAI Compatible"])

# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available RAG models",
    description=(
        "Returns one model per Milvus document collection. "
        "Each model ID has the format ``rag-{collection_name}``."
    ),
)
async def list_models() -> ModelListResponse:
    return await list_models_svc()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@router.post(
    "/chat/completions",
    summary="Chat completion with RAG",
    description=(
        "OpenAI-compatible chat completion endpoint. The model ID encodes "
        "the Milvus collection to query. "
        "Supports both streaming (SSE) and non-streaming responses."
    ),
)
async def chat_completions(request: ChatCompletionRequest):
    return await chat_completions_svc(request)
