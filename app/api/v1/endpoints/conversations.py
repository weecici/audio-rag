"""Conversation endpoints – multi-turn RAG chat with SSE streaming support."""

from typing import Optional

from fastapi import APIRouter, Query, status
from sse_starlette.sse import EventSourceResponse

from app.middleware.errors import ApiError
from app.schemas.conversations import (
    ConversationListItem,
    ConversationResponse,
    CreateConversationRequest,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.public.conversations import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    send_message,
    send_message_stream,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
    description=(
        "Create a new conversation linked to a document collection. "
        "An optional title can be provided; otherwise it will be auto-generated "
        "from the first user message."
    ),
)
async def create_conversation_endpoint(
    request: CreateConversationRequest,
) -> ConversationResponse:
    return await create_conversation(
        collection_name=request.collection_name,
        title=request.title,
    )


@router.get(
    "",
    response_model=list[ConversationListItem],
    summary="List conversations",
    description="List conversations, optionally filtered by collection name.",
)
async def list_conversations_endpoint(
    collection_name: Optional[str] = Query(
        None, description="Filter by document collection name."
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
) -> list[ConversationListItem]:
    return await list_conversations(
        collection_name=collection_name,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation",
    description="Load a conversation with its full message history.",
)
async def get_conversation_endpoint(
    conversation_id: str,
) -> ConversationResponse:
    conv = await get_conversation(conversation_id)
    if conv is None:
        raise ApiError(
            code="conversation_not_found",
            message=f"Conversation '{conversation_id}' not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return conv


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
    description="Delete a conversation and all its messages.",
)
async def delete_conversation_endpoint(
    conversation_id: str,
) -> None:
    deleted = await delete_conversation(conversation_id)
    if not deleted:
        raise ApiError(
            code="conversation_not_found",
            message=f"Conversation '{conversation_id}' not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=SendMessageResponse,
    summary="Send a message (RAG)",
    description=(
        "Send a user message and receive a RAG-augmented response. "
        "Set `stream=true` in the request body to receive Server-Sent Events "
        "with token-by-token streaming."
    ),
)
async def create_message(
    conversation_id: str,
    request: SendMessageRequest,
):
    if request.stream:
        return EventSourceResponse(
            send_message_stream(
                conversation_id=conversation_id,
                user_content=request.content,
                search_type=request.search_type,
                top_k=request.top_k,
                rerank=request.rerank,
            )
        )

    result = await send_message(
        conversation_id=conversation_id,
        user_content=request.content,
        search_type=request.search_type,
        top_k=request.top_k,
        rerank=request.rerank,
    )
    return result
