"""Public service: RAG conversation orchestration.

Coordinates:
1. Conversation CRUD (Milvus conversation repo)
2. Document retrieval (reuses existing search service)
3. LLM generation (internal generate service)
4. Message persistence

All heavy I/O (Milvus reads/writes, embedding, LLM calls) is async or
offloaded via ``run_in_executor`` so concurrent requests are never blocked.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from app.core.config import settings
from app.core.logging import logger
from app.models.conversation import ConversationMeta, Message
from app.repositories.milvus.conversations import (
    create_conversation as _create_conv,
    get_conversation as _get_conv,
    list_conversations as _list_convs,
    update_conversation_title as _update_title,
    delete_conversation as _delete_conv,
    save_messages as _save_msgs,
    get_messages as _get_msgs,
)
from app.schemas.conversations import (
    ConversationListItem,
    ConversationResponse,
    MessageResponse,
    SendMessageResponse,
    SourceDocument,
)
from app.services.internal.generate import (
    build_context_block,
    build_messages,
    generate,
    generate_stream,
)
from app.services.public.search import search_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg_to_response(msg: Message) -> MessageResponse:
    sources: list[SourceDocument] | None = None
    if msg.sources:
        sources = [SourceDocument(**s) for s in msg.sources]
    return MessageResponse(
        message_id=msg.message_id,
        role=msg.role,
        content=msg.content,
        sources=sources,
        created_at=msg.created_at,
    )


def _meta_to_list_item(meta: ConversationMeta) -> ConversationListItem:
    return ConversationListItem(
        conversation_id=meta.conversation_id,
        title=meta.title,
        collection_name=meta.collection_name,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


def _trim_history(
    messages: list[Message],
    max_turns: int,
) -> list[dict[str, str]]:
    """Return the last *max_turns* (user, assistant) pairs as dicts.

    Excludes system messages and the current user message (which will be
    appended separately with context).
    """
    relevant = [m for m in messages if m.role in ("user", "assistant")]
    trimmed = relevant[-(max_turns * 2) :]
    return [{"role": m.role, "content": m.content} for m in trimmed]


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


async def create_conversation(
    collection_name: str,
    title: str | None = None,
) -> ConversationResponse:
    """Create a new conversation and return its metadata."""
    conversation_id = str(uuid.uuid4())
    meta = ConversationMeta(
        conversation_id=conversation_id,
        title=title,
        collection_name=collection_name,
    )
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _create_conv, meta)
    logger.info(f"Created conversation {conversation_id} for '{collection_name}'")
    return ConversationResponse(
        conversation_id=meta.conversation_id,
        title=meta.title,
        collection_name=meta.collection_name,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        messages=[],
    )


async def get_conversation(conversation_id: str) -> ConversationResponse | None:
    """Load a conversation with its full message history."""
    loop = asyncio.get_running_loop()

    meta, messages = await asyncio.gather(
        loop.run_in_executor(None, _get_conv, conversation_id),
        loop.run_in_executor(None, _get_msgs, conversation_id),
    )

    if meta is None:
        return None

    return ConversationResponse(
        conversation_id=meta.conversation_id,
        title=meta.title,
        collection_name=meta.collection_name,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        messages=[_msg_to_response(m) for m in messages],
    )


async def list_conversations(
    collection_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ConversationListItem]:
    """List conversations, optionally filtered by collection."""
    loop = asyncio.get_running_loop()
    metas = await loop.run_in_executor(
        None, _list_convs, collection_name, limit, offset
    )
    return [_meta_to_list_item(m) for m in metas]


async def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation. Returns True if it existed."""
    loop = asyncio.get_running_loop()
    meta = await loop.run_in_executor(None, _get_conv, conversation_id)
    if meta is None:
        return False
    await loop.run_in_executor(None, _delete_conv, conversation_id)
    logger.info(f"Deleted conversation {conversation_id}")
    return True


# ---------------------------------------------------------------------------
# RAG chat — non-streaming
# ---------------------------------------------------------------------------


async def send_message(
    conversation_id: str,
    user_content: str,
    *,
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
    top_k: int = 5,
    rerank: bool = False,
) -> SendMessageResponse:
    """Full RAG pipeline: retrieve -> augment -> generate -> store.

    Steps:
    1. Load conversation metadata + history (concurrent)
    2. Retrieve relevant documents from the vector store
    3. Build prompt (system + history + context + question)
    4. Call LLM for answer
    5. Persist both messages + auto-title on first message
    """
    loop = asyncio.get_running_loop()

    # 1. Load conversation meta + existing messages concurrently
    meta, existing_msgs = await asyncio.gather(
        loop.run_in_executor(None, _get_conv, conversation_id),
        loop.run_in_executor(None, _get_msgs, conversation_id),
    )
    if meta is None:
        from app.middleware.errors import ApiError

        raise ApiError(
            code="conversation_not_found",
            message=f"Conversation '{conversation_id}' not found.",
            status_code=404,
        )

    # 2. Retrieve relevant documents
    search_results = await search_documents(
        query=user_content,
        collection_name=meta.collection_name,
        search_type=search_type,
        top_k=top_k,
        rerank=rerank,
    )

    sources: list[dict[str, Any]] = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "text": r.text,
            "score": r.score,
        }
        for r in search_results
    ]

    # 3. Build LLM messages
    history = _trim_history(existing_msgs, settings.GENERATION_HISTORY_TURNS)
    context_block = build_context_block(sources)
    llm_messages = build_messages(
        user_query=user_content,
        context_block=context_block,
        history=history,
    )

    # 4. Generate answer
    answer = await generate(llm_messages)

    # 5. Create Message objects
    now = datetime.now(timezone.utc)
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=user_content,
        created_at=now,
    )
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources if sources else None,
        created_at=now,
    )

    # 6. Persist messages + auto-title if first message
    await loop.run_in_executor(None, _save_msgs, [user_msg, assistant_msg])

    if not existing_msgs and not meta.title:
        # Auto-generate title from first user message (truncate to keep it short)
        auto_title = user_content[:100].strip()
        if len(user_content) > 100:
            auto_title += "..."
        await loop.run_in_executor(None, _update_title, conversation_id, auto_title)

    logger.info(
        f"Chat {conversation_id}: query={user_content!r:.80}, "
        f"sources={len(sources)}, answer_len={len(answer)}"
    )

    return SendMessageResponse(
        user_message=_msg_to_response(user_msg),
        assistant_message=_msg_to_response(assistant_msg),
    )


# ---------------------------------------------------------------------------
# RAG chat — streaming
# ---------------------------------------------------------------------------


async def send_message_stream(
    conversation_id: str,
    user_content: str,
    *,
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
    top_k: int = 5,
    rerank: bool = False,
):
    """Streaming RAG pipeline.  Yields SSE-formatted chunks.

    Same pipeline as ``send_message`` but streams the LLM response
    token-by-token.  The full answer is accumulated and persisted after
    the stream completes.

    Yields dicts suitable for ``sse_starlette.EventSourceResponse``:
    - ``{"event": "source", "data": ...}`` — retrieved documents (sent first)
    - ``{"event": "delta", "data": ...}`` — token deltas
    - ``{"event": "done", "data": ...}``  — final message IDs
    """
    import json

    loop = asyncio.get_running_loop()

    # 1. Load conversation
    meta, existing_msgs = await asyncio.gather(
        loop.run_in_executor(None, _get_conv, conversation_id),
        loop.run_in_executor(None, _get_msgs, conversation_id),
    )
    if meta is None:
        from app.middleware.errors import ApiError

        raise ApiError(
            code="conversation_not_found",
            message=f"Conversation '{conversation_id}' not found.",
            status_code=404,
        )

    # 2. Retrieve documents
    search_results = await search_documents(
        query=user_content,
        collection_name=meta.collection_name,
        search_type=search_type,
        top_k=top_k,
        rerank=rerank,
    )

    sources: list[dict[str, Any]] = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "text": r.text,
            "score": r.score,
        }
        for r in search_results
    ]

    # Yield sources event
    yield {
        "event": "sources",
        "data": json.dumps(sources),
    }

    # 3. Build prompt
    history = _trim_history(existing_msgs, settings.GENERATION_HISTORY_TURNS)
    context_block = build_context_block(sources)
    llm_messages = build_messages(
        user_query=user_content,
        context_block=context_block,
        history=history,
    )

    # 4. Stream generation
    queue = await generate_stream(llm_messages)
    full_answer_parts: list[str] = []

    while True:
        token = await queue.get()
        if token is None:
            break
        full_answer_parts.append(token)
        yield {
            "event": "delta",
            "data": json.dumps({"content": token}),
        }

    full_answer = "".join(full_answer_parts)

    # 5. Persist messages
    now = datetime.now(timezone.utc)
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=user_content,
        created_at=now,
    )
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=full_answer,
        sources=sources if sources else None,
        created_at=now,
    )
    await loop.run_in_executor(None, _save_msgs, [user_msg, assistant_msg])

    if not existing_msgs and not meta.title:
        auto_title = user_content[:100].strip()
        if len(user_content) > 100:
            auto_title += "..."
        await loop.run_in_executor(None, _update_title, conversation_id, auto_title)

    # 6. Done event
    yield {
        "event": "done",
        "data": json.dumps(
            {
                "user_message_id": user_msg.message_id,
                "assistant_message_id": assistant_msg.message_id,
            }
        ),
    }
