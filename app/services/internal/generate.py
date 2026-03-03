"""Internal service: LLM text generation via Cerebras API.

Provides both **non-streaming** and **streaming** generation.
The Cerebras SDK exposes an OpenAI-compatible ``chat.completions.create()``
interface, so the same patterns used in ``chunk.py`` for title generation
apply here for RAG answer generation.

All synchronous SDK calls are wrapped for ``asyncio.run_in_executor`` so the
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from cerebras.cloud.sdk import Cerebras

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_generation_client() -> Cerebras:
    """Return a cached Cerebras client for generation."""
    return Cerebras(api_key=settings.CEREBRAS_API_KEY)


# ---------------------------------------------------------------------------
# RAG system prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant. Answer the user's question based on the \
provided context documents. Follow these rules:

1. Use ONLY the information from the context to answer. If the context does \
not contain enough information, say so honestly.
2. Cite document titles or sources when referencing specific information.
3. Be concise and direct. Avoid unnecessary filler.
4. If the user asks a follow-up question, use conversation history for context \
but always ground answers in the retrieved documents.
5. Answer in the same language as the user's question.
"""


def build_context_block(sources: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the LLM prompt.

    Each source dict should have at least ``text`` and optionally ``title``
    and ``score``.
    """
    if not sources:
        return "(No relevant documents found.)"

    parts: list[str] = []
    for i, src in enumerate(sources, 1):
        title = src.get("title") or "Untitled"
        text = src.get("text", "")
        score = src.get("score")
        header = f"[Document {i}: {title}]"
        if score is not None:
            header += f" (relevance: {score:.2f})"
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def build_messages(
    *,
    user_query: str,
    context_block: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Assemble the full message list for ``chat.completions.create()``.

    Layout (follows Anthropic / OpenAI best practice for RAG):

    1. **System**: RAG instruction prompt
    2. **History**: previous (user, assistant) turns (trimmed to N)
    3. **User**: context block + current question

    The context is injected in the *latest user turn* rather than the system
    prompt so the model treats it as grounding material for the current
    question, not as a persistent instruction that might leak across turns.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
    ]

    # Append trimmed history
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    # Current user turn: context + question
    user_content = (
        f"Context documents:\n\n{context_block}\n\n---\n\nQuestion: {user_query}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Synchronous helpers (called via run_in_executor)
# ---------------------------------------------------------------------------


def _generate_sync(messages: list[dict[str, str]]) -> str:
    """Blocking call to Cerebras chat completion.  Returns the full response."""
    client = _get_generation_client()
    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL,
        messages=messages,
        max_tokens=settings.GENERATION_MAX_TOKENS,
        temperature=settings.GENERATION_TEMPERATURE,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _generate_stream_sync(
    messages: list[dict[str, str]],
) -> Iterator[str]:
    """Blocking *streaming* call.  Yields content delta strings."""
    client = _get_generation_client()
    stream = client.chat.completions.create(
        model=settings.GENERATION_MODEL,
        messages=messages,
        max_tokens=settings.GENERATION_MAX_TOKENS,
        temperature=settings.GENERATION_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------


async def generate(messages: list[dict[str, str]]) -> str:
    """Generate a complete response (non-streaming).

    Offloads the blocking Cerebras SDK call to the default thread-pool
    executor so the event loop stays free.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generate_sync, messages)


async def generate_stream(
    messages: list[dict[str, str]],
) -> asyncio.Queue[str | None]:
    """Start a streaming generation and return an ``asyncio.Queue``.

    The caller reads tokens from the queue.  A ``None`` sentinel signals
    end-of-stream.  The actual blocking iteration runs in a thread-pool
    executor.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer() -> None:
        try:
            for token in _generate_stream_sync(messages):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            logger.error(f"Streaming generation error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)
    return queue
