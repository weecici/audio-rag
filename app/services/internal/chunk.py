"""Text chunking with LangChain + title generation via Cerebras LLM."""

import asyncio
import os
from dataclasses import dataclass
from cerebras.cloud.sdk import Cerebras
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A chunk of text with its position metadata."""

    text: str
    index: int  # 0-based position within the source document
    source: str  # originating file path or identifier
    title: Optional[str] = None  # populated later by LLM


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split *text* into overlapping chunks using LangChain's recursive splitter."""
    chunk_size = chunk_size or settings.MAX_TOKENS
    chunk_overlap = chunk_overlap or settings.OVERLAP_TOKENS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    return [TextChunk(text=c, index=i, source=source) for i, c in enumerate(raw_chunks)]


# ---------------------------------------------------------------------------
# Title generation via Cerebras
# ---------------------------------------------------------------------------

_cerebras_client: Cerebras | None = None


def _get_cerebras_client() -> Cerebras:
    global _cerebras_client
    if _cerebras_client is None:
        _cerebras_client = Cerebras(api_key=settings.CEREBRAS_API_KEY)
    return _cerebras_client


_TITLE_SYSTEM_PROMPT = (
    "You are a concise title generator. Given a text chunk from a document, "
    f"produce a short, descriptive title (max {settings.TITLE_MAX_TOKENS} tokens). "
    "Output ONLY the title, nothing else."
)


def _generate_title_sync(text: str) -> str:
    """Call Cerebras chat completion to generate a title for a chunk."""
    client = _get_cerebras_client()
    try:
        response = client.chat.completions.create(
            model=settings.TITLE_MODEL,
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": text[:2000]},  # limit context
            ],
            temperature=0.0,
        )
        title = (response.choices[0].message.content or "").strip().strip("\"'")
        print(response)
        return title or None
    except Exception as exc:
        logger.warning(f"Cerebras title generation failed: {exc}")
        return None


async def generate_titles(chunks: list[TextChunk]) -> list[TextChunk]:
    """Generate titles for all chunks concurrently via Cerebras."""
    loop = asyncio.get_running_loop()

    async def _title_one(chunk: TextChunk) -> None:
        title = await loop.run_in_executor(None, _generate_title_sync, chunk.text)
        chunk.title = title

    await asyncio.gather(*[_title_one(c) for c in chunks])
    return chunks
