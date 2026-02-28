"""Dense embedding via Google Gemini Embedding API."""

import asyncio
from functools import lru_cache
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    """Return a cached Google GenAI client."""
    return genai.Client(api_key=settings.GOOGLE_API_KEY)


def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts synchronously using Google Gemini Embedding."""
    client = _get_genai_client()
    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=settings.DENSE_DIM,
        ),
    )
    return [list(e.values) for e in response.embeddings]


async def dense_encode(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching as needed.

    Returns a list of float vectors, one per input text, each of
    dimension ``settings.DENSE_DIM``.
    """
    if not texts:
        return []

    batch_size = settings.EMBEDDING_BATCH_SIZE
    loop = asyncio.get_running_loop()

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.debug(
            f"Embedding batch {start // batch_size + 1} "
            f"({len(batch)} texts, model={settings.EMBEDDING_MODEL})"
        )
        vectors = await loop.run_in_executor(None, _embed_batch_sync, batch)
        all_vectors.extend(vectors)

    return all_vectors
