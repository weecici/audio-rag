"""Dense embedding via Google Gemini Embedding API."""

import asyncio
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Return a cached Google Generative AI Embeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.EMBEDDING_DIM,
    )


def _embed_batch_sync(
    texts: list[str], titles: Optional[list[str]] = None
) -> list[list[float]]:
    """Embed a batch of texts synchronously using Google Generative AI Embeddings."""
    client = _get_embedding_client()
    return client.embed_documents(texts=texts, titles=titles)


async def dense_encode(
    texts: list[str], titles: Optional[list[str]] = None
) -> list[list[float]]:
    """Embed a list of texts, batching as needed.

    Returns a list of float vectors, one per input text, each of
    dimension ``settings.EMBEDDING_DIM``.
    """
    if not texts:
        return []

    if titles and len(titles) != len(texts):
        logger.warning("Length of titles does not match texts; expanding for embedding")
        titles = titles + [None] * (len(texts) - len(titles))

    batch_size = settings.EMBEDDING_BATCH_SIZE
    loop = asyncio.get_running_loop()

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_titles = titles[start : start + batch_size] if titles else None
        logger.debug(
            f"Embedding batch {start // batch_size + 1} "
            f"({len(batch_texts)} texts, model={settings.EMBEDDING_MODEL})"
        )
        vectors = await loop.run_in_executor(
            None, _embed_batch_sync, batch_texts, batch_titles
        )
        all_vectors.extend(vectors)

    return all_vectors
