"""Public service: execute vector-store searches.

Orchestrates query embedding (internal service) and Milvus search
(repository layer) to serve the ``POST /search/{collection_name}``
endpoint.

All heavy I/O (embedding API call, Milvus query) is offloaded via
``asyncio.run_in_executor`` so searches can happen concurrently.
"""

import asyncio
from typing import Literal

from app.core.logging import logger
from app.models import Document
from app.repositories.milvus import dense_search, sparse_search, hybrid_search
from app.services.internal.embed import embed_query
from app.schemas.search import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_to_result(
    doc: Document,
    score: float | None,
) -> SearchResult:
    """Convert a repository ``Document`` + optional score into a schema."""
    return SearchResult(
        doc_id=doc.doc_id,
        title=doc.title,
        text=doc.text,
        score=score,
        metadata=doc.metadata,
    )


# ---------------------------------------------------------------------------
# Search dispatchers (sync – run inside executor)
# ---------------------------------------------------------------------------


def _run_dense_search(
    query_vector: list[float],
    collection_name: str,
    top_k: int,
) -> list[tuple[Document, float | None]]:
    """Execute a dense (vector) search and return the first query's results."""
    batched = dense_search([query_vector], collection_name, top_k=top_k)
    return batched[0] if batched else []


def _run_sparse_search(
    query_text: str,
    collection_name: str,
    top_k: int,
) -> list[tuple[Document, float | None]]:
    """Execute a sparse (BM25) search and return the first query's results."""
    batched = sparse_search([query_text], collection_name, top_k=top_k)
    return batched[0] if batched else []


def _run_hybrid_search(
    query_vector: list[float],
    query_text: str,
    collection_name: str,
    top_k: int,
) -> list[tuple[Document, float | None]]:
    """Execute a hybrid (dense + BM25) search and return the first query's results."""
    batched = hybrid_search([query_vector], [query_text], collection_name, top_k=top_k)
    return batched[0] if batched else []


# ---------------------------------------------------------------------------
# Public async entry-point
# ---------------------------------------------------------------------------


async def search_documents(
    query: str,
    collection_name: str,
    *,
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
    top_k: int = 10,
) -> list[SearchResult]:
    """Run a search against the vector store and return ranked results.

    Concurrency:
    - Query embedding and Milvus search are both offloaded to the default
      thread-pool executor so multiple requests are served concurrently.
    - For *hybrid* search the query embedding is awaited first (needed as
      input), then the Milvus hybrid search runs in the executor.

    Args:
        query: Natural-language search query.
        collection_name: Target Milvus collection.
        search_type: One of ``"dense"``, ``"sparse"``, ``"hybrid"``.
        top_k: Maximum number of results to return.

    Returns:
        A list of :class:`SearchResult` in relevance order.
    """
    loop = asyncio.get_running_loop()

    if search_type == "sparse":
        hits = await loop.run_in_executor(
            None, _run_sparse_search, query, collection_name, top_k
        )

    elif search_type == "dense":
        query_vector = await embed_query(query)
        hits = await loop.run_in_executor(
            None, _run_dense_search, query_vector, collection_name, top_k
        )

    else:
        query_vector = await embed_query(query)
        hits = await loop.run_in_executor(
            None,
            _run_hybrid_search,
            query_vector,
            query,
            collection_name,
            top_k,
        )

    results = [_doc_to_result(doc, score) for doc, score in hits]

    logger.info(
        f"Search ({search_type}) on '{collection_name}': "
        f"query={query!r}, top_k={top_k}, returned={len(results)}"
    )
    return results
