import os
import json
from typing import Optional, Literal
from qdrant_client import models
from .storage import get_qdrant_client
from src.core import config


_vocab_cache: dict[str, dict[str, int]] = {}


def _load_vocab(collection_name: str) -> dict[str, int]:
    if collection_name in _vocab_cache:
        return _vocab_cache[collection_name]

    vocab_path = os.path.join(config.DISK_STORAGE_PATH, f"{collection_name}_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab: dict[str, int] = json.load(f)

    _vocab_cache[collection_name] = vocab
    return vocab


def _format_batch_results(
    batch_results: list[list[models.ScoredPoint]],
) -> list[list[dict]]:
    all_results: list[list[dict]] = []
    for search_result in batch_results:
        current_result: list[dict] = []

        for scored_point in search_result:
            payload = scored_point.payload or {}
            current_result.append(
                {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": payload,
                }
            )

        all_results.append(current_result)

    return all_results


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    filter: Optional[models.Filter] = None,
    dense_name: str = config.DENSE_MODEL,
) -> list[list[dict]]:

    client = get_qdrant_client()

    search_queries = []
    for query_embedding in query_embeddings:
        search_queries.append(
            models.SearchRequest(
                vector=models.NamedVector(name=dense_name, vector=query_embedding),
                limit=top_k,
                filter=filter,
                with_payload=True,
                with_vector=False,
            )
        )

    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
    )

    return _format_batch_results(batch_results)


def sparse_search(
    query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    filter: Optional[models.Filter] = None,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[dict]]:
    client = get_qdrant_client()
    vocab = _load_vocab(collection_name)

    search_queries = []
    for indices, values in query_embeddings:
        query_embedding = models.SparseVector(indices=indices, values=values)
        search_queries.append(
            models.SearchRequest(
                vector=models.NamedSparseVector(
                    name=sparse_name, vector=query_embedding
                ),
                limit=top_k,
                filter=filter,
                with_payload=True,
                with_vector=False,
            )
        )

    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
    )

    return _format_batch_results(batch_results)


def hybrid_search(
    dense_query_embeddings: list[list[float]],
    sparse_query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[dict]]:
    client = get_qdrant_client()

    if fusion_method.lower() == "dbsf":
        fm = models.Fusion.DBSF
    elif fusion_method.lower() == "rrf":
        fm = models.Fusion.RRF
    else:
        raise ValueError(f"Invalid fusion method: {fusion_method}")

    query_requests: list[models.QueryRequest] = []
    for i, (dense_embedding, (sparse_indices, sparse_values)) in enumerate(
        zip(dense_query_embeddings, sparse_query_embeddings)
    ):
        sparse_embedding = models.SparseVector(
            indices=sparse_indices, values=sparse_values
        )
        query_requests.append(
            models.QueryRequest(
                prefetch=[
                    models.Prefetch(
                        query=dense_embedding,
                        using=dense_name,
                        limit=int(top_k * overfetch_mul),
                    ),
                    models.Prefetch(
                        query=sparse_embedding,
                        using=sparse_name,
                        limit=int(top_k * overfetch_mul),
                    ),
                ],
                query=models.FusionQuery(fusion=fm),
                limit=top_k,
                with_payload=True,
                with_vector=False,
            )
        )

    batch_results = client.query_batch_points(
        collection_name=collection_name,
        requests=query_requests,
    )

    batch_results = [result.points for result in batch_results]
    return _format_batch_results(batch_results)
