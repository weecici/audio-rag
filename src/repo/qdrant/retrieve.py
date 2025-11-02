import os
import json
from typing import Optional, Literal
from qdrant_client import models
from .storage import get_qdrant_client
from src.core import config
from src.utils import tokenize
from collections import Counter


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


def sparse_query_vectorize(
    tokens: list[str], vocab: dict[str, int]
) -> models.SparseVector:
    word_counts = Counter(tokens)

    indices = [vocab[token] for token in word_counts.keys() if token in vocab]
    values = [
        float(word_counts[token]) for token in word_counts.keys() if token in vocab
    ]

    return models.SparseVector(indices=indices, values=values)


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    filter: Optional[models.Filter] = None,
    dense_name: str = config.DENSE_MODEL,
) -> list[list[dict]]:

    client = get_qdrant_client()

    search_queries = []
    for query_emb in query_embeddings:
        search_queries.append(
            models.SearchRequest(
                vector=models.NamedVector(name=dense_name, vector=query_emb),
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
    query_texts: list[str],
    collection_name: str,
    word_process_method: str = config.WORD_PROCESS_METHOD,
    top_k: int = 5,
    filter: Optional[models.Filter] = None,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[dict]]:
    client = get_qdrant_client()
    vocab = _load_vocab(collection_name)

    tokenized_query_texts = tokenize(
        texts=query_texts,
        word_process_method=word_process_method,
        return_ids=False,
    )

    search_queries = []
    for tokens in tokenized_query_texts:
        sparse_vector = sparse_query_vectorize(tokens=tokens, vocab=vocab)
        search_queries.append(
            models.SearchRequest(
                vector=models.NamedSparseVector(name=sparse_name, vector=sparse_vector),
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
    query_embeddings: list[list[float]],
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[dict]]:
    client = get_qdrant_client()
    vocab = _load_vocab(collection_name)

    tokenized_query_texts = tokenize(
        texts=query_texts,
        word_process_method=config.WORD_PROCESS_METHOD,
        return_ids=False,
    )

    if fusion_method.lower() == "dbsf":
        fm = models.Fusion.DBSF
    elif fusion_method.lower() == "rrf":
        fm = models.Fusion.RRF
    else:
        raise ValueError(f"Invalid fusion method: {fusion_method}")

    query_requests: list[models.QueryRequest] = []
    for i, tokens in enumerate(tokenized_query_texts):
        sparse_vector = sparse_query_vectorize(tokens=tokens, vocab=vocab)
        query_requests.append(
            models.QueryRequest(
                prefetch=[
                    models.Prefetch(
                        query=query_embeddings[i],
                        using=dense_name,
                        limit=int(top_k * overfetch_mul),
                    ),
                    models.Prefetch(
                        query=sparse_vector,
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
