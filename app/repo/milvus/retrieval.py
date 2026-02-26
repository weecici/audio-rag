from datetime import datetime, timezone
from app import schema
from app.core import config

from ._client import get_client


_OUTPUT_FIELDS = [
    "title",
    "author_info",
    "tags",
    "metadata",
    "text",
    "dense_vector",
    "created_at",
    "updated_at",
]


def _hit_to_document(hit: dict) -> schema.Document:
    entity = dict(hit.get("entity") or {})
    # PyMilvus returns primary key in `hit['id']` and also sometimes as `hit['doc_id']`.
    doc_id = hit.get("id")
    if doc_id is None:
        doc_id = hit.get("doc_id")
    if doc_id is None:
        doc_id = entity.get("doc_id")
    if doc_id is None:
        raise ValueError("Milvus search hit missing primary key (doc_id)")
    entity["doc_id"] = int(doc_id)
    if not entity.get("title"):
        entity["title"] = "untitled"
    entity.setdefault("created_at", datetime.now(timezone.utc))
    entity.setdefault("updated_at", None)
    entity.setdefault("sparse_vector", None)

    # TIMESTAMPTZ comes back as an ISO string.
    if isinstance(entity.get("created_at"), str):
        entity["created_at"] = datetime.fromisoformat(entity["created_at"])
    if isinstance(entity.get("updated_at"), str):
        entity["updated_at"] = datetime.fromisoformat(entity["updated_at"])

    return schema.Document(**entity)


def dense_search(
    query_vectors: list[list[float]],
    collection_name: str,
    top_k: int = 5,
) -> list[list[schema.Document]]:
    client = get_client()
    if not client.has_collection(collection_name):
        return [[] for _ in query_vectors]

    client.load_collection(collection_name)
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": config.MILVUS_HNSW_EF},
    }

    raw = client.search(
        collection_name=collection_name,
        data=query_vectors,
        anns_field="dense_vector",
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
        search_params=search_params,
    )

    return [[_hit_to_document(h) for h in hits] for hits in raw]


def sparse_search(
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
) -> list[list[schema.Document]]:
    client = get_client()
    if not client.has_collection(collection_name):
        return [[] for _ in query_texts]

    client.load_collection(collection_name)
    search_params = {"metric_type": "BM25", "params": {}}

    raw = client.search(
        collection_name=collection_name,
        data=query_texts,
        anns_field="sparse_vector",
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
        search_params=search_params,
    )

    return [[_hit_to_document(h) for h in hits] for hits in raw]


def hybrid_search(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Hybrid retrieval not implemented yet")
