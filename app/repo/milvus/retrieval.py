from datetime import datetime, timezone
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

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

    def _parse_timestamptz(v: object) -> object:
        if not isinstance(v, str):
            return v
        # Milvus often returns RFC3339 with trailing 'Z'.
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.fromisoformat(v)

    entity["created_at"] = _parse_timestamptz(entity.get("created_at"))
    entity["updated_at"] = _parse_timestamptz(entity.get("updated_at"))

    return schema.Document(**entity)


def dense_search(
    query_vectors: list[list[float]],
    collection_name: str,
    top_k: int = 5,
) -> list[list[schema.Document]]:
    client = get_client()
    if not client.has_collection(collection_name):
        return [[] for _ in range(len(query_vectors))]

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
        return [[] for _ in range(len(query_texts))]

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


def hybrid_search(
    query_vectors: list[list[float]],
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
) -> list[list[schema.Document]]:
    client = get_client()
    if not client.has_collection(collection_name):
        return [[] for _ in range(len(query_vectors))]

    client.load_collection(collection_name)

    if len(query_vectors) != len(query_texts):
        raise ValueError("query_vectors and query_texts must have same length")

    req_dense = AnnSearchRequest(
        data=query_vectors,
        anns_field="dense_vector",
        param={
            "metric_type": "COSINE",
            "params": {"ef": config.MILVUS_HNSW_EF},
        },
        limit=top_k,
    )
    req_sparse = AnnSearchRequest(
        data=query_texts,
        anns_field="sparse_vector",
        param={"metric_type": "BM25", "params": {}},
        limit=top_k,
    )

    # Keep config backwards-compatible: default FUSION_METHOD is 'dbsf'.
    fusion = config.FUSION_METHOD
    if fusion in ("weighted", "dbsf"):
        ranker = WeightedRanker(
            config.FUSION_ALPHA, 1 - config.FUSION_ALPHA, norm_score=True
        )
    elif fusion == "rrf":
        ranker = RRFRanker(k=max(1, int(config.RRF_K)))
    else:
        raise ValueError(f"Unsupported fusion method: {fusion}")

    raw = client.hybrid_search(
        collection_name=collection_name,
        reqs=[req_dense, req_sparse],
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
        ranker=ranker,
    )

    return [[_hit_to_document(h) for h in hits] for hits in raw]
