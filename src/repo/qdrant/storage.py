from functools import lru_cache
from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from typing import Optional
from src import schemas
from src.core import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=30)


def ensure_collection_exists(
    collection_name: str,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    dense_dim: int = config.DENSE_DIM,
) -> None:
    client = get_qdrant_client()

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_name: models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=100,
                ),
            ),
        },
        sparse_vectors_config={
            sparse_name: models.SparseVectorParams(),
        },
    )
    print(f"Created collection '{collection_name}'")


def upsert_data(
    nodes: list[BaseNode],
    dense_embeddings: list[list[float]],
    sparse_embeddings: Optional[list[tuple[list[int], list[float]]]],
    collection_name: str,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    dense_dim: int = config.DENSE_DIM,
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if len(dense_embeddings) != len(nodes):
        raise ValueError(
            f"The number of dense embeddings ({len(dense_embeddings)}) must match the number of nodes ({len(nodes)})"
        )

    if sparse_embeddings is not None and len(sparse_embeddings) != len(nodes):
        raise ValueError(
            f"The number of sparse embeddings ({len(sparse_embeddings)}) must match the number of nodes ({len(nodes)})"
        )

    client = get_qdrant_client()
    ensure_collection_exists(
        collection_name=collection_name,
        dense_name=dense_name,
        sparse_name=sparse_name,
        dense_dim=dense_dim,
    )

    points: list[models.PointStruct] = []
    for i, node in enumerate(nodes):
        vector_map: dict[str, object] = {
            dense_name: dense_embeddings[i],
            sparse_name: (
                models.SparseVector(
                    indices=sparse_embeddings[i][0], values=sparse_embeddings[i][1]
                )
                if sparse_embeddings is not None
                else models.SparseVector(indices=[], values=[])
            ),
        }

        payload = schemas.DocumentPayload(
            text=node.text,
            metadata=schemas.DocumentMetadata.model_validate(node.metadata),
        )

        points.append(
            models.PointStruct(
                id=node.id_,
                vector=vector_map,
                payload=payload.model_dump(),
            )
        )

    # Upsert points to Qdrant
    out = client.upsert(
        collection_name=collection_name,
        points=points,
    )

    # Qdrant returns UpdateStatus.ACKNOWLEDGED or COMPLETED
    if out.status not in (
        models.UpdateStatus.ACKNOWLEDGED,
        models.UpdateStatus.COMPLETED,
    ):
        raise RuntimeError(f"Failed to upsert nodes: {out}")
