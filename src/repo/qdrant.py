import json
from functools import lru_cache
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.core import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=30)


def ensure_collection_exists(
    collection_name: str, vector_size: int = config.EMBEDDING_DIM
) -> None:
    client = get_qdrant_client()

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        ),
    )
    print(f"Created collection '{collection_name}' with HNSW index for ANN search.")


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection_exists(collection_name)
    return QdrantVectorStore(client=client, collection_name=collection_name)


def upsert_nodes(
    nodes: list[BaseNode], collection_name: str, insert_batch_size: int = 512
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if not all(node.embedding is not None for node in nodes):
        raise ValueError("All nodes must have embeddings attached before upserting")

    vector_store = get_vector_store(collection_name)

    vector_store.add(nodes)

    print(
        f"Successfully upserted {len(nodes)} nodes into collection '{collection_name}'."
    )


def search_batch_similar_nodes(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    filters: models.Filter = None,
) -> list[tuple[list[BaseNode], list[float]]]:

    client = get_qdrant_client()

    search_queries = [
        models.SearchRequest(
            vector=query_emb,
            limit=top_k,
            filter=filters,
            with_payload=True,
            with_vector=True,
        )
        for query_emb in query_embeddings
    ]

    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
    )

    all_results = []
    for search_result in batch_results:
        current_result = []

        for scored_point in search_result:
            payload = scored_point.payload
            node_content_str = payload.get("_node_content")
            if node_content_str:
                node_content: dict = json.loads(node_content_str)

                important_content = {
                    k: node_content.get(k) for k in ["id_", "metadata", "text"]
                }
                important_content["score"] = scored_point.score

                current_result.append(important_content)
            else:
                raise ValueError(f"Missing node's contents")

        all_results.append(current_result)

    return all_results
