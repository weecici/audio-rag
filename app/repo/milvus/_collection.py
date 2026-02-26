from pymilvus import MilvusClient, DataType, Function, FunctionType, CollectionSchema

from app.core import config
from app.util import logger
from ._client import get_client


def _create_index_params(client: MilvusClient):
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vec_hnsw_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": config.MILVUS_HNSW_M,
            "efConstruction": config.MILVUS_HNSW_EF_CONSTRUCTION,
        },
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vec_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": config.MILVUS_BM25_K1,
            "bm25_b": config.MILVUS_BM25_B,
        },
    )

    return index_params


def _create_schema(client: MilvusClient) -> CollectionSchema:
    schema = client.create_schema()

    schema.add_field(
        field_name="doc_id", datatype=DataType.INT64, is_primary=True, auto_id=False
    )
    schema.add_field(
        field_name="title", datatype=DataType.VARCHAR, max_length=100, nullable=True
    )
    schema.add_field(
        field_name="author_info",
        datatype=DataType.VARCHAR,
        max_length=100,
        nullable=True,
    )
    schema.add_field(
        field_name="tags",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=30,
        nullable=True,
    )
    schema.add_field(field_name="metadata", datatype=DataType.JSON, nullable=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=10000,
        enable_analyzer=True,
    )
    schema.add_field(
        field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=config.DENSE_DIM
    )
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="created_at", datatype=DataType.TIMESTAMPTZ)
    schema.add_field(
        field_name="updated_at", datatype=DataType.TIMESTAMPTZ, nullable=True
    )

    bm25_fn = Function(
        name="chunk_text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )

    schema.add_function(bm25_fn)

    return schema


def create_collection(collection_name: str) -> None:
    client = get_client()

    if client.has_collection(collection_name):
        logger.info(
            f"Collection '{collection_name}' already exists. Skipping creation."
        )
        return

    logger.info(f"Creating collection '{collection_name}'...")

    schema = _create_schema(client)
    index_params = _create_index_params(client)
    client.create_collection(collection_name, schema=schema, index_params=index_params)


def delete_collection(collection_name: str) -> None:
    client = get_client()

    if not client.has_collection(collection_name):
        logger.info(
            f"Collection '{collection_name}' does not exist. Skipping deletion."
        )
        return

    logger.info(f"Deleting collection '{collection_name}'...")
    client.drop_collection(collection_name)
