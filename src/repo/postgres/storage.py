import os
import psycopg
from functools import lru_cache
from typing import Optional
from psycopg import sql
from pgvector import Vector, SparseVector
from pgvector.psycopg import register_vector
from llama_index.core.schema import BaseNode
from src import schemas
from src.core import config
from src.utils import logger


def _get_db_params() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "pg"),
        "dbname": os.getenv("POSTGRES_DB", "cs419_db"),
    }


@lru_cache(maxsize=1)
def get_pg_conn() -> psycopg.Connection:
    params = _get_db_params()
    conn = psycopg.connect(**params)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    register_vector(conn)
    return conn


def ensure_collection_exists(
    collection_name: str,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    dense_dim: int = config.DENSE_DIM,
    sparse_dim: int = config.SPARSE_DIM,
) -> None:
    conn = get_pg_conn()

    create_table = sql.SQL(
        """
		CREATE TABLE IF NOT EXISTS {table} (
			id TEXT PRIMARY KEY,
			text TEXT NOT NULL,
			document_id TEXT,
			title TEXT,
			file_name TEXT,
			file_path TEXT,
			{dense_col} vector({dense_dim}) NOT NULL,
			{sparse_col} sparsevec({sparse_dim})
		);
		"""
    ).format(
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
        sparse_col=sql.Identifier(sparse_name),
        dense_dim=sql.Literal(int(dense_dim)),
        sparse_dim=sql.Literal(int(sparse_dim)),
    )

    create_dense_index = sql.SQL(
        """
		DO $$
		BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM pg_class c
				JOIN pg_namespace n ON n.oid = c.relnamespace
				WHERE c.relname = {idx_name}
			) THEN
				CREATE INDEX {index} ON {table}
				USING hnsw ({dense_col} vector_cosine_ops) WITH (m = 32, ef_construction = 128);
			END IF;
		END $$;
		"""
    ).format(
        idx_name=sql.Literal(f"{collection_name}_{dense_name}_idx"),
        index=sql.Identifier(f"{collection_name}_{dense_name}_idx"),
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
    )

    with conn.cursor() as cur:
        cur.execute(create_table)
        cur.execute(create_dense_index)


def _to_sparsevec(indices: list[int], values: list[float]) -> SparseVector:
    # Convert indices/values to dict form required by SparseVector with dimension
    elem = {int(i): float(v) for i, v in zip(indices, values)}
    return SparseVector(elem, config.SPARSE_DIM)


def upsert_data(
    nodes: list[BaseNode],
    dense_embeddings: list[list[float]],
    sparse_embeddings: Optional[list[tuple[list[int], list[float]]]],
    collection_name: str,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    dense_dim: int = config.DENSE_DIM,
    sparse_dim: int = config.SPARSE_DIM,
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

    conn = get_pg_conn()
    ensure_collection_exists(
        collection_name=collection_name,
        dense_name=dense_name,
        sparse_name=sparse_name,
        dense_dim=dense_dim,
        sparse_dim=sparse_dim,
    )

    insert_stmt = sql.SQL(
        """
		INSERT INTO {table} (id, text, document_id, title, file_name, file_path, {dense_col}, {sparse_col})
		VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
		ON CONFLICT (id) DO UPDATE SET
			text = EXCLUDED.text,
			document_id = EXCLUDED.document_id,
			title = EXCLUDED.title,
			file_name = EXCLUDED.file_name,
			file_path = EXCLUDED.file_path,
			{dense_col} = EXCLUDED.{dense_col},
			{sparse_col} = EXCLUDED.{sparse_col};
		"""
    ).format(
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
        sparse_col=sql.Identifier(sparse_name),
    )

    rows = []
    for i, node in enumerate(nodes):
        payload = schemas.DocumentPayload(
            text=node.text,
            metadata=schemas.DocumentMetadata.model_validate(node.metadata),
        )

        dense_vec = Vector(dense_embeddings[i])
        sparse_vec = None
        if sparse_embeddings is not None:
            idxs, vals = sparse_embeddings[i]
            if len(idxs) > 0:
                sparse_vec = _to_sparsevec(idxs, vals)

        rows.append(
            (
                node.id_,
                payload.text,
                payload.metadata.document_id,
                payload.metadata.title,
                payload.metadata.file_name,
                payload.metadata.file_path,
                dense_vec,
                sparse_vec,
            )
        )

    with conn.cursor() as cur:
        cur.executemany(insert_stmt, rows)
