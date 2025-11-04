from typing import Optional, Literal
import math

import psycopg
from psycopg import sql
from pgvector import Vector, SparseVector
from pgvector.psycopg import register_vector

from src import schemas
from src.core import config
from src.services.internal import fuse_results
from .storage import get_pg_conn, ensure_collection_exists


def _rows_to_results(
    rows: list[tuple],
    *,
    distance_to_similarity: Optional[callable] = None,
) -> list[schemas.RetrievedDocument]:
    results: list[schemas.RetrievedDocument] = []
    for row in rows:
        # row: (id, score, text, document_id, title, file_name, file_path)
        (rid, score, text, document_id, title, file_name, file_path) = row
        # Convert distance to similarity if requested
        sim_score = float(score)
        if distance_to_similarity is not None:
            try:
                sim_score = float(distance_to_similarity(sim_score))
            except Exception:
                # Fallback to raw score if conversion fails
                sim_score = float(score)
        payload = schemas.DocumentPayload(
            text=text,
            metadata=schemas.DocumentMetadata(
                document_id=document_id or "",
                title=title or "",
                file_name=file_name or "",
                file_path=file_path or "",
            ),
        )
        results.append(
            schemas.RetrievedDocument(
                id=rid,
                score=sim_score,
                payload=payload,
            )
        )
    return results


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    dense_name: str = config.DENSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name, dense_name=dense_name)

    query_tmpl = sql.SQL(
        """
		SELECT id,
			   {dense_col} <=> %s AS score,
			   text,
			   document_id,
			   title,
			   file_name,
			   file_path
		FROM {table}
		ORDER BY {dense_col} <=> %s
		LIMIT %s;
		"""
    ).format(
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
    )

    all_results: list[list[schemas.RetrievedDocument]] = []
    with conn.cursor() as cur:
        for emb in query_embeddings:
            vec = Vector(emb)
            cur.execute(query_tmpl, (vec, vec, top_k))
            rows = cur.fetchall()
            # cosine distance -> similarity in [-1, 1] via (1 - distance)
            all_results.append(
                _rows_to_results(rows, distance_to_similarity=lambda d: 1.0 - d)
            )

    return all_results


def sparse_search(
    query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name, sparse_name=sparse_name)

    query_tmpl = sql.SQL(
        """
		SELECT id,
               {sparse_col} <#> %s AS score,
			   text,
			   document_id,
			   title,
			   file_name,
			   file_path
		FROM {table}
        ORDER BY {sparse_col} <#> %s
		LIMIT %s;
		"""
    ).format(
        table=sql.Identifier(collection_name),
        sparse_col=sql.Identifier(sparse_name),
    )

    all_results: list[list[schemas.RetrievedDocument]] = []
    with conn.cursor() as cur:
        for indices, values in query_embeddings:
            if len(indices) == 0:
                all_results.append([])
                continue
            vec = SparseVector(
                {int(i): float(v) for i, v in zip(indices, values)}, config.SPARSE_DIM
            )
            cur.execute(query_tmpl, (vec, vec, top_k))
            rows = cur.fetchall()
            # inner product operator <#> returns negative dot product by design
            # convert to positive similarity = -value (sum of weights)
            all_results.append(
                _rows_to_results(rows, distance_to_similarity=lambda v: -v)
            )

    return all_results


def hybrid_search(
    dense_query_embeddings: list[list[float]],
    sparse_query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    # Overfetch separately then fuse client-side
    overfetch_amount = max(top_k, int(top_k * overfetch_mul))

    dense_results = dense_search(
        query_embeddings=dense_query_embeddings,
        collection_name=collection_name,
        top_k=overfetch_amount,
        dense_name=dense_name,
    )
    sparse_results = sparse_search(
        query_embeddings=sparse_query_embeddings,
        collection_name=collection_name,
        top_k=overfetch_amount,
        sparse_name=sparse_name,
    )

    fused_results: list[list[schemas.RetrievedDocument]] = []
    for d_res, s_res in zip(dense_results, sparse_results):
        fused = fuse_results(results1=d_res, results2=s_res, method=fusion_method)
        fused_results.append(fused[:top_k])

    return fused_results
