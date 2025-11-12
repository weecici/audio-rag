from typing import Optional, Literal
from collections import Counter
from psycopg import sql
from pgvector import Vector
from src import schemas
from src.core import config
from src.utils import *
from ._storage import (
    get_pg_conn,
    ensure_collection_exists,
    POSTINGS_LIST_TABLE_SUFFIX,
    DOC_FREQ_TABLE_SUFFIX,
)


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
                id=str(rid),
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

    logger.info(
        f"pgvector.dense_search collection={collection_name} top_k={top_k} using={dense_name} op=<=> (cosine distance)"
    )

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
            if not rows:
                logger.debug("pgvector.dense_search: 0 candidates")
            else:
                preview = min(5, len(rows))
                for i in range(preview):
                    rid, dist, *_ = rows[i]
                    try:
                        d = float(dist)
                    except Exception:
                        d = 0.0
                    logger.debug(
                        f"dense cand[{i}] id={rid} dist={d:.6f} sim={1.0 - d:.6f}"
                    )
            # cosine distance -> similarity in [-1, 1] via (1 - distance)
            all_results.append(
                _rows_to_results(rows, distance_to_similarity=lambda d: 1.0 - d)
            )

    return all_results


def sparse_search(
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    scoring_method: Literal["tfidf", "okapi-bm25"] = "okapi-bm25",
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name)

    results_all: list[list[schemas.RetrievedDocument]] = []

    # Precompute corpus stats once
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COUNT(*)::int, AVG(doc_len) FROM {}; ").format(
                sql.Identifier(collection_name)
            )
        )
        row = cur.fetchone()
        N = int(row[0]) if row and row[0] is not None else 0
        avg_dl = float(row[1]) if row and row[1] is not None else 0.0

        if N == 0:
            return [[] for _ in query_texts]

        pl_table = f"{collection_name}_{POSTINGS_LIST_TABLE_SUFFIX}"
        df_table = f"{collection_name}_{DOC_FREQ_TABLE_SUFFIX}"

        tokenized_texts = tokenize(texts=query_texts)

        pl_select = sql.SQL("SELECT doc_id, freq FROM {} WHERE term = %s;").format(
            sql.Identifier(pl_table)
        )
        df_select = sql.SQL("SELECT doc_freq FROM {} WHERE term = %s;").format(
            sql.Identifier(df_table)
        )
        doc_select = sql.SQL(
            """
            SELECT id, text, document_id, title, file_name, file_path, doc_len
            FROM {}
            WHERE id = %s;
            """
        ).format(sql.Identifier(collection_name))

        # Simple caches to avoid redundant queries within the same request batch
        df_cache: dict[str, int] = {}
        dl_cache: dict[str, int] = {}

        for q_idx, tokens in enumerate(tokenized_texts):
            term_counts = Counter(tokens)
            # doc_id -> RetrievedDocument
            doc_scores: dict[str, schemas.RetrievedDocument] = {}

            for term, query_tf in term_counts.items():
                cur.execute(pl_select, (term,))
                postings = cur.fetchall()  # rows of (doc_id, freq)
                if not postings:
                    continue

                # fetch doc_freq (idf needs this)
                if term in df_cache:
                    df = df_cache[term]
                else:
                    cur.execute(df_select, (term,))
                    df_row = cur.fetchone()
                    df = int(df_row[0]) if df_row and df_row[0] is not None else 0
                    df_cache[term] = df

                for doc_id, tf in postings:
                    sid = str(doc_id)

                    # fetch payload + doc_len
                    if sid in dl_cache:
                        dl = dl_cache[sid]
                    else:
                        cur.execute(doc_select, (doc_id,))
                        drow = cur.fetchone()
                        if not drow:
                            continue
                        (
                            _,
                            text,
                            document_id,
                            title,
                            file_name,
                            file_path,
                            dl,
                        ) = drow

                        payload = schemas.DocumentPayload(
                            text=text,
                            metadata=schemas.DocumentMetadata(
                                document_id=document_id or "",
                                title=title or "",
                                file_name=file_name or "",
                                file_path=file_path or "",
                            ),
                        )

                        dl_cache[sid] = int(dl) if dl is not None else 0
                        doc_scores[sid] = schemas.RetrievedDocument(
                            id=sid,
                            score=0.0,
                            payload=payload,
                        )

                    idf = calc_idf(N=N, df=df)
                    if scoring_method == "tfidf":
                        doc_scores[sid].score += query_tf * calc_tfidf(tf=tf, idf=idf)
                    elif scoring_method == "okapi-bm25":
                        doc_scores[sid].score += query_tf * calc_okapi_bm25(
                            tf=tf, idf=idf, dl=dl, avg_dl=avg_dl
                        )
                    else:
                        raise ValueError(
                            f"Unsupported scoring_method: {scoring_method}"
                        )

            retrieved_docs = list(doc_scores.values())
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)

            results_all.append(retrieved_docs[:top_k])

    return results_all


def hybrid_search(
    dense_query_embeddings: list[list[float]],
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
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
        query_texts=query_texts,
        collection_name=collection_name,
        top_k=overfetch_amount,
    )

    fused_results: list[list[schemas.RetrievedDocument]] = []
    for d_res, s_res in zip(dense_results, sparse_results):
        fused = fuse_results(results1=d_res, results2=s_res, method=fusion_method)
        fused_results.append(fused[:top_k])

    return fused_results
