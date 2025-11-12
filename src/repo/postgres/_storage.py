import psycopg
from uuid import UUID
from functools import lru_cache
from psycopg import sql
from pgvector import Vector
from pgvector.psycopg import register_vector
from llama_index.core.schema import BaseNode
from src import schemas
from src.core import config

POSTINGS_LIST_TABLE_SUFFIX = "pl"
DOC_FREQ_TABLE_SUFFIX = "df"


def _get_db_params() -> dict:
    return {
        "host": config.POSTGRES_HOST,
        "port": config.POSTGRES_PORT,
        "user": config.POSTGRES_USER,
        "password": config.POSTGRES_PASSWORD,
        "dbname": config.POSTGRES_DB,
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
    dense_dim: int = config.DENSE_DIM,
    m: int = 32,
    ef_construction: int = 128,
) -> None:
    conn = get_pg_conn()

    df_table = f"{collection_name}_{DOC_FREQ_TABLE_SUFFIX}"
    pl_table = f"{collection_name}_{POSTINGS_LIST_TABLE_SUFFIX}"

    create_main_table = sql.SQL(
        """
		CREATE TABLE IF NOT EXISTS {main_table} (
			id UUID PRIMARY KEY,
			text TEXT NOT NULL,
			document_id TEXT,
			title TEXT,
			file_name TEXT,
			file_path TEXT,
			{dense_col} vector({dense_dim}) NOT NULL,
            doc_len INT NOT NULL
		);
		"""
    ).format(
        main_table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
        dense_dim=sql.Literal(dense_dim),
    )

    create_emb_index = sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {index} ON {table}
        USING hnsw ({col} {ops}) WITH (m = {m}, ef_construction = {ef_construction});
		"""
    )

    create_dense_index = create_emb_index.format(
        idx_name=sql.Literal(f"{collection_name}_{dense_name}_idx"),
        index=sql.Identifier(f"{collection_name}_{dense_name}_idx"),
        table=sql.Identifier(collection_name),
        col=sql.Identifier(dense_name),
        ops=sql.SQL("vector_cosine_ops"),
        m=sql.Literal(m),
        ef_construction=sql.Literal(ef_construction),
    )

    create_doc_freq_table = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {df_table} (
            term TEXT PRIMARY KEY,
            doc_freq INT NOT NULL
        );
        """
    ).format(
        df_table=sql.Identifier(df_table),
    )

    create_postings_list_table = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {pl_table} (
            term TEXT,
            doc_id UUID,
            freq INT NOT NULL,
            PRIMARY KEY (term, doc_id),
            FOREIGN KEY (term) REFERENCES {df_table}(term) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES {main_table}(id) ON DELETE CASCADE
        );
        """
    ).format(
        pl_table=sql.Identifier(pl_table),
        df_table=sql.Identifier(df_table),
        main_table=sql.Identifier(collection_name),
    )

    create_term_index = sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {term_index} ON {pl_table} (term);
        """
    ).format(
        term_index=sql.Identifier(
            f"{collection_name}_{POSTINGS_LIST_TABLE_SUFFIX}_term_idx"
        ),
        pl_table=sql.Identifier(pl_table),
    )

    with conn.cursor() as cur:
        cur.execute(create_main_table)
        cur.execute(create_dense_index)
        cur.execute(create_doc_freq_table)
        cur.execute(create_postings_list_table)
        cur.execute(create_term_index)


def upsert_data(
    nodes: list[BaseNode],
    dense_embeddings: list[list[float]],
    postings_list: dict[str, schemas.TermEntry],
    doc_lens: dict[str, int],
    collection_name: str,
    dense_name: str = config.DENSE_MODEL,
    dense_dim: int = config.DENSE_DIM,
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if len(dense_embeddings) != len(nodes):
        raise ValueError(
            f"The number of dense embeddings ({len(dense_embeddings)}) must match the number of nodes ({len(nodes)})"
        )

    conn = get_pg_conn()
    ensure_collection_exists(
        collection_name=collection_name,
        dense_name=dense_name,
        dense_dim=dense_dim,
    )

    insert_main_table = sql.SQL(
        """
		INSERT INTO {table} (id, text, document_id, title, file_name, file_path, {dense_col}, doc_len)
		VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        --ON CONFLICT (id) DO UPDATE SET
        --	text = EXCLUDED.text,
        --	document_id = EXCLUDED.document_id,
        --	title = EXCLUDED.title,
        --	file_name = EXCLUDED.file_name,
        --	file_path = EXCLUDED.file_path,
        --	{dense_col} = EXCLUDED.{dense_col};
		"""
    ).format(
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
    )

    insert_df_table = sql.SQL(
        """
        INSERT INTO {df_table} (term, doc_freq)
        VALUES (%s, %s)
        ON CONFLICT (term) DO UPDATE SET
            doc_freq = EXCLUDED.doc_freq;
        """
    ).format(
        df_table=sql.Identifier(f"{collection_name}_{DOC_FREQ_TABLE_SUFFIX}"),
    )

    insert_pl_table = sql.SQL(
        """
        INSERT INTO {pl_table} (term, doc_id, freq)
        VALUES (%s, %s, %s)
        --ON CONFLICT (term, doc_id) DO UPDATE SET
        --  freq = EXCLUDED.freq;
        """
    ).format(
        pl_table=sql.Identifier(f"{collection_name}_{POSTINGS_LIST_TABLE_SUFFIX}"),
    )

    main_rows = []
    df_rows = []
    pl_rows = []

    # prepare main table rows
    for i, node in enumerate(nodes):
        payload = schemas.DocumentPayload(
            text=node.text,
            metadata=schemas.DocumentMetadata.model_validate(node.metadata),
        )

        dense_vec = Vector(dense_embeddings[i])
        doc_len = doc_lens.get(node.id_, 0)

        main_rows.append(
            (
                UUID(node.id_),
                payload.text,
                payload.metadata.document_id,
                payload.metadata.title,
                payload.metadata.file_name,
                payload.metadata.file_path,
                dense_vec,
                doc_len,
            )
        )

    # prepare postings list and document frequency rows
    for term, term_entry in postings_list.items():
        df_rows.append(
            (
                term,
                term_entry.doc_freq,
            )
        )
        for posting in term_entry.postings:
            pl_rows.append(
                (
                    term,
                    UUID(posting.doc_id),
                    posting.term_freq,
                )
            )

    with conn.cursor() as cur:
        cur.executemany(insert_main_table, main_rows)
        cur.executemany(insert_df_table, df_rows)
        cur.executemany(insert_pl_table, pl_rows)
