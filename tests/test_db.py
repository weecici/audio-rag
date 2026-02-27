import pytest
from datetime import datetime, timezone

from app import models
from app.repositories.milvus._client import get_client
from app.repositories.milvus._collection import delete_collection
from app.repositories.milvus.storage import upsert_documents, delete_documents
from app.repositories.milvus.retrieval import dense_search, sparse_search, hybrid_search
from app.core import config


@pytest.fixture()
def collection_name() -> str:
    return f"test_docs_{int(datetime.now(timezone.utc).timestamp())}"


@pytest.fixture()
def ensure_collection_cleanup(collection_name: str):
    yield
    delete_collection(collection_name)


def _docs_for_tests() -> list[models.Document]:
    now = datetime.now(timezone.utc)
    dim = config.DENSE_DIM
    return [
        models.Document(
            doc_id=1,
            title="Milvus intro",
            author_info="a",
            tags=["vecdb"],
            metadata={"source": "unit"},
            text="Milvus is a vector database. It supports BM25 full-text search.",
            dense_vector=[0.0] * dim,
            sparse_vector=[0.0],
            created_at=now,
            updated_at=None,
        ),
        models.Document(
            doc_id=2,
            title="PyMilvus client",
            author_info="b",
            tags=["sdk"],
            metadata={"source": "unit"},
            text="PyMilvus is the Python SDK for Milvus.",
            dense_vector=[1.0] * dim,
            sparse_vector=[0.0],
            created_at=now,
            updated_at=None,
        ),
    ]


def test_milvus_connection():
    client = get_client()
    assert client is not None


def test_upsert_dense_and_delete_roundtrip(
    collection_name: str, ensure_collection_cleanup
):
    docs = _docs_for_tests()
    upsert_documents(docs, collection_name)

    results = dense_search(
        query_vectors=[[1.0] * config.DENSE_DIM],
        collection_name=collection_name,
        top_k=5,
    )
    assert len(results) == 1
    assert {d.doc_id for d in results[0]} >= {1, 2}

    deleted = delete_documents([2], collection_name)
    assert deleted == 1

    results_after = dense_search(
        query_vectors=[[1.0] * config.DENSE_DIM],
        collection_name=collection_name,
        top_k=5,
    )
    assert len(results_after) == 1
    assert 2 not in {d.doc_id for d in results_after[0]}


def test_sparse_bm25_search(collection_name: str, ensure_collection_cleanup):
    docs = _docs_for_tests()
    upsert_documents(docs, collection_name)

    results = sparse_search(
        query_texts=["vector database"],
        collection_name=collection_name,
        top_k=3,
    )

    assert len(results) == 1
    assert len(results[0]) >= 1
    assert any("vector database" in d.text.lower() for d in results[0])


def test_hybrid_search_returns_results(collection_name: str, ensure_collection_cleanup):
    docs = _docs_for_tests()
    upsert_documents(docs, collection_name)

    results = hybrid_search(
        query_vectors=[[1.0] * config.DENSE_DIM],
        query_texts=["vector database"],
        collection_name=collection_name,
        top_k=2,
    )

    assert len(results) == 1
    assert 1 <= len(results[0]) <= 2
    assert {d.doc_id for d in results[0]} <= {1, 2}
