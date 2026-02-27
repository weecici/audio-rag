from .storage import upsert_documents, delete_documents
from .retrieval import dense_search, sparse_search, hybrid_search


def upsert_data(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "upsert_data is not implemented in this Milvus backend; use upsert_documents instead"
    )
