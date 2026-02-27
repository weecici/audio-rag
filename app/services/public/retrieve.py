from fastapi import status

from app import schemas
from app.core.logging import logger
from app.services.internal import dense_encode, rerank
from app.repositories.milvus import (
    dense_search,
    sparse_search,
    hybrid_search,
)


async def retrieve_documents(
    request: schemas.RetrievalRequest,
) -> schemas.RetrievalResponse:
    raise NotImplementedError("Document retrieval service is not implemented yet.")
