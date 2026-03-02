"""Search endpoint – query the vector store for relevant documents."""

from fastapi import APIRouter

from app.schemas.search import SearchRequest, SearchResponse
from app.services.public.search import search_documents

router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "/{collection_name}",
    response_model=SearchResponse,
    summary="Search for relevant documents",
    description=(
        "Run dense, sparse, or hybrid search against the vector store. "
        "The endpoint is fully async — concurrent requests are served in parallel."
    ),
)
async def search(
    collection_name: str,
    request: SearchRequest,
) -> SearchResponse:
    results = await search_documents(
        query=request.query,
        collection_name=collection_name,
        search_type=request.search_type,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        collection_name=collection_name,
        search_type=request.search_type,
        total_results=len(results),
        results=results,
    )
