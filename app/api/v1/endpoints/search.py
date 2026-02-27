from fastapi import APIRouter, status

from app.middleware.errors import ApiError

router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "/",
    response_model=dict,
    summary="Search for relevant documents",
    description=(
        "Run dense, sparse, or hybrid search against the vector store and "
        "optionally rerank the results."
    ),
)
async def search(
    request: dict,
) -> dict:
    raise ApiError(
        code="not_implemented",
        message="Search is not implemented yet.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )
