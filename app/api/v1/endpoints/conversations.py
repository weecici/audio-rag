from fastapi import APIRouter, status

from app.middleware.errors import ApiError

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post(
    "/{conversation_id}/messages",
    response_model=dict,
    summary="Generate answers with RAG",
    description=(
        "Retrieve context documents and generate answers using a large "
        "language model (retrieval-augmented generation)."
    ),
)
async def create_message(
    conversation_id: str,
    request: dict,
) -> dict:
    raise ApiError(
        code="not_implemented",
        message="Generation is not implemented yet.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )
