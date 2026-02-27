from fastapi import APIRouter, status

from app.middleware.errors import ApiError

router = APIRouter(prefix="/files", tags=["Ingestion"])


@router.post(
    "/",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest files",
    description=("Accept files (text and audio) for ingestion into the vector store."),
)
async def ingest_files(
    request: dict,
) -> dict:
    raise ApiError(
        code="not_implemented",
        message="File ingestion is not implemented yet.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )
