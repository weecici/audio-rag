from fastapi import APIRouter, status

from app.middleware.errors import ApiError

router = APIRouter(prefix="/files", tags=["Ingestion"])


@router.post(
    "/text",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest text documents",
    description=(
        "Accept document file paths or a directory for background ingestion "
        "into the vector store."
    ),
)
async def ingest_text_files(
    request: dict,
) -> dict:
    raise ApiError(
        code="not_implemented",
        message="Text document ingestion is not implemented yet.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )


@router.post(
    "/audio",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest audio sources",
    description=(
        "Accept audio file paths or YouTube URLs for background transcription "
        "and ingestion into the vector store."
    ),
)
async def ingest_audio_files(
    request: dict,
) -> dict:
    raise ApiError(
        code="not_implemented",
        message="Audio ingestion is not implemented yet.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )
