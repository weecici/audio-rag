import app.services.public as public_svcs
from fastapi import APIRouter, status
from app import schemas
from app.api.middleware import ApiError

router = APIRouter()


@router.post(
    "/ingest/documents",
    response_model=schemas.IngestionResponse,
    summary="Document ingestion",
    description="Ingest documents from the specified file paths or directory.",
)
async def ingest_documents(
    request: schemas.DocumentIngestionRequest,
) -> schemas.IngestionResponse:
    try:
        return await public_svcs.ingest_documents(request)
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_failed",
            message="document ingestion failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


@router.post(
    "/ingest/audios",
    response_model=schemas.IngestionResponse,
    summary="Audio ingestion",
    description="Ingest audio files from the specified file paths or youtube links.",
)
async def ingest_audios(
    request: schemas.AudioIngestionRequest,
) -> schemas.IngestionResponse:
    try:
        return await public_svcs.ingest_audios(request)
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_failed",
            message="audio ingestion failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
