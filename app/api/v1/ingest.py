import app.service.public as public_svc
from fastapi import APIRouter, status, BackgroundTasks
from app import schema
from app.api.middleware import ApiError

router = APIRouter()


@router.post(
    "/ingest/documents",
    response_model=schema.IngestionResponse,
    summary="Document ingestion",
    description="Ingest documents from the specified file paths or directory in the background.",
)
async def ingest_documents(
    request: schema.DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
) -> schema.IngestionResponse:
    try:
        if not request.file_paths and not request.file_dir:
            raise ValueError("No file paths or directory provided in event data.")

        background_tasks.add_task(public_svc.ingest_documents, request)
        return schema.IngestionResponse(
            status=status.HTTP_202_ACCEPTED,
            message=f"Document ingestion for collection '{request.collection_name}' has been started in the background.",
        )
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_failed",
            message="document ingestion failed to start",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


@router.post(
    "/ingest/audios",
    response_model=schema.IngestionResponse,
    summary="Audio ingestion",
    description="Ingest audio files from the specified file paths or youtube links in the background.",
)
async def ingest_audios(
    request: schema.AudioIngestionRequest,
    background_tasks: BackgroundTasks,
) -> schema.IngestionResponse:
    try:
        if not request.file_paths and not request.urls:
            raise ValueError("No audio file paths or URLs provided in request data.")

        background_tasks.add_task(public_svc.ingest_audios, request)
        return schema.IngestionResponse(
            status=status.HTTP_202_ACCEPTED,
            message=f"Audio ingestion for collection '{request.collection_name}' has been started in the background.",
        )
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="ingest_failed",
            message="audio ingestion failed to start",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
