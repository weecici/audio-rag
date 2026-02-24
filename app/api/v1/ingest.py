import app.services.public as public_svcs
from fastapi import APIRouter
from app import schemas

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
    return await public_svcs.ingest_documents(request)


@router.post(
    "/ingest/audios",
    response_model=schemas.IngestionResponse,
    summary="Audio ingestion",
    description="Ingest audio files from the specified file paths or youtube links.",
)
async def ingest_audios(
    request: schemas.AudioIngestionRequest,
) -> schemas.IngestionResponse:
    return await public_svcs.ingest_audios(request)
