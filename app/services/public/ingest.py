from fastapi import status

from app import schemas
from app.utils import logger, download_audio
from app.repositories.milvus import upsert_data
from app.services.internal import (
    process_documents,
    dense_encode,
    build_inverted_index,
    transcribe_audio,
)


async def ingest_documents(
    request: schemas.DocumentIngestionRequest,
) -> schemas.IngestionResponse:
    raise NotImplementedError("Document ingestion service is not implemented yet.")


async def ingest_audios(
    request: schemas.AudioIngestionRequest,
) -> schemas.IngestionResponse:
    raise NotImplementedError("Audio ingestion service is not implemented yet.")
