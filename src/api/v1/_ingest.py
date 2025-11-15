import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="ingest-documents",
    trigger=inngest.TriggerEvent(event="rag/ingest-documents"),
    retries=0,
)
async def ingest_documents(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.IngestionRequest.model_validate(ctx.event.data)
    return public_svcs.ingest_documents(request).model_dump()


@router.post(
    "/ingest/documents",
    response_model=schemas.IngestionResponse,
    summary="Document ingestion",
    description="Ingest documents from the specified file paths or directory.",
)
async def ingest(
    request: schemas.DocumentIngestionRequest,
) -> schemas.IngestionResponse:
    return public_svcs.ingest_documents(request)


@inngest_client.create_function(
    fn_id="ingest-audios",
    trigger=inngest.TriggerEvent(event="rag/ingest-audios"),
    retries=0,
)
async def ingest_audios(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.AudioIngestionRequest.model_validate(ctx.event.data)
    return (await public_svcs.ingest_audios(request)).model_dump()


@router.post(
    "/ingest/audios",
    response_model=schemas.IngestionResponse,
    summary="Audio ingestion",
    description="Ingest audio files from the specified file paths or youtube links.",
)
async def ingest(request: schemas.AudioIngestionRequest) -> schemas.IngestionResponse:
    return await public_svcs.ingest_audios(request)
