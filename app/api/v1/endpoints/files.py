"""File ingestion endpoint – accepts uploads and spawns background processing."""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, status, File, UploadFile, BackgroundTasks

from app.middleware.errors import ApiError
from app.core.config import settings
from app.schemas import FileIngestionResponse, FileResult
from app.utils import save_upload
from app.services.public import ingest_files
from app.repositories.redis import create_job

router = APIRouter(prefix="/files", tags=["Files"])


async def _run_ingest(
    job_id: str,
    file_paths: list[str],
    filenames: list[str],
    collection_name: str,
) -> None:
    """Wrapper that runs the async ingest_files inside BackgroundTasks."""
    await ingest_files(job_id, file_paths, filenames, collection_name)


@router.post(
    "/{collection_name}",
    response_model=FileIngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest files to vector store",
    description="Accept files (text and audio) for ingestion into the vector store.",
)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    collection_name: str,
    files: list[UploadFile] = File(..., description="One or more files to ingest"),
) -> FileIngestionResponse:

    if not files:
        raise ApiError(
            code="validation_error",
            message="At least one file must be provided.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    base_dir = Path(settings.LOCAL_STORAGE_PATH) / "uploads"
    base_dir.mkdir(parents=True, exist_ok=True)

    allowed_prefixes = ("text/", "audio/")

    saved_paths: list[str] = []
    filenames: list[str] = []
    results: list[FileResult] = []

    for upload in files:
        content_type = (upload.content_type or "").lower()
        if content_type and not content_type.startswith(allowed_prefixes):
            raise ApiError(
                code="unsupported_media_type",
                message="Only text/* and audio/* uploads are supported by this endpoint.",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                details={
                    "content_type": upload.content_type,
                    "filename": upload.filename,
                },
            )

        saved_path, _size = await save_upload(upload, base_dir)
        fname = upload.filename or "unknown"
        saved_paths.append(saved_path)
        filenames.append(fname)
        results.append(FileResult(filename=fname, status="accepted", reason=None))

    # Create job in Redis and schedule background processing
    job_id = str(uuid.uuid4())
    create_job(job_id, collection_name, filenames)
    background_tasks.add_task(
        _run_ingest, job_id, saved_paths, filenames, collection_name
    )

    return FileIngestionResponse(
        job_id=job_id,
        collection_name=collection_name,
        results=results,
    )
