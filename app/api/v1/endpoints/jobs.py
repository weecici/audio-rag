"""Job status polling endpoint."""

from fastapi import APIRouter, status

from app.middleware.errors import ApiError
from app.schemas.jobs import JobStatusResponse, FileJobStatus
from app.services.public import get_job_status

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get ingestion job status",
    description="Poll the status of an asynchronous file ingestion job.",
)
async def get_job(job_id: str) -> JobStatusResponse:
    data = get_job_status(job_id)

    if data is None:
        raise ApiError(
            code="not_found",
            message=f"Job '{job_id}' not found or has expired.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # Build per-file status map
    files_map: dict[str, FileJobStatus] = {}
    for fname, fdata in data.get("files", {}).items():
        files_map[fname] = FileJobStatus(
            status=fdata.get("status", "pending"),
            error=fdata.get("error", ""),
            chunks=int(fdata.get("chunks", 0)),
        )

    return JobStatusResponse(
        job_id=job_id,
        status=data["status"],
        collection=data["collection"],
        total_files=data["total_files"],
        processed=data["processed"],
        failed_cnt=data["failed_cnt"],
        documents_ingested=data["documents_ingested"],
        error=data.get("error", ""),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        files=files_map,
    )
