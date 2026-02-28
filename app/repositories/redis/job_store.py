"""Redis-backed job store for tracking async ingestion jobs.

Job structure (Redis hash at key ``job:{job_id}``):
    status      – queued | processing | completed | failed
    collection  – target Milvus collection name
    total_files – number of files in the job
    processed   – number of files processed so far
    failed_cnt  – number of files that failed
    error       – top-level error message (empty when ok)
    created_at  – ISO-8601 UTC timestamp
    updated_at  – ISO-8601 UTC timestamp
    documents_ingested – total document chunks written to vector store

Per-file status stored at ``job:{job_id}:files:{filename}``:
    status  – pending | processing | completed | failed
    error   – error message (empty when ok)
    chunks  – number of chunks produced from this file
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from app.core.config import settings
from app.core.logging import logger
from ._client import get_redis_client


JobStatus = Literal["queued", "processing", "completed", "failed"]
FileStatus = Literal["pending", "processing", "completed", "failed"]

_KEY_PREFIX = "job"


def _job_key(job_id: str) -> str:
    return f"{_KEY_PREFIX}:{job_id}"


def _file_key(job_id: str, filename: str) -> str:
    return f"{_KEY_PREFIX}:{job_id}:files:{filename}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def create_job(
    job_id: str,
    collection_name: str,
    filenames: list[str],
) -> None:
    """Create a new job with status ``queued`` and per-file entries."""
    r = get_redis_client()
    ttl = settings.REDIS_JOB_TTL_SEC
    now = _now_iso()

    # Main job hash
    job_data = {
        "status": "queued",
        "collection": collection_name,
        "total_files": len(filenames),
        "processed": 0,
        "failed_cnt": 0,
        "error": "",
        "created_at": now,
        "updated_at": now,
        "documents_ingested": 0,
        "filenames": json.dumps(filenames),
    }
    jk = _job_key(job_id)
    r.hset(jk, mapping=job_data)
    r.expire(jk, ttl)

    # Per-file hashes
    for fname in filenames:
        fk = _file_key(job_id, fname)
        r.hset(fk, mapping={"status": "pending", "error": "", "chunks": 0})
        r.expire(fk, ttl)


def update_job_status(job_id: str, status: JobStatus) -> None:
    """Update top-level job status."""
    r = get_redis_client()
    r.hset(_job_key(job_id), mapping={"status": status, "updated_at": _now_iso()})


def update_file_status(
    job_id: str,
    filename: str,
    status: FileStatus,
    *,
    error: str = "",
    chunks: int = 0,
) -> None:
    """Update a single file's processing status and bump job counters."""
    r = get_redis_client()
    jk = _job_key(job_id)
    fk = _file_key(job_id, filename)

    file_data: dict[str, Any] = {"status": status, "error": error}
    if chunks:
        file_data["chunks"] = chunks
    r.hset(fk, mapping=file_data)

    if status == "completed":
        r.hincrby(jk, "processed", 1)
        if chunks:
            r.hincrby(jk, "documents_ingested", chunks)
    elif status == "failed":
        r.hincrby(jk, "processed", 1)
        r.hincrby(jk, "failed_cnt", 1)

    r.hset(jk, "updated_at", _now_iso())


def set_job_error(job_id: str, error: str) -> None:
    """Mark a job as failed with a top-level error."""
    r = get_redis_client()
    r.hset(
        _job_key(job_id),
        mapping={"status": "failed", "error": error, "updated_at": _now_iso()},
    )


def set_job_result(job_id: str, documents_ingested: int) -> None:
    """Mark a job as completed with final counts."""
    r = get_redis_client()
    r.hset(
        _job_key(job_id),
        mapping={
            "status": "completed",
            "documents_ingested": documents_ingested,
            "updated_at": _now_iso(),
        },
    )


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return the full job state including per-file statuses.

    Returns ``None`` if the job does not exist (expired or never created).
    """
    r = get_redis_client()
    jk = _job_key(job_id)
    data = r.hgetall(jk)
    if not data:
        return None

    # Parse numeric fields
    data["total_files"] = int(data.get("total_files", 0))
    data["processed"] = int(data.get("processed", 0))
    data["failed_cnt"] = int(data.get("failed_cnt", 0))
    data["documents_ingested"] = int(data.get("documents_ingested", 0))

    # Collect per-file statuses
    filenames: list[str] = json.loads(data.get("filenames", "[]"))
    files: dict[str, dict[str, Any]] = {}
    for fname in filenames:
        fdata = r.hgetall(_file_key(job_id, fname))
        if fdata:
            fdata["chunks"] = int(fdata.get("chunks", 0))
            files[fname] = fdata
    data["files"] = files

    return data
