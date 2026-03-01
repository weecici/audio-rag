"""Public service: retrieve job status from Redis."""

from typing import Any

from app.repositories.redis import get_job
from app.core.logging import logger


def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Return the current job state or ``None`` if not found / expired."""
    logger.debug(f"Getting job status for job_id={job_id}")
    return get_job(job_id)
