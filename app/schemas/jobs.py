"""Schemas for job status polling."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FileJobStatus(BaseModel):
    """Processing status of a single file within a job."""

    status: Literal["pending", "processing", "completed", "failed"]
    error: str = ""
    chunks: int = Field(0, description="Number of document chunks produced")


class JobStatusResponse(BaseModel):
    """Response returned when the client polls ``GET /jobs/{job_id}``."""

    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    collection: str
    total_files: int
    processed: int = Field(0, description="Files finished (success + failed)")
    failed_count: int = Field(0, alias="failed_cnt")
    documents_ingested: int = Field(
        0, description="Total chunks written to vector store"
    )
    error: str = Field("", description="Top-level error (empty when ok)")
    created_at: str
    updated_at: str
    files: dict[str, FileJobStatus] = Field(
        default_factory=dict, description="Per-file processing status"
    )

    model_config = {"populate_by_name": True}
