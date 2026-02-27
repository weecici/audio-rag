from pydantic import BaseModel, Field
from typing import Literal, Optional


class FileResult(BaseModel):
    filename: str
    status: Literal["accepted", "rejected"]
    reason: Optional[str]


class FileIngestionResponse(BaseModel):
    job_id: str = Field(..., description="Unique ID for the ingestion job")
    collection_name: str = Field(
        ..., description="Name of the collection the files are being ingested into"
    )
    results: list[FileResult] = Field(
        ...,
        description="List of received files with status and reasons for any rejections",
    )
    status: Literal["queued"] = Field(
        "queued",
        description=(
            "Overall status of the ingestion job. 'queued' indicates that the "
            "files have been received and the job is waiting to be processed."
        ),
    )
