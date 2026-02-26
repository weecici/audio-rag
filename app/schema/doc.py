from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
from datetime import datetime, timezone


class DocumentChunk(BaseModel):
    chunk_index: int = Field(
        ..., description="The index of the chunk within the document"
    )
    title: Optional[str] = Field(None, description="The title of the chunk")
    text: str = Field(..., description="The text content of the chunk")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata for the chunk (e.g. source info)"
    )
    dense_vector: Optional[list[float]] = Field(
        None, description="The dense vector embedding of the chunk"
    )
    sparse_vector: Optional[list[float]] = Field(
        None, description="The sparse vector embedding of the chunk (e.g. BM25)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)",
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last-updated timestamp (UTC)"
    )

    class Config:
        extra = "forbid"


class Document(BaseModel):
    doc_id: int = Field(..., description="The Document ID")
    title: str = Field(..., description="The title of the document")
    author_info: Optional[str] = Field(None, description="The author of the document")
    tags: list[str] = Field(
        default_factory=list, description="Free-form tags / categories"
    )
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata for the document (e.g. source info)"
    )
    chunks: list[DocumentChunk] = Field(
        default_factory=list, description="The list of document chunks"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)",
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last-updated timestamp (UTC)"
    )

    class Config:
        extra = "forbid"

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title must not be empty")
        return v.strip()
