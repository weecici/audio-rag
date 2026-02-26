from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Optional
from datetime import datetime, timezone


class Document(BaseModel):
    doc_id: int = Field(..., description="The Document ID")
    title: str = Field(..., description="The title of the document")
    author_info: Optional[str] = Field(None, description="The author of the document")
    tags: Optional[list[str]] = Field(None, description="Free-form tags / categories")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata for the document (e.g. source info)"
    )
    text: str = Field(..., description="The text content of the document")
    dense_vector: Optional[list[float]] = Field(
        None, description="The dense vector embedding of the document"
    )
    sparse_vector: Optional[list[float]] = Field(
        None, description="The sparse vector embedding of the document (e.g. BM25)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)",
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last-updated timestamp (UTC)"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title must not be empty")
        return v.strip()

    @field_validator("tags")
    @classmethod
    def num_tags_must_not_exceed_limit(
        cls, v: Optional[list[str]]
    ) -> Optional[list[str]]:
        if v and len(v) > 10:
            raise ValueError("tags must not exceed 10 items")
        return v
