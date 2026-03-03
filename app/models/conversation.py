"""Domain models for RAG conversations."""

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single message in a conversation (user or assistant)."""

    message_id: str = Field(..., description="Unique message identifier (UUID).")
    conversation_id: str = Field(..., description="Parent conversation identifier.")
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message role."
    )
    content: str = Field(..., description="Message text content.")
    sources: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Retrieved document sources used to generate this response.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )

    model_config = ConfigDict(extra="forbid")


class ConversationMeta(BaseModel):
    """Metadata for a conversation (lightweight summary)."""

    conversation_id: str = Field(..., description="Unique conversation identifier.")
    title: Optional[str] = Field(None, description="Auto-generated or user-set title.")
    collection_name: str = Field(
        ..., description="The document collection this conversation queries."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last-updated timestamp (UTC)."
    )

    model_config = ConfigDict(extra="forbid")
