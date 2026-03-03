"""Request / response schemas for the conversations endpoint."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CreateConversationRequest(BaseModel):
    collection_name: str = Field(
        ...,
        min_length=1,
        description="The document collection to query for RAG.",
    )
    title: Optional[str] = Field(
        None,
        max_length=256,
        description="Optional conversation title. Auto-generated from first message if omitted.",
    )


class SendMessageRequest(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="The user message content.",
    )
    search_type: Literal["dense", "sparse", "hybrid"] = Field(
        "hybrid",
        description="Search strategy for document retrieval.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of documents to retrieve.",
    )
    stream: bool = Field(
        False,
        description="If true, response is streamed via SSE.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """A retrieved document used as context for the response."""

    doc_id: int = Field(..., description="Document identifier.")
    title: Optional[str] = Field(None, description="Document chunk title.")
    text: str = Field(..., description="Document text snippet.")
    score: Optional[float] = Field(None, description="Relevance score.")


class MessageResponse(BaseModel):
    """A single message in the conversation."""

    message_id: str = Field(..., description="Unique message identifier.")
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message role."
    )
    content: str = Field(..., description="Message text.")
    sources: Optional[list[SourceDocument]] = Field(
        None,
        description="Retrieved documents (only for assistant messages).",
    )
    created_at: datetime = Field(..., description="Creation timestamp.")


class ConversationResponse(BaseModel):
    """Full conversation with metadata and message history."""

    conversation_id: str = Field(..., description="Unique conversation identifier.")
    title: Optional[str] = Field(None, description="Conversation title.")
    collection_name: str = Field(..., description="Document collection name.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")
    messages: list[MessageResponse] = Field(
        default_factory=list, description="Message history."
    )


class ConversationListItem(BaseModel):
    """Summary item for conversation listing."""

    conversation_id: str = Field(..., description="Conversation identifier.")
    title: Optional[str] = Field(None, description="Conversation title.")
    collection_name: str = Field(..., description="Document collection name.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class SendMessageResponse(BaseModel):
    """Response after sending a message (non-streaming)."""

    user_message: MessageResponse = Field(
        ..., description="The user's message as stored."
    )
    assistant_message: MessageResponse = Field(
        ..., description="The assistant's RAG-augmented response."
    )
