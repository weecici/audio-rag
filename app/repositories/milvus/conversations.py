"""Milvus repository for conversation storage.

Two collections:

- **_conversation_meta** — one row per conversation (id, title, collection, timestamps).
- **_conversation_messages** — one row per message (id, conversation_id, role, content,
  sources, timestamp).  Scalar-filtered by ``conversation_id``.

Both are auto-created on first write (lazy initialization).
"""

import json
from datetime import datetime, timezone
from typing import Any
from pymilvus import CollectionSchema, DataType, MilvusClient

from app.core.config import settings
from app.core.logging import logger
from app.models.conversation import ConversationMeta, Message
from ._client import get_client


# ---------------------------------------------------------------------------
# Collection names (from settings)
# ---------------------------------------------------------------------------

_META_COL = settings.CONVERSATION_META_COLLECTION
_MSG_COL = settings.CONVERSATION_MSG_COLLECTION


# ---------------------------------------------------------------------------
# Schema creation (idempotent)
# ---------------------------------------------------------------------------


def _ensure_meta_collection(client: MilvusClient) -> None:
    if client.has_collection(_META_COL):
        return
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(
        field_name="conversation_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_primary=True,
    )
    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=256,
        nullable=True,
    )
    schema.add_field(
        field_name="collection_name",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(
        field_name="updated_at", datatype=DataType.VARCHAR, max_length=64, nullable=True
    )
    schema.add_field(field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="conversation_id",
        index_name="conv_id_idx",
        index_type="TRIE",
    )
    index_params.add_index(
        field_name="dummy_vector",
        index_name="dummy_vector_idx",
        index_type="HNSW",
        metric_type="COSINE",
    )
    client.create_collection(_META_COL, schema=schema, index_params=index_params)
    logger.info(f"Created conversation meta collection '{_META_COL}'")


def _ensure_msg_collection(client: MilvusClient) -> None:
    if client.has_collection(_MSG_COL):
        return
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(
        field_name="message_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_primary=True,
    )
    schema.add_field(
        field_name="conversation_id",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="role",
        datatype=DataType.VARCHAR,
        max_length=16,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="sources",
        datatype=DataType.JSON,
        nullable=True,
    )
    schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="message_id",
        index_name="msg_id_idx",
        index_type="TRIE",
    )
    index_params.add_index(
        field_name="conversation_id",
        index_name="conv_msg_id_idx",
        index_type="TRIE",
    )
    index_params.add_index(
        field_name="dummy_vector",
        index_name="dummy_vector_idx",
        index_type="HNSW",
        metric_type="COSINE",
    )

    client.create_collection(_MSG_COL, schema=schema, index_params=index_params)
    logger.info(f"Created conversation messages collection '{_MSG_COL}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_now_iso = lambda: datetime.now(timezone.utc).isoformat()


def _meta_to_entity(meta: ConversationMeta) -> dict[str, Any]:
    return {
        "conversation_id": meta.conversation_id,
        "title": meta.title,
        "collection_name": meta.collection_name,
        "created_at": meta.created_at.isoformat(),
        "updated_at": meta.updated_at.isoformat() if meta.updated_at else None,
        "dummy_vector": [0.0, 0.0],  # Placeholder for required vector field
    }


def _entity_to_meta(entity: dict[str, Any]) -> ConversationMeta:
    return ConversationMeta(
        conversation_id=entity["conversation_id"],
        title=entity.get("title"),
        collection_name=entity["collection_name"],
        created_at=datetime.fromisoformat(entity["created_at"]),
        updated_at=(
            datetime.fromisoformat(entity["updated_at"])
            if entity.get("updated_at")
            else None
        ),
    )


def _msg_to_entity(msg: Message) -> dict[str, Any]:
    return {
        "message_id": msg.message_id,
        "conversation_id": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "sources": msg.sources,
        "created_at": msg.created_at.isoformat(),
        "dummy_vector": [0.0, 0.0],  # Placeholder for required vector field
    }


def _entity_to_msg(entity: dict[str, Any]) -> Message:
    return Message(
        message_id=entity["message_id"],
        conversation_id=entity["conversation_id"],
        role=entity["role"],
        content=entity["content"],
        sources=entity.get("sources"),
        created_at=datetime.fromisoformat(entity["created_at"]),
    )


# ---------------------------------------------------------------------------
# CRUD — Conversation Meta
# ---------------------------------------------------------------------------

_META_OUTPUT_FIELDS = [
    "conversation_id",
    "title",
    "collection_name",
    "created_at",
    "updated_at",
]


def create_conversation(meta: ConversationMeta) -> None:
    """Insert a new conversation metadata row."""
    client = get_client()
    _ensure_meta_collection(client)
    client.insert(_META_COL, [_meta_to_entity(meta)])
    client.flush(_META_COL)


def get_conversation(conversation_id: str) -> ConversationMeta | None:
    """Retrieve conversation metadata by ID.  Returns ``None`` if not found."""
    client = get_client()
    _ensure_meta_collection(client)
    client.load_collection(_META_COL)
    results = client.query(
        _META_COL,
        filter=f'conversation_id == "{conversation_id}"',
        output_fields=_META_OUTPUT_FIELDS,
        limit=1,
    )
    if not results:
        return None
    return _entity_to_meta(results[0])


def list_conversations(
    collection_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ConversationMeta]:
    """List conversations, optionally filtered by collection_name."""
    client = get_client()
    _ensure_meta_collection(client)
    client.load_collection(_META_COL)

    filt = f'collection_name == "{collection_name}"' if collection_name else ""

    results = client.query(
        _META_COL,
        filter=filt or "",
        output_fields=_META_OUTPUT_FIELDS,
        limit=limit,
        offset=offset,
    )
    return [_entity_to_meta(r) for r in results]


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Update the title and updated_at of a conversation."""
    client = get_client()
    _ensure_meta_collection(client)
    # Milvus upsert: must re-insert the full row.
    meta = get_conversation(conversation_id)
    if meta is None:
        return
    meta.title = title
    meta.updated_at = datetime.now(timezone.utc)
    client.upsert(_META_COL, [_meta_to_entity(meta)])
    client.flush(_META_COL)


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation and all its messages."""
    client = get_client()

    # Delete meta
    _ensure_meta_collection(client)
    client.delete(_META_COL, filter=f'conversation_id == "{conversation_id}"')
    client.flush(_META_COL)

    # Delete messages
    _ensure_msg_collection(client)
    client.delete(_MSG_COL, filter=f'conversation_id == "{conversation_id}"')
    client.flush(_MSG_COL)


# ---------------------------------------------------------------------------
# CRUD — Messages
# ---------------------------------------------------------------------------

_MSG_OUTPUT_FIELDS = [
    "message_id",
    "conversation_id",
    "role",
    "content",
    "sources",
    "created_at",
]


def save_message(msg: Message) -> None:
    """Insert a single message."""
    client = get_client()
    _ensure_msg_collection(client)
    client.insert(_MSG_COL, [_msg_to_entity(msg)])
    client.flush(_MSG_COL)


def save_messages(msgs: list[Message]) -> None:
    """Insert multiple messages in a batch."""
    if not msgs:
        return
    client = get_client()
    _ensure_msg_collection(client)
    client.insert(_MSG_COL, [_msg_to_entity(m) for m in msgs])
    client.flush(_MSG_COL)


def get_messages(
    conversation_id: str,
    limit: int = 200,
) -> list[Message]:
    """Retrieve messages for a conversation, ordered by created_at ascending."""
    client = get_client()
    _ensure_msg_collection(client)
    client.load_collection(_MSG_COL)
    results = client.query(
        _MSG_COL,
        filter=f'conversation_id == "{conversation_id}"',
        output_fields=_MSG_OUTPUT_FIELDS,
        limit=limit,
    )
    msgs = [_entity_to_msg(r) for r in results]
    msgs.sort(key=lambda m: m.created_at)
    return msgs
