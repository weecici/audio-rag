"""Tests for the RAG conversation pipeline.

Covers:
- Internal generate service (build_context_block, build_messages, generate, generate_stream)
- Public conversation service (CRUD + RAG send_message / send_message_stream)
- API endpoint integration via TestClient
- Edge cases: conversation not found (404), validation errors, empty search results
- SSE streaming via EventSourceResponse

All external services (Milvus, Cerebras, embedding) are mocked.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock, call

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.conversation import ConversationMeta, Message
from app.schemas.conversations import (
    ConversationListItem,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    SourceDocument,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _make_meta(
    conversation_id: str = "conv-1",
    title: str | None = "Test Conversation",
    collection_name: str = "test_col",
) -> ConversationMeta:
    return ConversationMeta(
        conversation_id=conversation_id,
        title=title,
        collection_name=collection_name,
        created_at=NOW,
        updated_at=None,
    )


def _make_message(
    message_id: str = "msg-1",
    conversation_id: str = "conv-1",
    role: str = "user",
    content: str = "Hello",
    sources: list | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        created_at=NOW,
    )


def _make_source_dicts(n: int = 2) -> list[dict]:
    return [
        {
            "doc_id": i,
            "title": f"Document {i}",
            "text": f"Content of document {i}.",
            "score": 0.9 - i * 0.1,
        }
        for i in range(1, n + 1)
    ]


# ===================================================================
# 1. Internal generate service — unit tests
# ===================================================================


class TestBuildContextBlock:
    """Test build_context_block() formatting."""

    def test_empty_sources(self):
        from app.services.internal.generate import build_context_block

        result = build_context_block([])
        assert result == "(No relevant documents found.)"

    def test_single_source(self):
        from app.services.internal.generate import build_context_block

        sources = [{"title": "My Doc", "text": "Some content", "score": 0.95}]
        result = build_context_block(sources)
        assert "[Document 1: My Doc]" in result
        assert "(relevance: 0.95)" in result
        assert "Some content" in result

    def test_multiple_sources(self):
        from app.services.internal.generate import build_context_block

        sources = _make_source_dicts(3)
        result = build_context_block(sources)
        assert "[Document 1:" in result
        assert "[Document 2:" in result
        assert "[Document 3:" in result
        # Sources are separated by ---
        assert "---" in result

    def test_source_without_title(self):
        from app.services.internal.generate import build_context_block

        sources = [{"text": "No title here"}]
        result = build_context_block(sources)
        assert "[Document 1: Untitled]" in result

    def test_source_without_score(self):
        from app.services.internal.generate import build_context_block

        sources = [{"title": "Doc", "text": "Content"}]
        result = build_context_block(sources)
        assert "relevance:" not in result


class TestBuildMessages:
    """Test build_messages() prompt assembly."""

    def test_basic_structure(self):
        from app.services.internal.generate import build_messages, RAG_SYSTEM_PROMPT

        messages = build_messages(
            user_query="What is AI?",
            context_block="[Doc 1] AI is ...",
            history=[],
        )
        # System + user = 2 messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == RAG_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "What is AI?" in messages[1]["content"]
        assert "[Doc 1] AI is ..." in messages[1]["content"]

    def test_with_history(self):
        from app.services.internal.generate import build_messages

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = build_messages(
            user_query="Follow-up",
            context_block="context",
            history=history,
        )
        # System + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous answer"
        assert messages[3]["role"] == "user"
        assert "Follow-up" in messages[3]["content"]

    def test_context_in_last_user_message(self):
        from app.services.internal.generate import build_messages

        messages = build_messages(
            user_query="My question",
            context_block="THE CONTEXT",
            history=[],
        )
        # Context is in the user turn, not in system prompt
        assert "THE CONTEXT" not in messages[0]["content"]
        assert "THE CONTEXT" in messages[-1]["content"]
        assert "My question" in messages[-1]["content"]


class TestGenerateAsync:
    """Test the async generate() wrapper."""

    @pytest.mark.asyncio
    async def test_generate_calls_sync_helper(self):
        from app.services.internal.generate import generate

        with patch(
            "app.services.internal.generate._generate_sync",
            return_value="Generated answer",
        ) as mock_sync:
            result = await generate([{"role": "user", "content": "test"}])

        assert result == "Generated answer"
        mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_returns_string(self):
        from app.services.internal.generate import generate

        with patch(
            "app.services.internal.generate._generate_sync",
            return_value="Hello world",
        ):
            result = await generate([{"role": "user", "content": "hi"}])

        assert isinstance(result, str)


class TestGenerateStreamAsync:
    """Test the async generate_stream() wrapper that returns a queue."""

    @pytest.mark.asyncio
    async def test_generate_stream_returns_queue(self):
        from app.services.internal.generate import generate_stream

        def fake_stream(messages):
            yield "Hello "
            yield "world"

        with patch(
            "app.services.internal.generate._generate_stream_sync",
            side_effect=fake_stream,
        ):
            queue = await generate_stream([{"role": "user", "content": "hi"}])

        assert isinstance(queue, asyncio.Queue)

        tokens = []
        while True:
            token = await asyncio.wait_for(queue.get(), timeout=2.0)
            if token is None:
                break
            tokens.append(token)

        assert tokens == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_generate_stream_sends_none_sentinel(self):
        from app.services.internal.generate import generate_stream

        def fake_stream(messages):
            yield "one"

        with patch(
            "app.services.internal.generate._generate_stream_sync",
            side_effect=fake_stream,
        ):
            queue = await generate_stream([{"role": "user", "content": "x"}])

        tokens = []
        while True:
            token = await asyncio.wait_for(queue.get(), timeout=2.0)
            if token is None:
                break
            tokens.append(token)

        assert tokens == ["one"]

    @pytest.mark.asyncio
    async def test_generate_stream_error_still_sends_sentinel(self):
        """Even if the stream raises, None sentinel is sent."""
        from app.services.internal.generate import generate_stream

        def failing_stream(messages):
            yield "partial"
            raise RuntimeError("Stream broke")

        with patch(
            "app.services.internal.generate._generate_stream_sync",
            side_effect=failing_stream,
        ):
            queue = await generate_stream([{"role": "user", "content": "x"}])

        tokens = []
        while True:
            token = await asyncio.wait_for(queue.get(), timeout=2.0)
            if token is None:
                break
            tokens.append(token)

        # We should still get "partial" before the stream ended
        assert "partial" in tokens


class TestGenerateSyncHelper:
    """Test _generate_sync directly (mocking the Cerebras client)."""

    def test_calls_cerebras_client(self):
        from app.services.internal.generate import _generate_sync

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "app.services.internal.generate._get_generation_client",
            return_value=mock_client,
        ):
            result = _generate_sync([{"role": "user", "content": "q"}])

        assert result == "Answer"
        mock_client.chat.completions.create.assert_called_once()

    def test_strips_whitespace(self):
        from app.services.internal.generate import _generate_sync

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  Answer with spaces  "

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "app.services.internal.generate._get_generation_client",
            return_value=mock_client,
        ):
            result = _generate_sync([{"role": "user", "content": "q"}])

        assert result == "Answer with spaces"

    def test_none_content_returns_empty(self):
        from app.services.internal.generate import _generate_sync

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "app.services.internal.generate._get_generation_client",
            return_value=mock_client,
        ):
            result = _generate_sync([{"role": "user", "content": "q"}])

        assert result == ""


# ===================================================================
# 2. Public conversation service — CRUD tests
# ===================================================================


class TestCreateConversation:
    """Test services.public.conversations.create_conversation."""

    @pytest.mark.asyncio
    async def test_create_returns_response(self):
        from app.services.public.conversations import create_conversation

        with patch(
            "app.services.public.conversations._create_conv",
        ) as mock_create:
            result = await create_conversation(
                collection_name="my_docs",
                title="My Chat",
            )

        assert isinstance(result, ConversationResponse)
        assert result.collection_name == "my_docs"
        assert result.title == "My Chat"
        assert result.messages == []
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_generates_uuid(self):
        from app.services.public.conversations import create_conversation

        with patch("app.services.public.conversations._create_conv"):
            result = await create_conversation(collection_name="col")

        # Should be a valid UUID
        uuid.UUID(result.conversation_id)

    @pytest.mark.asyncio
    async def test_create_no_title(self):
        from app.services.public.conversations import create_conversation

        with patch("app.services.public.conversations._create_conv"):
            result = await create_conversation(collection_name="col")

        assert result.title is None


class TestGetConversation:
    """Test services.public.conversations.get_conversation."""

    @pytest.mark.asyncio
    async def test_get_existing_conversation(self):
        from app.services.public.conversations import get_conversation

        meta = _make_meta()
        messages = [
            _make_message(message_id="m1", role="user", content="Hi"),
            _make_message(message_id="m2", role="assistant", content="Hello!"),
        ]

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=messages,
            ),
        ):
            result = await get_conversation("conv-1")

        assert result is not None
        assert result.conversation_id == "conv-1"
        assert result.title == "Test Conversation"
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        from app.services.public.conversations import get_conversation

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=None,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
        ):
            result = await get_conversation("nonexistent")

        assert result is None


class TestListConversations:
    """Test services.public.conversations.list_conversations."""

    @pytest.mark.asyncio
    async def test_list_returns_items(self):
        from app.services.public.conversations import list_conversations

        metas = [
            _make_meta(conversation_id="c1", title="Chat 1"),
            _make_meta(conversation_id="c2", title="Chat 2"),
        ]

        with patch(
            "app.services.public.conversations._list_convs",
            return_value=metas,
        ):
            result = await list_conversations()

        assert len(result) == 2
        assert all(isinstance(r, ConversationListItem) for r in result)
        assert result[0].conversation_id == "c1"
        assert result[1].title == "Chat 2"

    @pytest.mark.asyncio
    async def test_list_empty(self):
        from app.services.public.conversations import list_conversations

        with patch(
            "app.services.public.conversations._list_convs",
            return_value=[],
        ):
            result = await list_conversations()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_collection_filter(self):
        from app.services.public.conversations import list_conversations

        with patch(
            "app.services.public.conversations._list_convs",
            return_value=[],
        ) as mock_list:
            await list_conversations(collection_name="filtered_col", limit=10, offset=5)

        mock_list.assert_called_once_with("filtered_col", 10, 5)


class TestDeleteConversation:
    """Test services.public.conversations.delete_conversation."""

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self):
        from app.services.public.conversations import delete_conversation

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch("app.services.public.conversations._delete_conv") as mock_del,
        ):
            result = await delete_conversation("conv-1")

        assert result is True
        mock_del.assert_called_once_with("conv-1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self):
        from app.services.public.conversations import delete_conversation

        with patch(
            "app.services.public.conversations._get_conv",
            return_value=None,
        ):
            result = await delete_conversation("nonexistent")

        assert result is False


# ===================================================================
# 3. Public conversation service — RAG send_message tests
# ===================================================================


class TestSendMessage:
    """Test the full non-streaming RAG pipeline: send_message."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """send_message retrieves docs, builds prompt, generates, and stores."""
        from app.services.public.conversations import send_message
        from app.schemas.search import SearchResult

        meta = _make_meta()
        search_results = [
            SearchResult(doc_id=1, title="Doc 1", text="Content 1", score=0.9),
            SearchResult(doc_id=2, title="Doc 2", text="Content 2", score=0.8),
        ]

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=search_results,
            ) as mock_search,
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="LLM answer based on docs",
            ) as mock_gen,
            patch(
                "app.services.public.conversations._save_msgs",
            ) as mock_save,
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            result = await send_message("conv-1", "What is AI?")

        assert isinstance(result, SendMessageResponse)
        assert result.user_message.content == "What is AI?"
        assert result.user_message.role == "user"
        assert result.assistant_message.content == "LLM answer based on docs"
        assert result.assistant_message.role == "assistant"
        assert result.assistant_message.sources is not None
        assert len(result.assistant_message.sources) == 2

        mock_search.assert_awaited_once()
        mock_gen.assert_awaited_once()
        mock_save.assert_called_once()
        # Auto-title should be set since no existing messages and no title
        mock_title.assert_not_called()  # meta already has title="Test Conversation"

    @pytest.mark.asyncio
    async def test_auto_title_on_first_message(self):
        """Auto-title is generated when conversation has no title and no messages."""
        from app.services.public.conversations import send_message
        from app.schemas.search import SearchResult

        meta = _make_meta(title=None)  # No title

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],  # No existing messages
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            await send_message("conv-1", "What is machine learning?")

        mock_title.assert_called_once_with("conv-1", "What is machine learning?")

    @pytest.mark.asyncio
    async def test_no_auto_title_when_messages_exist(self):
        """No auto-title when there are existing messages."""
        from app.services.public.conversations import send_message
        from app.schemas.search import SearchResult

        meta = _make_meta(title=None)
        existing = [_make_message(message_id="old-1")]

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=existing,
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            await send_message("conv-1", "Follow-up")

        mock_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_conversation_not_found_raises(self):
        """send_message raises ApiError 404 when conversation not found."""
        from app.services.public.conversations import send_message
        from app.middleware.errors import ApiError

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=None,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            pytest.raises(ApiError) as exc_info,
        ):
            await send_message("nonexistent", "hello")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        """Pipeline works with zero search results."""
        from app.services.public.conversations import send_message

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="No docs found answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch("app.services.public.conversations._update_title"),
        ):
            result = await send_message("conv-1", "question")

        assert result.assistant_message.sources is None

    @pytest.mark.asyncio
    async def test_search_type_passed_through(self):
        """The search_type parameter is forwarded to search_documents."""
        from app.services.public.conversations import send_message

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_search,
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch("app.services.public.conversations._update_title"),
        ):
            await send_message("conv-1", "question", search_type="dense", top_k=3)

        mock_search.assert_awaited_once_with(
            query="question",
            collection_name="test_col",
            search_type="dense",
            top_k=3,
        )

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        """Conversation history is trimmed and passed to build_messages."""
        from app.services.public.conversations import send_message

        existing = [
            _make_message(message_id=f"m{i}", role=r, content=f"msg-{i}")
            for i, r in enumerate(["user", "assistant"] * 15)
        ]

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=existing,
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ) as mock_gen,
            patch("app.services.public.conversations._save_msgs"),
            patch("app.services.public.conversations._update_title"),
            patch(
                "app.services.public.conversations.build_messages",
                wraps=__import__(
                    "app.services.internal.generate", fromlist=["build_messages"]
                ).build_messages,
            ) as mock_build,
        ):
            await send_message("conv-1", "new question")

        # build_messages should have been called with history
        mock_build.assert_called_once()
        kwargs = mock_build.call_args
        history = kwargs.kwargs.get("history") or kwargs[1].get("history", [])
        # History should be trimmed (default GENERATION_HISTORY_TURNS * 2 messages)


class TestSendMessageStream:
    """Test the streaming RAG pipeline: send_message_stream."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        """send_message_stream yields sources, deltas, and done events."""
        from app.services.public.conversations import send_message_stream
        from app.schemas.search import SearchResult

        meta = _make_meta()
        search_results = [
            SearchResult(doc_id=1, title="Doc", text="Content", score=0.9),
        ]

        async def fake_generate_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("Hello ")
            q.put_nowait("world")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=search_results,
            ),
            patch(
                "app.services.public.conversations.generate_stream",
                side_effect=fake_generate_stream,
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch("app.services.public.conversations._update_title"),
        ):
            events = []
            async for event in send_message_stream("conv-1", "question"):
                events.append(event)

        # First event: sources
        assert events[0]["event"] == "sources"
        sources_data = json.loads(events[0]["data"])
        assert len(sources_data) == 1
        assert sources_data[0]["doc_id"] == 1

        # Middle events: deltas
        delta_events = [e for e in events if e["event"] == "delta"]
        assert len(delta_events) == 2
        assert json.loads(delta_events[0]["data"])["content"] == "Hello "
        assert json.loads(delta_events[1]["data"])["content"] == "world"

        # Last event: done
        done_event = events[-1]
        assert done_event["event"] == "done"
        done_data = json.loads(done_event["data"])
        assert "user_message_id" in done_data
        assert "assistant_message_id" in done_data

    @pytest.mark.asyncio
    async def test_stream_conversation_not_found(self):
        """send_message_stream raises ApiError 404 if conversation not found."""
        from app.services.public.conversations import send_message_stream
        from app.middleware.errors import ApiError

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=None,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
        ):
            with pytest.raises(ApiError) as exc_info:
                async for _ in send_message_stream("nonexistent", "hi"):
                    pass

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_persists_messages(self):
        """After streaming completes, messages are persisted."""
        from app.services.public.conversations import send_message_stream

        async def fake_generate_stream(messages):
            q: asyncio.Queue[str | None] = asyncio.Queue()
            q.put_nowait("Response")
            q.put_nowait(None)
            return q

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate_stream",
                side_effect=fake_generate_stream,
            ),
            patch(
                "app.services.public.conversations._save_msgs",
            ) as mock_save,
            patch("app.services.public.conversations._update_title"),
        ):
            async for _ in send_message_stream("conv-1", "test"):
                pass

        mock_save.assert_called_once()
        saved_msgs = mock_save.call_args[0][0]
        assert len(saved_msgs) == 2
        assert saved_msgs[0].role == "user"
        assert saved_msgs[0].content == "test"
        assert saved_msgs[1].role == "assistant"
        assert saved_msgs[1].content == "Response"


# ===================================================================
# 4. Helper function tests
# ===================================================================


class TestTrimHistory:
    """Test _trim_history helper."""

    def test_empty_messages(self):
        from app.services.public.conversations import _trim_history

        result = _trim_history([], max_turns=5)
        assert result == []

    def test_filters_system_messages(self):
        from app.services.public.conversations import _trim_history

        messages = [
            _make_message(role="system", content="System msg"),
            _make_message(role="user", content="User msg"),
            _make_message(role="assistant", content="Asst msg"),
        ]
        result = _trim_history(messages, max_turns=5)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_trims_to_max_turns(self):
        from app.services.public.conversations import _trim_history

        # 6 messages = 3 turns, but max_turns=1 => only last 2 messages
        messages = [
            _make_message(role="user", content=f"Q{i}")
            if i % 2 == 0
            else _make_message(role="assistant", content=f"A{i}")
            for i in range(6)
        ]
        result = _trim_history(messages, max_turns=1)
        assert len(result) == 2

    def test_returns_dicts(self):
        from app.services.public.conversations import _trim_history

        messages = [_make_message(role="user", content="Hi")]
        result = _trim_history(messages, max_turns=5)
        assert isinstance(result[0], dict)
        assert "role" in result[0]
        assert "content" in result[0]


class TestMsgToResponse:
    """Test _msg_to_response helper."""

    def test_basic_conversion(self):
        from app.services.public.conversations import _msg_to_response

        msg = _make_message(role="user", content="Hello")
        result = _msg_to_response(msg)
        assert isinstance(result, MessageResponse)
        assert result.role == "user"
        assert result.content == "Hello"
        assert result.sources is None

    def test_with_sources(self):
        from app.services.public.conversations import _msg_to_response

        sources = _make_source_dicts(2)
        msg = _make_message(role="assistant", content="Answer", sources=sources)
        result = _msg_to_response(msg)
        assert result.sources is not None
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], SourceDocument)
        assert result.sources[0].doc_id == 1


# ===================================================================
# 5. Schema validation tests
# ===================================================================


class TestConversationSchemas:
    """Test Pydantic schema validation for conversation requests."""

    def test_create_request_valid(self):
        req = CreateConversationRequest(collection_name="my_docs")
        assert req.collection_name == "my_docs"
        assert req.title is None

    def test_create_request_with_title(self):
        req = CreateConversationRequest(collection_name="col", title="My Chat")
        assert req.title == "My Chat"

    def test_create_request_empty_collection_rejected(self):
        with pytest.raises(Exception):
            CreateConversationRequest(collection_name="")

    def test_send_message_request_defaults(self):
        req = SendMessageRequest(content="Hello")
        assert req.search_type == "hybrid"
        assert req.top_k == 5
        assert req.stream is False

    def test_send_message_request_all_fields(self):
        req = SendMessageRequest(
            content="test",
            search_type="dense",
            top_k=10,
            stream=True,
        )
        assert req.search_type == "dense"
        assert req.top_k == 10
        assert req.stream is True

    def test_send_message_empty_content_rejected(self):
        with pytest.raises(Exception):
            SendMessageRequest(content="")

    def test_send_message_top_k_zero_rejected(self):
        with pytest.raises(Exception):
            SendMessageRequest(content="test", top_k=0)

    def test_send_message_top_k_over_50_rejected(self):
        with pytest.raises(Exception):
            SendMessageRequest(content="test", top_k=51)

    def test_send_message_invalid_search_type_rejected(self):
        with pytest.raises(Exception):
            SendMessageRequest(content="test", search_type="invalid")

    def test_source_document_model(self):
        src = SourceDocument(doc_id=1, text="content", score=0.9)
        assert src.doc_id == 1
        assert src.title is None

    def test_conversation_response_model(self):
        resp = ConversationResponse(
            conversation_id="c1",
            title="Chat",
            collection_name="col",
            created_at=NOW,
            messages=[],
        )
        assert resp.conversation_id == "c1"
        assert resp.messages == []


# ===================================================================
# 6. API endpoint integration tests
# ===================================================================


class TestCreateConversationEndpoint:
    """POST /api/v1/conversations"""

    def test_create_201(self, client: TestClient):
        mock_response = ConversationResponse(
            conversation_id="new-id",
            title="My Chat",
            collection_name="docs",
            created_at=NOW,
            messages=[],
        )

        with patch(
            "app.api.v1.endpoints.conversations.create_conversation",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_svc:
            response = client.post(
                "/api/v1/conversations",
                json={"collection_name": "docs", "title": "My Chat"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["conversation_id"] == "new-id"
        assert data["collection_name"] == "docs"
        assert data["title"] == "My Chat"
        assert data["messages"] == []

        mock_svc.assert_awaited_once_with(
            collection_name="docs",
            title="My Chat",
        )

    def test_create_without_title_201(self, client: TestClient):
        mock_response = ConversationResponse(
            conversation_id="id-2",
            title=None,
            collection_name="col",
            created_at=NOW,
            messages=[],
        )

        with patch(
            "app.api.v1.endpoints.conversations.create_conversation",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/api/v1/conversations",
                json={"collection_name": "col"},
            )

        assert response.status_code == 201
        assert response.json()["title"] is None

    def test_create_empty_collection_422(self, client: TestClient):
        response = client.post(
            "/api/v1/conversations",
            json={"collection_name": ""},
        )
        assert response.status_code == 422

    def test_create_missing_collection_422(self, client: TestClient):
        response = client.post(
            "/api/v1/conversations",
            json={},
        )
        assert response.status_code == 422


class TestListConversationsEndpoint:
    """GET /api/v1/conversations"""

    def test_list_200(self, client: TestClient):
        mock_items = [
            ConversationListItem(
                conversation_id="c1",
                title="Chat 1",
                collection_name="col",
                created_at=NOW,
            ),
            ConversationListItem(
                conversation_id="c2",
                title="Chat 2",
                collection_name="col",
                created_at=NOW,
            ),
        ]

        with patch(
            "app.api.v1.endpoints.conversations.list_conversations",
            new_callable=AsyncMock,
            return_value=mock_items,
        ):
            response = client.get("/api/v1/conversations")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["conversation_id"] == "c1"

    def test_list_with_filter(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.conversations.list_conversations",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_svc:
            response = client.get(
                "/api/v1/conversations",
                params={"collection_name": "my_col", "limit": 10, "offset": 5},
            )

        assert response.status_code == 200
        mock_svc.assert_awaited_once_with(
            collection_name="my_col",
            limit=10,
            offset=5,
        )

    def test_list_empty_200(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.conversations.list_conversations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.get("/api/v1/conversations")

        assert response.status_code == 200
        assert response.json() == []


class TestGetConversationEndpoint:
    """GET /api/v1/conversations/{conversation_id}"""

    def test_get_200(self, client: TestClient):
        mock_response = ConversationResponse(
            conversation_id="conv-1",
            title="Test",
            collection_name="col",
            created_at=NOW,
            messages=[
                MessageResponse(
                    message_id="m1",
                    role="user",
                    content="Hi",
                    created_at=NOW,
                ),
            ],
        )

        with patch(
            "app.api.v1.endpoints.conversations.get_conversation",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.get("/api/v1/conversations/conv-1")

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv-1"
        assert len(data["messages"]) == 1

    def test_get_404(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.conversations.get_conversation",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/api/v1/conversations/nonexistent")

        assert response.status_code == 404


class TestDeleteConversationEndpoint:
    """DELETE /api/v1/conversations/{conversation_id}"""

    def test_delete_204(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.conversations.delete_conversation",
            new_callable=AsyncMock,
            return_value=True,
        ):
            response = client.delete("/api/v1/conversations/conv-1")

        assert response.status_code == 204

    def test_delete_404(self, client: TestClient):
        with patch(
            "app.api.v1.endpoints.conversations.delete_conversation",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.delete("/api/v1/conversations/nonexistent")

        assert response.status_code == 404


class TestSendMessageEndpoint:
    """POST /api/v1/conversations/{conversation_id}/messages"""

    def test_send_message_200(self, client: TestClient):
        """Non-streaming message returns JSON with user + assistant messages."""
        mock_response = SendMessageResponse(
            user_message=MessageResponse(
                message_id="um-1",
                role="user",
                content="What is AI?",
                created_at=NOW,
            ),
            assistant_message=MessageResponse(
                message_id="am-1",
                role="assistant",
                content="AI is artificial intelligence.",
                sources=[
                    SourceDocument(doc_id=1, title="Doc", text="AI text", score=0.9),
                ],
                created_at=NOW,
            ),
        )

        with patch(
            "app.api.v1.endpoints.conversations.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_svc:
            response = client.post(
                "/api/v1/conversations/conv-1/messages",
                json={
                    "content": "What is AI?",
                    "search_type": "dense",
                    "top_k": 3,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["user_message"]["content"] == "What is AI?"
        assert data["assistant_message"]["content"] == "AI is artificial intelligence."
        assert len(data["assistant_message"]["sources"]) == 1

        mock_svc.assert_awaited_once_with(
            conversation_id="conv-1",
            user_content="What is AI?",
            search_type="dense",
            top_k=3,
        )

    def test_send_message_default_params(self, client: TestClient):
        """Default search_type=hybrid, top_k=5, stream=False."""
        mock_response = SendMessageResponse(
            user_message=MessageResponse(
                message_id="um",
                role="user",
                content="test",
                created_at=NOW,
            ),
            assistant_message=MessageResponse(
                message_id="am",
                role="assistant",
                content="response",
                created_at=NOW,
            ),
        )

        with patch(
            "app.api.v1.endpoints.conversations.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_svc:
            response = client.post(
                "/api/v1/conversations/conv-1/messages",
                json={"content": "test"},
            )

        assert response.status_code == 200
        mock_svc.assert_awaited_once_with(
            conversation_id="conv-1",
            user_content="test",
            search_type="hybrid",
            top_k=5,
        )

    def test_send_message_empty_content_422(self, client: TestClient):
        response = client.post(
            "/api/v1/conversations/conv-1/messages",
            json={"content": ""},
        )
        assert response.status_code == 422

    def test_send_message_missing_content_422(self, client: TestClient):
        response = client.post(
            "/api/v1/conversations/conv-1/messages",
            json={},
        )
        assert response.status_code == 422

    def test_send_message_invalid_search_type_422(self, client: TestClient):
        response = client.post(
            "/api/v1/conversations/conv-1/messages",
            json={"content": "test", "search_type": "invalid"},
        )
        assert response.status_code == 422


class TestStreamingEndpoint:
    """POST /api/v1/conversations/{id}/messages with stream=true"""

    def test_streaming_returns_sse(self, client: TestClient):
        """When stream=true, the endpoint returns an SSE event stream."""

        async def fake_stream(*args, **kwargs):
            yield {
                "event": "sources",
                "data": json.dumps([{"doc_id": 1, "text": "doc"}]),
            }
            yield {
                "event": "delta",
                "data": json.dumps({"content": "Hello"}),
            }
            yield {
                "event": "delta",
                "data": json.dumps({"content": " world"}),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {"user_message_id": "u1", "assistant_message_id": "a1"}
                ),
            }

        with patch(
            "app.api.v1.endpoints.conversations.send_message_stream",
            side_effect=fake_stream,
        ):
            response = client.post(
                "/api/v1/conversations/conv-1/messages",
                json={"content": "test", "stream": True},
            )

        assert response.status_code == 200
        # SSE content type
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events from body
        body = response.text
        assert "event: sources" in body
        assert "event: delta" in body
        assert "event: done" in body

    def test_streaming_calls_stream_service(self, client: TestClient):
        """Verify stream=true calls send_message_stream, not send_message."""

        async def fake_stream(*args, **kwargs):
            yield {"event": "done", "data": "{}"}

        with (
            patch(
                "app.api.v1.endpoints.conversations.send_message_stream",
                side_effect=fake_stream,
            ) as mock_stream,
            patch(
                "app.api.v1.endpoints.conversations.send_message",
                new_callable=AsyncMock,
            ) as mock_non_stream,
        ):
            client.post(
                "/api/v1/conversations/conv-1/messages",
                json={"content": "test", "stream": True},
            )

        mock_stream.assert_called_once_with(
            conversation_id="conv-1",
            user_content="test",
            search_type="hybrid",
            top_k=5,
        )
        mock_non_stream.assert_not_awaited()

    def test_non_streaming_does_not_call_stream(self, client: TestClient):
        """stream=false should call send_message, not send_message_stream."""
        mock_response = SendMessageResponse(
            user_message=MessageResponse(
                message_id="u",
                role="user",
                content="x",
                created_at=NOW,
            ),
            assistant_message=MessageResponse(
                message_id="a",
                role="assistant",
                content="y",
                created_at=NOW,
            ),
        )

        with (
            patch(
                "app.api.v1.endpoints.conversations.send_message",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_non_stream,
            patch(
                "app.api.v1.endpoints.conversations.send_message_stream",
            ) as mock_stream,
        ):
            client.post(
                "/api/v1/conversations/conv-1/messages",
                json={"content": "x", "stream": False},
            )

        mock_non_stream.assert_awaited_once()
        mock_stream.assert_not_called()


# ===================================================================
# 7. Concurrency tests
# ===================================================================


class TestConversationConcurrency:
    """Verify conversation operations don't block the event loop."""

    @pytest.mark.asyncio
    async def test_concurrent_creates(self):
        """Multiple conversation creations can run concurrently."""
        from app.services.public.conversations import create_conversation

        with patch("app.services.public.conversations._create_conv"):
            results = await asyncio.gather(
                create_conversation(collection_name="col1"),
                create_conversation(collection_name="col2"),
                create_conversation(collection_name="col3"),
            )

        assert len(results) == 3
        # All should have different IDs
        ids = [r.conversation_id for r in results]
        assert len(set(ids)) == 3

    @pytest.mark.asyncio
    async def test_concurrent_gets(self):
        """Multiple get operations can run concurrently."""
        from app.services.public.conversations import get_conversation

        meta = _make_meta()

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=meta,
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
        ):
            results = await asyncio.gather(
                get_conversation("conv-1"),
                get_conversation("conv-1"),
            )

        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_send_messages(self):
        """Multiple send_message calls can run concurrently."""
        from app.services.public.conversations import send_message

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch("app.services.public.conversations._update_title"),
        ):
            results = await asyncio.gather(
                send_message("conv-1", "q1"),
                send_message("conv-1", "q2"),
            )

        assert len(results) == 2
        assert results[0].user_message.content == "q1"
        assert results[1].user_message.content == "q2"


# ===================================================================
# 8. Auto-title edge cases
# ===================================================================


class TestAutoTitle:
    """Test auto-title truncation behavior."""

    @pytest.mark.asyncio
    async def test_long_message_title_truncated(self):
        """Messages longer than 100 chars get truncated with ellipsis."""
        from app.services.public.conversations import send_message

        long_content = "A" * 150

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(title=None),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            await send_message("conv-1", long_content)

        call_args = mock_title.call_args[0]
        title = call_args[1]
        assert len(title) == 103  # 100 chars + "..."
        assert title.endswith("...")

    @pytest.mark.asyncio
    async def test_short_message_no_ellipsis(self):
        """Messages under 100 chars don't get ellipsis."""
        from app.services.public.conversations import send_message

        short_content = "Short question"

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(title=None),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            await send_message("conv-1", short_content)

        call_args = mock_title.call_args[0]
        title = call_args[1]
        assert title == "Short question"
        assert "..." not in title

    @pytest.mark.asyncio
    async def test_no_auto_title_when_title_exists(self):
        """No auto-title when conversation already has a title."""
        from app.services.public.conversations import send_message

        with (
            patch(
                "app.services.public.conversations._get_conv",
                return_value=_make_meta(title="Existing Title"),
            ),
            patch(
                "app.services.public.conversations._get_msgs",
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.search_documents",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.public.conversations.generate",
                new_callable=AsyncMock,
                return_value="Answer",
            ),
            patch("app.services.public.conversations._save_msgs"),
            patch(
                "app.services.public.conversations._update_title",
            ) as mock_title,
        ):
            await send_message("conv-1", "First message")

        mock_title.assert_not_called()
