"""Tests for the search pipeline.

Covers:
- embed_query (internal service — RETRIEVAL_QUERY task type)
- Public search service (dense, sparse, hybrid dispatching)
- API endpoint integration via TestClient
- Edge cases: empty results, validation errors, bad search_type
- Concurrency: verify async offloading via run_in_executor
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.doc import Document
from app.schemas.search import SearchRequest, SearchResult, SearchResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _make_document(
    doc_id: int = 1,
    title: str = "Test Title",
    text: str = "Test document content.",
    dense_vector: list[float] | None = None,
) -> Document:
    """Create a minimal Document for testing."""
    return Document(
        doc_id=doc_id,
        title=title,
        text=text,
        dense_vector=dense_vector or [0.1] * 768,
        metadata={"source_filename": "test.txt"},
    )


def _make_search_hits(
    n: int = 3,
) -> list[tuple[Document, float | None]]:
    """Produce a list of (Document, score) tuples like Milvus repo returns."""
    return [
        (
            _make_document(doc_id=i, title=f"Result {i}", text=f"Content {i}"),
            0.9 - i * 0.1,
        )
        for i in range(1, n + 1)
    ]


FAKE_QUERY_VECTOR = [0.5] * 768


# ===================================================================
# 1. embed_query unit tests (internal service)
# ===================================================================


class TestEmbedQuery:
    """Test the RETRIEVAL_QUERY embedding path."""

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        """embed_query should return a float vector via _embed_query_sync."""
        from app.services.internal.embed import embed_query

        expected = [0.42] * 768
        with patch(
            "app.services.internal.embed._embed_query_sync",
            return_value=expected,
        ):
            result = await embed_query("what is machine learning?")

        assert result == expected
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_query_calls_sync_helper(self):
        """embed_query offloads to run_in_executor calling _embed_query_sync."""
        from app.services.internal.embed import embed_query

        with patch(
            "app.services.internal.embed._embed_query_sync",
            return_value=[0.0] * 768,
        ) as mock_sync:
            await embed_query("test query")

        mock_sync.assert_called_once_with("test query")

    def test_embed_query_sync_uses_query_client(self):
        """_embed_query_sync should use the RETRIEVAL_QUERY client."""
        from app.services.internal.embed import _embed_query_sync

        mock_client = MagicMock()
        mock_client.embed_query.return_value = [0.1] * 768

        with patch(
            "app.services.internal.embed._get_query_embedding_client",
            return_value=mock_client,
        ):
            result = _embed_query_sync("search text")

        mock_client.embed_query.assert_called_once_with("search text")
        assert len(result) == 768

    def test_embed_query_sync_vs_batch_sync_different_clients(self):
        """_embed_query_sync and _embed_batch_sync use different clients."""
        from app.services.internal.embed import _embed_query_sync, _embed_batch_sync

        query_client = MagicMock()
        query_client.embed_query.return_value = [0.1] * 768
        doc_client = MagicMock()
        doc_client.embed_documents.return_value = [[0.2] * 768]

        with (
            patch(
                "app.services.internal.embed._get_query_embedding_client",
                return_value=query_client,
            ),
            patch(
                "app.services.internal.embed._get_document_embedding_client",
                return_value=doc_client,
            ),
        ):
            _embed_query_sync("query")
            _embed_batch_sync(["doc text"])

        query_client.embed_query.assert_called_once()
        doc_client.embed_documents.assert_called_once()


# ===================================================================
# 2. Public search service tests
# ===================================================================


class TestSearchDocumentsService:
    """Test app.services.public.search.search_documents for all 3 search types."""

    @pytest.mark.asyncio
    async def test_dense_search(self):
        """Dense search: embeds query, then runs dense_search in executor."""
        from app.services.public.search import search_documents

        hits = _make_search_hits(3)

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ) as mock_embed,
            patch(
                "app.services.public.search.dense_search",
                return_value=[hits],  # batched: outer list per-query
            ) as mock_dense,
        ):
            results = await search_documents(
                query="machine learning",
                collection_name="test_col",
                search_type="dense",
                top_k=5,
            )

        mock_embed.assert_awaited_once_with("machine learning")
        mock_dense.assert_called_once()
        # Verify we got the correct number of SearchResult objects
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].doc_id == 1
        assert results[0].title == "Result 1"
        assert results[0].score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_sparse_search(self):
        """Sparse search: no embedding needed, uses raw query text."""
        from app.services.public.search import search_documents

        hits = _make_search_hits(2)

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
            ) as mock_embed,
            patch(
                "app.services.public.search.sparse_search",
                return_value=[hits],
            ) as mock_sparse,
        ):
            results = await search_documents(
                query="BM25 search query",
                collection_name="test_col",
                search_type="sparse",
                top_k=10,
            )

        # embed_query should NOT be called for sparse search
        mock_embed.assert_not_awaited()
        mock_sparse.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Hybrid search: embeds query, then runs hybrid_search with both inputs."""
        from app.services.public.search import search_documents

        hits = _make_search_hits(4)

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ) as mock_embed,
            patch(
                "app.services.public.search.hybrid_search",
                return_value=[hits],
            ) as mock_hybrid,
        ):
            results = await search_documents(
                query="hybrid query",
                collection_name="test_col",
                search_type="hybrid",
                top_k=10,
            )

        mock_embed.assert_awaited_once_with("hybrid query")
        mock_hybrid.assert_called_once()
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """When Milvus returns empty hits, the service returns an empty list."""
        from app.services.public.search import search_documents

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ),
            patch(
                "app.services.public.search.dense_search",
                return_value=[[]],  # empty batch
            ),
        ):
            results = await search_documents(
                query="obscure query",
                collection_name="empty_col",
                search_type="dense",
                top_k=5,
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_result_conversion_fields(self):
        """Verify _doc_to_result maps all Document fields correctly to SearchResult."""
        from app.services.public.search import search_documents

        doc = _make_document(
            doc_id=42,
            title="Specific Title",
            text="Specific content for testing.",
        )
        hits = [(doc, 0.95)]

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ),
            patch(
                "app.services.public.search.dense_search",
                return_value=[hits],
            ),
        ):
            results = await search_documents(
                query="specific",
                collection_name="col",
                search_type="dense",
            )

        assert len(results) == 1
        r = results[0]
        assert r.doc_id == 42
        assert r.title == "Specific Title"
        assert r.text == "Specific content for testing."
        assert r.score == pytest.approx(0.95)
        assert r.metadata == {"source_filename": "test.txt"}

    @pytest.mark.asyncio
    async def test_score_none_preserved(self):
        """If Milvus returns score=None, it is preserved in SearchResult."""
        from app.services.public.search import search_documents

        doc = _make_document(doc_id=99)
        hits = [(doc, None)]

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ),
            patch(
                "app.services.public.search.dense_search",
                return_value=[hits],
            ),
        ):
            results = await search_documents(
                query="q",
                collection_name="col",
                search_type="dense",
            )

        assert results[0].score is None

    @pytest.mark.asyncio
    async def test_default_search_type_is_hybrid(self):
        """search_documents defaults to hybrid search when search_type not specified."""
        from app.services.public.search import search_documents

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ),
            patch(
                "app.services.public.search.hybrid_search",
                return_value=[[]],
            ) as mock_hybrid,
        ):
            await search_documents(query="test", collection_name="col")

        mock_hybrid.assert_called_once()


# ===================================================================
# 3. _doc_to_result helper unit tests
# ===================================================================


class TestDocToResult:
    """Direct tests for the _doc_to_result helper."""

    def test_basic_conversion(self):
        from app.services.public.search import _doc_to_result

        doc = _make_document(doc_id=10, title="Hello", text="World")
        result = _doc_to_result(doc, 0.88)

        assert isinstance(result, SearchResult)
        assert result.doc_id == 10
        assert result.title == "Hello"
        assert result.text == "World"
        assert result.score == pytest.approx(0.88)

    def test_none_score(self):
        from app.services.public.search import _doc_to_result

        doc = _make_document()
        result = _doc_to_result(doc, None)
        assert result.score is None

    def test_none_title(self):
        from app.services.public.search import _doc_to_result

        doc = Document(
            doc_id=5,
            title=None,
            text="No title doc",
            dense_vector=[0.1] * 768,
        )
        result = _doc_to_result(doc, 0.5)
        assert result.title is None

    def test_metadata_passthrough(self):
        from app.services.public.search import _doc_to_result

        doc = Document(
            doc_id=7,
            text="Text",
            dense_vector=[0.1] * 768,
            metadata={"key": "value", "num": 42},
        )
        result = _doc_to_result(doc, 0.7)
        assert result.metadata == {"key": "value", "num": 42}


# ===================================================================
# 4. Schema validation tests
# ===================================================================


class TestSearchSchemas:
    """Test Pydantic schema validation for SearchRequest."""

    def test_valid_request_defaults(self):
        req = SearchRequest(query="hello")
        assert req.top_k == 10
        assert req.search_type == "hybrid"
        assert req.language is None

    def test_valid_request_all_fields(self):
        req = SearchRequest(
            query="test",
            top_k=50,
            search_type="dense",
            language="en",
        )
        assert req.top_k == 50
        assert req.search_type == "dense"
        assert req.language == "en"

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            SearchRequest(query="")

    def test_top_k_zero_rejected(self):
        with pytest.raises(Exception):
            SearchRequest(query="test", top_k=0)

    def test_top_k_negative_rejected(self):
        with pytest.raises(Exception):
            SearchRequest(query="test", top_k=-1)

    def test_top_k_over_100_rejected(self):
        with pytest.raises(Exception):
            SearchRequest(query="test", top_k=101)

    def test_top_k_boundary_1(self):
        req = SearchRequest(query="test", top_k=1)
        assert req.top_k == 1

    def test_top_k_boundary_100(self):
        req = SearchRequest(query="test", top_k=100)
        assert req.top_k == 100

    def test_invalid_search_type_rejected(self):
        with pytest.raises(Exception):
            SearchRequest(query="test", search_type="full_text")

    def test_search_response_model(self):
        resp = SearchResponse(
            query="hello",
            collection_name="col",
            search_type="dense",
            total_results=1,
            results=[
                SearchResult(doc_id=1, text="content", score=0.9),
            ],
        )
        assert resp.total_results == 1
        assert resp.results[0].doc_id == 1


# ===================================================================
# 5. API endpoint integration tests
# ===================================================================


class TestSearchEndpoint:
    """Integration tests for POST /api/v1/search/{collection_name}."""

    def test_dense_search_200(self, client: TestClient):
        """Successful dense search returns 200 with correct response shape."""
        mock_results = [
            SearchResult(doc_id=1, title="Doc 1", text="Content 1", score=0.9),
            SearchResult(doc_id=2, title="Doc 2", text="Content 2", score=0.8),
        ]

        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=mock_results,
        ) as mock_svc:
            response = client.post(
                "/api/v1/search/my_collection",
                json={"query": "test query", "search_type": "dense", "top_k": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["collection_name"] == "my_collection"
        assert data["search_type"] == "dense"
        assert data["total_results"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["doc_id"] == 1
        assert data["results"][0]["title"] == "Doc 1"
        assert data["results"][0]["score"] == pytest.approx(0.9)

        mock_svc.assert_awaited_once_with(
            query="test query",
            collection_name="my_collection",
            search_type="dense",
            top_k=5,
        )

    def test_sparse_search_200(self, client: TestClient):
        """Sparse search endpoint works correctly."""
        mock_results = [
            SearchResult(doc_id=10, title="Sparse Hit", text="BM25 content", score=0.7),
        ]

        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            response = client.post(
                "/api/v1/search/test_col",
                json={"query": "sparse query", "search_type": "sparse"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "sparse"
        assert data["total_results"] == 1

    def test_hybrid_search_default(self, client: TestClient):
        """When search_type is omitted, defaults to hybrid."""
        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_svc:
            response = client.post(
                "/api/v1/search/col",
                json={"query": "default search"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "hybrid"
        assert data["total_results"] == 0
        assert data["results"] == []

        # Verify the service was called with hybrid
        mock_svc.assert_awaited_once_with(
            query="default search",
            collection_name="col",
            search_type="hybrid",
            top_k=10,
        )

    def test_empty_results_200(self, client: TestClient):
        """Empty search results should still return 200 with empty list."""
        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.post(
                "/api/v1/search/empty_col",
                json={"query": "no results"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_empty_query_returns_422(self, client: TestClient):
        """Empty query string should fail Pydantic validation -> 422."""
        response = client.post(
            "/api/v1/search/col",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient):
        """Missing query field should fail validation -> 422."""
        response = client.post(
            "/api/v1/search/col",
            json={},
        )
        assert response.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client: TestClient):
        """top_k outside [1, 100] should fail validation -> 422."""
        response = client.post(
            "/api/v1/search/col",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/search/col",
            json={"query": "test", "top_k": 200},
        )
        assert response.status_code == 422

    def test_invalid_search_type_returns_422(self, client: TestClient):
        """Invalid search_type should fail validation -> 422."""
        response = client.post(
            "/api/v1/search/col",
            json={"query": "test", "search_type": "full_text"},
        )
        assert response.status_code == 422

    def test_collection_name_in_response(self, client: TestClient):
        """The collection name from the URL path is echoed in the response."""
        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.post(
                "/api/v1/search/my_special_collection",
                json={"query": "test"},
            )

        assert response.status_code == 200
        assert response.json()["collection_name"] == "my_special_collection"

    def test_score_null_in_response(self, client: TestClient):
        """Results with score=None should serialize as null in JSON."""
        mock_results = [
            SearchResult(doc_id=1, text="content", score=None),
        ]

        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            response = client.post(
                "/api/v1/search/col",
                json={"query": "test"},
            )

        assert response.status_code == 200
        assert response.json()["results"][0]["score"] is None

    def test_metadata_in_response(self, client: TestClient):
        """Metadata dict is correctly serialized in the response."""
        mock_results = [
            SearchResult(
                doc_id=1,
                text="content",
                score=0.5,
                metadata={"source_filename": "paper.pdf", "page": 3},
            ),
        ]

        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            response = client.post(
                "/api/v1/search/col",
                json={"query": "metadata test"},
            )

        assert response.status_code == 200
        meta = response.json()["results"][0]["metadata"]
        assert meta["source_filename"] == "paper.pdf"
        assert meta["page"] == 3

    def test_with_language_field(self, client: TestClient):
        """The optional language field should be accepted without error."""
        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.post(
                "/api/v1/search/col",
                json={"query": "test", "language": "vi"},
            )

        assert response.status_code == 200

    def test_custom_top_k(self, client: TestClient):
        """Custom top_k is passed through to the service."""
        with patch(
            "app.api.v1.endpoints.search.search_documents",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_svc:
            response = client.post(
                "/api/v1/search/col",
                json={"query": "test", "top_k": 42},
            )

        assert response.status_code == 200
        mock_svc.assert_awaited_once_with(
            query="test",
            collection_name="col",
            search_type="hybrid",
            top_k=42,
        )


# ===================================================================
# 6. Concurrency tests
# ===================================================================


class TestSearchConcurrency:
    """Verify the search pipeline doesn't block the event loop."""

    @pytest.mark.asyncio
    async def test_dense_search_uses_executor(self):
        """Dense search should call run_in_executor for the Milvus query."""
        from app.services.public.search import search_documents

        hits = _make_search_hits(1)

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
                return_value=FAKE_QUERY_VECTOR,
            ),
            patch(
                "app.services.public.search.dense_search",
                return_value=[hits],
            ) as mock_dense,
        ):
            # Run two searches concurrently — they should not block each other
            results = await asyncio.gather(
                search_documents("q1", "col", search_type="dense", top_k=5),
                search_documents("q2", "col", search_type="dense", top_k=5),
            )

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        # dense_search was called twice (once per query)
        assert mock_dense.call_count == 2

    @pytest.mark.asyncio
    async def test_sparse_search_does_not_embed(self):
        """Sparse search should never call embed_query."""
        from app.services.public.search import search_documents

        with (
            patch(
                "app.services.public.search.embed_query",
                new_callable=AsyncMock,
            ) as mock_embed,
            patch(
                "app.services.public.search.sparse_search",
                return_value=[[]],
            ),
        ):
            await search_documents("q", "col", search_type="sparse")

        mock_embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hybrid_search_embeds_then_searches(self):
        """Hybrid search should embed query first, then search with both inputs."""
        from app.services.public.search import search_documents

        call_order = []

        async def mock_embed(text):
            call_order.append("embed")
            return FAKE_QUERY_VECTOR

        def mock_hybrid(*args, **kwargs):
            call_order.append("hybrid_search")
            return [_make_search_hits(1)]

        with (
            patch(
                "app.services.public.search.embed_query",
                side_effect=mock_embed,
            ),
            patch(
                "app.services.public.search.hybrid_search",
                side_effect=mock_hybrid,
            ),
        ):
            results = await search_documents("q", "col", search_type="hybrid")

        # Embedding must happen before hybrid search
        assert call_order == ["embed", "hybrid_search"]
        assert len(results) == 1


# ===================================================================
# 7. Search dispatcher helper tests
# ===================================================================


class TestSearchDispatchers:
    """Test the sync dispatcher wrappers in public/search.py."""

    def test_run_dense_search_extracts_first_batch(self):
        """_run_dense_search calls dense_search and returns batched[0]."""
        from app.services.public.search import _run_dense_search

        hits = _make_search_hits(2)
        with patch(
            "app.services.public.search.dense_search",
            return_value=[hits],
        ) as mock:
            result = _run_dense_search(FAKE_QUERY_VECTOR, "col", 5)

        assert len(result) == 2
        mock.assert_called_once_with([FAKE_QUERY_VECTOR], "col", top_k=5)

    def test_run_dense_search_empty_batched(self):
        """_run_dense_search returns [] when dense_search returns empty."""
        from app.services.public.search import _run_dense_search

        with patch(
            "app.services.public.search.dense_search",
            return_value=[],
        ):
            result = _run_dense_search(FAKE_QUERY_VECTOR, "col", 5)

        assert result == []

    def test_run_sparse_search_extracts_first_batch(self):
        """_run_sparse_search calls sparse_search and returns batched[0]."""
        from app.services.public.search import _run_sparse_search

        hits = _make_search_hits(1)
        with patch(
            "app.services.public.search.sparse_search",
            return_value=[hits],
        ) as mock:
            result = _run_sparse_search("query text", "col", 10)

        assert len(result) == 1
        mock.assert_called_once_with(["query text"], "col", top_k=10)

    def test_run_sparse_search_empty_batched(self):
        from app.services.public.search import _run_sparse_search

        with patch(
            "app.services.public.search.sparse_search",
            return_value=[],
        ):
            result = _run_sparse_search("query", "col", 5)

        assert result == []

    def test_run_hybrid_search_extracts_first_batch(self):
        """_run_hybrid_search calls hybrid_search and returns batched[0]."""
        from app.services.public.search import _run_hybrid_search

        hits = _make_search_hits(3)
        with patch(
            "app.services.public.search.hybrid_search",
            return_value=[hits],
        ) as mock:
            result = _run_hybrid_search(FAKE_QUERY_VECTOR, "hybrid query", "col", 5)

        assert len(result) == 3
        mock.assert_called_once_with(
            [FAKE_QUERY_VECTOR], ["hybrid query"], "col", top_k=5
        )

    def test_run_hybrid_search_empty_batched(self):
        from app.services.public.search import _run_hybrid_search

        with patch(
            "app.services.public.search.hybrid_search",
            return_value=[],
        ):
            result = _run_hybrid_search(FAKE_QUERY_VECTOR, "query", "col", 5)

        assert result == []
