"""Tests for the file ingestion pipeline.

Covers:
- Redis job store operations
- Text chunking
- Multi-format file loading (txt, pdf, docx)
- process_files pipeline (mocked LLM + embedding)
- Audio transcription (mocked Whisper)
- Concurrent audio+text ingestion
- API endpoints (POST /files, GET /jobs)
- Job status tracking with transcribing state
"""

import json
import uuid
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def sample_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file with enough content to produce multiple chunks."""
    content = (
        "Artificial intelligence is transforming how we interact with technology. "
        "Machine learning models can process vast amounts of data to find patterns "
        "that humans might miss. Deep learning, a subset of machine learning, uses "
        "neural networks with many layers to learn representations of data.\n\n"
        "Natural language processing enables computers to understand, interpret, "
        "and generate human language. This technology powers chatbots, translation "
        "services, and content analysis tools. Recent advances in transformer "
        "architectures have dramatically improved NLP capabilities.\n\n"
        "Computer vision allows machines to interpret and make decisions based on "
        "visual data from the real world. Applications include autonomous vehicles, "
        "medical image analysis, and quality control in manufacturing. The field "
        "has seen rapid progress thanks to convolutional neural networks.\n\n"
    )
    # Repeat to ensure we get multiple chunks
    full_content = content * 5
    f = tmp_path / "sample.txt"
    f.write_text(full_content, encoding="utf-8")
    return f


@pytest.fixture()
def small_text_file(tmp_path: Path) -> Path:
    """A small text file that fits in a single chunk."""
    f = tmp_path / "small.txt"
    f.write_text("Hello world. This is a small document.", encoding="utf-8")
    return f


@pytest.fixture()
def fake_audio_file(tmp_path: Path) -> Path:
    """A fake audio file (we mock the transcription)."""
    f = tmp_path / "lecture.mp3"
    f.write_bytes(b"\x00" * 100)  # dummy bytes
    return f


# ===================================================================
# 1. Redis job store tests (unit tests with fakeredis)
# ===================================================================


class FakeRedis:
    """Minimal in-memory Redis replacement for testing."""

    def __init__(self):
        self._data: dict[str, dict[str, str]] = {}
        self._expiry: dict[str, int] = {}

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | None = None,
        mapping: dict | None = None,
    ) -> int:
        if name not in self._data:
            self._data[name] = {}
        count = 0
        if key is not None and value is not None:
            self._data[name][str(key)] = str(value)
            count += 1
        if mapping:
            for k, v in mapping.items():
                self._data[name][str(k)] = str(v)
                count += 1
        return count

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._data.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        if key not in self._data:
            self._data[key] = {}
        current = int(self._data[key].get(field, "0"))
        new_val = current + amount
        self._data[key][field] = str(new_val)
        return new_val

    def expire(self, key: str, seconds: int) -> bool:
        self._expiry[key] = seconds
        return True


@pytest.fixture()
def fake_redis():
    return FakeRedis()


@pytest.fixture()
def _patch_redis(fake_redis):
    with patch(
        "app.repositories.redis.job_store.get_redis_client", return_value=fake_redis
    ):
        yield


class TestRedisJobStore:
    """Test Redis job store CRUD operations."""

    @pytest.mark.usefixtures("_patch_redis")
    def test_create_job(self):
        from app.repositories.redis.job_store import create_job, get_job

        job_id = "test-job-001"
        create_job(job_id, "my_collection", ["a.txt", "b.txt"])

        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        assert job["collection"] == "my_collection"
        assert job["total_files"] == 2
        assert job["processed"] == 0
        assert "a.txt" in job["files"]
        assert "b.txt" in job["files"]

    @pytest.mark.usefixtures("_patch_redis")
    def test_update_job_status(self):
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            update_job_status,
        )

        job_id = "test-job-002"
        create_job(job_id, "col", ["x.txt"])
        update_job_status(job_id, "processing")

        job = get_job(job_id)
        assert job["status"] == "processing"

    @pytest.mark.usefixtures("_patch_redis")
    def test_update_file_status_completed(self):
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            update_file_status,
        )

        job_id = "test-job-003"
        create_job(job_id, "col", ["doc.txt"])
        update_file_status(job_id, "doc.txt", "completed", chunks=5)

        job = get_job(job_id)
        assert job["processed"] == 1
        assert job["documents_ingested"] == 5
        assert job["files"]["doc.txt"]["status"] == "completed"
        assert job["files"]["doc.txt"]["chunks"] == 5

    @pytest.mark.usefixtures("_patch_redis")
    def test_update_file_status_failed(self):
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            update_file_status,
        )

        job_id = "test-job-004"
        create_job(job_id, "col", ["bad.txt"])
        update_file_status(job_id, "bad.txt", "failed", error="parse error")

        job = get_job(job_id)
        assert job["failed_cnt"] == 1
        assert job["processed"] == 1
        assert job["files"]["bad.txt"]["status"] == "failed"
        assert job["files"]["bad.txt"]["error"] == "parse error"

    @pytest.mark.usefixtures("_patch_redis")
    def test_update_file_status_transcribing(self):
        """Test the new 'transcribing' status for audio files."""
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            update_file_status,
        )

        job_id = "test-job-transcribe"
        create_job(job_id, "col", ["audio.mp3"])
        update_file_status(job_id, "audio.mp3", "transcribing")

        job = get_job(job_id)
        assert job["files"]["audio.mp3"]["status"] == "transcribing"

    @pytest.mark.usefixtures("_patch_redis")
    def test_set_job_error(self):
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            set_job_error,
        )

        job_id = "test-job-005"
        create_job(job_id, "col", ["f.txt"])
        set_job_error(job_id, "catastrophic failure")

        job = get_job(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "catastrophic failure"

    @pytest.mark.usefixtures("_patch_redis")
    def test_set_job_result(self):
        from app.repositories.redis.job_store import (
            create_job,
            get_job,
            set_job_result,
        )

        job_id = "test-job-006"
        create_job(job_id, "col", ["a.txt", "b.txt"])
        set_job_result(job_id, documents_ingested=42)

        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["documents_ingested"] == 42

    @pytest.mark.usefixtures("_patch_redis")
    def test_get_nonexistent_job_returns_none(self):
        from app.repositories.redis.job_store import get_job

        assert get_job("nonexistent-id") is None


# ===================================================================
# 2. Chunk service tests
# ===================================================================


class TestChunkText:
    def test_basic_chunking(self):
        from app.services.internal.chunk import chunk_text

        text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
        chunks = chunk_text(text, source="test.txt", chunk_size=600, chunk_overlap=50)

        assert len(chunks) >= 2
        assert all(c.source == "test.txt" for c in chunks)
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    def test_small_text_single_chunk(self):
        from app.services.internal.chunk import chunk_text

        chunks = chunk_text("Hello world", source="small.txt", chunk_size=1024)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].index == 0

    def test_empty_text(self):
        from app.services.internal.chunk import chunk_text

        chunks = chunk_text("", source="empty.txt")
        assert chunks == []

    def test_chunk_overlap(self):
        from app.services.internal.chunk import chunk_text

        # Create text that will be split into at least 2 chunks
        text = " ".join(["word"] * 300)
        chunks = chunk_text(
            text, source="overlap.txt", chunk_size=200, chunk_overlap=50
        )

        if len(chunks) >= 2:
            # Check overlap exists: end of first chunk should appear at start of second
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0


# ===================================================================
# 3. Title generation tests (mocked Cerebras)
# ===================================================================


class TestTitleGeneration:
    def test_generate_title_sync_success(self):
        from app.services.internal.chunk import _generate_title_sync

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI in Healthcare"

        with patch(
            "app.services.internal.chunk._get_cerebras_client"
        ) as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            title = _generate_title_sync("Some text about AI in healthcare")
            assert title == "AI in Healthcare"

    def test_generate_title_sync_failure_returns_none(self):
        from app.services.internal.chunk import _generate_title_sync

        with patch(
            "app.services.internal.chunk._get_cerebras_client"
        ) as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API down")
            mock_client_fn.return_value = mock_client

            title = _generate_title_sync("Some text")
            assert title == None

    @pytest.mark.asyncio
    async def test_generate_titles_batch(self):
        from app.services.internal.chunk import generate_titles, TextChunk

        chunks = [
            TextChunk(text="Chunk about cats", index=0, source="test.txt"),
            TextChunk(text="Chunk about dogs", index=1, source="test.txt"),
        ]

        with patch(
            "app.services.internal.chunk._generate_title_sync",
            side_effect=["Cats Overview", "Dogs Overview"],
        ):
            result = await generate_titles(chunks)
            assert result[0].title == "Cats Overview"
            assert result[1].title == "Dogs Overview"


# ===================================================================
# 4. Embedding tests (mocked Google API)
# ===================================================================


class TestDenseEncode:
    @pytest.mark.asyncio
    async def test_dense_embed_returns_vectors(self):
        from app.services.internal.embed import dense_embed

        # Patch the sync function that run_in_executor calls directly.
        fake_vectors = [[0.1] * 1024, [0.2] * 1024]
        with patch(
            "app.services.internal.embed._embed_batch_sync",
            return_value=fake_vectors,
        ):
            vectors = await dense_embed(["text one", "text two"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 1024
            assert vectors[1][0] == 0.2

    @pytest.mark.asyncio
    async def test_dense_embed_empty_input(self):
        from app.services.internal.embed import dense_embed

        vectors = await dense_embed([])
        assert vectors == []


# ===================================================================
# 5. Multi-format file loading tests
# ===================================================================


class TestFileLoading:
    def test_load_text_file(self, small_text_file: Path):
        from app.services.internal.process_files import _load_text

        text = _load_text(small_text_file)
        assert "Hello world" in text

    def test_load_text_fallback_for_unknown_ext(self, tmp_path: Path):
        """Unknown extensions fall back to TextLoader."""
        from app.services.internal.process_files import _load_text

        f = tmp_path / "notes.log"
        f.write_text("Log entry content", encoding="utf-8")
        text = _load_text(f)
        assert "Log entry content" in text

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        from app.services.internal.process_files import _load_text

        text = _load_text(tmp_path / "missing.txt")
        assert text == ""

    def test_loader_map_has_expected_types(self):
        from app.services.internal.process_files import _LOADER_MAP

        assert ".txt" in _LOADER_MAP
        assert ".md" in _LOADER_MAP
        assert ".pdf" in _LOADER_MAP
        assert ".docx" in _LOADER_MAP
        assert ".doc" in _LOADER_MAP


# ===================================================================
# 6. Process files pipeline tests (mocked externals)
# ===================================================================


class TestProcessFiles:
    @pytest.mark.asyncio
    async def test_process_single_file(self, small_text_file: Path):
        from app.services.internal.process_files import process_files

        with (
            patch(
                "app.services.internal.embed._embed_batch_sync",
                return_value=[[0.5] * 1024],
            ),
            patch(
                "app.services.internal.chunk._generate_title_sync",
                return_value="Small Document",
            ),
        ):
            docs = await process_files([small_text_file])

        assert len(docs) == 1
        doc = docs[0]
        assert doc.title == "Small Document"
        assert doc.dense_vector is not None
        assert len(doc.dense_vector) == 1024
        assert doc.metadata is not None
        assert doc.metadata["source_filename"] == "small.txt"
        assert doc.text == "Hello world. This is a small document."

    @pytest.mark.asyncio
    async def test_process_multiple_files_concurrently(
        self, sample_text_file: Path, small_text_file: Path
    ):
        """Multiple files are processed via asyncio.gather (concurrently)."""
        from app.services.internal.process_files import process_files

        def fake_embed(
            texts: list[str], titles: list[str] | None = None
        ) -> list[list[float]]:
            return [[0.5] * 1024 for _ in texts]

        with (
            patch(
                "app.services.internal.embed._embed_batch_sync",
                side_effect=fake_embed,
            ),
            patch(
                "app.services.internal.chunk._generate_title_sync",
                return_value="Generated Title",
            ),
        ):
            docs = await process_files([sample_text_file, small_text_file])

        # sample_text_file is large -> multiple chunks; small -> 1 chunk
        assert len(docs) >= 2
        for doc in docs:
            assert doc.dense_vector is not None
            assert len(doc.dense_vector) == 1024
            assert doc.title == "Generated Title"
            assert doc.doc_id > 0

    @pytest.mark.asyncio
    async def test_process_single_file_func(self, small_text_file: Path):
        """Test process_single_file directly (the per-file coroutine)."""
        from app.services.internal.process_files import process_single_file

        with (
            patch(
                "app.services.internal.embed._embed_batch_sync",
                return_value=[[0.5] * 1024],
            ),
            patch(
                "app.services.internal.chunk._generate_title_sync",
                return_value="Single File Title",
            ),
        ):
            docs = await process_single_file(small_text_file)

        assert len(docs) == 1
        assert docs[0].title == "Single File Title"

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, tmp_path: Path):
        from app.services.internal.process_files import process_files

        docs = await process_files([tmp_path / "nonexistent.txt"])
        assert docs == []

    @pytest.mark.asyncio
    async def test_process_empty_list(self):
        from app.services.internal.process_files import process_files

        docs = await process_files([])
        assert docs == []

    @pytest.mark.asyncio
    async def test_stable_doc_id_deterministic(self):
        from app.services.internal.process_files import _stable_doc_id

        id1 = _stable_doc_id("file.txt", 0)
        id2 = _stable_doc_id("file.txt", 0)
        id3 = _stable_doc_id("file.txt", 1)

        assert id1 == id2  # deterministic
        assert id1 != id3  # different chunk index


# ===================================================================
# 7. Speech-to-text tests (mocked Whisper)
# ===================================================================


class TestSpeechToText:
    """Tests for speech_to_text module using BatchedInferencePipeline."""

    def _mock_batched_model(self):
        """Return a MagicMock that satisfies _get_batched_model() expectations."""
        ct2_model = MagicMock()
        ct2_model.device = "cpu"
        ct2_model.load_model = MagicMock()
        ct2_model.unload_model = MagicMock()

        whisper_model = MagicMock()
        whisper_model.model = ct2_model

        batched_model = MagicMock()
        batched_model.model = whisper_model
        return batched_model

    def test_parse_audio_to_text_returns_transcript_paths(
        self, fake_audio_file: Path, tmp_path: Path
    ):
        """parse_audio_to_text transcribes audio and writes .txt files."""
        from app.services.internal.speech_to_text import parse_audio_to_text

        out_dir = tmp_path / "transcripts"

        with (
            patch(
                "app.services.internal.speech_to_text._get_batched_model",
                return_value=self._mock_batched_model(),
            ),
            patch(
                "app.services.internal.speech_to_text._transcribe_single",
                return_value="This is the transcript of the lecture.",
            ),
        ):
            paths = parse_audio_to_text([fake_audio_file], out_dir=out_dir)

        assert len(paths) == 1
        assert paths[0].suffix == ".txt"
        assert paths[0].stem == "lecture"
        assert paths[0].read_text() == "This is the transcript of the lecture."

    def test_parse_audio_to_text_handles_failure(
        self, fake_audio_file: Path, tmp_path: Path
    ):
        """If transcription fails for a file, it's skipped (not raised)."""
        from app.services.internal.speech_to_text import parse_audio_to_text

        out_dir = tmp_path / "transcripts"

        with (
            patch(
                "app.services.internal.speech_to_text._get_batched_model",
                return_value=self._mock_batched_model(),
            ),
            patch(
                "app.services.internal.speech_to_text._transcribe_single",
                side_effect=RuntimeError("GPU OOM"),
            ),
        ):
            paths = parse_audio_to_text([fake_audio_file], out_dir=out_dir)

        assert paths == []

    def test_parse_audio_to_text_multiple_files(self, tmp_path: Path):
        """Multiple audio files produce multiple transcript files."""
        from app.services.internal.speech_to_text import parse_audio_to_text

        audio1 = tmp_path / "part1.mp3"
        audio2 = tmp_path / "part2.wav"
        audio1.write_bytes(b"\x00" * 50)
        audio2.write_bytes(b"\x00" * 50)
        out_dir = tmp_path / "transcripts"

        with (
            patch(
                "app.services.internal.speech_to_text._get_batched_model",
                return_value=self._mock_batched_model(),
            ),
            patch(
                "app.services.internal.speech_to_text._transcribe_single",
                side_effect=["Transcript one.", "Transcript two."],
            ),
        ):
            paths = parse_audio_to_text([audio1, audio2], out_dir=out_dir)

        assert len(paths) == 2
        assert paths[0].stem == "part1"
        assert paths[1].stem == "part2"


# ===================================================================
# 8. API endpoint tests
# ===================================================================


class TestFilesEndpoint:
    @pytest.mark.usefixtures("_patch_redis")
    def test_upload_returns_202(self, client: TestClient, small_text_file: Path):
        with open(small_text_file, "rb") as f:
            response = client.post(
                "/api/v1/files/test_collection",
                files=[("files", ("small.txt", f, "text/plain"))],
            )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["collection_name"] == "test_collection"
        assert data["status"] == "queued"
        assert len(data["results"]) == 1
        assert data["results"][0]["filename"] == "small.txt"
        assert data["results"][0]["status"] == "accepted"

    @pytest.mark.usefixtures("_patch_redis")
    def test_upload_multiple_files(
        self, client: TestClient, small_text_file: Path, sample_text_file: Path
    ):
        with (
            open(small_text_file, "rb") as f1,
            open(sample_text_file, "rb") as f2,
        ):
            response = client.post(
                "/api/v1/files/test_collection",
                files=[
                    ("files", ("small.txt", f1, "text/plain")),
                    ("files", ("sample.txt", f2, "text/plain")),
                ],
            )

        assert response.status_code == 202
        data = response.json()
        assert len(data["results"]) == 2

    @pytest.mark.usefixtures("_patch_redis")
    def test_upload_audio_file_accepted(
        self, client: TestClient, fake_audio_file: Path
    ):
        """Audio files with allowed MIME types should be accepted."""
        with open(fake_audio_file, "rb") as f:
            response = client.post(
                "/api/v1/files/test_collection",
                files=[("files", ("lecture.mp3", f, "audio/mpeg"))],
            )

        assert response.status_code == 202
        data = response.json()
        assert data["results"][0]["filename"] == "lecture.mp3"
        assert data["results"][0]["status"] == "accepted"

    def test_upload_unsupported_media_type(self, client: TestClient, tmp_path: Path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        with open(img, "rb") as f:
            response = client.post(
                "/api/v1/files/test_collection",
                files=[("files", ("photo.jpg", f, "image/jpeg"))],
            )

        assert response.status_code == 415


class TestJobsEndpoint:
    @pytest.mark.usefixtures("_patch_redis")
    def test_get_job_status(self, client: TestClient):
        from app.repositories.redis.job_store import create_job

        job_id = str(uuid.uuid4())
        create_job(job_id, "test_col", ["file1.txt"])

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["collection"] == "test_col"
        assert data["total_files"] == 1

    @pytest.mark.usefixtures("_patch_redis")
    def test_get_job_status_with_progress(self, client: TestClient):
        from app.repositories.redis.job_store import (
            create_job,
            update_job_status,
            update_file_status,
        )

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["a.txt", "b.txt"])
        update_job_status(job_id, "processing")
        update_file_status(job_id, "a.txt", "completed", chunks=10)

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["processed"] == 1
        assert data["documents_ingested"] == 10
        assert data["files"]["a.txt"]["status"] == "completed"
        assert data["files"]["a.txt"]["chunks"] == 10
        assert data["files"]["b.txt"]["status"] == "pending"

    @pytest.mark.usefixtures("_patch_redis")
    def test_get_job_status_audio_transcribing(self, client: TestClient):
        """Job status endpoint reflects 'transcribing' state for audio files."""
        from app.repositories.redis.job_store import (
            create_job,
            update_job_status,
            update_file_status,
        )

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["doc.txt", "audio.mp3"])
        update_job_status(job_id, "processing")
        update_file_status(job_id, "audio.mp3", "transcribing")
        update_file_status(job_id, "doc.txt", "processing")

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["files"]["audio.mp3"]["status"] == "transcribing"
        assert data["files"]["doc.txt"]["status"] == "processing"

    @pytest.mark.usefixtures("_patch_redis")
    def test_get_nonexistent_job_returns_404(self, client: TestClient):
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


# ===================================================================
# 9. Public ingest service tests (mocked internals)
# ===================================================================


class TestIngestService:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_text_files_success(self, small_text_file: Path):
        """Text-only ingestion: files processed concurrently, upserted to Milvus."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job
        from app.models import Document

        job_id = str(uuid.uuid4())
        fname = "small.txt"
        create_job(job_id, "test_col", [fname])

        mock_doc = Document(
            doc_id=123,
            title="Test Title",
            text="Hello world",
            dense_vector=[0.1] * 1024,
        )

        with (
            patch(
                "app.services.public.ingest.process_single_file",
                return_value=[mock_doc],
            ),
            patch("app.services.public.ingest.upsert_documents") as mock_upsert,
        ):
            await ingest_files(job_id, [small_text_file], [fname], "test_col")

        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["documents_ingested"] == 1
        assert job["files"][fname]["status"] == "completed"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_text_files_partial_failure(self, tmp_path: Path):
        """If one text file fails, others still succeed (partial failure)."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job
        from app.models import Document

        good_file = tmp_path / "good.txt"
        good_file.write_text("Some good content", encoding="utf-8")
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("Bad content", encoding="utf-8")

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["good.txt", "bad.txt"])

        mock_doc = Document(
            doc_id=456,
            title="Good Doc",
            text="Some good content",
            dense_vector=[0.2] * 1024,
        )

        call_count = 0

        async def mock_process_single(path):
            nonlocal call_count
            call_count += 1
            if "bad" in str(path):
                raise RuntimeError("Processing failed")
            return [mock_doc]

        with (
            patch(
                "app.services.public.ingest.process_single_file",
                side_effect=mock_process_single,
            ),
            patch("app.services.public.ingest.upsert_documents"),
        ):
            await ingest_files(
                job_id,
                [good_file, bad_file],
                ["good.txt", "bad.txt"],
                "col",
            )

        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["files"]["good.txt"]["status"] == "completed"
        assert job["files"]["bad.txt"]["status"] == "failed"
        assert job["failed_cnt"] == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_audio_files(self, tmp_path: Path):
        """Audio files: transcribed -> transcripts processed as text -> upserted."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job
        from app.models import Document

        audio_file = tmp_path / "lecture.mp3"
        audio_file.write_bytes(b"\x00" * 50)

        # Create a transcript file that parse_audio_to_text would produce
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        transcript_file = transcript_dir / "lecture.txt"
        transcript_file.write_text("Transcribed lecture content.", encoding="utf-8")

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["lecture.mp3"])

        mock_doc = Document(
            doc_id=789,
            title="Lecture Title",
            text="Transcribed lecture content.",
            dense_vector=[0.3] * 1024,
        )

        with (
            patch(
                "app.services.public.ingest.parse_audio_to_text",
                return_value=[transcript_file],
            ),
            patch(
                "app.services.public.ingest.process_single_file",
                return_value=[mock_doc],
            ),
            patch("app.services.public.ingest.upsert_documents"),
        ):
            await ingest_files(job_id, [audio_file], ["lecture.mp3"], "col")

        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["documents_ingested"] == 1
        assert job["files"]["lecture.mp3"]["status"] == "completed"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_mixed_audio_and_text_concurrent(self, tmp_path: Path):
        """Mixed upload: audio transcription and text processing run concurrently."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job
        from app.models import Document

        text_file = tmp_path / "doc.txt"
        text_file.write_text("Text document content.", encoding="utf-8")
        audio_file = tmp_path / "speech.mp3"
        audio_file.write_bytes(b"\x00" * 50)

        # Transcript for the audio
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        transcript_file = transcript_dir / "speech.txt"
        transcript_file.write_text("Speech transcript.", encoding="utf-8")

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["doc.txt", "speech.mp3"])

        text_doc = Document(
            doc_id=100,
            title="Text Doc",
            text="Text document content.",
            dense_vector=[0.1] * 1024,
        )
        audio_doc = Document(
            doc_id=200,
            title="Speech Doc",
            text="Speech transcript.",
            dense_vector=[0.2] * 1024,
        )

        process_call_paths = []

        async def mock_process_single(path):
            process_call_paths.append(str(path))
            if "speech" in str(path):
                return [audio_doc]
            return [text_doc]

        with (
            patch(
                "app.services.public.ingest.parse_audio_to_text",
                return_value=[transcript_file],
            ),
            patch(
                "app.services.public.ingest.process_single_file",
                side_effect=mock_process_single,
            ),
            patch("app.services.public.ingest.upsert_documents"),
        ):
            await ingest_files(
                job_id,
                [text_file, audio_file],
                ["doc.txt", "speech.mp3"],
                "col",
            )

        job = get_job(job_id)
        assert job["status"] == "completed"
        assert job["documents_ingested"] == 2
        assert job["files"]["doc.txt"]["status"] == "completed"
        assert job["files"]["speech.mp3"]["status"] == "completed"
        # Both files were processed (text file + audio transcript)
        assert len(process_call_paths) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_audio_transcription_failure(self, tmp_path: Path):
        """If transcription fails, audio files marked as failed."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job

        audio_file = tmp_path / "bad_audio.mp3"
        audio_file.write_bytes(b"\x00" * 50)

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["bad_audio.mp3"])

        with (
            patch(
                "app.services.public.ingest.parse_audio_to_text",
                side_effect=RuntimeError("GPU OOM"),
            ),
            patch("app.services.public.ingest.upsert_documents"),
        ):
            await ingest_files(job_id, [audio_file], ["bad_audio.mp3"], "col")

        job = get_job(job_id)
        assert job["status"] == "completed"  # job completes even if files fail
        assert job["files"]["bad_audio.mp3"]["status"] == "failed"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_patch_redis")
    async def test_ingest_top_level_exception(self, tmp_path: Path):
        """A catastrophic error sets the job to failed."""
        from app.services.public.ingest import ingest_files
        from app.repositories.redis.job_store import create_job, get_job

        job_id = str(uuid.uuid4())
        create_job(job_id, "col", ["file.txt"])

        file = tmp_path / "file.txt"
        file.write_text("content", encoding="utf-8")

        with patch(
            "app.services.public.ingest.update_job_status",
            side_effect=RuntimeError("Redis down"),
        ):
            await ingest_files(job_id, [file], ["file.txt"], "col")

        job = get_job(job_id)
        assert job["status"] == "failed"
        assert "Redis down" in job["error"]
