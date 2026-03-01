"""Public service: orchestrate file ingestion with job tracking via Redis.

Concurrency design:
- Files are split into **text** files and **audio** files.
- Text files are processed concurrently (each file independently via
  ``process_single_file`` inside ``asyncio.gather``).
- Audio files are transcribed to text (GPU-bound, serialised internally)
  via ``parse_audio_to_text`` in a thread-pool executor.
- **Transcription and text-file processing run concurrently** -- the event
  loop kicks off both branches at the same time with ``asyncio.gather``.
- Once transcription finishes, the resulting transcript .txt files are
  processed like any other text file (also concurrently).
- Per-file Redis status is updated throughout so clients can poll progress.
"""

import asyncio
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.models import Document
from app.repositories.milvus import upsert_documents
from app.repositories.redis import (
    update_job_status,
    update_file_status,
    set_job_error,
    set_job_result,
)
from app.services.internal import process_single_file, parse_audio_to_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUDIO_EXTS = set(settings.ALLOWED_AUDIO_EXTS)  # e.g. {".mp3", ".wav", ...}


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


# ---------------------------------------------------------------------------
# Per-file processing with Redis status tracking
# ---------------------------------------------------------------------------


async def _process_text_file(
    job_id: str,
    fpath: Path,
    fname: str,
    collection_name: str,
) -> int:
    """Process a single text file and upsert results. Returns chunk count."""
    update_file_status(job_id, fname, "processing")
    try:
        docs = await process_single_file(fpath)
        if docs:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, upsert_documents, docs, collection_name)
        chunks = len(docs)
        update_file_status(job_id, fname, "completed", chunks=chunks)
        logger.info(f"[job={job_id}] File '{fname}' ingested: {chunks} chunks")
        return chunks
    except Exception as exc:
        logger.error(f"[job={job_id}] Failed to process file '{fname}': {exc}")
        update_file_status(job_id, fname, "failed", error=str(exc))
        return 0


async def _transcribe_and_process_audio(
    job_id: str,
    audio_paths: list[Path],
    audio_names: list[str],
    collection_name: str,
) -> int:
    """Transcribe audio files, then process the transcripts as text.

    Returns total chunk count across all audio files.
    """
    if not audio_paths:
        return 0

    # Mark all audio files as "transcribing"
    for fname in audio_names:
        update_file_status(job_id, fname, "transcribing")

    # Run transcription in a thread (GPU-bound, blocks)
    loop = asyncio.get_running_loop()
    try:
        transcript_paths: list[Path] = await loop.run_in_executor(
            None, parse_audio_to_text, audio_paths, None
        )
    except Exception as exc:
        logger.error(f"[job={job_id}] Audio transcription failed: {exc}")
        for fname in audio_names:
            update_file_status(job_id, fname, "failed", error=str(exc))
        return 0

    # Build a mapping: original audio name -> transcript path
    # parse_audio_to_text returns paths in order, skipping failures
    # We match by stem name since transcript is <audio_stem>.txt
    transcript_by_stem: dict[str, Path] = {}
    for tp in transcript_paths:
        transcript_by_stem[tp.stem] = tp

    total_chunks = 0

    # Process each transcript (concurrently, like text files)
    async def _process_one_transcript(audio_path: Path, audio_name: str) -> int:
        stem = audio_path.stem
        tp = transcript_by_stem.get(stem)
        if tp is None:
            update_file_status(
                job_id, audio_name, "failed", error="Transcription produced no output"
            )
            return 0
        return await _process_text_file(job_id, tp, audio_name, collection_name)

    results = await asyncio.gather(
        *[_process_one_transcript(ap, an) for ap, an in zip(audio_paths, audio_names)],
        return_exceptions=True,
    )

    for an, result in zip(audio_names, results):
        if isinstance(result, Exception):
            logger.error(
                f"[job={job_id}] Failed transcript processing '{an}': {result}"
            )
            update_file_status(job_id, an, "failed", error=str(result))
        else:
            total_chunks += result

    return total_chunks


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


async def ingest_files(
    job_id: str,
    file_paths: list[Path],
    filenames: list[str],
    collection_name: str,
) -> None:
    """Process uploaded files and ingest them into the vector store.

    This function is designed to be called from a background task.
    It updates Redis job state throughout the pipeline so that clients
    can poll ``GET /jobs/{job_id}`` for progress.

    Concurrency model:
    - Text files are all processed concurrently (fan-out via gather).
    - Audio files are transcribed (GPU-serialised), then the transcripts
      are processed concurrently.
    - Text processing and audio transcription start at the **same time**.

    Args:
        job_id: The unique job identifier (already created in Redis).
        file_paths: On-disk paths to the saved uploads.
        filenames: Original filenames (same order as *file_paths*).
        collection_name: Target Milvus collection.
    """
    try:
        update_job_status(job_id, "processing")

        # --- Split into text vs audio ---
        text_paths: list[Path] = []
        text_names: list[str] = []
        audio_paths: list[Path] = []
        audio_names: list[str] = []

        for fpath, fname in zip(file_paths, filenames):
            if _is_audio(fpath):
                audio_paths.append(fpath)
                audio_names.append(fname)
            else:
                text_paths.append(fpath)
                text_names.append(fname)

        logger.info(
            f"[job={job_id}] Ingestion started: "
            f"{len(text_paths)} text file(s), {len(audio_paths)} audio file(s)"
        )

        # --- Run both branches concurrently ---
        text_coros = [
            _process_text_file(job_id, fp, fn, collection_name)
            for fp, fn in zip(text_paths, text_names)
        ]

        audio_coro = _transcribe_and_process_audio(
            job_id, audio_paths, audio_names, collection_name
        )

        # Gather: [text_result_0, text_result_1, ..., audio_total_chunks]
        all_results = await asyncio.gather(
            *text_coros, audio_coro, return_exceptions=True
        )

        total_docs_ingested = 0
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"[job={job_id}] Task {i} raised: {result}")
            else:
                total_docs_ingested += result

        set_job_result(job_id, documents_ingested=total_docs_ingested)
        logger.info(
            f"[job={job_id}] Ingestion complete: "
            f"{total_docs_ingested} chunks into '{collection_name}'"
        )

    except Exception as exc:
        logger.exception(f"[job={job_id}] Ingestion job failed: {exc}")
        set_job_error(job_id, str(exc))
