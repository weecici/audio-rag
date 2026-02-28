"""Public service: orchestrate file ingestion with job tracking via Redis."""

from pathlib import Path

from app.core.logging import logger
from app.repositories.milvus import upsert_documents
from app.repositories.redis import (
    update_job_status,
    update_file_status,
    set_job_error,
    set_job_result,
)
from app.services.internal import process_files


async def ingest_files(
    job_id: str,
    file_paths: list[str],
    filenames: list[str],
    collection_name: str,
) -> None:
    """Process uploaded files and ingest them into the vector store.

    This function is designed to be called from a background task.
    It updates Redis job state throughout the pipeline so that clients
    can poll ``GET /jobs/{job_id}`` for progress.

    Args:
        job_id: The unique job identifier (already created in Redis).
        file_paths: On-disk paths to the saved uploads.
        filenames: Original filenames (same order as *file_paths*).
        collection_name: Target Milvus collection.
    """
    try:
        update_job_status(job_id, "processing")

        total_docs_ingested = 0

        for fpath, fname in zip(file_paths, filenames):
            update_file_status(job_id, fname, "processing")
            try:
                docs = await process_files([Path(fpath)])

                if docs:
                    upsert_documents(docs, collection_name)
                    total_docs_ingested += len(docs)

                update_file_status(job_id, fname, "completed", chunks=len(docs))
                logger.info(
                    f"[job={job_id}] File '{fname}' ingested: {len(docs)} chunks"
                )

            except Exception as file_exc:
                logger.error(
                    f"[job={job_id}] Failed to process file '{fname}': {file_exc}"
                )
                update_file_status(job_id, fname, "failed", error=str(file_exc))

        set_job_result(job_id, documents_ingested=total_docs_ingested)
        logger.info(
            f"[job={job_id}] Ingestion complete: "
            f"{total_docs_ingested} chunks into '{collection_name}'"
        )

    except Exception as exc:
        logger.exception(f"[job={job_id}] Ingestion job failed: {exc}")
        set_job_error(job_id, str(exc))
