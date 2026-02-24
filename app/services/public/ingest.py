from fastapi import status
from app import schemas
from app.utils import logger, download_audio
from app.repo.postgres import upsert_data
from app.services.internal import (
    process_documents,
    dense_encode,
    build_inverted_index,
    transcribe_audio,
)


async def ingest_documents(
    request: schemas.DocumentIngestionRequest,
) -> schemas.IngestionResponse:
    try:
        if not request.file_paths and not request.file_dir:
            raise ValueError("No file paths or directory provided in event data.")

        logger.info(
            f"Starting documents ingestion process to the collection '{request.collection_name}'..."
        )

        nodes = await process_documents(
            file_paths=request.file_paths, file_dir=request.file_dir
        )

        if len(nodes) == 0:
            raise ValueError("No nodes were created from the provided documents.")

        logger.info(f"Processed {len(nodes)} chunks with UUIDs and metadata.")

        texts = [node.text for node in nodes]
        titles = [node.metadata.get("title", "none") for node in nodes]
        full_texts = [f"{title}\n\n{text}" for title, text in zip(titles, texts)]

        # Create dense embeddings for the docs
        dense_embeddings = dense_encode(
            text_type="document",
            texts=texts,
            titles=titles,
        )

        logger.info(
            f"Generated {len(dense_embeddings)} dense embeddings with each embedding's size is: {len(dense_embeddings[0])}"
        )

        # Build inverted index (postings list) for the docs
        postings_list, doc_lens = build_inverted_index(
            doc_ids=[node.id_ for node in nodes],
            texts=full_texts,
        )

        logger.info(f"Built inverted index with size: {len(postings_list)}")

        # Upsert data (embeddings + postings list) into DB
        upsert_data(
            nodes=nodes,
            dense_embeddings=dense_embeddings,
            postings_list=postings_list,
            doc_lens=doc_lens,
            collection_name=request.collection_name,
        )

        logger.info(
            f"Completed ingestion process of {len(nodes)} documents for collection '{request.collection_name}'."
        )

        return schemas.IngestionResponse(
            status=status.HTTP_201_CREATED,
            message=f"Successfully ingested {len(nodes)} nodes into collection '{request.collection_name}'.",
        )

    except Exception as e:
        logger.error(f"Error while ingesting documents: {e}")

        return schemas.IngestionResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)
        )


async def ingest_audios(
    request: schemas.AudioIngestionRequest,
) -> schemas.IngestionResponse:
    try:
        if not request.file_paths and not request.urls:
            raise ValueError("No audio file paths or URLs provided in request data.")

        logger.info(
            f"Starting audio ingestion process to the collection '{request.collection_name}'..."
        )

        download_filepaths = download_audio(urls=request.urls)
        total_filepaths = request.file_paths + download_filepaths

        transcript_paths = transcribe_audio(audio_paths=total_filepaths)
        if len(transcript_paths) == 0:
            raise ValueError("No transcripts were generated from the provided audios.")

        doc_ingest_request = schemas.DocumentIngestionRequest(
            collection_name=request.collection_name, file_paths=transcript_paths
        )

        transcript_ingest_response = await ingest_documents(request=doc_ingest_request)

        if transcript_ingest_response.status != status.HTTP_201_CREATED:
            raise ValueError(
                f"Document ingestion failed during audio ingestion: {transcript_ingest_response.message}"
            )

        return schemas.IngestionResponse(
            status=status.HTTP_200_OK,
            message=f"Ingested {len(total_filepaths)} audio files into collection '{request.collection_name}'.",
        )
    except Exception as e:
        logger.error(f"Error while ingesting audios: {e}")

        return schemas.IngestionResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)
        )
