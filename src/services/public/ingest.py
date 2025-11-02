import inngest
from fastapi import status
from src import schemas
from src.services.internal import (
    process_documents,
    dense_encode,
    sparse_encode,
    build_inverted_index,
)
from src.repo.qdrant import upsert_data
from src.repo.local import store_index


def ingest_documents(ctx: inngest.Context) -> schemas.IngestionResponse:
    try:
        request = schemas.IngestRequest.model_validate(ctx.event.data)
        if not request.file_paths and not request.file_dir:
            raise ValueError("No file paths or directory provided in event data.")

        ctx.logger.info(
            f"Starting documents ingestion process to the collection '{request.collection_name}'."
        )

        nodes = process_documents(
            file_paths=request.file_paths, file_dir=request.file_dir
        )

        if len(nodes) == 0:
            raise ValueError("No nodes were created from the provided documents.")

        ctx.logger.info(f"Processed {len(nodes)} chunks with UUIDs and metadata.")

        # Create dense embeddings for the docs
        dense_embeddings = dense_encode(
            texts=[node.text for node in nodes],
            titles=[node.metadata.get("title", "none") for node in nodes],
            text_type="document",
        )

        ctx.logger.info(
            f"Generated {len(dense_embeddings)} dense embeddings with each embedding's size is: {len(dense_embeddings[0])}"
        )

        if request.sparse_process_method == "sparse_embedding":
            # Create sparse embeddings for the docs
            sparse_embeddings = sparse_encode(
                text_type="document",
                texts=[node.text for node in nodes],
            )

            ctx.logger.info(f"Generated {len(sparse_embeddings)} sparse embeddings.")

            upsert_data(
                nodes=nodes,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                collection_name=request.collection_name,
            )
        else:
            # Build inverted index for the docs
            indexed_docs = build_inverted_index(
                texts=[node.text for node in nodes],
                uuids=[node.id_ for node in nodes],
                metadata=[node.metadata for node in nodes],
            )

            ctx.logger.info(
                f"Built inverted index with vocab size: {len(indexed_docs['vocab'])}"
            )

            store_index(
                collection_name=request.collection_name,
                indexed_docs=indexed_docs,
            )

            upsert_data(
                nodes=nodes,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=None,
                collection_name=request.collection_name,
            )

        ctx.logger.info(
            f"Completed ingestion process of {len(nodes)} documents for collection '{request.collection_name}'."
        )

        return schemas.IngestionResponse(
            status=status.HTTP_201_CREATED,
            message=f"Successfully ingested {len(nodes)} nodes into collection '{request.collection_name}'.",
        )

    except Exception as e:
        ctx.logger.error(f"Error while ingesting documents: {e}")

        return schemas.IngestionResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)
        )
