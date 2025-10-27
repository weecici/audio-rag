import inngest
from fastapi import status
from src import schemas
from src.services.internal import process_documents, dense_encode
from src.repo.qdrant import upsert_nodes


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

        embeddings = dense_encode(
            texts=[node.text for node in nodes], prefix="document"
        )

        if len(embeddings) != len(nodes):
            raise ValueError(
                f"Embeddings generation failed or returned incorrect count: {len(embeddings)}"
            )

        ctx.logger.info(
            f"Generated {len(nodes)} embeddings with each embedding's length is: {len(embeddings[0])}"
        )

        # Attach embeddings to nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        # Upsert nodes
        upsert_nodes(nodes=nodes, collection_name=request.collection_name)
        ctx.logger.info(
            f"Upserted {len(nodes)} nodes from {len(request.file_paths)} documents to Qdrant collection '{request.collection_name}'."
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
