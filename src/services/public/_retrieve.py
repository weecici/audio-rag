from fastapi import status
from src import schemas
from src.utils import logger
from src.services.internal import dense_encode, rerank
from src.repo.postgres import (
    dense_search,
    sparse_search,
    hybrid_search,
)


async def retrieve_documents(
    request: schemas.RetrievalRequest,
) -> schemas.RetrievalResponse:
    try:
        if not request.queries:
            raise ValueError("No query text provided in event data.")

        logger.info(
            f"Starting document retrieval process for the {len(request.queries)} input queries..."
        )

        # Generate dense embeddings for the queries
        if request.mode in ["dense", "hybrid"]:
            dense_query_embeddings = dense_encode(
                texts=request.queries, text_type="query"
            )
            if len(dense_query_embeddings) != len(request.queries):
                raise ValueError(
                    f"Query embeddings generation failed or returned incorrect count: {len(dense_query_embeddings)}"
                )
            logger.info(
                f"Generated {len(dense_query_embeddings)} dense query embeddings with each embedding's length is: {len(dense_query_embeddings[0])}"
            )

        # Retrieve documents based on the specified mode
        logger.info(
            f"Performing '{request.mode}' retrieval from collection '{request.collection_name}'."
        )
        if request.mode == "dense":
            results = dense_search(
                query_embeddings=dense_query_embeddings,
                collection_name=request.collection_name,
                top_k=request.top_k,
            )
        elif request.mode == "sparse":
            results = sparse_search(
                query_texts=request.queries,
                collection_name=request.collection_name,
                top_k=request.top_k,
            )
        elif request.mode == "hybrid":
            results = hybrid_search(
                dense_query_embeddings=dense_query_embeddings,
                query_texts=request.queries,
                collection_name=request.collection_name,
                top_k=request.top_k,
                overfetch_mul=request.overfetch_mul,
            )
        else:
            raise ValueError(f"Invalid retrieval mode: {request.mode}")

        if results is None or len(results) != len(request.queries):
            raise ValueError(
                f"Retrieval failed or returned incorrect number of results: {len(results)}"
            )
        logger.info(
            f"Retrieved top {request.top_k} similar documents for each of the {len(request.queries)} queries from collection '{request.collection_name}'."
        )

        # Rerank results
        if request.rerank_enabled:
            logger.info("Starting reranking of retrieved results.")
            results = rerank(queries=request.queries, candidates=results)

        return schemas.RetrievalResponse(
            status=status.HTTP_200_OK,
            results=results,
        )

    except Exception as e:
        logger.error(f"Error in retrieve_documents: {e}")
        return schemas.RetrievalResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, results=[]
        )
