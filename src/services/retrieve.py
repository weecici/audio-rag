import inngest
from src import schemas
from fastapi import status
from . import encode_texts
from src.repo.qdrant import search_batch_similar_nodes


def retrieve_documents(ctx: inngest.Context) -> schemas.RetrievalResponse:
    try:
        request = schemas.RetrievalQuery.model_validate(ctx.event.data)

        if not request.queries:
            raise ValueError("No query text provided in event data.")

        ctx.logger.info(
            f"Starting document retrieval process for the {len(request.queries)} input queries."
        )

        query_embeddings = encode_texts(texts=request.queries, prefix="query")

        if len(query_embeddings) != len(request.queries):
            raise ValueError(
                f"Query embeddings generation failed or returned incorrect count: {len(query_embeddings)}"
            )

        ctx.logger.info(
            f"Generated {len(query_embeddings)} query embeddings with each embedding's length is: {len(query_embeddings[0])}"
        )

        results = search_batch_similar_nodes(
            query_embeddings=query_embeddings,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )

        if len(results) != len(request.queries):
            raise ValueError(
                f"Retrieval failed or returned incorrect number of results: {len(results)}"
            )

        ctx.logger.info(
            f"Retrieved top {request.top_k} similar documents for each of the {len(request.queries)} queries from collection '{request.collection_name}'."
        )

        return schemas.RetrievalResponse(
            status=status.HTTP_200_OK,
            results=results,
        )

    except Exception as e:
        ctx.logger.error(f"Error in retrieve_documents: {e}")
        return schemas.RetrievalResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR, results=[]
        )
