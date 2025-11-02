import inngest
from fastapi import status
from src import schemas
from src.services.internal import dense_encode, sparse_encode, rerank
from src.repo.qdrant import dense_search, sparse_search, hybrid_search
from src.repo.local import index_retrieve
from src.services.internal import fuse_results


def retrieve_documents(ctx: inngest.Context) -> schemas.RetrievalResponse:
    try:
        request = schemas.RetrievalQuery.model_validate(ctx.event.data)

        if not request.queries:
            raise ValueError("No query text provided in event data.")

        ctx.logger.info(
            f"Starting document retrieval process for the {len(request.queries)} input queries."
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
            ctx.logger.info(
                f"Generated {len(dense_query_embeddings)} query embeddings with each embedding's length is: {len(dense_query_embeddings[0])}"
            )

        # Generate sparse embeddings for the queries
        if (
            request.mode in ["sparse", "hybrid"]
            and request.sparse_process_method == "sparse_embedding"
        ):
            sparse_query_embeddings = sparse_encode(
                texts=request.queries, text_type="query"
            )
            if len(sparse_query_embeddings) != len(request.queries):
                raise ValueError(
                    f"Sparse query embeddings generation failed or returned incorrect count: {len(sparse_query_embeddings)}"
                )
            ctx.logger.info(
                f"Generated {len(sparse_query_embeddings)} sparse query embeddings"
            )

        # Retrieve documents based on the specified mode
        ctx.logger.info(
            f"Performing '{request.mode}' retrieval from collection '{request.collection_name}'."
        )
        results: list[list[dict]] | None = None
        if request.mode == "dense":
            results = dense_search(
                query_embeddings=dense_query_embeddings,
                collection_name=request.collection_name,
                top_k=request.top_k,
            )
        elif request.mode == "sparse":
            if request.sparse_process_method == "sparse_embedding":
                results = sparse_search(
                    query_embeddings=sparse_query_embeddings,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                )
            else:
                results = index_retrieve(
                    query_texts=request.queries,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                )
        elif request.mode == "hybrid":
            if request.sparse_process_method == "sparse_embedding":
                results = hybrid_search(
                    dense_query_embeddings=dense_query_embeddings,
                    sparse_query_embeddings=sparse_query_embeddings,
                    collection_name=request.collection_name,
                    top_k=request.top_k,
                    overfetch_mul=request.overfetch_mul,
                )
            else:
                overfetch_amount = int(request.top_k * request.overfetch_mul)

                dense_results = dense_search(
                    query_embeddings=dense_query_embeddings,
                    collection_name=request.collection_name,
                    top_k=overfetch_amount,
                )
                sparse_results = index_retrieve(
                    query_texts=request.queries,
                    collection_name=request.collection_name,
                    top_k=overfetch_amount,
                )
                results = [
                    fuse_results(results1=dense_result, results2=sparse_result)
                    for dense_result, sparse_result in zip(
                        dense_results, sparse_results
                    )
                ]
        else:
            raise ValueError(f"Invalid retrieval mode: {request.mode}")

        if results is None or len(results) != len(request.queries):
            raise ValueError(
                f"Retrieval failed or returned incorrect number of results: {len(results)}"
            )
        ctx.logger.info(
            f"Retrieved top {request.top_k} similar documents for each of the {len(request.queries)} queries from collection '{request.collection_name}'."
        )

        # Rerank results
        results = rerank(
            queries=request.queries,
            candidates=results,
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
