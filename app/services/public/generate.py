from fastapi import status
from app import schema
from app.services.internal import (
    get_augmented_prompts,
    generate,
    get_summarization_prompts,
    parse_summarization_responses,
)
from app.utils import logger
from .retrieve import retrieve_documents


async def generate_responses(
    request: schema.GenerationRequest,
) -> schema.GenerationResponse:
    try:
        logger.info("Starting response generation process...")
        retrieved_docs = (await retrieve_documents(request)).results

        qa_prompts = get_augmented_prompts(
            queries=request.queries,
            contexts=retrieved_docs,
        )

        qa_responses = generate(
            prompts=qa_prompts,
            model=request.model_name,
        )

        docs = retrieved_docs

        if request.summarization_enabled:
            documents_text = [
                [doc.payload.text for doc in docs] for docs in retrieved_docs
            ]

            sum_prompts = get_summarization_prompts(documents_list=documents_text)

            sum_responses = generate(prompts=sum_prompts, model=request.model_name)

            parsed_summaries_list = parse_summarization_responses(
                responses=sum_responses,
                documents_list=retrieved_docs,
            )
            docs = parsed_summaries_list

        return schema.GenerationResponse(
            status=status.HTTP_200_OK,
            responses=qa_responses,
            summarized_docs_list=docs,
        )
    except Exception as e:
        logger.error(f"Error during generation process: {str(e)}")
        return schema.GenerationResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            responses=[],
            summarized_docs_list=[],
        )
