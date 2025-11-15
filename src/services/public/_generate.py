from fastapi import status
from src import schemas
from src.services.internal import augment_prompts, generate
from src.utils import logger
from ._retrieve import retrieve_documents


async def generate_responses(
    request: schemas.GenerationRequest,
) -> schemas.GenerationResponse:
    try:
        logger.info("Starting response generation process...")
        retrieved_docs = retrieve_documents(request).results

        prompts = augment_prompts(
            queries=request.queries,
            contexts=retrieved_docs,
        )

        responses = generate(
            prompts=prompts,
            model=request.model_name,
        )

        return schemas.GenerationResponse(
            status=status.HTTP_200_OK,
            responses=responses,
        )
    except Exception as e:
        logger.error(f"Error during generation process: {str(e)}")
        return schemas.GenerationResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            responses=[],
        )
