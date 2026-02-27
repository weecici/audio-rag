from fastapi import status

from app import schemas
from app.services.internal import (
    get_augmented_prompts,
    generate,
    get_summarization_prompts,
    parse_summarization_responses,
)
from app.utils import logger
from .retrieve import retrieve_documents


async def generate_responses(
    request: schemas.GenerationRequest,
) -> schemas.GenerationResponse:
    raise NotImplementedError("Generation service is not implemented yet.")
