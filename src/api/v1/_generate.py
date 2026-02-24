import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas

router = APIRouter()


@router.post(
    "/generate",
    response_model=schemas.GenerationResponse,
    summary="Generate responses from documents",
    description="Generate responses based on the provided queries and retrieved documents.",
)
async def generate(request: schemas.GenerationRequest) -> schemas.GenerationResponse:
    return await public_svcs.generate_responses(request)
