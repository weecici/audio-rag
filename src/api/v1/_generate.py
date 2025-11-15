import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="generate-responses",
    trigger=inngest.TriggerEvent(event="rag/generate-responses"),
    retries=0,
)
async def generate_responses(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.GenerationRequest.model_validate(ctx.event.data)
    return (await public_svcs.generate_responses(request)).model_dump()


@router.post(
    "/generate",
    response_model=schemas.GenerationResponse,
    summary="Generate responses from documents",
    description="Generate responses based on the provided queries and retrieved documents.",
)
async def generate(request: schemas.GenerationRequest) -> schemas.GenerationResponse:
    return await public_svcs.generate_responses(request)
