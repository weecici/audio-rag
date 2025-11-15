import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="retrieve-documents",
    trigger=inngest.TriggerEvent(event="rag/retrieve-documents"),
    retries=0,
)
async def retrieve_documents(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.RetrievalRequest.model_validate(ctx.event.data)
    return (await public_svcs.retrieve_documents(request)).model_dump()


@router.post(
    "/retrieve",
    response_model=schemas.RetrievalResponse,
    summary="Retrieve relevant documents",
    description="Retrieve documents based on the provided queries.",
)
async def retrieve(request: schemas.RetrievalRequest) -> schemas.RetrievalResponse:
    return await public_svcs.retrieve_documents(request)
