import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas

router = APIRouter()


@router.post(
    "/retrieve",
    response_model=schemas.RetrievalResponse,
    summary="Retrieve relevant documents",
    description="Retrieve documents based on the provided queries.",
)
async def retrieve(request: schemas.RetrievalRequest) -> schemas.RetrievalResponse:
    return await public_svcs.retrieve_documents(request)
