import app.service.public as public_svc
from fastapi import APIRouter, status
from app import schema
from app.api.middleware import ApiError

router = APIRouter()


@router.post(
    "/retrieve",
    response_model=schema.RetrievalResponse,
    summary="Retrieve relevant documents",
    description="Retrieve documents based on the provided queries.",
)
async def retrieve(request: schema.RetrievalRequest) -> schema.RetrievalResponse:
    try:
        return await public_svc.retrieve_documents(request)
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="retrieve_failed",
            message="retrieval failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
