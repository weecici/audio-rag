import app.services.public as public_svcs
from fastapi import APIRouter, status
from app import schema
from app.api.middleware import ApiError

router = APIRouter()


@router.post(
    "/generate",
    response_model=schema.GenerationResponse,
    summary="Generate responses from documents",
    description="Generate responses based on the provided queries and retrieved documents.",
)
async def generate(request: schema.GenerationRequest) -> schema.GenerationResponse:
    try:
        return await public_svcs.generate_responses(request)
    except ValueError as exc:
        raise ApiError(
            code="invalid_request",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise ApiError(
            code="generate_failed",
            message="generation failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
