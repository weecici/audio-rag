"""Structured error types and FastAPI exception handlers."""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.logging import logger, request_id_ctx


class ApiError(Exception):
    """Base error raised by endpoint / service code to produce a JSON error body."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class RateLimitError(ApiError):
    """Convenience subclass returned by the rate-limit middleware."""

    def __init__(self, message: str = "rate limit exceeded") -> None:
        super().__init__(
            code="rate_limited",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


# ── helpers ──────────────────────────────────────────────────────────


def _build_error_response(
    *,
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": request_id,
            }
        },
    )


# ── handlers ─────────────────────────────────────────────────────────


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    request_id = request_id_ctx.get()
    logger.warning(
        "api error",
        extra={
            "request_id": request_id,
            "code": exc.code,
            "status_code": exc.status_code,
        },
    )
    return _build_error_response(
        request_id=request_id,
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_ctx.get()
    logger.exception("unhandled exception", extra={"request_id": request_id})
    return _build_error_response(
        request_id=request_id,
        code="internal_error",
        message="internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
