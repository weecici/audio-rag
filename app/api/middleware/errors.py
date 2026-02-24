from fastapi import Request
from fastapi.responses import JSONResponse
from app.utils.logging import logger, request_id_ctx


class ApiError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class RateLimitError(ApiError):
    def __init__(self, message: str = "rate limit exceeded") -> None:
        super().__init__(code="rate_limited", message=message, status_code=429)


def _error_response(
    *,
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
) -> JSONResponse:
    payload = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


async def api_error_handler(request: Request, exc: Exception):
    if not isinstance(exc, ApiError):
        return await unhandled_error_handler(request, exc)
    request_id = request_id_ctx.get()
    logger.warning(
        "api error",
        extra={
            "request_id": request_id,
            "code": exc.code,
            "status_code": exc.status_code,
        },
    )
    return _error_response(
        request_id=request_id,
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = request_id_ctx.get()
    logger.exception("unhandled exception", extra={"request_id": request_id})
    return _error_response(
        request_id=request_id,
        code="internal_error",
        message="internal server error",
        status_code=500,
    )
