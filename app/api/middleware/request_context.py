import time
import uuid
from fastapi import Request
from app.utils.logging import logger, request_id_ctx


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code if response else 500,
                "duration_ms": round(elapsed_ms, 2),
            },
        )
        request_id_ctx.reset(token)

    response.headers["x-request-id"] = request_id
    return response
