import time
from fastapi import Request
from starlette.responses import JSONResponse
from app.util.logging import request_id_ctx
from .errors import RateLimitError


class RateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}

    def check(self, key: str, now: float) -> bool:
        bucket = self._buckets.get(key, [])
        cutoff = now - self.window_seconds
        bucket = [ts for ts in bucket if ts >= cutoff]
        allowed = len(bucket) < self.max_requests
        if allowed:
            bucket.append(now)
        self._buckets[key] = bucket
        return allowed


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


async def rate_limit_middleware(request: Request, call_next):
    now = time.time()
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    if not rate_limiter.check(key, now):
        request_id = request_id_ctx.get() or "unknown"
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limited",
                    "message": "Too many requests. Please slow down.",
                    "request_id": request_id,
                }
            },
            headers={"x-request-id": request_id},
        )
    return await call_next(request)
