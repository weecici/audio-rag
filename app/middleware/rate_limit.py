"""In-memory sliding-window rate limiter (per IP + path)."""

from __future__ import annotations

import time

from fastapi import Request, status
from starlette.responses import JSONResponse

from app.core.logging import request_id_ctx


class RateLimiter:
    """Simple sliding-window counter keyed by an arbitrary string."""

    def __init__(self, *, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}

    def is_allowed(self, key: str, now: float | None = None) -> bool:
        now = now or time.monotonic()
        cutoff = now - self.window_seconds
        bucket = [ts for ts in self._buckets.get(key, []) if ts >= cutoff]
        if len(bucket) >= self.max_requests:
            self._buckets[key] = bucket
            return False
        bucket.append(now)
        self._buckets[key] = bucket
        return True

    def reset(self) -> None:
        self._buckets.clear()


rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"

    if not rate_limiter.is_allowed(key):
        request_id = request_id_ctx.get()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
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
