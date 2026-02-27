"""Static API-key authentication middleware."""

from __future__ import annotations

import os

from fastapi import Request
from starlette.responses import JSONResponse

_STATIC_API_KEY: str = os.getenv("API_KEY", "dev-secret-key")

# Paths that bypass authentication.
_PUBLIC_PATHS: frozenset[str] = frozenset({"/docs", "/redoc", "/openapi.json"})
_PUBLIC_PREFIXES: tuple[str, ...] = ("/api/v1/health", "/api/v1/ready")


async def auth_middleware(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS or request.url.path.startswith(
        _PUBLIC_PREFIXES
    ):
        return await call_next(request)

    token = _extract_token(request)

    if not token or token != _STATIC_API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": (
                        "Invalid or missing API key. Provide "
                        "'Authorization: Bearer <key>' or 'X-API-Key: <key>'."
                    ),
                }
            },
        )

    return await call_next(request)


def _extract_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ")

    return request.headers.get("X-API-Key")
