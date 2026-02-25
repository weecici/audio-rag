import os
from fastapi import Request
from starlette.responses import JSONResponse
from .errors import ApiError

# Can be loaded from config
STATIC_API_KEY = os.getenv("API_KEY", "dev-secret-key")


async def auth_middleware(request: Request, call_next):
    # Skip auth for docs, openapi, or health endpoints
    if request.url.path in [
        "/docs",
        "/redoc",
        "/openapi.json",
    ] or request.url.path.startswith("/api/v1/health"):
        return await call_next(request)

    # Check for Bearer token or x-api-key header
    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")

    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif api_key_header:
        token = api_key_header

    if not token or token != STATIC_API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "Invalid or missing API key. Please provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
                }
            },
        )

    response = await call_next(request)
    return response
