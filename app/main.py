"""FastAPI application factory and wiring."""

from fastapi import FastAPI

from app.api import router
from app.middleware import (
    ApiError,
    api_error_handler,
    auth_middleware,
    rate_limit_middleware,
    request_context_middleware,
    unhandled_error_handler,
)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Audio RAG Service",
        version="1.0.0",
        description="Retrieval-Augmented Generation API for audio and document sources.",
    )

    # Middleware executes in reverse registration order.
    # request_context first (outermost), then rate-limit, then auth (innermost).
    # application.middleware("http")(auth_middleware)
    # application.middleware("http")(rate_limit_middleware)
    # application.middleware("http")(request_context_middleware)

    # Exception handlers
    application.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    application.add_exception_handler(Exception, unhandled_error_handler)  # type: ignore[arg-type]

    # Routers
    application.include_router(router)

    return application


app = create_app()
