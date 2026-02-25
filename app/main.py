from app.core import app
from app.api.v1 import api_v1
from app.api.middleware import (
    api_error_handler,
    rate_limit_middleware,
    request_context_middleware,
    auth_middleware,
    unhandled_error_handler,
    ApiError,
)

# Middleware is executed in reverse order of adding
app.middleware("http")(auth_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(request_context_middleware)

app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(router=api_v1, prefix="/api/v1")
