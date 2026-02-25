import contextvars
import logging

request_id_ctx = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

for handler in list(logger.handlers):
    handler.addFilter(RequestIdFilter())
