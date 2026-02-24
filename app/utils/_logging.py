import logging


class NoInngestSyncFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage().lower()
        if "syncing app" in msg or "out-of-band sync" in msg:
            return False
        # optionally drop messages coming from the inngest module
        if record.name.startswith("inngest"):
            return False
        return True


logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)


for h in list(logger.handlers):
    h.addFilter(NoInngestSyncFilter())
