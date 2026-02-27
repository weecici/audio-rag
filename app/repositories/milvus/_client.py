from functools import lru_cache
from pymilvus import MilvusClient

from app.core.config import settings


def _build_client_kwargs() -> dict:
    kwargs: dict = {
        "uri": settings.MILVUS_URI,
        "db_name": settings.MILVUS_DB_NAME,
        "timeout": settings.MILVUS_TIMEOUT_SEC,
    }

    # Prefer token auth when provided.
    if settings.MILVUS_TOKEN:
        kwargs["token"] = settings.MILVUS_TOKEN
        return kwargs

    # Fallback to user/password if provided.
    if settings.MILVUS_USER or settings.MILVUS_PASSWORD:
        kwargs["user"] = settings.MILVUS_USER
        kwargs["password"] = settings.MILVUS_PASSWORD

    return kwargs


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    return MilvusClient(**_build_client_kwargs())
