from functools import lru_cache
from pymilvus import MilvusClient

from app.core import config


def _build_client_kwargs() -> dict:
    kwargs: dict = {
        "uri": config.MILVUS_URI,
        "db_name": config.MILVUS_DB_NAME,
        "timeout": config.MILVUS_TIMEOUT_SEC,
    }

    # Prefer token auth when provided.
    if config.MILVUS_TOKEN:
        kwargs["token"] = config.MILVUS_TOKEN
        return kwargs

    # Fallback to user/password if provided.
    if config.MILVUS_USER or config.MILVUS_PASSWORD:
        kwargs["user"] = config.MILVUS_USER
        kwargs["password"] = config.MILVUS_PASSWORD

    return kwargs


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    return MilvusClient(**_build_client_kwargs())
