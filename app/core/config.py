import random
from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # llm provider api keys
    CEREBRAS_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # chunking config
    MAX_TOKENS: int = 1024
    OVERLAP_TOKENS: int = 200

    # title generation (Cerebras)
    TITLE_MODEL: str = "gpt-oss-120b"
    TITLE_MAX_TOKENS: int = 50

    # embedding
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 768  # must match the actual dimension of the embedding model

    # utils
    FUSION_METHOD: Literal["weighted", "dbsf", "rrf"] = "weighted"
    RRF_K: int = 2
    FUSION_ALPHA: float = 0.7

    # milvus connection
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_DB_NAME: str = "default"
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_TOKEN: str = ""
    MILVUS_TIMEOUT_SEC: float = 30.0

    # milvus index / search
    MILVUS_METRIC_TYPE: str = "COSINE"
    MILVUS_INDEX_TYPE: str = "HNSW"
    MILVUS_HNSW_M: int = 16
    MILVUS_HNSW_EF_CONSTRUCTION: int = 200
    MILVUS_HNSW_EF: int = 64

    # milvus BM25
    MILVUS_BM25_K1: float = 1.5
    MILVUS_BM25_B: float = 0.75

    # milvus collection
    MILVUS_INSERT_BATCH_SIZE: int = 512
    MILVUS_ENABLE_FULLTEXT: bool = False
    MILVUS_TEXT_MAX_LENGTH: int = 9000

    # redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "dev-redis-password"
    REDIS_DB: int = 0
    REDIS_JOB_TTL_SEC: int = 86400  # 24 hours

    # local storage
    LOCAL_STORAGE_PATH: str = "./.storage"

    @field_validator("FUSION_ALPHA")
    @classmethod
    def fusion_alpha_must_be_between_0_and_1(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("FUSION_ALPHA must be between 0 and 1.")
        return v

    @field_validator("RRF_K")
    @classmethod
    def rrf_k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RRF_K must be a positive integer.")
        return v

    # Derived storage paths (computed, not read from env)
    @property
    def AUDIO_STORAGE_PATH(self) -> str:
        return str(Path(self.LOCAL_STORAGE_PATH) / "audios")

    @property
    def TRANSCRIPT_STORAGE_PATH(self) -> str:
        return str(Path(self.LOCAL_STORAGE_PATH) / "transcripts")

    @property
    def CHUNKED_TRANSCRIPT_STORAGE_PATH(self) -> str:
        return str(Path(self.LOCAL_STORAGE_PATH) / "chunked_transcripts")


settings = Settings()

# Shared RNG (seeded for reproducibility)
rng = random.Random(42)
