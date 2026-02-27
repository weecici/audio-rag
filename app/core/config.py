import random
from pathlib import Path
from typing import Literal

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
    CEREBRAS_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    # chunking config
    MAX_TOKENS: int = 1024
    OVERLAP_TOKENS: int = 200

    # dense model
    DENSE_MODEL: str = "embeddinggemma-300m"
    DENSE_MODEL_PATH: str = "google/embeddinggemma-300m"
    DENSE_DIM: int = 768

    # reranking model
    RERANKING_MODEL: str = "bge-reranker-v2-m3"
    RERANKING_MODEL_PATH: str = "BAAI/bge-reranker-v2-m3"

    # utils
    WORD_PROCESS_METHOD: str = "stem"
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

    # local storage
    LOCAL_STORAGE_PATH: str = "./.storage"

    # speech to text
    SPEECH2TEXT_MODEL: str = "small"

    # chunking
    CHUNKING_MODEL: str = "gemini-flash-latest"

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
