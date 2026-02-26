import os
import random
from dotenv import load_dotenv

load_dotenv()

rng = random.Random(42)

# llm provider api key
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# chunking config
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "200"))

# dense model
DENSE_MODEL = os.getenv("DENSE_MODEL", "embeddinggemma-300m")
DENSE_MODEL_PATH = os.getenv("DENSE_MODEL_PATH", "google/embeddinggemma-300m")
DENSE_DIM = int(os.getenv("DENSE_DIM", 768))

# reranking model
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "bge-reranker-v2-m3")
RERANKING_MODEL_PATH = os.getenv("RERANKING_MODEL_PATH", "BAAI/bge-reranker-v2-m3")

# utils
WORD_PROCESS_METHOD = os.getenv("WORD_PROCESS_METHOD", "stem")
FUSION_METHOD = os.getenv("FUSION_METHOD", "dbsf")
RRF_K = int(os.getenv("RRF_K", 2))
if not RRF_K > 0:
    raise ValueError("RRF_K must be a positive integer.")
FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", 0.7))


# milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_TIMEOUT_SEC = float(os.getenv("MILVUS_TIMEOUT_SEC", "30"))

MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
MILVUS_HNSW_M = int(os.getenv("MILVUS_HNSW_M", "16"))
MILVUS_HNSW_EF_CONSTRUCTION = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "200"))
MILVUS_HNSW_EF = int(os.getenv("MILVUS_HNSW_EF", "64"))

MILVUS_BM25_K1 = float(os.getenv("MILVUS_BM25_K1", "1.5"))
MILVUS_BM25_B = float(os.getenv("MILVUS_BM25_B", "0.75"))

MILVUS_INSERT_BATCH_SIZE = int(os.getenv("MILVUS_INSERT_BATCH_SIZE", "512"))
MILVUS_ENABLE_FULLTEXT = os.getenv("MILVUS_ENABLE_FULLTEXT", "false").lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
MILVUS_TEXT_MAX_LENGTH = int(os.getenv("MILVUS_TEXT_MAX_LENGTH", "9000"))

# local storage
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "./.storage")
AUDIO_STORAGE_PATH = os.path.join(LOCAL_STORAGE_PATH, "audios")
TRANSCRIPT_STORAGE_PATH = os.path.join(LOCAL_STORAGE_PATH, "transcripts")
CHUNKED_TRANSCRIPT_STORAGE_PATH = os.path.join(
    LOCAL_STORAGE_PATH, "chunked_transcripts"
)

# speech to text
SPEECH2TEXT_MODEL = os.getenv("SPEECH2TEXT_MODEL", "small")

# chunking
CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "gemini-flash-latest")
