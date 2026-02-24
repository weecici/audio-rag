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

# postgres
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pg")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cs419_db")

# local storage
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "./.storage")
AUDIO_STORAGE_PATH = os.path.join(LOCAL_STORAGE_PATH, "audios")
TRANSCRIPT_STORAGE_PATH = os.path.join(LOCAL_STORAGE_PATH, "transcripts")
CHUNKED_TRANSCRIPT_STORAGE_PATH = os.path.join(
    LOCAL_STORAGE_PATH, "chunked_transcripts"
)

# speech to text
SPEECH2TEXT_MODEL = os.getenv("SPEECH2TEXT_MODEL", "small")
