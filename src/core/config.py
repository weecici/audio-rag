import os
import random
from dotenv import load_dotenv

load_dotenv()

rng = random.Random(42)

# llm provider api key
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

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

# postgres
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pg")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cs419_db")

# local storage
LOCAL_STORAGE_PATH = os.getenv("LOCAL_STORAGE_PATH", "./.storage")
