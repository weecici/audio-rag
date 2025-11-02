import os
import random
from dotenv import load_dotenv, find_dotenv

load_dotenv()

rng = random.Random(42)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", None)
DENSE_MODEL = os.getenv("DENSE_MODEL", "embeddinggemma-300m")
DENSE_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "google/embeddinggemma-300m")
DENSE_DIM = int(os.getenv("DENSE_DIM", 768))
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "splade-v3")
SPARSE_MODEL_PATH = os.getenv("SPARSE_MODEL_PATH", "naver/splade-v3")
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "bge-reranker-v2-m3")
RERANKING_MODEL_PATH = os.getenv("RERANKING_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
DISK_STORAGE_PATH = os.getenv("DISK_STORAGE_PATH", "./.storage")
WORD_PROCESS_METHOD = os.getenv("WORD_PROCESS_METHOD", "stem")
FUSION_METHOD = os.getenv("FUSION_METHOD", "dbsf")

RRF_K = int(os.getenv("RRF_K", 2))
if not RRF_K > 0:
    raise ValueError("RRF_K must be a positive integer.")
