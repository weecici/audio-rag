import torch
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.core import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model '{config.EMBEDDING_MODEL}' on device: {device}")
    model = SentenceTransformer(
        model_name_or_path=config.EMBEDDING_MODEL_PATH,
        device=device,
    )
    return model


def dense_encode(
    texts: list[str], prefix: str, dim: int = config.EMBEDDING_DIM, batch_size: int = 8
) -> list[list[float]]:
    model = get_embedding_model()

    prefixed_texts = [f"{prefix}: {t}" for t in texts]

    embeddings = model.encode(
        sentences=prefixed_texts,
        truncate_dim=dim,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    return embeddings.tolist()
