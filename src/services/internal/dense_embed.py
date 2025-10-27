import torch
import numpy as np
from functools import lru_cache
from typing import Literal, Union
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
    text_type: Literal["document", "query"],
    texts: list[str],
    titles: list[str] = [],
    dim: int = config.EMBEDDING_DIM,
    batch_size: int = 8,
) -> list[list[float]]:
    model = get_embedding_model()

    # Add the provided prefix to the texts
    if text_type == "query":
        processed_prompts = [f"task: search result | query: {text}" for text in texts]
    elif text_type == "document":
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")

        processed_prompts = [
            f"title: {title} | text: {text}" for text, title in zip(texts, titles)
        ]

    embeddings = model.encode(
        sentences=processed_prompts,
        truncate_dim=dim,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    return embeddings.tolist()
