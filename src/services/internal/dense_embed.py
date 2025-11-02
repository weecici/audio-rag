import torch
from functools import lru_cache
from typing import Literal
from sentence_transformers import SentenceTransformer
from src.core import config


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading dense embedding model {config.DENSE_MODEL} on device: {device}")
    model = SentenceTransformer(
        model_name_or_path=config.DENSE_MODEL_PATH,
        device=device,
    )
    return model


def dense_encode(
    text_type: Literal["document", "query"],
    texts: list[str],
    titles: list[str] = [],
    dim: int = config.DENSE_DIM,
    batch_size: int = 8,
) -> list[list[float]]:
    model = _get_embedding_model()

    embeddings: torch.Tensor = torch.Tensor([])

    if text_type == "query":
        # Add the provided prefix to the texts
        processed_prompts = [f"task: search result | query: {text}" for text in texts]

        embeddings = model.encode_query(
            sentences=processed_prompts,
            truncate_dim=dim,
            batch_size=batch_size,
            convert_to_tensor=True,
        )
    elif text_type == "document":
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")

        # Add the provided prefix to the texts
        processed_prompts = [
            f"title: {title} | text: {text}" for text, title in zip(texts, titles)
        ]

        embeddings = model.encode_document(
            sentences=processed_prompts,
            truncate_dim=dim,
            batch_size=batch_size,
            convert_to_tensor=True,
        )
    else:
        raise ValueError(f"Unsupported text_type: {text_type}")

    final_embeddings = embeddings.cpu().tolist()

    return final_embeddings
