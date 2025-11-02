import torch
from sentence_transformers import SparseEncoder
from functools import lru_cache
from typing import Literal
from src.core import config


@lru_cache(maxsize=1)
def _get_embedding_model() -> SparseEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading sparse embedding model {config.SPARSE_MODEL} on device: {device}")
    model = SparseEncoder(
        model_name_or_path=config.SPARSE_MODEL_PATH,
        device=device,
    )
    return model


def sparse_encode(
    text_type: Literal["document", "query"],
    texts: list[str],
    batch_size: int = 8,
) -> list[tuple[list[int], list[float]]]:
    model = _get_embedding_model()
    embeddings: torch.Tensor = torch.Tensor([])

    if text_type == "query":
        embeddings = model.encode_query(sentences=texts, batch_size=batch_size)
    elif text_type == "document":
        embeddings = model.encode_document(sentences=texts, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported text_type: {text_type}")

    embeddings = embeddings.cpu().coalesce()

    final_embeddings = []
    for i in range(embeddings.shape[0]):
        item_mask = embeddings.indices()[0] == i
        indices = embeddings.indices()[1][item_mask].tolist()
        values = embeddings.values()[item_mask].tolist()
        final_embeddings.append((indices, values))

    return final_embeddings
