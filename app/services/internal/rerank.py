"""Reranking using a CrossEncoder from Hugging Face's sentence-transformers library."""

import torch
from functools import lru_cache
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.gpu import gpu_lock as _gpu_lock
from app.core.logging import logger


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Load and return a CrossEncoder model for reranking."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = settings.RERANKER_MODEL
    logger.info(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name, device=device)


def _rerank_single(
    model: CrossEncoder, query: str, candidates: list[str], batch_size: int = 32
) -> list[tuple[int, float]]:
    """Rerank candidates based on relevance to the query.

    Returns a list of (candidate_index, score) tuples, sorted by descending score.
    """

    out = model.rank(query=query, documents=candidates, batch_size=batch_size)
    ranking = [(item["corpus_id"], item["score"]) for item in out]

    return ranking


def rerank(
    queries: list[str], candidate_lists: list[list[str]], batch_size: int = 32
) -> list[list[tuple[int, float]]]:
    """Rerank each candidate list based on relevance to each query.

    Returns a len(queries) element list of list of (candidate_index, score) tuples, sorted by descending score.

    GPU lifecycle:
        Acquires the shared GPU lock, moves the model to CUDA, performs
        reranking, then moves back to CPU and releases VRAM.  This prevents
        OOM when another service (e.g. speech-to-text) also needs the GPU.
    """

    model = _get_model()

    with _gpu_lock:
        if torch.cuda.is_available():
            model.to("cuda")

        rankings = []
        for query, candidates in zip(queries, candidate_lists):
            ranking = _rerank_single(model, query, candidates, batch_size=batch_size)
            rankings.append(ranking)

        # free up GPU memory after reranking
        if model.device.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()

    return rankings
