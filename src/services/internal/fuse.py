import numpy as np
from typing import Literal
from src.core import config

FusionMethod = Literal["rrf", "dbsf"]


from collections import defaultdict


def _fuse_rrf(
    results1: list[dict], results2: list[dict], k: int = config.RRF_K, **kwargs
) -> list[dict]:
    """
    Fuses results using Reciprocal Rank Fusion (RRF).

    RRF score for a document is the sum of 1 / (k + rank) over all ranked lists.
    """
    fused_scores = defaultdict(float)
    all_docs = {}

    for rank, doc in enumerate(results1):
        fused_scores[doc["id"]] += 1 / (k + rank)
        if doc["id"] not in all_docs:
            all_docs[doc["id"]] = doc

    for rank, doc in enumerate(results2):
        fused_scores[doc["id"]] += 1 / (k + rank)
        if doc["id"] not in all_docs:
            all_docs[doc["id"]] = doc

    if not fused_scores:
        return []

    fused_results = [
        {
            "id": doc_id,
            "score": score,
            "payload": all_docs[doc_id].get("payload"),
        }
        for doc_id, score in fused_scores.items()
    ]
    fused_results.sort(key=lambda x: x["score"], reverse=True)

    return fused_results


def _fuse_dbsf(results1: list[dict], results2: list[dict], **kwargs) -> list[dict]:
    """
    Fuses results using Distribution-Based Score Fusion (DBSF).
    Normalizes scores based on their distribution (mean and std dev) before fusing.
    """
    fused_scores = defaultdict(float)
    all_docs = {}

    def _normalize_and_fuse(results: list[dict]):
        scores = np.array([d["score"] for d in results if d.get("score") is not None])
        if scores.size == 0:
            return

        mean, std = np.mean(scores), np.std(scores, ddof=1)
        # Handle case where all scores are the same
        if std == 0:
            for doc in results:
                if doc["id"] not in all_docs:
                    all_docs[doc["id"]] = doc
                fused_scores[doc["id"]] += 0.5
            return

        ub, lb = mean + 3 * std, mean - 3 * std
        score_range = ub - lb

        for doc in results:
            if doc["id"] not in all_docs:
                all_docs[doc["id"]] = doc
            if doc.get("score") is not None:
                normalized_score = (doc["score"] - lb) / score_range
                fused_scores[doc["id"]] += normalized_score

    _normalize_and_fuse(results1)
    _normalize_and_fuse(results2)

    if not fused_scores:
        return []

    fused_results = [
        {
            "id": doc_id,
            "score": score,
            "payload": all_docs[doc_id].get("payload"),
        }
        for doc_id, score in fused_scores.items()
    ]
    fused_results.sort(key=lambda x: x["score"], reverse=True)

    return fused_results


def fuse_results(
    results1: list[dict],
    results2: list[dict],
    method: FusionMethod = config.FUSION_METHOD,
    **kwargs,
) -> list[dict]:
    """
    Fuses two lists of retrieval results using the specified method.

    Each result in the lists is a dictionary with "id" and "score".

    Args:
        results1: The first list of retrieval results.
        results2: The second list of retrieval results.
        method: The fusion method to use. Can be "rrf", "weighted_sum", or "dbsf".
        **kwargs: Additional arguments for the fusion method.
            For "rrf":
                k (int): The ranking constant (default: 60).
            For "weighted_sum":
                w1 (float): Weight for results1 (default: 0.5).
                w2 (float): Weight for results2 (default: 0.5).

    Returns:
        A single fused and re-ranked list of results.
    """
    if method == "rrf":
        return _fuse_rrf(results1, results2, **kwargs)
    elif method == "dbsf":
        return _fuse_dbsf(results1, results2, **kwargs)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
