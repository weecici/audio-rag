import numpy as np
from typing import Literal, Union
from src import schemas
from src.core import config

FusionMethod = Literal["rrf", "dbsf"]


from collections import defaultdict


def _fuse_rrf(
    results1: list[schemas.RetrievedDocument],
    results2: list[schemas.RetrievedDocument],
    k: int = config.RRF_K,
) -> list[schemas.RetrievedDocument]:
    fused_scores = defaultdict(float)
    all_docs: dict[Union[str, int], schemas.RetrievedDocument] = {}

    for rank, doc in enumerate(results1):
        fused_scores[doc.id] += 1 / (k + rank)
        if doc.id not in all_docs:
            all_docs[doc.id] = doc

    for rank, doc in enumerate(results2):
        fused_scores[doc.id] += 1 / (k + rank)
        if doc.id not in all_docs:
            all_docs[doc.id] = doc

    if not fused_scores:
        return []

    fused_results = [
        schemas.RetrievedDocument(
            id=doc_id,
            score=score,
            payload=all_docs[doc_id].payload,
        )
        for doc_id, score in fused_scores.items()
    ]
    fused_results.sort(key=lambda x: x.score, reverse=True)

    return fused_results


def _fuse_dbsf(
    results1: list[schemas.RetrievedDocument], results2: list[schemas.RetrievedDocument]
) -> list[schemas.RetrievedDocument]:
    fused_scores = defaultdict(float)
    all_docs: dict[Union[str, int], schemas.RetrievedDocument] = {}

    def _normalize_and_fuse(results: list[schemas.RetrievedDocument]) -> None:
        scores = np.array([doc.score for doc in results])
        if scores.size == 0:
            return

        mean, std = np.mean(scores), np.std(scores, ddof=1)
        # Handle case where all scores are the same
        if std == 0:
            for doc in results:
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc
                fused_scores[doc.id] += 0.5
            return

        ub, lb = mean + 3 * std, mean - 3 * std
        score_range = ub - lb

        for doc in results:
            if doc.id not in all_docs:
                all_docs[doc.id] = doc
            normalized_score = (doc.score - lb) / score_range
            fused_scores[doc.id] += normalized_score

    _normalize_and_fuse(results1)
    _normalize_and_fuse(results2)

    if not fused_scores:
        return []

    fused_results = [
        schemas.RetrievedDocument(
            id=doc_id,
            score=score,
            payload=all_docs[doc_id].payload,
        )
        for doc_id, score in fused_scores.items()
    ]
    fused_results.sort(key=lambda x: x.score, reverse=True)

    return fused_results


def fuse_results(
    results1: list[schemas.RetrievedDocument],
    results2: list[schemas.RetrievedDocument],
    method: FusionMethod = config.FUSION_METHOD,
) -> list[schemas.RetrievedDocument]:
    if method == "rrf":
        return _fuse_rrf(results1, results2)
    elif method == "dbsf":
        return _fuse_dbsf(results1, results2)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
