import torch
from functools import lru_cache
from sentence_transformers import CrossEncoder
from app import schemas
from app.core import config
from app.utils import logger


@lru_cache(maxsize=1)
def _get_reranking_model() -> CrossEncoder:
    logger.info(f"Loading reranking model: {config.RERANKING_MODEL}")
    model = CrossEncoder(model_name_or_path=config.RERANKING_MODEL_PATH, device="cpu")
    return model


def rerank(
    queries: list[str],
    candidates: list[list[schemas.RetrievedDocument]],
    batch_size: int = 8,
) -> list[list[schemas.RetrievedDocument]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_reranking_model()
    model = model.to(device=device)

    if len(candidates) == 0:
        return [[] for _ in queries]

    sentence_pairs = []
    for i, query in enumerate(queries):
        for candidate in candidates[i]:
            sentence_pairs.append([query, candidate.payload.text])

    if not sentence_pairs:
        return []

    scores = model.predict(
        sentence_pairs, batch_size=batch_size, convert_to_tensor=True
    )
    scores: list[float] = scores.cpu().tolist()

    reranked_results: list[list[schemas.RetrievedDocument]] = []
    score_idx = 0
    for i, _ in enumerate(queries):
        num_candidates = len(candidates[i])
        query_scores = scores[score_idx : score_idx + num_candidates]
        score_idx += num_candidates

        scored_candidates = list(zip(candidates[i], query_scores))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        current_reranked = []
        for candidate, score in scored_candidates:
            candidate.score = score
            current_reranked.append(candidate)

        reranked_results.append(current_reranked)

    # move to cpu to save gpu memory
    if model.device.type == "cuda":
        model = model.to(device="cpu")
        torch.cuda.empty_cache()

    return reranked_results
