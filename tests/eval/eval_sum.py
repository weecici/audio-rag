import os
import json
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
from app.core import config
from app.service.internal import get_summarization_prompts, generate

DATA_DIR = "data/sum"
DATA_PATH = os.path.join(DATA_DIR, "raw.json")
STEP = 5


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    print(f"Loading dense embedding model: {config.DENSE_MODEL}")
    model = SentenceTransformer(
        model_name_or_path=config.DENSE_MODEL_PATH, device="cpu"
    )
    return model


def dense_encode(
    texts: list[str],
    dim: int = config.DENSE_DIM,
    batch_size: int = 8,
) -> list[list[float]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_embedding_model()
    model = model.to(device=device)

    embeddings: torch.Tensor = torch.Tensor([])

    embeddings = model.encode_document(
        sentences=texts,
        truncate_dim=dim,
        batch_size=batch_size,
        convert_to_tensor=True,
    )
    final_embeddings = embeddings.cpu().tolist()

    if model.device.type == "cuda":
        model = model.to(device="cpu")
        torch.cuda.empty_cache()

    return final_embeddings


def load_data(path: Union[str, Path]) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data: any = json.load(f)

    texts: list[str] = []

    for item in data:
        if isinstance(item, str):
            texts.append(item)
            continue
        text = item.get("text")
        texts.append(text)

    return texts


def parse_summarization_responses(
    responses: list[str],
    raw_texts_list: list[list[str]],
) -> list[str]:

    separator = "\n=========="
    sep_pattern = re.compile(separator, flags=re.MULTILINE)

    summaries_list: list[str] = []

    for i, (response, raw_texts) in enumerate(zip(responses, raw_texts_list)):
        summaries = [
            seg.strip() for seg in re.split(sep_pattern, response) if seg.strip() != ""
        ]

        if len(summaries) != len(raw_texts):
            raise ValueError(
                f"Mismatch at batch {i}: expected {len(raw_texts)} summaries, got {len(summaries)}."
            )
        summaries_list.extend(summaries)

    return summaries_list


if __name__ == "__main__":
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4.zip")
    except LookupError:
        nltk.download("omw-1.4")

    raw_texts: list[str] = load_data(DATA_PATH)
    raw_texts_list: list[list[str]] = [
        raw_texts[i : i + STEP] for i in range(0, len(raw_texts), STEP)
    ]
    prompts = get_summarization_prompts(documents_list=raw_texts_list)

    for i, prompt in enumerate(prompts):
        sum_path = os.path.join(DATA_DIR, "summaries", f"summaries_{i}.json")
        if os.path.exists(sum_path):
            print(f"Summaries for batch {i} already exist. Skipping...")
            continue
        try:
            res = generate(prompts=[prompt], model="gpt-oss-120b")
            summaries_list = parse_summarization_responses(
                res, raw_texts_list[i : i + 1]
            )
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summaries_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error during generation at batch {i}: {e}")
            continue

    summaries: list[str] = []
    for i in range(len(prompts)):
        sum_path = os.path.join(DATA_DIR, "summaries", f"summaries_{i}.json")
        summaries.extend(load_data(sum_path))

    if len(summaries) != len(raw_texts):
        raise ValueError(
            f"Total summaries count ({len(summaries)}) does not match raw texts count ({len(raw_texts)})."
        )

    metrics = {}

    # Compute embeddings
    embed_raw = dense_encode(raw_texts)
    embed_sum = dense_encode(summaries)

    # Evaluate embedding normalized cosine similarity
    tensor_raw = torch.tensor(embed_raw)
    tensor_sum = torch.tensor(embed_sum)

    scores = F.cosine_similarity(tensor_raw, tensor_sum, dim=1)
    metrics["avg_cosine_similarity"] = scores.mean().item()

    # Evaluate ROUGE scores
    rouge = Rouge()
    scores = rouge.get_scores(summaries, raw_texts, avg=True)

    metrics["rouge-1-p"] = scores["rouge-1"]["p"]
    metrics["rouge-1-r"] = scores["rouge-1"]["r"]
    metrics["rouge-1-f"] = scores["rouge-1"]["f"]

    metrics["rouge-2-p"] = scores["rouge-2"]["p"]
    metrics["rouge-2-r"] = scores["rouge-2"]["r"]
    metrics["rouge-2-f"] = scores["rouge-2"]["f"]

    metrics["rouge-l-p"] = scores["rouge-l"]["p"]
    metrics["rouge-l-r"] = scores["rouge-l"]["r"]
    metrics["rouge-l-f"] = scores["rouge-l"]["f"]

    # Evaluate METEOR, Repetition Rate, and Compression Ratio

    meteor_scores = []
    repetition_rates = []
    compression_ratios = []

    for ref, hyp in zip(raw_texts, summaries):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

        if not hyp_tokens:
            meteor_scores.append(0.0)
            repetition_rates.append(0.0)
            compression_ratios.append(0.0)
            continue

        # METEOR
        m_score = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(m_score)

        # Repetition Rate (3-grams)
        ngrams = [" ".join(hyp_tokens[i : i + 3]) for i in range(len(hyp_tokens) - 2)]
        if ngrams:
            rep_rate = 1.0 - (len(set(ngrams)) / len(ngrams))
        else:
            rep_rate = 0.0
        repetition_rates.append(rep_rate)

        if len(ref_tokens) > 0:
            ratio = len(hyp_tokens) / len(ref_tokens)
            compression_ratios.append(ratio)

    metrics["avg_meteor"] = sum(meteor_scores) / len(meteor_scores)
    metrics["avg_repetition_rate_3gram"] = sum(repetition_rates) / len(repetition_rates)
    metrics["avg_compression_ratio"] = sum(compression_ratios) / len(compression_ratios)

    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

    with open(os.path.join(DATA_DIR, "summarization_eval_results.txt"), "w") as f:
        for key, val in metrics.items():
            f.write(f"{key}: {val:.4f}\n")
