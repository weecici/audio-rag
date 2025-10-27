"""
Evaluate retrieval performance against the Cranfield test set.

This script will:
- Read queries from data/cranfield/test/query.txt
- For each query, retrieve top-k results from Qdrant via the internal services
- Aggregate chunk-level hits to document-level using max score per document
- Compare with ground-truth relevance in data/cranfield/test/results/{qid}.txt
- Report Recall@k, MAP@k, and nDCG@k (averaged over queries with at least 1 relevant doc)
"""

import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path so 'src' is importable when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.services.internal import dense_encode  # noqa: E402
from src.repo.qdrant import dense_search  # noqa: E402


def read_queries(path: str) -> List[Tuple[str, str]]:
    """Read queries in the format: `<qid>\t<text>` per line."""
    queries: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first tab; fallback to whitespace
            if "\t" in line:
                qid, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                qid, text = parts
            queries.append((qid.strip(), text.strip()))
    return queries


def read_qrels_file(path: str) -> Dict[str, int]:
    """Read qrels file with lines like: `<qid> <docid>\t<rel>` (whitespace tolerant).

    Returns mapping of doc_id -> graded relevance for rel > 0. Ignores rel <= 0.
    """
    qrels: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) < 3:
                continue
            _qid, doc_id, rel_str = parts[0], parts[1], parts[2]
            try:
                rel = int(rel_str)
            except ValueError:
                continue
            if rel > 0:
                # Store max grade if duplicates
                qrels[doc_id] = max(qrels.get(doc_id, 0), rel)
    return qrels


def precision_at_k(pred: List[str], truth_set: set, k: int) -> float:
    if k == 0:
        return 0.0
    pred_k = pred[:k]
    hits = sum(1 for d in pred_k if d in truth_set)
    return hits / k


def average_precision_at_k(pred: List[str], truth_set: set, k: int) -> float:
    if not truth_set:
        return 0.0
    ap = 0.0
    hits = 0
    for i, d in enumerate(pred[:k], start=1):
        if d in truth_set:
            hits += 1
            ap += hits / i
    return ap / max(1, len(truth_set))


def ndcg_at_k(pred: List[str], rel_map: Dict[str, int], k: int) -> float:
    # Graded DCG: sum((2^rel - 1) / log2(rank+1))
    def dcg(items: List[int]) -> float:
        if not items:
            return 0.0
        items = items[:k]
        gains = [(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(items)]
        return float(np.sum(gains))

    pred_rels = [rel_map.get(d, 0) for d in pred[:k]]
    idcg_items = sorted(rel_map.values(), reverse=True)[:k]
    denom = dcg(idcg_items)
    if denom == 0.0:
        return 0.0
    return dcg(pred_rels) / denom


def recall_at_k(pred: List[str], truth_set: set, k: int) -> float:
    if not truth_set:
        return 0.0
    pred_k = set(pred[:k])
    return len(pred_k & truth_set) / len(truth_set)


def aggregate_to_docs(results: List[Dict], top_k: int) -> List[Tuple[str, float]]:
    """Aggregate chunk-level hits to document-level by max score and return top_k.

    Each result entry is expected to have a 'metadata' dict with a 'document_id' key
    and a 'score' float at the top level.
    """
    per_doc: Dict[str, float] = {}
    for r in results:
        meta = r.get("metadata", {})
        doc_id = str(
            meta.get("document_id") or meta.get("file_name", "").split(".")[0]
        ).strip()

        if not doc_id:
            # Fallback: try from node id
            doc_id = str(r.get("id_", ""))

        score = float(r.get("score", 0.0))
        if doc_id:
            if doc_id not in per_doc or score > per_doc[doc_id]:
                per_doc[doc_id] = score
    # Sort by max score desc
    ranked = sorted(per_doc.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on Cranfield test set"
    )
    parser.add_argument(
        "--queries",
        default=os.path.join(PROJECT_ROOT, "data/cranfield/test/query.txt"),
        help="Path to query.txt",
    )
    parser.add_argument(
        "--qrels_dir",
        default=os.path.join(PROJECT_ROOT, "data/cranfield/test/results"),
        help="Directory of per-query relevance files (<qid>.txt)",
    )
    parser.add_argument(
        "--collection", default="cranfield", help="Qdrant collection name"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Top-K documents to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding"
    )
    args = parser.parse_args()

    queries = read_queries(args.queries)
    if not queries:
        print("No queries found.")
        raise SystemExit(1)

    query_texts = [q[1] for q in queries]

    # Embed queries
    embeddings = dense_encode(
        texts=query_texts, text_type="query", batch_size=args.batch_size
    )

    # Retrieve chunk-level results
    batch_results = dense_search(
        query_embeddings=embeddings,
        collection_name=args.collection,
        top_k=args.top_k * 5,  # fetch more chunks to ensure enough distinct docs
    )

    # Evaluate per query
    k = args.top_k
    precs: List[float] = []
    recs: List[float] = []
    maps: List[float] = []
    ndcgs: List[float] = []
    evaluated = 0

    for (qid, _text), res in zip(queries, batch_results):
        # Aggregate chunks to documents
        doc_rank = aggregate_to_docs(res, top_k=k)
        pred_docs = [doc_id for doc_id, _ in doc_rank]

        # Load qrels for this qid
        qrels_path = os.path.join(args.qrels_dir, f"{qid}.txt")
        if not os.path.exists(qrels_path):
            # Skip if no ground truth
            continue
        rel_map = read_qrels_file(qrels_path)
        truth = {d for d, r in rel_map.items() if r > 0}

        if not truth:
            # Skip queries with no relevant docs to avoid divide-by-zero in AP/Recall
            continue

        evaluated += 1
        precs.append(precision_at_k(pred_docs, truth, k))
        recs.append(recall_at_k(pred_docs, truth, k))
        maps.append(average_precision_at_k(pred_docs, truth, k))
        ndcgs.append(ndcg_at_k(pred_docs, rel_map, k))

    if evaluated == 0:
        print("No queries evaluated (no ground truth found).")
        raise SystemExit(1)

    def avg(x: List[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    print("Evaluation summary (averaged over", evaluated, "queries):")
    print(f"  Precision@{k}: {avg(precs):.4f}")
    print(f"  Recall@{k}:    {avg(recs):.4f}")
    print(f"  MAP@{k}:       {avg(maps):.4f}")
    print(f"  nDCG@{k}:      {avg(ndcgs):.4f}")


if __name__ == "__main__":
    main()
