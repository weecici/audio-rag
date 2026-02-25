import asyncio
import json
import math
import os
from pathlib import Path
from app import schema
from app.services.public import retrieve_documents

DATA_DIR = "data/ret"
POSSIBLE_K = [5, 10, 15, 20]


def load_data(path: str) -> list[dict[str, any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data: list[dict[str, any]] = json.load(f)
    return data


def get_overlap(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


def get_union_length(intervals):
    if not intervals:
        return 0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in sorted_intervals:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start < last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
    return sum(end - start for start, end in merged)


def calculate_dcg(scores, k):
    return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores[:k]))


def calculate_ndcg(scores, k):
    dcg = calculate_dcg(scores, k)
    idcg = calculate_dcg(sorted(scores, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0


if __name__ == "__main__":

    req = schema.RetrievalRequest(
        collection_name="cs431",
        queries=[],
        mode="hybrid",
        rerank_enabled=True,
    )

    data = load_data(os.path.join(DATA_DIR, "data.json"))
    queries = [item["question"] for item in data]
    req.queries = queries
    print(f"Loaded {len(queries)} queries for evaluation.")

    for k in POSSIBLE_K:
        print(f"Evaluating retrieval with top_k={k}")
        req.top_k = k
        res = asyncio.run(retrieve_documents(request=req))

        metrics_sum = {
            "cumulative_recall": 0.0,
            "oracle_recall": 0.0,
            "ndcg_recall": 0.0,
            "iou": 0.0,
            "oracle_iou": 0.0,
            "ndcg_iou": 0.0,
            "mrr": 0.0,
        }

        for i, docs in enumerate(res.results):
            gt = data[i]
            gt_start = gt["start"]
            gt_end = gt["end"]
            gt_video_id = gt["video_id"]
            gt_len = gt_end - gt_start

            overlaps = []
            recalls = []
            ious = []

            retrieved_by_video = {}
            first_relevant_rank = None

            for idx, doc in enumerate(docs):
                try:
                    _, start, end = doc.payload.metadata.title.split("||")
                    start, end = int(start), int(end)
                    doc_video_id = doc.payload.metadata.document_id
                except Exception:
                    overlaps.append(0)
                    recalls.append(0)
                    ious.append(0)
                    print(
                        "Error parsing document title (start, end):",
                        doc.payload.metadata.title,
                    )
                    continue

                # For IoU@k calculation
                if doc_video_id not in retrieved_by_video:
                    retrieved_by_video[doc_video_id] = []
                retrieved_by_video[doc_video_id].append((start, end))

                # For individual metrics
                if doc_video_id != gt_video_id:
                    overlap = 0
                    union = gt_len + (end - start)
                else:
                    overlap = get_overlap(gt_start, gt_end, start, end)
                    union = gt_len + (end - start) - overlap

                if overlap > 0 and first_relevant_rank is None:
                    first_relevant_rank = idx + 1

                overlaps.append(overlap)
                recalls.append(overlap / gt_len if gt_len > 0 else 0)
                ious.append(overlap / union if union > 0 else 0)

            # MRR
            if first_relevant_rank:
                metrics_sum["mrr"] += 1.0 / first_relevant_rank

            # Cumulative Recall
            metrics_sum["cumulative_recall"] += sum(recalls)

            # Oracle
            metrics_sum["oracle_recall"] += max(recalls) if recalls else 0
            metrics_sum["oracle_iou"] += max(ious) if ious else 0
            # IoU@k
            total_retrieved_len = 0
            for vid, intervals in retrieved_by_video.items():
                total_retrieved_len += get_union_length(intervals)

            matching_intervals = retrieved_by_video.get(gt_video_id, [])

            # Merge matching intervals to get disjoint set for intersection calculation
            sorted_matching = sorted(matching_intervals, key=lambda x: x[0])
            merged_matching = []
            for s, e in sorted_matching:
                if not merged_matching:
                    merged_matching.append((s, e))
                else:
                    ls, le = merged_matching[-1]
                    if s < le:
                        merged_matching[-1] = (ls, max(le, e))
                    else:
                        merged_matching.append((s, e))

            intersection_len = 0
            for s, e in merged_matching:
                intersection_len += get_overlap(gt_start, gt_end, s, e)

            union_len = gt_len + total_retrieved_len - intersection_len
            metrics_sum["iou"] += intersection_len / union_len if union_len > 0 else 0

            # nDCG
            metrics_sum["ndcg_recall"] += calculate_ndcg(recalls, k)
            metrics_sum["ndcg_iou"] += calculate_ndcg(ious, k)

        num_queries = len(res.results)
        if num_queries == 0:
            print(f"No results returned for k={k}")
            continue

        print(f"Results for k={k}:")
        for key, val in metrics_sum.items():
            print(f"  avg_{key}: {val / num_queries:.4f}")

        with open(os.path.join(DATA_DIR, "retrieval_eval_results.txt"), "a") as f:
            f.write(f"Results for k={k}:\n")
            for key, val in metrics_sum.items():
                f.write(f"  avg_{key}: {val / num_queries:.4f}\n")
            f.write("\n")
