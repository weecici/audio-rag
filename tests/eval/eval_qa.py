import os
import json
import asyncio
import time
from pathlib import Path
from app.services.public import generate_responses
from app import schemas

DATA_DIR = "data/qa"
POSSIBLE_K = [5, 10]


def load_data(path: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    return data


def get_full_question(item: dict) -> str:
    q_type = item.get("type", "")
    question = item.get("question", "")

    full_question = f'[{q_type}] {question} ([single] answer format: option_label; [multiple] answer format: option_label_1 option_label_2 option_label_3 ...; or "I don\'t know."; no explanation is needed.)'

    for option in item.get("options", []):
        label = option.get("label", "")
        text = option.get("text", "")
        full_question += f"\n{label}. {text}"
    return full_question


if __name__ == "__main__":
    exit()
    final_results_path = os.path.join(DATA_DIR, "final_results.txt")

    data = load_data(os.path.join(DATA_DIR, "data.json"))
    questions = [get_full_question(item) for item in data]
    raw_questions = [item.get("question", "") for item in data]
    answers = [item.get("correct", []) for item in data]

    for k in POSSIBLE_K:
        print(f"Evaluating for top_k={k}")

        results_path = os.path.join(DATA_DIR, f"results_k{k}.json")

        if not os.path.exists(results_path):
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        with open(results_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if len(records) == len(questions):
            print("All questions have been evaluated. Skipping...")
        else:
            for i, (question, answer) in enumerate(zip(questions, answers)):
                if str(i) in records:
                    print(f" + Q{i+1}: already evaluated, skipping.")
                    continue

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        req = schemas.GenerationRequest(
                            queries=[question],
                            collection_name="cs431",
                            top_k=k,
                            mode="hybrid",
                            rerank_enabled=True,
                            model_name="gpt-oss-120b",
                        )

                        res = asyncio.run(generate_responses(req))
                        gen_answer = res.responses[0].strip()

                        if "i don't know" in gen_answer.lower():
                            judge = "skipped"
                        else:
                            gen_answer = gen_answer.split()

                            if gen_answer == answer:
                                judge = "correct"
                            else:
                                judge = "incorrect"

                        print(
                            f" + Q{i+1}: {judge} - Generated: {gen_answer}, Expected: {answer}"
                        )

                        record = {
                            "question": raw_questions[i],
                            "generated_answer": gen_answer,
                            "expected_answer": answer,
                            "judge": judge,
                        }
                        records[i] = record
                        break

                    except Exception as e:
                        print(
                            f" + Q{i+1}: Error (Attempt {attempt+1}/{max_retries}) - {str(e)}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(2)

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)

        # Metrics calculation
        total = len(records)
        skipped = sum(1 for r in records.values() if r["judge"] == "skipped")
        correct = sum(1 for r in records.values() if r["judge"] == "correct")
        answered = total - skipped

        accuracy = correct / answered if answered > 0 else 0.0
        coverage = answered / total if total > 0 else 0.0

        with open(final_results_path, "a", encoding="utf-8") as f:
            f.write(f"Metrics for k={k}:\n")
            f.write(
                f"  Accuracy | Answered: {accuracy:.4f} ({correct}/{answered})\n  Coverage: {coverage:.4f} ({answered}/{total})\n"
            )

        print(f"Metrics for k={k}:")
        print(f"  Accuracy (on answered): {accuracy:.2%} ({correct}/{answered})")
        print(f"  Coverage: {coverage:.2%} ({answered}/{total})")
