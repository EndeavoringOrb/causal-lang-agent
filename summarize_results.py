import json
import os
from collections import defaultdict

BENCHMARK_PATH = "QRData/benchmark/"


def summarize_results():
    # Load benchmark metadata
    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize nested structure for tracking results
    stats = defaultdict(
        lambda: {
            "numerical": {"correct": 0, "total": 0},
            "categorical": {"correct": 0, "total": 0},
        }
    )

    # Read results.jsonl line by line
    path = os.path.join(BENCHMARK_PATH, "results.jsonl")
    with open(path) as resultfile:
        for line in resultfile:
            result = json.loads(line)
            idx = result["idx"]
            model = result["model"]
            prompt_type = result["prompt_type"]
            correct = int(result["correct"])
            qtype = data[idx]["meta_data"]["question_type"]

            key = (model, prompt_type)
            if qtype == "numerical":
                stats[key]["numerical"]["total"] += 1
                stats[key]["numerical"]["correct"] += correct
            else:
                stats[key]["categorical"]["total"] += 1
                stats[key]["categorical"]["correct"] += correct

    # Print results by model and prompt_type
    for (model, prompt_type), values in stats.items():
        num_total = values["numerical"]["total"]
        num_correct = values["numerical"]["correct"]
        cat_total = values["categorical"]["total"]
        cat_correct = values["categorical"]["correct"]

        num_acc = num_correct / num_total if num_total else 0
        cat_acc = cat_correct / cat_total if cat_total else 0

        print(f"Model: {model} | Prompt: {prompt_type}")
        print(f"  Numerical Accuracy    : {num_acc:.2%} ({num_correct}/{num_total})")
        print(f"  Multiple Choice Acc   : {cat_acc:.2%} ({cat_correct}/{cat_total})")
        print()


if __name__ == "__main__":
    summarize_results()
