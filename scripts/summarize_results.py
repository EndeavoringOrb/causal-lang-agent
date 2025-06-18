import json
import os
from collections import defaultdict

BENCHMARK_PATH = "QRData/benchmark/"


def summarize_results(results_filepath: str):
    # Load benchmark metadata
    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize nested structure for tracking results
    stats = defaultdict(
        lambda: {
            "numerical": {"correct": 0, "total": 0},
            "categorical": {"correct": 0, "total": 0},
            "causal": {"correct": 0, "total": 0},
            "statistical": {"correct": 0, "total": 0},
            "causal and numerical":{"correct": 0, "total": 0}
        }
    )

    # Read results.jsonl line by line
    path = os.path.join(results_filepath)
    with open(path) as resultfile:
        for line in resultfile:
            result = json.loads(line)
            idx = result["idx"]
            model = result["model"]
            correct = int(result["correct"])
            qtype = data[idx]["meta_data"]["question_type"]

            key = (model)
            if qtype == "numerical":
                stats[key]["numerical"]["total"] += 1
                stats[key]["numerical"]["correct"] += correct
            else:
                stats[key]["categorical"]["total"] += 1
                stats[key]["categorical"]["correct"] += correct
            if "Causality" in data[idx]["meta_data"]["keywords"]:
                stats[key]["causal"]['total'] += 1
                stats[key]["causal"]['correct'] += correct
            else:
                stats[key]["statistical"]['total'] += 1
                stats[key]["statistical"]['correct'] += correct
            if "Causality" in data[idx]["meta_data"]["keywords"] and qtype == "numerical":
                stats[key]["causal and numerical"]["total"] += 1
                stats[key]["causal and numerical"]["correct"] += correct


    # Print results by model and prompt_type
    for (model), values in stats.items():
        num_total = values["numerical"]["total"]
        num_correct = values["numerical"]["correct"]
        cat_total = values["categorical"]["total"]
        cat_correct = values["categorical"]["correct"]
        cau_total = values["causal"]["total"]
        cau_correct = values["causal"]["correct"]
        caunum_total = values["causal and numerical"]["total"]
        caunum_correct = values["causal and numerical"]["correct"]
        

        num_acc = num_correct / num_total if num_total else 0
        cat_acc = cat_correct / cat_total if cat_total else 0
        cau_acc = cau_correct / cau_total if cau_total else 0
        caunum_acc = caunum_correct / caunum_total if caunum_total else 0

        print(f"Model: {model}")
        print(f"  Numerical Accuracy    : {num_acc:.2%} ({num_correct}/{num_total})")
        print(f"  Multiple Choice Acc   : {cat_acc:.2%} ({cat_correct}/{cat_total})")
        print(f"  Causal Accuracy    : {cau_acc:.2%} ({cau_correct}/{cau_total})")
        print(f"  Causal + Numerical Accuracy    : {caunum_acc:.2%} ({caunum_correct}/{caunum_total})")

        print()


if __name__ == "__main__":
    summarize_results("tmp.jsonl")
