import json
import os
from collections import defaultdict
from typing import Optional, Set

def summarize_results(
    results_filepath: str, data, special_idxs: Optional[Set[int]] = None
):
    # Initialize nested structure for tracking results
    stats = defaultdict(
        lambda: {
            "numerical": {"correct": 0, "total": 0},
            "categorical": {"correct": 0, "total": 0},
            "causal": {"correct": 0, "total": 0},
            "statistical": {"correct": 0, "total": 0},
            "causal and numerical": {"correct": 0, "total": 0},
            "special subset": {"correct": 0, "total": 0},
        }
    )

    # Read results.jsonl and keep only the last occurrence of each idx
    path = os.path.join(results_filepath)
    latest_results = {}
    with open(path) as resultfile:
        for line in resultfile:
            result = json.loads(line)
            idx = result["idx"]
            if idx in latest_results:
                print(
                    f"ERROR: Duplicate idx {idx} detected. Overwriting with latest. (correct {'matches' if latest_results[idx]['correct'] == result['correct'] else 'does not match'})"
                )
            latest_results[idx] = result  # Overwrite previous if exists

    # Process the latest results only
    for idx, result in latest_results.items():
        model = result["model"]
        correct = int(result["correct"])
        qtype = data[idx]["meta_data"]["question_type"]
        keywords = data[idx]["meta_data"]["keywords"]

        key = model
        if qtype == "numerical":
            stats[key]["numerical"]["total"] += 1
            stats[key]["numerical"]["correct"] += correct
        else:
            stats[key]["categorical"]["total"] += 1
            stats[key]["categorical"]["correct"] += correct

        if "Causality" in keywords:
            stats[key]["causal"]["total"] += 1
            stats[key]["causal"]["correct"] += correct
        else:
            stats[key]["statistical"]["total"] += 1
            stats[key]["statistical"]["correct"] += correct

        if "Causality" in keywords and qtype == "numerical":
            stats[key]["causal and numerical"]["total"] += 1
            stats[key]["causal and numerical"]["correct"] += correct

        if special_idxs and idx in special_idxs:
            stats[key]["special subset"]["total"] += 1
            stats[key]["special subset"]["correct"] += correct

    # Print results by model
    for model, values in stats.items():

        def fmt(acc, correct, total):
            return f"{acc:.2%} ({correct}/{total})"

        print(f"Model: {model}")
        print(
            f"  Numerical Accuracy         : {fmt(values['numerical']['correct'] / values['numerical']['total'] if values['numerical']['total'] else 0, values['numerical']['correct'], values['numerical']['total'])}"
        )
        print(
            f"  Multiple Choice Accuracy   : {fmt(values['categorical']['correct'] / values['categorical']['total'] if values['categorical']['total'] else 0, values['categorical']['correct'], values['categorical']['total'])}"
        )
        print(
            f"  Causal Accuracy            : {fmt(values['causal']['correct'] / values['causal']['total'] if values['causal']['total'] else 0, values['causal']['correct'], values['causal']['total'])}"
        )
        print(
            f"  Causal + Numerical Accuracy: {fmt(values['causal and numerical']['correct'] / values['causal and numerical']['total'] if values['causal and numerical']['total'] else 0, values['causal and numerical']['correct'], values['causal and numerical']['total'])}"
        )
        if special_idxs:
            print(
                f"  Special Subset Accuracy    : {fmt(values['special subset']['correct'] / values['special subset']['total'] if values['special subset']['total'] else 0, values['special subset']['correct'], values['special subset']['total'])}"
            )
        print()


if __name__ == "__main__":
    ###########################################################################
    # Settings
    ###########################################################################
    result_dir = "results"
    special_idxs = set()  # Optional special idxs to treat as separate category
    BENCHMARK_PATH = "QRData/benchmark/"
    ###########################################################################

    result_files = [f"{result_dir}/{file}" for file in os.listdir(result_dir)]
    result_files = [file for file in result_files if file.endswith(".jsonl")]

    # Load benchmark metadata
    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    for file in result_files:
        print(f"File: {file}")
        summarize_results(file, data, special_idxs=special_idxs)
