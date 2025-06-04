import json
import os

BENCHMARK_PATH = "QRData/benchmark/"

def summarize_results():
    numerical_correct, numerical, cat_correct, cat = 0,0,0,0
    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    path = os.path.join(BENCHMARK_PATH, "results.jsonl")
    res = {}
    with open(path) as resultfile:
        for line in resultfile:
            result = json.loads(line)
            if data[result["idx"]]["meta_data"]["question_type"] == "numerical":
                numerical += 1
                numerical_correct += int(result["correct"])
            else:
                cat += 1
                cat_correct += int(result["correct"])
    print(f"Numerical Accuracy: {numerical_correct/numerical} Multiple Choice Acc: {cat/cat_correct}")
