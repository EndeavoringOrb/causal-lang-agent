import json
import random
from data import pretty_print
from solver import solve

CLADDER_PATH = "data/cladder-v1-q-balanced.json"
RANDOMIZE = False

with open(CLADDER_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def format_example(example):
    if not example["reasoning"]:
        return None, None

    output = {"graph": {"nodes": [], "edges": [], "given_info": {}}, "query": ""}

    # Get given info
    given_info_str = example["reasoning"]["step4"].strip()

    for info in given_info_str.split("\n"):
        k, v = [item.strip() for item in info.rsplit("=", 1)]
        output["graph"]["given_info"][k] = float(v)

    # Get nodes
    nodes_str = example["reasoning"]["step0"].strip()[4:-1]  # "Let " ... "."

    for node in nodes_str.split(";"):
        name, alias = [item.strip() for item in node.split("=")]
        # Make sure node is actually used
        used = False
        for item in output["graph"]["given_info"]:
            if name in item:
                used = True
                break
        if used:
            output["graph"]["nodes"].append({name: alias})

    # Get edges
    edges_str = example["reasoning"]["step1"].strip()

    for edge in edges_str.split(","):
        src, dst = [item.strip() for item in edge.split("->")]
        output["graph"]["edges"].append((src, dst))

    # Get query
    output["query"] = example["reasoning"]["step2"].strip()

    # Get other info
    info = {
        "expected_result": float(example["reasoning"]["step5"].split("=")[1].strip()),
        "query_type": example["meta"]["query_type"].strip(),
    }

    return output, info


if RANDOMIZE:
    random.shuffle(data)

incorrect_ids = [15, 60, 133, 143, 180, 1587, 1588, 1618, 1835, 218, 265]
error_ids = [997]

for example in data:
    # if example["meta"]["query_type"] not in ["ate"]:
    #     continue
    if example['question_id'] in incorrect_ids or example['question_id'] in error_ids:
        continue
    output, info = format_example(example)
    if output is None or info is None:
        continue
    # input("Press Enter to solve example.")
    result = solve(output)
    if result is None:
        pretty_print(example)
        result = solve(output)
    
    if abs(result - info["expected_result"]) > 0.011:
        print(f"Question ID {example['question_id']}: Expected '{info['expected_result']}' != '{result}'")
