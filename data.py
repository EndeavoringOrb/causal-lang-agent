import json
import random


def pretty_print(example):
    text = f"Question ID: {example['question_id']}\n"
    text += f"Description ID: {example['desc_id']}\n"
    text += f"Given Info: {example['given_info'].strip()}\n"
    text += f"Question: {example['question'].strip()}\n"
    text += f"Answer: {example['answer'].strip()}\n"
    if example["reasoning"]:
        text += "Reasoning:\n"
        for k, v in example["reasoning"].items():
            text += f"  {k.strip()}: {v.strip()}\n"
    text += f"Meta: {example['meta']}"
    print(f"\n{text}\n")


def iterate(data):
    for example in data:
        input("Press Enter to show example.")
        pretty_print(example)


def find_equations(data):
    all_items = set()

    for example in data:
        item = f"{example['meta']['query_type']} {example['meta']['formal_form']}"
        if item not in all_items:
            print(item)
            all_items.add(item)

    print(f"Processed {len(data):,} items")
    print(f"Found {len(all_items):,} unique (query_type, formal_form) pairs")


if __name__ == "__main__":
    CLADDER_PATH = "data/cladder-v1-q-balanced.json"
    RANDOMIZE = True

    with open(CLADDER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if RANDOMIZE:
        random.shuffle(data)

    find_equations(data)
    # iterate(data)
