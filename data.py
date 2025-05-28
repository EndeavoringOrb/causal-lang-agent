import json

CLADDER_PATH = "data/cladder-v1-q-balanced.json"

with open(CLADDER_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def pretty_print(example):
    text = f"Question ID: {example['question_id']}\n"
    text += f"Given Info: {example['given_info'].strip()}\n"
    text += f"Question: {example['question'].strip()}\n"
    text += "Reasoning:\n"
    for k, v in example['reasoning'].items():
        text += f"  {k.strip()}: {v.strip()}\n"
    text += f"Answer: {example['answer'].strip()}"
    print(f"\n{text}\n")

for example in data:
    input("Press Enter to show example.")
    pretty_print(example)