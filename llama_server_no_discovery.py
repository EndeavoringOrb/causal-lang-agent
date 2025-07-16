import os
import json
import csv
import argparse
from utils.llama_server_client import LlamaServerClient
from utils.utils import exec_with_output, is_correct, save_result, extract_code
from prompts.prompts import format_QRData_item

################################################################
# Settings
################################################################
DEFAULT_PORT = 55552
DEFAULT_HOST = "http://localhost"
BENCHMARK_PATH = (
    "QRData/benchmark"  # Path to the folder containing data/ and QRData.json
)
LOG = True  # If true, LlamaServerClient will print model responses in the terminal
MAX_NUM_EXAMPLES = (
    -1
)  # Max number of items from QRData to process. -1 means process all items
MAX_EXTRA_TURNS = 3  # The max number of retries the model gets for writing code
THINK = True  # Set to true if the model you are using outputs <think></think> tags
PROMPT_OPTIONS = {
    "prompt": "combined",
    "example": False,
    "rows": 10,
}
SKIP_RESULTS_PATH = None  # i.e. "results/results_24.jsonl"
################################################################

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="Unknown",
    help="Path to the model used in llama-server",
)
parser.add_argument(
    "--port",
    type=int,
    help="Port number for llama-server (default is 55552)",
)
args = parser.parse_args()

# Extract model name
MODEL_NAME = args.model.split("/")[-1]

# Construct server URL
port = args.port if args.port is not None else DEFAULT_PORT
LLAMA_CPP_SERVER_BASE_URL = f"{DEFAULT_HOST}:{port}"

# Make sure results folder exists
os.makedirs("results", exist_ok=True)
num_results = len(os.listdir("results"))
RESULTS_PATH = os.path.join("results", f"results_{num_results}.jsonl")


def generate_code(
    client: LlamaServerClient,
    think: bool,
    messages: list[dict[str, str]],
    answer_start: str,
):
    answer = ""
    if think:
        for chunk in client.generate(
            prompt=messages + [{"role": "assistant", "content": "<think>"}],
            stream=True,
            log_response=LOG,
            text_only=True,
            stop=["</think>"],
        ):
            answer += chunk

        answer += "</think>\n"

    answer += answer_start

    for chunk in client.generate(
        prompt=messages + [{"role": "assistant", "content": answer}],
        stream=True,
        log_response=LOG,
        text_only=True,
        stop=["```"],
    ):
        answer += chunk

    answer += "```"

    return answer


def process(
    client: LlamaServerClient,
    data,
    think: bool,
    max_num_examples=1,
    max_extra_turns=3,
):
    if max_num_examples == -1:
        max_num_examples = len(data)
    max_num_examples = min(len(data), max_num_examples)
    for idx, item in data[:max_num_examples]:
        prompt, answer_start = format_QRData_item(
            BENCHMARK_PATH, item, **PROMPT_OPTIONS
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        output, stdout, stderr = "", "", ""

        for step_idx in range(max_extra_turns + 1):
            if output != "" and stderr == "" and (step_idx != 0):
                break

            print(f"Answer: {output}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")

            answer = generate_code(client, think, messages, answer_start)

            answer, code, no_code = extract_code(answer)
            messages.append({"role": "assistant", "content": answer})
            if no_code:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": "No code parsed in response. Please format code as ```python\n...\n```",
                        },
                    ]
                )
                output, stdout, stderr = "", "", ""
            else:
                output, stdout, stderr = exec_with_output(
                    code, os.path.join(BENCHMARK_PATH, "data")
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Output: {output}\nSTDOUT: {stdout}\nSTDERR: {stderr}\nThis is the result of running the last python code, please fix the code according to the result.",
                    }
                )

        print(f"Final Answer: {output}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

        correct = is_correct(output, item)

        result_record = {
            "model": MODEL_NAME,
            "idx": idx,
            "answer": item["answer"],
            "pred": output,
            "correct": correct,
        }
        save_result(RESULTS_PATH, result_record)

        logpath = f"results/logs/{MODEL_NAME}_Q{idx}_log.jsonl"
        for r in messages:
            save_result(logpath, r)


if __name__ == "__main__":
    client = LlamaServerClient(LLAMA_CPP_SERVER_BASE_URL)

    with open(
        os.path.join(BENCHMARK_PATH, "QRData_cleaned.json"), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    skip_idxs = set()
    if SKIP_RESULTS_PATH:
        with open(SKIP_RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                skip_idxs.add(line["idx"])
        print(f"Skipping {len(skip_idxs):,} items that have already been processed")

    print(f"Loaded {len(data):,} items")
    data = [
        item
        for item in enumerate(data)
        if (
            ("Causality" in item[1]["meta_data"]["keywords"])
            and (item[1]["meta_data"]["question_type"] == "numerical")
            and item[0] not in skip_idxs
        )
    ]

    def count_columns(csv_path):
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            return len(header)

    data = sorted(
        data,
        key=lambda item: count_columns(
            os.path.join(BENCHMARK_PATH, "data", item[1]["data_files"][0])
        ),
    )
    print(f"Filtered for {len(data):,} causal numerical items")
    print(data[0])

    process(client, data, THINK, MAX_NUM_EXAMPLES, max_extra_turns=MAX_EXTRA_TURNS)
