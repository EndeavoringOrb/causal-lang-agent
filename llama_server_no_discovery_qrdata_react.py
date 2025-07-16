import os
import json
import pandas as pd
import argparse
from utils.llama_server_client import LlamaServerClient
from utils.utils import is_correct, save_result
from tools.shell import PersistentShell
from prompts.prompts import format_QRData_item_react

################################################################
# Settings
################################################################
DEFAULT_PORT = 55553
DEFAULT_HOST = "http://localhost"
BENCHMARK_PATH = "QRData/benchmark"  # Path to the folder containing data/ and QRData.json
LOG = True  # If true, LlamaServerClient will print model responses in the terminal
MAX_NUM_EXAMPLES = (
    -1
)  # Max number of items from QRData to process. -1 means process all items
MAX_EXTRA_TURNS = 9  # The max number of retries the model gets for writing code
THINK = True  # Set to true if the model you are using outputs <think></think> tags
PROMPT_OPTIONS = {
    "prompt": "qrdata_react",
    "rows": 5,
}
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
    help="Port number for llama-server (default is 55553)",
)
args = parser.parse_args()

# Extract model name
MODEL_NAME = args.model.split("/")[-1]

# Construct server URL
port = args.port if args.port is not None else DEFAULT_PORT
LLAMA_CPP_SERVER_BASE_URL = f"{DEFAULT_HOST}:{port}"

# Make sure results folder exists
RESULTS_PATH = "/home/azbelikoff/projects/2025_Summer/results"
os.makedirs(RESULTS_PATH, exist_ok=True)
num_results = len(os.listdir(RESULTS_PATH))
RESULTS_PATH = os.path.join(RESULTS_PATH, f"results_{num_results}.jsonl")
print(f"Saving results to {RESULTS_PATH}")


def react_turn(
    client: LlamaServerClient,
    think: bool,
    messages: list[dict[str, str]],
    answer_start: str = "Thought:",
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
        stop=["Observation:"],
    ):
        answer += chunk

    return answer


def process_react_answer(answer: str):
    """
    Takes in an answer string.
    Returns a list of dictionaries, where each dictionary can be of type:
    - {"type": "Thought", "content": ...}
    - {"type": "Action", "content": ...}
    - {"type": "Action Input", "content": ...}
    - {"type": "Final Answer", "content": ...}
    """

    # Remove thinking from answer
    if answer.find("</think>") != -1:
        answer = answer[answer.find("</think>") + len("</think>") :].strip()
    # Remove Observation start from answer
    if answer.endswith("Observation"):
        answer = answer[: -len("Observation")].strip()

    keywords = ["Thought:", "Action:", "Action Input:", "Final Answer:"]

    found_keywords = []
    for keyword in keywords:
        start_index = 0
        while True:
            index = answer.find(keyword, start_index)
            if index == -1:
                break
            found_keywords.append((index, keyword))
            start_index = index + 1

    found_keywords.sort()

    if not found_keywords:
        return []

    result = []
    for i in range(len(found_keywords)):
        start_index = found_keywords[i][0] + len(found_keywords[i][1])
        end_index = (
            found_keywords[i + 1][0] if i + 1 < len(found_keywords) else len(answer)
        )

        content = answer[start_index:end_index].strip()
        keyword = found_keywords[i][1]

        type_name = keyword[:-1]

        if type_name == "Action":
            action_name = content.split("\n")[0].strip()
            result.append({"type": type_name, "content": action_name})
        else:
            result.append({"type": type_name, "content": content})

    # If an action input is present, truncate the list after it.
    action_input_index = -1
    for i, part in enumerate(result):
        if part["type"] == "Action Input":
            action_input_index = i
            break

    if action_input_index != -1:
        return result[: action_input_index + 1]

    return result


def process(
    client: LlamaServerClient,
    data,
    think: bool,
    max_num_examples=1,
    max_extra_turns=3,
):
    shell = PersistentShell(log=True)
    shell.set_working_directory(BENCHMARK_PATH)
    if max_num_examples == -1:
        max_num_examples = len(data)
    max_num_examples = min(len(data), max_num_examples)
    for idx, item in data[:max_num_examples]:
        prompt, df = format_QRData_item_react(BENCHMARK_PATH, item, **PROMPT_OPTIONS)
        shell.reset()
        shell.set_variable("df", df)
        messages = [
            {"role": "user", "content": prompt},
        ]
        output = ""

        for turn_num in range(max_extra_turns + 1):
            answer = react_turn(client, think, messages)
            messages.append({"role": "assistant", "content": answer})

            parsed_answer = process_react_answer(answer)

            if not parsed_answer:
                output = "The model did not produce a parsable response."
                break

            final_answer_found = False
            observation = ""

            action_name = None
            action_input = None

            for part in parsed_answer:
                if part["type"] == "Action":
                    action_name = part["content"]
                elif part["type"] == "Action Input":
                    action_input = part["content"]
                elif part["type"] == "Final Answer":
                    output = part["content"]
                    final_answer_found = True
                    break

            if final_answer_found:
                break

            if action_name:
                if action_name != "python_repl_ast":
                    observation = f'Action "{action_name}" is not a valid action.'
                elif action_input is None:
                    observation = "No Action Input specified."
                else:
                    observation = shell.run(action_input)
            else:
                observation = "No Action specified."

            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}",
                }
            )
            print(f"Observation: {observation}")

            if turn_num == max_extra_turns:
                output = observation

        print(f"Final Answer: {output}")

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

    print(f"Loaded {len(data):,} items")
    data = [
        item
        for item in enumerate(data)
        if (("Causality" in item[1]["meta_data"]["keywords"]))
        and (item[1]["meta_data"]["question_type"] == "numerical")
    ]

    def count_columns(csv_path):
        df = pd.read_csv(csv_path)
        return len(df.columns)

    data = sorted(
        data,
        key=lambda item: count_columns(
            os.path.join(BENCHMARK_PATH, "data", item[1]["data_files"][0])
        ),
    )
    print(f"Filtered for {len(data):,} causal numerical items")
    print(data[0])

    process(client, data, THINK, MAX_NUM_EXAMPLES, max_extra_turns=MAX_EXTRA_TURNS)
