import os
import argparse
from utils.llama_server_client import LlamaServerClient
from utils.utils import is_correct, save_result, load_data
from tools.shell import PersistentShell
from prompts.prompts import format_QRData_item_react

################################################################
# Settings
################################################################
config = {
    "default_port": 55553,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",  # Path to the folder containing data/ and QRData.json
    "qrdata_file": "QRData_cleaned.json",
    "log": True,  # If true, LlamaServerClient will print model responses in the terminal
    "max_num_examples": -1,  # Max number of items from QRData to process. -1 means process all items
    "max_extra_turns": 9,  # The max number of retries the model gets for writing code
    "think": True,  # Set to true if the model you are using outputs <think></think> tags
    "prompt_options": {"prompt": "qrdata_react", "rows": 5},
    "skip_results_path": None,  # Path to a file containing results that should be skipped. If None, no results will be skipped.
    "data_filters": [
        "Causal",
        "Num",
    ],  # Possible filters: "Causal", "Num", "Multiple Choice"
    "results_path": "results",  # Path to the folder where results will be saved
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
port = args.port if args.port is not None else config["default_port"]
LLAMA_CPP_SERVER_BASE_URL = f"{config['default_host']}:{port}"

# Make sure results folder exists
os.makedirs(config["results_path"], exist_ok=True)
num_results = len(os.listdir(config["results_path"]))
config["results_path"] = os.path.join(
    config["results_path"], f"results_{num_results}.jsonl"
)
print(f"Saving results to {config['results_path']}")


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
            log_response=config["log"],
            text_only=True,
            stop=["</think>"],
        ):
            answer += chunk

        answer += "</think>\n"

    answer += answer_start

    for chunk in client.generate(
        prompt=messages + [{"role": "assistant", "content": answer}],
        stream=True,
        log_response=config["log"],
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
    shell.set_working_directory(config["benchmark_path"])
    if max_num_examples == -1:
        max_num_examples = len(data)
    max_num_examples = min(len(data), max_num_examples)
    for idx, item in data[:max_num_examples]:
        prompt, df = format_QRData_item_react(
            config["benchmark_path"], item, **config["prompt_options"]
        )
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
        save_result(config["results_path"], result_record)

        logpath = f"results/logs/{MODEL_NAME}_Q{idx}_log.jsonl"
        for r in messages:
            save_result(logpath, r)


if __name__ == "__main__":
    client = LlamaServerClient(LLAMA_CPP_SERVER_BASE_URL)

    data = load_data(
        config["benchmark_path"],
        config["qrdata_file"],
        config["skip_results_path"],
        config["data_filters"],
    )

    process(
        client,
        data,
        config["think"],
        config["max_num_examples"],
        max_extra_turns=config["max_extra_turns"],
    )
