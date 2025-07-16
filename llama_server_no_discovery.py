import os
import argparse
from utils.llama_server_client import LlamaServerClient
from utils.utils import (
    exec_with_output,
    is_correct,
    save_result,
    extract_code,
    load_data,
)
from prompts.prompts import format_QRData_item

################################################################
# Settings
################################################################
config = {
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",  # Path to the folder containing data/ and QRData.json
    "qrdata_file": "QRData_cleaned.json",
    "log": True,  # If true, LlamaServerClient will print model responses in the terminal
    "max_num_examples": -1,  # Max number of items from QRData to process. -1 means process all items
    "max_extra_turns": 3,  # The max number of retries the model gets for writing code
    "think": True,  # Set to true if the model you are using outputs <think></think> tags
    "prompt_options": {"prompt": "combined", "example": False, "rows": 10},
    "skip_results_path": None,  # Path to a file containing results that should be skipped. If None, no results will be skipped.
    "data_filters": [
        "Causal",
        "Num",
    ],  # Possible filters: "Causal", "Num", "Multiple Choice"
    "results_path": "results",  # Path to the folder where results will be saved
    "method": "POT",  # "POT" or "process"
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
    help="Port number for llama-server (default is 55552)",
)
args = parser.parse_args()

# Extract model name
MODEL_NAME = args.model.split("/")[-1].strip("gguf")

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
        stop=["```"],
    ):
        answer += chunk

    answer += "```"

    return answer


def process(
    client: LlamaServerClient,
    data,
    think: bool,
    max_extra_turns=3,
):
    for idx, item in data:
        prompt, answer_start = format_QRData_item(
            config["benchmark_path"], item, **config["prompt_options"]
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
                    code, os.path.join(config["benchmark_path"], "data")
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
        save_result(config["results_path"], result_record)

        logpath = f"results/logs/{MODEL_NAME}_Q{idx}_log.jsonl"
        for r in messages:
            save_result(logpath, r)


def POT(client, data):
    # Program of thoughts
    for idx, item in data:
        prompt = format_QRData_item(
            config["benchmark_path"], item, **config["prompt_options"]
        )
        try:
            answer = "def solution():\n"
            for chunk in client.generate(
                prompt=prompt,
                stream=True,
                log_response=config["log"],
                text_only=True,
                stop=["```"],
            ):
                answer += chunk

            lines = answer.splitlines()
            answer = []
            for line in lines:
                if line.strip().startswith("return"):
                    answer.append(line)
                    break
                answer.append(line)
            answer = "\n".join(answer)

            final_answer, stdout, stderr = exec_with_output(
                answer, os.path.join(config["benchmark_path"], "data")
            )
            print(f"Final Answer: {final_answer}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")

            correct = is_correct(final_answer, item)

            result_record = {
                "model": MODEL_NAME,
                "idx": idx,
                "answer": item["answer"],
                "pred": final_answer,
                "correct": correct,
            }
        except Exception:
            result_record = {
                "model": MODEL_NAME,
                "idx": idx,
                "answer": item["answer"],
                "pred": "err",
                "correct": False,
            }
        save_result(config["results_path"], result_record)


if __name__ == "__main__":
    client = LlamaServerClient(LLAMA_CPP_SERVER_BASE_URL)

    data = load_data(
        config["benchmark_path"],
        config["qrdata_file"],
        config["skip_results_path"],
        config["data_filters"],
        config["max_num_examples"],
    )

    if config["method"] == "POT":
        POT(client, data)
    elif config["method"] == "process":
        process(
            client,
            data,
            config["think"],
            max_extra_turns=config["max_extra_turns"],
        )
