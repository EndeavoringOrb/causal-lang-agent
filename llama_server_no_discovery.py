import os
import json
import pandas as pd
import argparse
from utils.llama_server_client import LlamaServerClient
from utils.utils import exec_with_output, is_correct, save_result, extract_code
from prompts import format_QRData_item


################################################################
# Settings
################################################################
LLAMA_CPP_SERVER_BASE_URL = "http://localhost:55552"  # The llama-server url
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
    "prompt": "identify_common_causes_effect_modifiers",
    "example": False,
    "rows": 10,
}
################################################################

# Make sure results folder exists
os.makedirs("results", exist_ok=True)
num_results = len(os.listdir("results"))
RESULTS_PATH = os.path.join("results", f"results_{num_results}.jsonl")

# Try to get model name from arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="Unknown",
    help="Path to the model used in llama-server",
)
args = parser.parse_args()
MODEL_NAME = args.model.split("/")[-1]


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
        if think:
            answer = ""
            messages = [
                {"role": "user", "content": prompt},
            ]
            for chunk in client.generate(
                prompt=messages + [{"role": "assistant", "content": "<think>"}],
                stream=True,
                log_response=LOG,
                text_only=True,
                stop=["</think>"],
            ):
                answer += chunk

            answer += "</think>\n" + answer_start

            for chunk in client.generate(
                prompt=messages + [{"role": "assistant", "content": answer}],
                stream=True,
                log_response=LOG,
                text_only=True,
                stop=["```"],
            ):
                answer += chunk

            answer += "```"
        else:
            answer = answer_start
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_start},
            ]
            stop = ["```"]

            for chunk in client.generate(
                prompt=messages,
                stream=True,
                log_response=LOG,
                text_only=True,
                stop=stop,
            ):
                answer += chunk

            answer += "```"

        answer, code = extract_code(answer)

        output, stdout, stderr = exec_with_output(
            code, os.path.join(BENCHMARK_PATH, "data")
        )

        for _ in range(max_extra_turns):
            if output != "" and stderr == "":
                break

            print(f"Answer: {output}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")

            if think:
                messages.extend(
                    [
                        {"role": "assistant", "content": answer},
                        {
                            "role": "user",
                            "content": f"Output: {output}\nSTDOUT: {stdout}\nSTDERR: {stderr}",
                        },
                    ]
                )
                answer = ""

                for chunk in client.generate(
                    prompt=messages + [{"role": "assistant", "content": "<think>"}],
                    stream=True,
                    log_response=LOG,
                    text_only=True,
                    stop=["</think>"],
                ):
                    answer += chunk

                answer += "</think>\n" + answer_start

                for chunk in client.generate(
                    prompt=messages + [{"role": "assistant", "content": answer}],
                    stream=True,
                    log_response=LOG,
                    text_only=True,
                    stop=["```"],
                ):
                    answer += chunk

                answer += "```"
            else:
                messages[-1] = {"role": "assistant", "content": answer}
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": f"Output: {output}\nSTDOUT: {stdout}\nSTDERR: {stderr}",
                        },
                        {"role": "assistant", "content": answer_start},
                    ]
                )
                answer = answer_start
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer_start},
                ]
                stop = ["```"]

                for chunk in client.generate(
                    prompt=messages,
                    stream=True,
                    log_response=LOG,
                    text_only=True,
                    stop=stop,
                ):
                    answer += chunk

                answer += "```"

            answer, code = extract_code(answer)

            output, stdout, stderr = exec_with_output(
                code, os.path.join(BENCHMARK_PATH, "data")
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


if __name__ == "__main__":
    client = LlamaServerClient(LLAMA_CPP_SERVER_BASE_URL)

    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} items")
    data = [
        item
        for item in enumerate(data)
        if (
            ("Causality" in item[1]["meta_data"]["keywords"])
            and (item[1]["meta_data"]["question_type"] == "numerical")
        )
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
