import os
import io
import re
import json
import requests
import traceback
import contextlib
import pandas as pd
from typing import List
import argparse

import discovery.config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="Unknown",
    help="Path to the model used in llama-server",
)
args = parser.parse_args()

MODEL_NAME = args.model.split("/")[-1].strip("gguf")

USE_BOXED = True
discovery.config.LLAMA_CPP_SERVER_BASE_URL = "http://localhost:55551"
discovery.config.MAX_PARALLEL_REQUESTS = 1
# MODEL_NAME = "unsloth/gemma-3-27b-it-UD-Q8_K_XL"
BENCHMARK_PATH = "QRData/benchmark"
RESULTS_PATH = os.path.join(BENCHMARK_PATH, "results.jsonl")
LOG = True
THINK = True
MAX_NUM_EXAMPLES = -1
MAX_EXTRA_TURNS = 3
import discovery.discover as discover


class LlamaServerClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
    ) -> None:
        """
        Initialize the LlamaCPPServerClient to interface with a llama.cpp server.

        :param base_url: The base URL of the llama.cpp server API (default is http://localhost:8080).
        :param model: A model identifier (optional, kept for consistency with other clients,
                      llama.cpp server typically doesn't use this directly for /v1/completions).
        """
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        stream: bool = True,
        stop: List[str] = [],
        log_response: bool = False,
        text_only: bool = True,
        **kwargs,
    ):
        """
        Internal method to interact with the llama.cpp server's /v1/completions endpoint.
        This method is adapted from the LlamaServerClient example provided.

        :param prompt: The input prompt string.
        :param stream: Whether to stream the response (currently collects full response if True).
        :param stop: A list of stop sequences.
        :param log_response: If stream is true, prints responses as they are streamed back.
        :param text_only: If true, only return text (assumed for this client).
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for controlling randomness.
        :param kwargs: Additional parameters to pass to the API.
        :return: The generated text as a string.
        """
        payload = {
            "stream": stream,
            **kwargs,
        }

        if isinstance(prompt, str):
            url = f"{self.base_url}/v1/completions"
            payload["prompt"] = prompt
            is_chat = False
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload["messages"] = prompt
            is_chat = True
        if stop:
            payload["stop"] = stop

        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line == "data: [DONE]":
                        break
                    try:
                        line_data = json.loads(line[5:])  # Strip "data: " prefix
                        if text_only:
                            if is_chat:
                                if "content" not in line_data["choices"][0]["delta"]:
                                    continue
                                content = line_data["choices"][0]["delta"]["content"]
                                if content is None:
                                    continue
                            else:
                                content = line_data["choices"][0]["text"]
                        else:
                            content = (
                                line_data  # Handle non-text_only structure if needed
                            )
                        if log_response:
                            print(content, end="", flush=True)
                        yield content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
        else:
            return response.json()["choices"][0]["text"]


def format_QRData_item(benchmark_path, item, rows=10):
    with open("causal_model_docs.py", "r", encoding="utf-8") as f:
        causal_model_docs = f.read().strip()
    text = f"""You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You must use the DoWhy library to build a causal model and perform effect estimation. The returned value of the program should be 
the answer. The path to the causal graph is graph.gml. After the solution function is written, don't write any more code and enter ```. The general format of the code should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)

    model = CausalModel(
        data = data,
        treatment = "treatment_col"
        outcome = "outcome_col"
        graph = "graph.gml"
    )
    identified_estimand = model.identify_effect()
    answer = model.estimate_effect(identified_estimand, method_name=...)
    return answer.value
```
Here is documentation for the CausalModel class:
```python
{causal_model_docs}
```""".strip()

    text += "\nData Description:\n"
    text += item["data_description"]

    text += f"\nFirst {rows} rows of the data:\n"
    for file_name in item["data_files"]:
        text += file_name.strip(".csv") + ":\n"
        df = pd.read_csv(os.path.join(benchmark_path, f"data/{file_name}"))
        df = df.sample(frac=1, random_state=42)  # Shuffle the rows
        text += str(df.head(rows)).strip() + "\n"

    text += "\nQuestion:\n"
    text += item["question"].strip()

    text += "\nResponse\n```python\ndef solution():\n"

    return text


def format_QRData_item_ReAct(benchmark_path, item, rows=10):
    with open("causal_model_docs.py", "r", encoding="utf-8") as f:
        causal_model_docs = f.read().strip()
    text = f"""You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You must use the DoWhy library to build a causal model and perform effect estimation. The returned value of the program should be 
the answer. The path to the causal graph is graph.gml. After the solution function is written, don't write any more code and enter ```. The general format of the code should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)

    model = CausalModel(
        data = data,
        treatment = "treatment_col"
        outcome = "outcome_col"
        graph = "graph.gml"
    )
    identified_estimand = model.identify_effect()
    answer = model.estimate_effect(identified_estimand, method_name=...)
    return answer.value
```
Here is documentation for the CausalModel class:
```python
{causal_model_docs}
```""".strip()

    text += "\nData Description:\n"
    text += item["data_description"]

    text += f"\nFirst {rows} rows of the data:\n"
    for file_name in item["data_files"]:
        text += file_name.strip(".csv") + ":\n"
        df = pd.read_csv(os.path.join(benchmark_path, f"data/{file_name}"))
        df = df.sample(frac=1, random_state=42)  # Shuffle the rows
        text += str(df.head(rows)).strip() + "\n"

    text += "\nQuestion:\n"
    text += item["question"].strip()

    return text, "```python\ndef solution():\n"


def extract_boxed_content(text):
    # Match content inside \boxed{...}
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1)
    return None


def extract_answer(question, answer):
    text = f'Extract the final answer from the given solution as a numeric value or a short phrase for the question. If you cannot extract an answer, return "None". '
    if USE_BOXED:
        text += "You should put your answer in $\\boxed{}$, i.e. $\\boxed{1.5}$ or $\\boxed{numerical}$ or $\\boxed{None}$.\n"
    else:
        text += 'You should either return "None" or the final answer without any additional words.\n'
    text += f"Question: {question.strip()}\n"
    text += f"Solution: {answer.strip()}\n"
    text += "Final Answer: "
    if USE_BOXED:
        text += "$\\boxed{"
    return text


def exec_with_output(code: str, working_dir: str | None = None):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    result = ""
    original_dir = os.getcwd()

    try:
        if working_dir:
            os.chdir(working_dir)

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            sol = {}
            exec(code, globals(), sol)
            result = sol["solution"]()
    except Exception:
        traceback.print_exc(file=stderr_buffer)
    finally:
        if working_dir:
            os.chdir(original_dir)

    return result, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def extract_first_number(string):
    # This regular expression will match any number including decimals and negative numbers,
    # and possibly followed by a percentage sign.
    match = re.search(r"-?\d+\.?\d*%?", string)
    if match:
        return match.group()
    else:
        return None


def is_correct(pred, data):
    gold = data["answer"]

    error_scale = 0.03

    pred = str(pred).strip().strip(".").strip("}")

    if data["meta_data"]["question_type"] == "numerical":
        if gold[-1] != "%":
            gold_float = float(gold)
        else:
            gold_float = float(gold[:-1]) / 100

        try:
            pred_float = extract_first_number(pred)
            if pred_float is None:
                return False

            if pred_float[-1] != "%":
                pred_float = float(pred_float)
            else:
                pred_float = float(pred_float[:-1]) / 100

            lower_bound = min(
                gold_float * (1 - error_scale), gold_float * (1 + error_scale)
            )
            upper_bound = max(
                gold_float * (1 - error_scale), gold_float * (1 + error_scale)
            )

            return lower_bound < pred_float and upper_bound > pred_float
        except:
            # cannot extract number from the prediction
            return False
    else:  # question type is multiple choice
        return gold == pred


def save_result(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")


def build_graph_dot(adj_mat, labels, working_dir):
    with open(os.path.join(working_dir, "graph.gml"), "w") as f:
        f.write("graph [\ndirected 1\n")

        # Write nodes with labels
        for i, label in enumerate(labels):
            f.write(f'node [\nid {i}\nlabel "{label}"\n]\n')

        # Write directed edges
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat[i])):
                if adj_mat[i][j] != 0:
                    f.write(f"edge [\nsource {i}\ntarget {j}\n]\n")

        f.write("]")
        f.close()


def POT(client, data, max_num_examples=3):
    # Program of thoughts
    for idx, item in data[: min(len(data), max_num_examples)]:
        prompt = format_QRData_item(BENCHMARK_PATH, item)
        causal_graph, labels = discover.discover(
            os.path.join(BENCHMARK_PATH, "data", item["data_files"][0]),
            item["data_description"],
        )
        build_graph_dot(causal_graph, labels, os.path.join(BENCHMARK_PATH, "data"))
        answer = "def solution():\n"
        for chunk in client.generate(
            prompt=prompt,
            stream=True,
            log_response=LOG,
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
            answer, os.path.join(BENCHMARK_PATH, "data")
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
        save_result(RESULTS_PATH, result_record)


def ReAct(client: LlamaServerClient, data, max_num_examples=1, max_extra_turns=3):
    # Program of thoughts
    for idx, item in data[: min(len(data), max_num_examples)]:
        prompt, answer_start = format_QRData_item_ReAct(BENCHMARK_PATH, item)
        causal_graph, labels = discover.discover(
            os.path.join(BENCHMARK_PATH, "data", item["data_files"][0]),
            item["data_description"],
        )
        build_graph_dot(causal_graph, labels, os.path.join(BENCHMARK_PATH, "data"))
        answer = answer_start
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_start},
        ]
        for chunk in client.generate(
            prompt=messages,
            stream=True,
            log_response=LOG,
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
            if line.strip() == "```python":
                continue
            answer.append(line)
        answer = "\n".join(answer)

        final_answer, stdout, stderr = exec_with_output(
            answer, os.path.join(BENCHMARK_PATH, "data")
        )

        for i in range(max_extra_turns):
            if stderr == "":
                break

            print(f"Answer: {final_answer}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")

            messages[-1] = {"role": "assistant", "content": answer + "```"}
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"Output: {final_answer}\nSTDOUT: {stdout}\nSTDERR: {stderr}",
                    },
                    {"role": "assistant", "content": answer_start},
                ]
            )
            answer = answer_start
            for chunk in client.generate(
                prompt=messages,
                stream=True,
                log_response=LOG,
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
                if line.strip() == "```python":
                    continue
                answer.append(line)
            answer = "\n".join(answer)

            final_answer, stdout, stderr = exec_with_output(
                answer, os.path.join(BENCHMARK_PATH, "data")
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
        save_result(RESULTS_PATH, result_record)


def ReAct_think(client: LlamaServerClient, data, max_num_examples=3, max_extra_turns=3):
    # Program of thoughts
    if max_num_examples == -1:
        max_num_examples = len(data)
    max_num_examples = min(len(data), max_num_examples)
    for idx, item in data[:max_num_examples]:
        prompt, _ = format_QRData_item_ReAct(BENCHMARK_PATH, item)
        causal_graph, labels = discover.discover(
            os.path.join(BENCHMARK_PATH, "data", item["data_files"][0]),
            item["data_description"],
        )
        build_graph_dot(causal_graph, labels, os.path.join(BENCHMARK_PATH, "data"))
        answer = ""
        messages = [
            {"role": "user", "content": prompt},
        ]
        for chunk in client.generate(
            prompt=messages,
            stream=True,
            log_response=LOG,
            text_only=True,
        ):
            answer += chunk

        # Remove thinking from answer
        if answer.find("</think>") != -1:
            full_answer = answer[answer.find("</think>") + len("</think>") :]

        # Extract code from answer
        code_start = full_answer.rindex("```python")
        code_end = full_answer.rindex("```")
        lines = full_answer[code_start + len("```python") : code_end].splitlines()
        answer = []
        for line in lines:
            if line.strip().startswith("return"):
                answer.append(line)
                break
            if line.strip() == "```python":
                continue
            answer.append(line)
        answer = "\n".join(answer)

        full_answer = (
            full_answer[:code_start]
            + "```python"
            + answer
            + "```"
            + full_answer[code_end + 1 :]
        )

        output, stdout, stderr = exec_with_output(
            answer, os.path.join(BENCHMARK_PATH, "data")
        )

        for _ in range(max_extra_turns):
            if stderr == "":
                break

            print(f"Answer: {output}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")

            messages.extend(
                [
                    {"role": "assistant", "content": full_answer},
                    {
                        "role": "user",
                        "content": f"Output: {output}\nSTDOUT: {stdout}\nSTDERR: {stderr}",
                    },
                ]
            )
            answer = ""
            for chunk in client.generate(
                prompt=messages,
                stream=True,
                log_response=LOG,
                text_only=True,
            ):
                answer += chunk

            # Remove thinking from answer
            full_answer = answer[answer.index("</think>") + 1 :]

            # Extract code from answer
            code_start = full_answer.rindex("```python")
            code_end = full_answer.rindex("```")
            lines = full_answer[code_start:code_end].splitlines()
            answer = []
            for line in lines:
                if line.strip().startswith("return"):
                    answer.append(line)
                    break
                if line.strip() == "```python":
                    continue
                answer.append(line)
            answer = "\n".join(answer)

            full_answer = (
                full_answer[:code_start]
                + "```python"
                + answer
                + "```"
                + full_answer[code_end + 1 :]
            )

            output, stdout, stderr = exec_with_output(
                answer, os.path.join(BENCHMARK_PATH, "data")
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


def test_is_correct():
    data = {
        "answer": "treatment group",
        "meta_data": {
            "question_type": "multiple_choice",
        },
    }
    assert is_correct("treatment group", data) == True
    assert is_correct(" treatment group", data) == True
    assert is_correct(" treatment group.", data) == True
    assert is_correct("treatment group}", data) == True


if __name__ == "__main__":
    client = LlamaServerClient(discovery.config.LLAMA_CPP_SERVER_BASE_URL)

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

    if THINK:
        ReAct_think(client, data, MAX_NUM_EXAMPLES, max_extra_turns=MAX_EXTRA_TURNS)
    else:
        ReAct(client, data, MAX_NUM_EXAMPLES, max_extra_turns=MAX_EXTRA_TURNS)
