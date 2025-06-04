import os
import io
import re
import json
import requests
import contextlib
import pandas as pd
from typing import List, Optional, Generator, Union


class LlamaServerClient:
    def __init__(self, base_url: str = "http://localhost:55551"):
        """
        Initialize the LlamaServerClient with a specific base URL.

        :param base_url: The base URL of the llama-server API.
        """
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        stop: List[str] = [],
        log_response: bool = False,
        text_only: bool = True,
        **kwargs,
    ):
        """
        Generate text from the given prompt using the specified model.

        :param prompt: The input prompt string.
        :param stream: Whether to stream the response.
        :param stop: A list of stop sequences.
        :param log_response: If stream is true, prints responses as they are streamed back
        :param text_only: If true, only return text
        :param kwargs: Additional parameters to pass to the API.
        :return: The generated text as a string or a generator yielding strings if streaming.
        """
        url = f"{self.base_url}/v1/completions"
        payload = {
            "prompt": prompt,
            "stream": stream,
            "stop": stop,
            **kwargs,
        }

        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()

        if stream:

            def stream_generator():
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line == "data: [DONE]":
                            break
                        line = json.loads(line[5:])
                        if text_only:
                            line = line["choices"][0]["text"]
                        if log_response:
                            print(line, end="", flush=True)
                        yield line

            return stream_generator()
        else:
            return response.json()["choices"][0]["text"]

    def get_model_info(self):
        """
        Retrieve model and server information from the llama-server.

        :return: A dictionary containing server and model details.
        """
        url = f"{self.base_url}/v1/info"
        response = requests.post(url, headers={"accept": "application/json"})
        response.raise_for_status()
        return response.json()


def format_QRData_item(benchmark_path, item, rows=10):
    text = "Data Description:\n"
    text += item["data_description"]

    for file_name in item["data_files"]:
        text += file_name.strip(".csv") + ":\n"
        df = pd.read_csv(os.path.join(benchmark_path, f"data/{file_name}"))
        df = df.sample(frac=1, random_state=42)  # Shuffle the rows
        text += str(df.head(rows)).strip() + "\n"

    text += "Task:\n"
    text += "You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the "
    "provided data. The description and the table are listed above. Please analyze the table to answer the question. Do not write "
    "any code in your answer. Ensure that your final answer is positioned at the very end of your output, adhering to the format "
    "'Final answer: [answer]'. The final answer should be a number or a short phrase and should be written in a new line."

    text += "\nQuestion:\n"
    text += item["question"].strip()

    text += "\nResponse\nLet's think step by step."

    return text


def format_QRData_item_POT(benchmark_path, item, rows=10):
    text = """You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You can use any python library. The returned value of the program is supposed to be 
the answer. The format of the code should be
```python
def solution():
    # import libraries if needed
    # load data
    # write code to get the answer
    # return answer
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
    except Exception as e:
        print(f"Exception: {e}", file=stderr_buffer)
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


def COT(client, data):
    # Chain of thought
    for idx, item in enumerate(data):
        prompt = format_QRData_item(BENCHMARK_PATH, item)
        answer = ""
        for chunk in client.generate(
            prompt=prompt, stream=True, log_response=LOG, text_only=True
        ):
            answer += chunk

        extract_prompt = extract_answer(item["question"], answer)
        final_answer = ""
        for chunk in client.generate(
            prompt=extract_prompt,
            stream=True,
            log_response=LOG,
            text_only=True,
            stop=["$"] if USE_BOXED else ["\n"],
        ):
            final_answer += chunk

        correct = is_correct(final_answer, item)

        result_record = {
            "model": MODEL_NAME,
            "prompt_type": PROMPT_TYPE,
            "idx": idx,
            "answer": item["answer"],
            "pred": final_answer,
            "correct": correct,
        }
        save_result(RESULTS_PATH, result_record)


def POT(client, data):
    # Program of thoughts
    for idx, item in enumerate(data):
        prompt = format_QRData_item_POT(BENCHMARK_PATH, item)
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

        correct = is_correct(final_answer, item)

        result_record = {
            "model": MODEL_NAME,
            "prompt_type": PROMPT_TYPE,
            "idx": idx,
            "answer": item["answer"],
            "pred": final_answer,
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
    USE_BOXED = True
    MODEL_NAME = "unsloth/Qwen3-8B-Q5_K_M"
    BENCHMARK_PATH = "QRData/benchmark"
    PROMPT_TYPE = "POT"  # COT, POT
    RESULTS_PATH = os.path.join(BENCHMARK_PATH, "results.jsonl")
    LOG = True
    client = LlamaServerClient()

    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    if PROMPT_TYPE == "COT":
        COT(client, data)
    elif PROMPT_TYPE == "POT":
        POT(client, data)
