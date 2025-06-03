import os
import io
import re
import json
import requests
import contextlib
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from typing import List, Optional, Generator, Union


class HuggingFaceModel:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", device: str = "auto"):
        """
        Initialize the Hugging Face model client.

        :param model_name: The name or path of the Hugging Face model.
        :param device: 'cpu', 'cuda', or 'auto'. If 'auto', uses GPU if available.
        """
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        stop: List[str] = [],
        log_response: bool = False,
        text_only: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from the given prompt using the Hugging Face model.

        :param prompt: The input prompt string.
        :param stream: Whether to stream the response (uses TextStreamer).
        :param stop: A list of stop sequences (not fully supported in streaming mode).
        :param log_response: If stream is true, prints responses as they are streamed back.
        :param text_only: If true, only return text (standard mode only).
        :param max_new_tokens: Maximum number of tokens to generate.
        :param kwargs: Additional generation arguments.
        :return: The generated text or a streaming generator.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            **kwargs,
        }

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer
            self.model.generate(**generation_kwargs)
            return None  # TextStreamer prints directly to stdout
        else:
            output_ids = self.model.generate(**generation_kwargs)[0]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            # Truncate the prompt part if necessary
            if text_only and output_text.startswith(prompt):
                output_text = output_text[len(prompt):]

            # Apply stop sequences
            for stop_seq in stop:
                idx = output_text.find(stop_seq)
                if idx != -1:
                    output_text = output_text[:idx]
                    break

            return output_text.strip()

    def get_model_info(self):
        """
        Return basic model and device info.
        """
        return {
            "model_name": self.model.config.name_or_path,
            "device": str(self.device),
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }



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


def extract_answer(question, answer):
    text = f'Extract the final answer from the given solution as a numeric value or a short phrase for the question. If you cannot extract an answer, return "None". '
    text += 'You should either return "None" or the final answer without any additional words.\n'
    text += f"Question: {question.strip()}\n"
    text += f"Solution: {answer.strip()}\n"
    text += "Final Answer:"
    return text


def exec_with_output(code: str, working_dir: str | None = None):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    result = None
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
            if lower_bound < pred_float and upper_bound > pred_float:
                return True
        except:
            # cannot extract number from the prediction
            return False

    else:  # question type is multiple choice
        return gold == pred[: len(gold)]


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
            stop=["\n"],
        ):
            final_answer += chunk

        correct = is_correct(final_answer, item)

        result_record = {
            "model": MODEL_NAME,
            "idx": idx,
            "answer": item["answer"],
            "pred": final_answer,
            "correct": correct,
        }
        save_result(RESULTS_PATH, result_record)


def POT(client, data):
    # Program of thoughts
    for idx, item in enumerate(data[:2]):
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

        final_answer, stdout, stderr = exec_with_output(
            answer, os.path.join(BENCHMARK_PATH, "data")
        )

        correct = is_correct(final_answer, item)

        result_record = {
            "model": MODEL_NAME,
            "idx": idx,
            "answer": item["answer"],
            "pred": final_answer,
            "correct": correct,
        }
        save_result(RESULTS_PATH, result_record)


if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    BENCHMARK_PATH = "QRData/benchmark"
    PROMPT_TYPE = "COT"  # COT, POT
    RESULTS_PATH = os.path.join(BENCHMARK_PATH, "results.jsonl")
    LOG = True
    client = HuggingFaceModel("meta-llama/Llama-3.2-1B")

    with open(os.path.join(BENCHMARK_PATH, "QRData.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    if PROMPT_TYPE == "COT":
        COT(client, data)
    elif PROMPT_TYPE == "POT":
        POT(client, data)
