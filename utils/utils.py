import contextlib
import traceback
import json
import csv
import os
import io
import re

# for exec
import econml
import dowhy


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def extract_code(answer: str):
    # Remove thinking from answer
    if answer.find("</think>") != -1:
        answer = answer[answer.find("</think>") + len("</think>") :]
    if answer.find("```python") == -1 or answer[len("```python") :].find("```") == -1:
        return answer, "", True
    code_start = answer.rindex("```python")
    code_end = answer.rindex("```")
    lines = answer[code_start + len("```python") : code_end].splitlines()
    code = []
    for line in lines:
        if line.strip().startswith("return"):
            code.append(line)
            break
        if line.strip() == "```python":
            continue
        code.append(line)
    code = "\n".join(code).strip()

    return answer, code, False

def count_columns(csv_path: str):
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        return len(header)
    
def load_data(BENCHMARK_PATH: str, QRDATA_FILE: str, SKIP_RESULTS_PATH: str | None, DATA_FILTERS: list[str], MAX_NUM_EXAMPLES: int = -1):
    with open(os.path.join(BENCHMARK_PATH, QRDATA_FILE), "r", encoding="utf-8") as f:
        data = json.load(f)

    skip_idxs = set()
    if SKIP_RESULTS_PATH:
        with open(SKIP_RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                skip_idxs.add(line["idx"])
        print(f"Skipping {len(skip_idxs):,} items that have already been processed")

    print(f"Loaded {len(data):,} items")
    new_data = []
    for item in enumerate(data):
        if "Causal" in DATA_FILTERS and (
            "Causality" not in item[1]["meta_data"]["keywords"]
        ):
            continue
        if "Num" in DATA_FILTERS and (
            item[1]["meta_data"]["question_type"] != "numerical"
        ):
            continue
        if "Multiple Choice" in DATA_FILTERS and (
            item[1]["meta_data"]["question_type"] == "numerical"
        ):
            continue
        if item[0] in skip_idxs:
            continue
        new_data.append(item)
    data = new_data
    print(f"Filtered for {len(data):,} items using filters {DATA_FILTERS}")

    data = sorted(
        data,
        key=lambda item: count_columns(
            os.path.join(BENCHMARK_PATH, "data", item[1]["data_files"][0])
        ),
    )
    print(f"Sorted data by # variables ascending")

    if MAX_NUM_EXAMPLES == -1:
        MAX_NUM_EXAMPLES = len(data)
    MAX_NUM_EXAMPLES = min(MAX_NUM_EXAMPLES, len(data))
    data = data[:MAX_NUM_EXAMPLES]
    print(f"Truncated data to {MAX_NUM_EXAMPLES:,} items")

    return data