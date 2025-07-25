# causal-lang-agent

## Setup

### Python Environment
1. Create virtual environment
```bash
python -m venv .venv
```
2. Activate
#### Windows
```bash
.venv\Scripts\activate
```
#### Linux
```bash
source .venv/bin/activate
```
3. Install
```bash
pip install -r requirements.txt
```

### Data
1. Get data
```bash
git clone https://github.com/xxxiaol/QRData
```
2. Extract `QRData/benchmark/data.zip` -> `QRData/benchmark/data`
3. Copy `data/QRData_cleaned.json` -> `QRData/benchmark/QRData_cleaned.json`

## Running
1. Run llama-server with the following command:
```bash
llama-server -m /path/to/model.gguf --host localhost --port 55552
```
2. Run script (`llama_server_no_discovery.py` or `llama_server_no_discovery_qrdata_react.py`)

## Reproducing results

### results/Qwen3-14B-UD-Q6_K_XL-Ours-Num-NoInstructions.jsonl 
Run `llama_server_no_discovery.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData_cleaned.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 3,
    "think": True,
    "prompt_options": {"prompt": "combined", "rows": 10},
    "skip_results_path": None,
    "data_filters": [
        "Causal",
        "Num"],
    "results_path": "results",
    "method": "process"
}
```

### results/Qwen3-14B-UD-Q6_K_XL-ReAct-Num-NoInstructions.jsonl 
Run `llama_server_no_discovery_qrdata_react.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData_cleaned.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 9,
    "think": True,
    "prompt_options": {"prompt": "qrdata_react", "rows": 5},
    "skip_results_path": None,
    "data_filters": [
        "Causal",
        "Num"],
    "results_path": "results",
}
```

### results/Qwen3-14B-UD-Q6_K_XL-ReAct-Ours.jsonl 
Run `llama_server_no_discovery_qrdata_react.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData_cleaned.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 9,
    "think": True,
    "prompt_options": {
        "prompt": "qrdata_react", 
        "rows": 5,
        "api_docs": True,
        "tool_docs": True},
    "skip_results_path": None,
    "data_filters": [
        "Causal",
        "Num"],
    "results_path": "results",
}
```

### results/Qwen3-14B-UD-Q6_K_XL-ReAct.jsonl 
Run `llama_server_no_discovery_qrdata_react.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 9,
    "think": True,
    "prompt_options": {"prompt": "qrdata_react", "rows": 5},
    "skip_results_path": None,
    "data_filters": [
        "Causal"],
    "results_path": "results",
}
```

### results/Qwen3-14B-UD-Q6_K_XL-Ours.jsonl 
Run `llama_server_no_discovery.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 3,
    "think": True,
    "prompt_options": {"prompt": "combined", "rows": 10},
    "skip_results_path": None,
    "data_filters": [
        "Causal"],
    "results_path": "results",
    "method": "process"
}
```

### results/Qwen3-32B-UD-Q5_K_XL-POT.jsonl
Run `llama_server_no_discovery.py` with config:
```python
{
    "default_port": 55552,
    "default_host": "http://localhost",
    "benchmark_path": "QRData/benchmark",
    "qrdata_file": "QRData.json",
    "log": True,
    "max_num_examples": -1,
    "max_extra_turns": 3,
    "think": True,
    "prompt_options": {"prompt": "original", "rows": 10},
    "skip_results_path": None,
    "data_filters": [
        "Causal"],
    "results_path": "results",
    "method": "POT"
}
```