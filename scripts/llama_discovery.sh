#!/bin/bash
#SBATCH --job-name=llama_discovery
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%j.out

set -e

# Initialize timer
START_TIME=$(date +%s)

echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"

# Load CUDA modules
module purge
module load cuda12.6
module load cudnn8.9-cuda12.3
module reload

# Activate environment
echo "Activating python environment"
source .venv/bin/activate

echo "GPU info:"
nvidia-smi

# Set model path
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/Qwen3-32B-Q5_K_M.gguf"
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/gemma-3-27b-it-qat-Q4_0.gguf"
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/gemma-3-27b-it-UD-Q8_K_XL.gguf"
MODEL_PATH="/home/azbelikoff/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/672575d5a4634e1c6f2a12b5a05e18f5a86f227f/Qwen3-8B-Q5_K_M.gguf"

# Function to clean up llama-server
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$LLAMA_SERVER_PID" ]]; then
        echo "Killing llama-server process with PID $LLAMA_SERVER_PID"
        kill $LLAMA_SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT SIGINT SIGTERM

# Start llama-server in the background
./llama.cpp/llama-server --model "$MODEL_PATH" --host localhost --port 55551 -ngl 999 -c 32768 -np 2 &

LLAMA_SERVER_PID=$!

# Wait for the server to become ready
python scripts/wait_for_llama_server.py
if [ $? -ne 0 ]; then
    echo "llama-server failed to become ready."
    exit 1
fi

# Run inference script, passing model path as an argument
python llama_server_discovery.py --model "$MODEL_PATH"
