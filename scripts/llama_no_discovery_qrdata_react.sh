#!/bin/bash
#SBATCH --job-name=react_14B
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH -C "A100|H100|H200"
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

# Set variables
PORT=55552
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/Qwen3-32B-UD-Q5_K_XL.gguf"
MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/Qwen3-14B-UD-Q6_K_XL.gguf"
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/gemma-3-27b-it-UD-Q8_K_XL.gguf"
# MODEL_PATH="/home/azbelikoff/projects/2025_Summer/models/Qwen3-8B-UD-Q5_K_XL.gguf"

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
./llama.cpp/llama-server --model "$MODEL_PATH" --host localhost --port "$PORT" -ngl 999 -c 32768 &

LLAMA_SERVER_PID=$!

# Wait for the server to become ready
python scripts/wait_for_llama_server.py --port "$PORT"
if [ $? -ne 0 ]; then
    echo "llama-server failed to become ready."
    exit 1
fi

# Run inference script, passing model path as an argument
echo "Running llama_server_no_discovery_qrdata_react.py"
python llama_server_no_discovery_qrdata_react.py --model "$MODEL_PATH" --port "$PORT"
