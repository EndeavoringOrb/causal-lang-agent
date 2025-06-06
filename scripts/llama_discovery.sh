#!/bin/bash
#SBATCH --job-name=QRData_llama_server
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
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
./llama.cpp/llama-server --model /home/azbelikoff/.cache/huggingface/hub/models--unsloth--Qwen3-8B-GGUF/snapshots/672575d5a4634e1c6f2a12b5a05e18f5a86f227f/Qwen3-8B-Q5_K_M.gguf --host localhost --port 55551 -ngl 999 &

LLAMA_SERVER_PID=$!

# Wait for the server to become ready
python scripts/wait_for_llama_server.py
if [ $? -ne 0 ]; then
    echo "llama-server failed to become ready."
    exit 1
fi

# Run inference script
python llama_server_discovery.py