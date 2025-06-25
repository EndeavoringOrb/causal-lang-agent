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
source venv/bin/activate

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

./llama.cpp/llama-server --model /home/mlamborne/reu/huggingface_models/Qwen3-14B-UD-Q6_K_XL --host localhost --port 55552 -ngl 999 -c 8192 &


LLAMA_SERVER_PID=$!

# Wait for the server to become ready
python scripts/wait_for_llama_server.py
if [ $? -ne 0 ]; then
    echo "llama-server failed to become ready."
    exit 1
fi

cd matmcd

# Run inference script
python child_dataset_experiment.py