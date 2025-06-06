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

# Run inference script
python train_grpo_lora_unsloth.py