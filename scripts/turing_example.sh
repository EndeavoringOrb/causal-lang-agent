#!/bin/bash
#SBATCH --job-name=turing_example
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%j.out

echo "Job started at $(date)"
echo "Running on host: $(hostname)"

# Load modules
echo "Loading modules"
module load python/3.13.2
module load cudnn8.9-cuda12.3/8.9.7.29
module load cuda12.3/blas/12.3.2
module load cuda12.3/fft/12.3.2
module load cuda12.3/toolkit/12.3.2
module reload

# Activate environment
echo "Activating python environment"
source ~/projects/povertyIndex/llm_env/bin/activate
echo "Python Version:"
python --version
echo "Python Executable Path:"
which python

echo "GPU info:"
nvidia-smi
nvcc --version

# Tests
echo "GPU debug:"
gpu_debug

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Run your inference script
python inference.py

echo "Job finished at $(date)"
