#!/bin/bash
#SBATCH --job-name=build_llama_cpp
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%j.out

# Load necessary modules
echo Initial Modules:
module list
module load gcc/12.1.0
module load cmake
module load cudnn8.9-cuda12.3
module load cuda12.3/blas/12.3.2
module load cuda12.3/fft/12.3.2
module load cuda12.3/toolkit/12.3.2
module load curl/8.6.0
module reload
echo Final Modules:
module list

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Configure with CMake 
cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;86;89;90" \

# Build targets
cmake --build build --config Release -j --target llama-server llama-quantize llama-cli llama-gguf-split # --clean-first

# Copy binaries to main directory for convenience
cp build/bin/llama-* ./

echo "Setup complete!"