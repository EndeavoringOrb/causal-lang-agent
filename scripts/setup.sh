module load python/3.13.2
module load cudnn8.9-cuda12.3/8.9.7.29
module load cuda12.3/blas/12.3.2
module load cuda12.3/fft/12.3.2
module load cuda12.3/toolkit/12.3.2
module load cmake
python3 -m venv venv
source venv/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
mkdir slurm_logs

echo "Profile > Settings > Access Tokens > Create new token > Fine Grained > Create token"
huggingface-cli login