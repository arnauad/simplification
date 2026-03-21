#!/bin/bash
#SBATCH -J inference_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j.%N.out
#SBATCH -e logs/%j.%N.err

echo "START TIME: $(date)"

module load conda
eval "$(conda shell.bash hook)"
conda activate vllm

nvidia-smi

# HuggingFace cache
export TORCH_COMPILE_CACHE=/home/aayguade/.cache/torch_compile_cache
export HF_HOME=/data/upftfg34/aayguade/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Scratch directory for this job
export JOB_SCRATCH=$SCRATCH/$SLURM_JOB_ID
mkdir -p $JOB_SCRATCH

# TorchInductor cache (this fixes your error)
export TORCHINDUCTOR_CACHE_DIR=$JOB_SCRATCH/torchinductor
mkdir -p $TORCHINDUCTOR_CACHE_DIR

# Optional but recommended for SLURM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "SCRATCH: $JOB_SCRATCH"
echo "TORCH CACHE: $TORCHINDUCTOR_CACHE_DIR"

python rule_search.py


echo "END TIME: $(date)"