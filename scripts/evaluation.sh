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
#conda activate vllm
conda activate bleurt_tf212

nvidia-smi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export TORCH_COMPILE_CACHE=/home/aayguade/.cache/torch_compile_cache
export HF_HOME=/data/upftfg34/aayguade/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

export JOB_SCRATCH=$SCRATCH/$SLURM_JOB_ID
mkdir -p $JOB_SCRATCH

export TORCHINDUCTOR_CACHE_DIR=$JOB_SCRATCH/torchinductor
mkdir -p $TORCHINDUCTOR_CACHE_DIR

export VLLM_CACHE_DIR=$JOB_SCRATCH/vllm_cache
mkdir -p $VLLM_CACHE_DIR

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export XDG_CACHE_HOME=$JOB_SCRATCH/.cache
mkdir -p $XDG_CACHE_HOME

echo "SCRATCH: $JOB_SCRATCH"
echo "TORCH CACHE: $TORCHINDUCTOR_CACHE_DIR"

python -u ../src/evaluate/bleurt/bleurt_eval_vllm.py

echo "END TIME: $(date)"