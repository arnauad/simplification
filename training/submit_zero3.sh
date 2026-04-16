#!/bin/bash
#SBATCH -J grpo_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START TIME: $(date)"

# -------------------------
# ENV SETUP
# -------------------------
module load conda
eval "$(conda shell.bash hook)"
conda activate trl

mkdir -p logs

# -------------------------
# PATHS
# -------------------------
export DATASET_PATH="/data/upftfg34/aayguade/dataset/"
export LANG="EN"
export MODEL="/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct"
export OUT_MODEL="/home/aayguade/simplification/training/tmp/output"
export GRPO_RUNS="/home/aayguade/simplification/training/logs/checkpoints"
export REWARDS="/home/aayguade/simplification/training/logs/rewards.jsonl"

# -------------------------
# CACHE
# -------------------------
export HF_HOME=/home/aayguade/simplification/training/tmp/huggingface
export HF_HUB_CACHE=$HF_HOME/hub 
export HF_DATASETS_CACHE=/tmp/$USER/hf_datasets_$SLURM_PROCID

export TRITON_CACHE_DIR=/tmp/$USER/triton/$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"


NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# -------------------------
# LAUNCH
# -------------------------
echo "Launching distributed job..."


srun --nodes=2 --ntasks=2 --nodelist=${NODELIST[0]},${NODELIST[1]} --gres=gpu:2 bash -c '
  LOG=logs/gpu_${SLURM_JOB_ID}_$(hostname).log
  (
    while true; do
      echo "==== $(date) ====" >> $LOG
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \
                 --format=csv >> $LOG
      sleep 10
    done
  ) &

  accelerate launch \
    --config_file tmp/deepspeed_zero3.yaml \
    --num_processes 4 \
    --num_machines 2 \
    --main_process_ip '"${NODELIST[0]}"' \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    grpo_zero3.py
'
# -------------------------
# CLEANUP
# -------------------------
echo "END TIME: $(date)"
