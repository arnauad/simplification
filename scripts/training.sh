#!/bin/bash
#SBATCH -J grpo_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH -o logs-es/%j.out
#SBATCH -e logs-es/%j.err

echo "START TIME: $(date)"

# -------------------------
# ENV SETUP
# -------------------------
module load conda
eval "$(conda shell.bash hook)"
conda activate trl

mkdir -p logs-es

# -------------------------
# PATHS
# -------------------------
export DATASET_PATH="/data/upftfg34/aayguade/dataset/"
export LANG="ES"
export MODEL="/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct"
export OUT_MODEL="/data/upftfg34/aayguade/models/tuned-Iberian/Iberian-LLM-ES"
export GRPO_RUNS="/data/upftfg34/aayguade/models/tuned-Iberian/checkpoints-ES"
export REWARDS="/data/upftfg34/aayguade/models/tuned-Iberian/rewards-ES.jsonl"

# -------------------------
# CACHE
# -------------------------
export HF_HOME=/home/aayguade/simplification/training/tmp/huggingface
export HF_HUB_CACHE=$HF_HOME/hub 
export HF_DATASETS_CACHE=/tmp/$USER/hf_datasets_${SLURM_JOB_ID}_${SLURM_PROCID}

export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor/$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/$USER/triton/$SLURM_JOB_ID
export XDG_CACHE_HOME=/tmp/$USER/xdg_cache/$SLURM_JOB_ID
mkdir -p $TORCHINDUCTOR_CACHE_DIR
mkdir -p $TRITON_CACHE_DIR
mkdir -p $XDG_CACHE_HOME


NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# -------------------------
# LAUNCH
# -------------------------
echo "Launching distributed job..."


srun --nodes=2 --ntasks=2 --nodelist=${NODELIST[0]},${NODELIST[1]} --gres=gpu:2 bash -c '
  LOG=logs-es/gpu_${SLURM_JOB_ID}_$(hostname).log
  (
    while true; do
      echo "==== $(date) ====" >> $LOG
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \
                 --format=csv >> $LOG
      sleep 10
    done
  ) &

  accelerate launch \
    --config_file ../src/training/utils/deepspeed_zero3.yaml \
    --num_processes 4 \
    --num_machines 2 \
    --main_process_ip '"${NODELIST[0]}"' \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    ../src/training/grpo_zero3.py
'
# -------------------------
# CLEANUP
# -------------------------
echo "END TIME: $(date)"
