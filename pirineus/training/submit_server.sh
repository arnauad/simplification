#!/bin/bash
#SBATCH -J inference_test
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH -o logs2/%j.%N.out
#SBATCH -e logs2/%j.%N.err

echo "START TIME: $(date)"

module load conda
eval "$(conda shell.bash hook)"
conda activate vllm

mkdir -p logs2

# ENV
export DATASET_PATH="/data/upftfg34/aayguade/dataset/"
export LANG="EN"
export MODEL="/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct"
export OUT_MODEL="/home/aayguade/simplification/training/tmp/output"
export GRPO_RUNS="/home/aayguade/simplification/training/logs2/checkpoints"
export REWARDS="/home/aayguade/simplification/training/logs2/rewards.jsonl"

export HF_HOME=/data/upftfg34/aayguade/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# NODE ASSIGNMENT
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

TRAIN_NODE=${NODELIST[0]}
VLLM_NODE=${NODELIST[1]}

echo "TRAIN NODE: $TRAIN_NODE"
echo "VLLM NODE: $VLLM_NODE"

# GET IPs
VLLM_HOST=$(srun --nodes=1 --ntasks=1 -w $VLLM_NODE hostname)
MAIN_IP=$(srun --nodes=1 --ntasks=1 -w $TRAIN_NODE hostname -I | awk '{print $1}')

echo "VLLM HOST: $VLLM_HOST"
echo "MAIN IP: $MAIN_IP"

# START vLLM NODE
srun --nodes=1 --ntasks=1 -w $VLLM_NODE --gres=gpu:2 bash -c "
export CUDA_VISIBLE_DEVICES=0,1

echo 'Starting GPU monitor (vLLM node)...'
while true; do
    echo '==== \$(date) ====' >> logs/gpu_vllm_${SLURM_JOB_ID}.log
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> logs/gpu_vllm_${SLURM_JOB_ID}.log
    sleep 5
done &
MON_PID=\$!

echo 'Starting vLLM server...'
trl vllm-serve \
    --model $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    > logs/vllm_${SLURM_JOB_ID}.log 2>&1

kill \$MON_PID
" &

# WAIT FOR vLLM
echo "Waiting for vLLM..."
sleep 5

until curl -s http://$VLLM_HOST:8000/health > /dev/null; do
    echo "Still waiting..."
    sleep 5
done

echo "vLLM READY"

# START TRAINING NODE
srun --nodes=1 --ntasks=1 -w $TRAIN_NODE --gres=gpu:2 bash -c "
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_HOST=$VLLM_HOST

echo 'Starting GPU monitor (TRAIN node)...'
while true; do
    echo '==== \$(date) ====' >> logs/gpu_train_${SLURM_JOB_ID}.log
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> logs/gpu_train_${SLURM_JOB_ID}.log
    sleep 5
done &
MON_PID=\$!

echo 'Starting GRPO training...'
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_ip $MAIN_IP \
    --machine_rank 0 \
    --rdzv_backend c10d \
    grpo_server.py

kill \$MON_PID
"

wait

echo "END TIME: $(date)"