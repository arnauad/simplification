import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REWARDS = "../../data/rewards/rewards-EN.jsonl"
TOTAL_STEPS = 112_048
BATCH_SIZE = 512

len_penalties_triggered = []

# Extract length penalties from logs
with open(REWARDS, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            length_penalty = obj.get("length_penalty")
            if length_penalty is None:
                continue

            # 1 means triggered, 0 means not triggered
            len_penalties_triggered.append(1 if float(length_penalty) < 1.0 else 0)

        except Exception:
            continue

num_batches = len(len_penalties_triggered) // BATCH_SIZE

batch_steps = []
length_penalty_rates = []

# Calculate activation rates per batch
for i in range(num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    
    len_rate = (sum(len_penalties_triggered[start_idx:end_idx]) / BATCH_SIZE) * 100
    length_penalty_rates.append(len_rate)
    
    batch_center_idx = start_idx + (BATCH_SIZE / 2)
    step_mapping = (batch_center_idx / len(len_penalties_triggered)) * TOTAL_STEPS
    batch_steps.append(step_mapping)

plt.figure(figsize=(12, 6))

plt.plot(
    batch_steps, 
    length_penalty_rates, 
    color="tab:red", 
    linewidth=2.5, 
    label="Length Penalty ($P_{\\text{len}} < 1.0$)"
)

plt.xlim(0, TOTAL_STEPS)
plt.ylim(-5, 105)

plt.xlabel("Training Generation Step")
plt.ylabel("Constraint Activation Rate (%)")
plt.title("Programmatic Constraint Activation Rate During GRPO with English ASSET")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()