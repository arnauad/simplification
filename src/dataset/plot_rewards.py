import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REWARDS = "../../data/rewards/rewards-EN.jsonl"
TOTAL_STEPS = 112_048

rewards = []

with open(REWARDS, "r", encoding="utf-8", errors="ignore") as f:
    for line_num, line in enumerate(f, start=1):

        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)

            if not isinstance(obj, dict):
                continue

            reward = obj.get("reward")
            if reward is None:
                continue

            rewards.append(float(reward))

        except Exception:
            continue

print(f"Loaded {len(rewards)} rewards")

if not rewards:
    raise ValueError("No valid reward values were found.")

df = pd.DataFrame({"reward": rewards})

df["step"] = np.linspace(1, TOTAL_STEPS, len(df))

window = 100
df["reward_smooth"] = ( df["reward"].rolling(window=window, min_periods=1).mean())

plt.figure(figsize=(12, 6))

plt.plot(
    df["step"],
    df["reward"],
    alpha=0.05,
    linewidth=0.5,
    label="Raw rewards"
)

plt.plot(
    df["step"],
    df["reward_smooth"],
    linewidth=2,
    label=f"Rolling mean ({window})"
)

plt.xlim(0, TOTAL_STEPS)

plt.xlabel("Training generation")
plt.ylabel("Reward")
plt.title("GRPO Reward Trend English")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()