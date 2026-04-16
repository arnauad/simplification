import json
import matplotlib.pyplot as plt

steps = []
train_rewards = []
eval_steps = []
eval_rewards = []

with open("trainer_output/metrics.jsonl") as f:
    for line in f:
        data = json.loads(line)

        if "train_reward" in data:
            steps.append(data["step"])
            train_rewards.append(data["train_reward"])

        if "eval_reward" in data:
            eval_steps.append(data["step"])
            eval_rewards.append(data["eval_reward"])

plt.plot(steps, train_rewards, label="Train Reward")
plt.plot(eval_steps, eval_rewards, label="Eval Reward")

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()
plt.title("Training vs Validation Reward")
plt.show()