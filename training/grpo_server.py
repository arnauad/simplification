import json
import os
import time
import torch
import requests
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from prompt import (
    SYSTEM_PROMPT_CA, USER_TEMPLATE_CA, FEW_SHOTS_CA,
    SYSTEM_PROMPT_EN, USER_TEMPLATE_EN, FEW_SHOTS_EN,
    SYSTEM_PROMPT_ES, USER_TEMPLATE_ES, FEW_SHOTS_ES
)
from metrics import compute_reward


DATASET_PATH = os.environ.get("DATASET_PATH")
LANG = os.environ.get("LANG")

MODEL = os.environ.get("MODEL")
OUT_MODEL = os.environ.get("OUT_MODEL")
GRPO_RUNS = os.environ.get("GRPO_RUNS")
REWARDS = os.environ.get("REWARDS")

PROMPTS = {
    "CA": (SYSTEM_PROMPT_CA, USER_TEMPLATE_CA, FEW_SHOTS_CA),
    "EN": (SYSTEM_PROMPT_EN, USER_TEMPLATE_EN, FEW_SHOTS_EN),
    "ES": (SYSTEM_PROMPT_ES, USER_TEMPLATE_ES, FEW_SHOTS_ES),
}

SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOTS = PROMPTS[LANG]


def load_datasets():
    with open(DATASET_PATH + f"train_{LANG}.json", encoding="utf-8") as f:
        train = json.load(f)

    with open(DATASET_PATH + f"test_{LANG}.json", encoding="utf-8") as f:
        test = json.load(f)

    return train, test


def build_prompt(tokenizer, sentence):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(sentence=sentence)
    })

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def build_grpo_dataset(dataset, tokenizer):
    grpo_data = []

    for item in dataset:
        prompt = build_prompt(tokenizer, item["original"])

        grpo_data.append({
            "prompt": prompt,
            "source": item["original"],
            "reference": item["simplification"],
        })

    return grpo_data



class HPCTrainer(GRPOTrainer):
    def __init__(self, *args, train_dataset=None, **kwargs):
        super().__init__(*args, train_dataset=train_dataset, **kwargs)

        self._start_time = time.time()
        self._step_times = []
        self._train_dataset_ref = train_dataset

    def training_step(self, *args, **kwargs):
        t0 = time.time()
        out = super().training_step(*args, **kwargs)
        t1 = time.time()

        step_time = t1 - t0
        self._step_times.append(step_time)

        step = self.state.global_step
        elapsed = t1 - self._start_time

        # stable average
        window = min(len(self._step_times), 20)
        avg_step = sum(self._step_times[-window:]) / window

        steps_per_sec = 1.0 / avg_step if avg_step > 0 else 0

        # robust total steps
        total_steps = self.state.max_steps
        if not total_steps or total_steps <= 0:
            total_steps = self.args.num_train_epochs * (
                len(self._train_dataset_ref) //
                (self.args.per_device_train_batch_size *
                 self.args.gradient_accumulation_steps)
            )

        remaining_steps = max(total_steps - step, 0)
        eta = remaining_steps * avg_step

        print(
            f"[STEP {step:5d}] "
            f"step={step_time:.2f}s | avg={avg_step:.2f}s | "
            f"{steps_per_sec:.2f} step/s | "
            f"elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m"
        )

        return out

    def log(self, logs, *args, **kwargs):
        step = self.state.global_step

        record = {"step": step}

        if "loss" in logs:
            record["loss"] = float(logs["loss"])

        if "reward" in logs:
            record["train_reward"] = float(logs["reward"])

        if "eval_reward" in logs:
            record["eval_reward"] = float(logs["eval_reward"])

        if "learning_rate" in logs:
            record["lr"] = float(logs["learning_rate"])

        with open(REWARDS, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[LOG {step}] {record}")

        return super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, *args, **kwargs):
    step = self.state.global_step

    checkpoint_folder = f"checkpoint-{step}"
    output_dir = self.args.output_dir
    checkpoint_path = os.path.join(output_dir, checkpoint_folder)

    print(f"\nSaving checkpoint at step {step}")
    print(f"Path: {checkpoint_path}\n")

    t0 = time.time()
    # Save locally 
    super()._save_checkpoint(*args, **kwargs)

    # Trigger vLLM to reload the checkpoint
    payload = {"model_path": checkpoint_path}
    try:
        r = requests.post(f"http://{os.environ['VLLM_HOST']}:8000/update_model/", json=payload)
        if r.status_code == 200:
            print(f"[SYNC] vLLM weights updated at step {step}")
        else:
            print(f"[SYNC] Failed to update vLLM weights: {r.text}")
    except Exception as e:
        print(f"[SYNC] Exception during vLLM update: {e}")
    t1 = time.time()
    print(f"Checkpoint saving and vLLM reload took {t1 - t0} seconds\n")

    def evaluate(self, *args, **kwargs):
        print(f"\n Running evaluation at step {self.state.global_step}\n")
        return super().evaluate(*args, **kwargs)



if __name__ == "__main__":
    print(f"Loading {LANG} dataset")
    train, test = load_datasets()

    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("Building datasets...")
    train = Dataset.from_list(build_grpo_dataset(train, tokenizer))
    test = Dataset.from_list(build_grpo_dataset(test, tokenizer))

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=os.environ.get("VLLM_HOST"),
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.85,

        output_dir=GRPO_RUNS,

        logging_strategy="steps",
        logging_steps=1,
        report_to="none",

        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,

        eval_strategy="steps",
        eval_steps=300,

        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=5e-6,
        gradient_checkpointing=True,

        num_generations=8,
        max_completion_length=64,
        beta=0,

        per_device_eval_batch_size=8,

        generation_kwargs={
            "temperature": 0.9,
            "stop": ["\n\n", "\n \n"],
        },
    )

    steps_per_epoch = len(train) // (
        training_args.per_device_train_batch_size *
        training_args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs

    print("Train size:", len(train))
    print("Eval size:", len(test))
    print("Steps/epoch:", steps_per_epoch)
    print("Total steps:", total_steps)


    trainer = HPCTrainer(
        model=model,
        reward_funcs=compute_reward,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(OUT_MODEL)
    tokenizer.save_pretrained(OUT_MODEL)