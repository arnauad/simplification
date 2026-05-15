import json
import os
import time
import torch
import numbers
import torch.distributed as dist
from datasets import Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

from src.prompts.training import (
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

def _to_jsonable(v):
    if isinstance(v, numbers.Number):
        return float(v)
    if hasattr(v, "item"):  # torch scalar
        try:
            return float(v.item())
        except Exception:
            pass
    return str(v)

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


MAX_PROMPT_TOKENS = 3072

def build_grpo_dataset(dataset, tokenizer):
    grpo_data = []

    for i, item in enumerate(dataset):
        src = item["original"]
        ref = item["simplification"]
        idx = item["id"]

        prompt = build_prompt(tokenizer, src)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        if prompt_tokens > MAX_PROMPT_TOKENS:
            print(f"[SKIP] idx={idx} prompt_tokens={prompt_tokens} src_chars={len(src)}")
            continue

        grpo_data.append({
            "id": idx,
            "prompt": prompt,
            "source": src,
            "reference": ref,
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
        step = int(self.state.global_step)
        record = {"step": step}

        for k, v in logs.items():
            record[k] = _to_jsonable(v)

        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(REWARDS), exist_ok=True)
            with open(REWARDS, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[LOG {step}] {record}", flush=True)
        return super().log(logs, *args, **kwargs)
    
    def _save_checkpoint(self, *args, **kwargs):
        step = self.state.global_step

        checkpoint_folder = f"checkpoint-{step}"
        output_dir = self.args.output_dir
        checkpoint_path = os.path.join(output_dir, checkpoint_folder)

        print(f"\nSaving checkpoint at step {step}")
        print(f"Path: {checkpoint_path}\n")

        return super()._save_checkpoint(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        print(f"\n Running evaluation at step {self.state.global_step}\n")
        return super().evaluate(*args, **kwargs)



if __name__ == "__main__":
    acc = Accelerator()

    print("Distributed type:", acc.state.distributed_type)
    print("Num processes:", acc.state.num_processes)
    print("Mixed precision:", acc.state.mixed_precision)

    print(f"Loading {LANG} dataset")
    train, test = load_datasets()

    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building datasets...")
    train = Dataset.from_list(build_grpo_dataset(train, tokenizer))
    test = Dataset.from_list(build_grpo_dataset(test, tokenizer))


    training_args = GRPOConfig(
        output_dir=GRPO_RUNS,

        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,

        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=True,

        logging_steps=1,

        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        shuffle_dataset=False,

        num_generations=4,
        max_completion_length=32,

        beta=0,
        loss_type="dr_grpo",

        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=2,

        generation_kwargs={
            "temperature": 0.9,
            #"stop": ["\n\n", "\n \n"],
            "top_p": 0.95,
            #"use_cache": True,
        },

    )

    trainer = HPCTrainer(
        model=MODEL,
        reward_funcs=compute_reward,
        args=training_args,
        train_dataset=train,
    )

    print("model:", type(trainer.model))
    print("model_wrapped:", type(trainer.model_wrapped))

    total = sum(p.numel() for p in trainer.model.parameters())
    local = sum(p.numel() for p in trainer.model.parameters() if p.device.type == "cuda")

    print(f"Total params: {total}")
    print(f"Local params on this GPU: {local}")

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(OUT_MODEL)
    tokenizer.save_pretrained(OUT_MODEL)