import json
import os
import time
import torch
import random
import math
from datasets import Dataset
from vllm import LLM, SamplingParams
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

MODELS_DIR = os.environ.get("MODELS_DIR")
MODEL = os.environ.get("MODEL")
OUT_MODEL = os.environ.get("OUT_MODEL")
GRPO_RUNS = os.environ.get("GRPO_RUNS")

PROMPTS = {
    "CA": (SYSTEM_PROMPT_CA, USER_TEMPLATE_CA, FEW_SHOTS_CA),
    "EN": (SYSTEM_PROMPT_EN, USER_TEMPLATE_EN, FEW_SHOTS_EN),
    "ES": (SYSTEM_PROMPT_ES, USER_TEMPLATE_ES, FEW_SHOTS_ES),
}

SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOTS = PROMPTS[LANG]

def load_datasets():
    train_p = DATASET_PATH + "train_" + LANG + ".json"
    with open(train_p, encoding="utf-8") as f:
        train = json.load(f)
    
    test_p = DATASET_PATH + "test_" + LANG + ".json"
    with open(test_p, encoding="utf-8") as f:
        test = json.load(f)

    return train, test


def build_prompt(tokenizer, sentence):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples as conversation turns
    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Add actual query
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
        sentence = item["original"]
        reference = item["simplification"]

        prompt = build_prompt(tokenizer, sentence)

        grpo_data.append({
            "prompt": prompt,
            "source": sentence,
            "reference": reference,
        })

    return grpo_data


class TimedTrainer(GRPOTrainer):
    def training_step(self, *args, **kwargs):
        t0 = time.time()
        out = super().training_step(*args, **kwargs)
        t1 = time.time()

        step = self.state.global_step
        print(f"[STEP {step}] time={t1 - t0:.2f}s")

        return out


if __name__ == "__main__":
    print(f"Loading {LANG} dataset")
    train, test = load_datasets()

    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("Converting to correct dataset format")
    train = build_grpo_dataset(train, tokenizer)
    test = build_grpo_dataset(test, tokenizer)

    train = Dataset.from_list(train)
    test = Dataset.from_list(test)

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("Initializing GRPO Config")
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=os.environ.get("VLLM_HOST"),
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.85,
        output_dir=GRPO_RUNS,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=10,
        report_to="tensorboard",
        logging_dir="logs/tb",
        gradient_checkpointing=True,
        num_generations=8,
        max_completion_length=64,
        beta=0,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        generation_kwargs={
            "temperature":0.9,
            "stop": ["\n\n", "\n \n"],
        },
    )

    print("Initializing trainer")
    trainer = TimedTrainer(
        model=model,
        reward_funcs=compute_reward,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
    )

    model_test = trainer.model if hasattr(trainer, "model") else None
    if model_test is not None:
        dtypes = set(p.dtype for p in model_test.parameters())
        print("MODEL DTYPES:", dtypes)

    print("CUDA memory before train():")
    print(torch.cuda.memory_summary())

    print("Trainer model:", type(trainer.model))
    if hasattr(trainer, "ref_model"):
        print("Reference model exists:", trainer.ref_model is not None)

    print("Starting training!!!")
    trainer.train()
    
    trainer.save_model(OUT_MODEL)
    tokenizer.save_pretrained(OUT_MODEL)
