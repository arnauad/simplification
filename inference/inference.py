import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOTS, SYSTEM_PROMPT_BEST_RULES

DATA = "/data/upftfg34/aayguade/dataset/"
BASE_MODEL = "/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct"
MODEL = "/home/aayguade/simplification/training/trainer_output/checkpoint-4440"
OUT_FILE = "/home/aayguade/simplification/inference/results/new_model_test_ca.json"
BATCH_SIZE = 128


def get_dataset(tokenizer):
    dataset_path = os.path.join(DATA, "test_CA.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def build_prompt(tokenizer, sentence):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_BEST_RULES}]

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


def construct_prompts(dataset, tokenizer):
    prompts = []
    index_map = []

    for idx, item in enumerate(dataset):
        sentence = item["original"]

        prompt = build_prompt(tokenizer, sentence)

        prompts.append(prompt)
        index_map.append(idx)

    return index_map, prompts


def inference(llm, sampling_params, tokenizer):
    dataset = get_dataset(tokenizer)

    index_map, prompts = construct_prompts(dataset, tokenizer)

    for start in range(0, len(prompts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_prompts = prompts[start:end]

        batch_outputs = llm.generate(batch_prompts, sampling_params)

        for output, data_idx in zip(batch_outputs, index_map[start:end]):
            generations = []

            for candidate in output.outputs:
                text = candidate.text.strip()
                if not text:
                    text = "No generació"
                generations.append(text)

            dataset[data_idx]["simplification"] = generations

        print(f"Processed batch from {start} to {end}")

    return dataset

def save_dataset(dataset):
    output_json = os.path.join(DATA, OUT_FILE)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved simplified dataset to: {output_json}")


if __name__ == "__main__":
    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


    print("Sampling params")
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=80,
        n=1,
        stop=["\n\n", "\n \n", "\n"]
        #stop_token_ids=[tokenizer.eos_token_id]
    )

    print("Loading model")
    llm = LLM(
        model=MODEL,
        tokenizer=BASE_MODEL,
        gpu_memory_utilization=0.85,
    )

    print("Runing inference")
    dataset = inference(llm, sampling_params, tokenizer)

    save_dataset(dataset)
    