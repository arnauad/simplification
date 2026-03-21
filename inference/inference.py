import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOTS

DATA = "/data/upftfg34/aayguade/dataset/"
MODEL = "/data/upftfg34/aayguade/models/salamandra-2b-instruct"
OUT_FILE = "CAT_salamandra7B.json"
BATCH_SIZE = 128


def get_dataset(tokenizer):
    dataset_path = os.path.join(DATA, "CAT_semantic.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def build_prompt(tokenizer, sentence):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    """# Add few-shot examples as conversation turns
    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})"""

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
        sentence = item["original_sentence"]

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

            dataset[data_idx]["simplified_sentences"] = generations

    return dataset

def save_dataset(dataset):
    output_json = os.path.join(DATA, OUT_FILE)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved simplified dataset to: {output_json}")


if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=200,
        n=5
    )

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    dataset = inference(llm, sampling_params, tokenizer)

    save_dataset(dataset)
    