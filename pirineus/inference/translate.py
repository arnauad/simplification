import os
import json
from vllm import LLM, SamplingParams

DATA_INPUT = "/data/upftfg34/aayguade/dataset/asset.json"
DATA_OUTPUT = "/data/upftfg34/aayguade/dataset/asset_es.json"

MODEL_DIR = "/data/upftfg34/aayguade/models/salamandraTA-7B-instruct"
MODEL_NAME = "salamandraTA_7B_inst_q4.gguf"
BATCH_SIZE = 512

PROMPT_TEMPLATE = """\
Translate the following text from English into Spanish.
English: {sentence}
Spanish:
"""

def get_dataset():
    with open(DATA_INPUT, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def make_prompts(dataset):
    prompts = []
    index_map = []

    for idx, item in enumerate(dataset):
        prompts.append(
            PROMPT_TEMPLATE.format(sentence=item["original"])
        )
        index_map.append((idx, "original"))

        prompts.append(
            PROMPT_TEMPLATE.format(sentence=item["simplification"])
        )
        index_map.append((idx, "simplification"))

    return prompts, index_map


if __name__ == "__main__":

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=200,
        n=1
    )

    llm = LLM(
        model=os.path.join(MODEL_DIR, MODEL_NAME),
        tokenizer=MODEL_DIR,
        gpu_memory_utilization=0.85,
    )

    dataset = get_dataset()

    prompts, index_map = make_prompts(dataset)

    for start in range(0, len(prompts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_prompts = prompts[start:end]

        batch_outputs = llm.generate(batch_prompts, sampling_params)

        for output, (data_idx, target_field) in zip(batch_outputs, index_map[start:end]):
            translation = output.outputs[0].text.strip()
            dataset[data_idx][target_field] = translation

    with open(DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved translated dataset to: {DATA_OUTPUT}")