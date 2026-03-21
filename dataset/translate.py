import os
import json
from vllm import LLM, SamplingParams

DATA = "/data/upftfg34/aayguade/dataset/"
BATCH_SIZE = 128

PROMPT_TEMPLATE = """\
Ets un assistent que tradueix textos del català a l'anglès.
Tradueix la frase següent mantenint exactament el mateix significat.
Retorna NOMÉS la frase traduïda.

Frase:
{sentence}

Traducció:
"""

def get_dataset():
    dataset_path = os.path.join(DATA, "CAT_processed.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


if __name__ == "__main__":

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        seed=1,
        max_tokens=200,
        repetition_penalty=1.2,
        stop=["\n"]
    )

    llm = LLM(
        model="/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct",
        gpu_memory_utilization=0.85,
    )

    dataset = get_dataset()

    prompts = []
    index_map = []  # (data_idx, field_name)

    for idx, item in enumerate(dataset):
        # Translate original sentence
        if "original_sentence" in item:
            prompts.append(
                PROMPT_TEMPLATE.format(sentence=item["original_sentence"])
            )
            index_map.append((idx, "original_sentence_en"))

        # Translate simplified sentence
        if "simplified_sentence" in item:
            prompts.append(
                PROMPT_TEMPLATE.format(sentence=item["simplified_sentence"])
            )
            index_map.append((idx, "simplified_sentence_en"))

    for start in range(0, len(prompts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_prompts = prompts[start:end]

        batch_outputs = llm.generate(batch_prompts, sampling_params)

        for output, (data_idx, target_field) in zip(
            batch_outputs, index_map[start:end]
        ):
            translation = output.outputs[0].text.strip()

            if not translation:
                translation = "No translation"

            dataset[data_idx][target_field] = translation

    output_json = os.path.join(DATA, "og_results_translated_Ib.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved translated dataset to: {output_json}")
