import os
import json
from itertools import combinations
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import RULES, SYSTEM_PROMPT_RULES, USER_TEMPLATE, FEW_SHOTS

DATA = "/data/upftfg34/aayguade/dataset/"
OUT_DIR = os.path.join(DATA, "rule_search_salamandra")
MODEL = "/data/upftfg34/aayguade/models/salamandra-7b-instruct"

BATCH_SIZE = 512
MAX_RULES = 4


def get_dataset():
    dataset_path = os.path.join(DATA, "CAT_semantic.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def build_prompt(tokenizer, sentence, rules):
    rules_text = "\n".join(f"- {r}" for r in rules)

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT_RULES.format(rules_text=rules_text)
    }]

    # Few-shot examples
    for ex in FEW_SHOTS:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Actual query
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(sentence=sentence)
    })

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def construct_prompts(dataset, tokenizer, rules):
    prompts = []
    index_map = []

    for idx, item in enumerate(dataset):
        sentence = item["original_sentence"]

        prompt = build_prompt(tokenizer, sentence, rules)

        prompts.append(prompt)
        index_map.append(idx)

    return index_map, prompts


def inference(llm, sampling_params, tokenizer, dataset, rules):
    index_map, prompts = construct_prompts(dataset, tokenizer, rules)

    predictions = []

    for start in range(0, len(prompts), BATCH_SIZE):
        end = start + BATCH_SIZE

        batch_prompts = prompts[start:end]
        batch_indices = index_map[start:end]

        batch_outputs = llm.generate(batch_prompts, sampling_params)

        for output, data_idx in zip(batch_outputs, batch_indices):
            item = dataset[data_idx]

            text = output.outputs[0].text.strip()
            if not text:
                text = "No generació"

            predictions.append({
                "sample_id": item["sample_id"],
                "original_sentence_id": item["original_sentence_id"],
                "original_sentence": item["original_sentence"],
                "simplified_sentence": text
            })

    return predictions


def save_predictions(predictions, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved generations to: {out_path}")


def generate_rule_sets(rules, max_rules=4):
    all_sets = []
    for r in range(1, max_rules + 1):
        print(f"Rule size: {r}")
        for combo in combinations(rules, r):
            all_sets.append(list(combo))
    return all_sets


WINNERS = [1921, 1371, 1144]

def print_winners(rule_sets):
    for winner in WINNERS:
        print(f"Rule set {winner} is {rule_sets[winner]}")


if __name__ == "__main__":
    print("Initializing sampling params...")
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=200,
    )

    print("Initializing LLM...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("Loading dataset...")
    dataset = get_dataset()

    print("Generating rule sets...")
    rule_sets = generate_rule_sets(RULES, max_rules=MAX_RULES)

    print_winners(rule_sets)

    """print("Starting inference...")
    for i, rule_set in enumerate(rule_sets):

        predictions = inference(
            llm,
            sampling_params,
            tokenizer,
            dataset,
            rule_set
        )

        out_path = os.path.join(OUT_DIR, f"{i}.json")
        save_predictions(predictions, out_path)"""

    