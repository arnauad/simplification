import json
import evaluate
import numpy as np
import os

dataset_path = "../data/CAT_semantic.json"
predictions_folder = "../data/rule_search_catallama"
results = "results/rule_search/model_ranking_catallama.json"


sari = evaluate.load("sari")


def evaluate_model(predictions_path, sources_data, batch_size=32):
    
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    pred_lookup = {
        (item["sample_id"], item["original_sentence_id"]): item["simplified_sentence"]
        for item in predictions_data
    }

    per_sample_results = []

    sources_batch = []
    predictions_batch = []
    references_batch = []
    meta_batch = []

    for item in sources_data:
        key = (item["sample_id"], item["original_sentence_id"])

        if key not in pred_lookup:
            continue

        sources_batch.append(item["original_sentence"])
        predictions_batch.append(pred_lookup[key])
        references_batch.append([item["simplified_sentence"]])
        meta_batch.append(item)

        if len(sources_batch) == batch_size:

            result = sari.compute(
                sources=sources_batch,
                predictions=predictions_batch,
                references=references_batch
            )

            batch_sari = result["sari"]

            for m in meta_batch:
                per_sample_results.append(batch_sari)

            sources_batch = []
            predictions_batch = []
            references_batch = []
            meta_batch = []

    if sources_batch:
        result = sari.compute(
            sources=sources_batch,
            predictions=predictions_batch,
            references=references_batch
        )

        batch_sari = result["sari"]

        for m in meta_batch:
            per_sample_results.append(batch_sari)

    avg = np.mean(per_sample_results)
    std = np.std(per_sample_results)

    return avg, std


# Load dataset once
with open(dataset_path, "r", encoding="utf-8") as f:
    sources_data = json.load(f)


best_model = None
best_score = -1
all_results = []

for file in os.listdir(predictions_folder):

    if not file.endswith(".json"):
        continue

    model_path = os.path.join(predictions_folder, file)

    print(f"Evaluating {file}...")

    avg, std = evaluate_model(model_path, sources_data)

    print(f"{file} -> Avg SARI: {avg:.3f}")

    all_results.append({
        "model": file,
        "avg_sari": avg,
        "std": std
    })

    if avg > best_score:
        best_score = avg
        best_model = file


print("\n BEST MODEL ")
print(f"{best_model} with Avg SARI {best_score:.3f}")


with open(f"{results}", "w") as f:
    json.dump(sorted(all_results, key=lambda x: x["avg_sari"], reverse=True), f, indent=4)