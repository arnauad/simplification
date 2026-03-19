import json
import evaluate
import numpy as np

MODEL_NAME = "CAT_iberian_best"
dataset_path = "../data/CAT_processed.json"
model_path = f"../data/{MODEL_NAME}.json"

def save_results_to_json(results, output_path, average_score=None, std_score=None):
    output = {
        "results": results
    }

    if average_score is not None and std_score is not None:
        output["statistics"] = {
            "average_score": average_score,
            "std_dev": std_score,
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

with open(dataset_path, "r", encoding="utf-8") as f:
    sources_data = json.load(f)

with open(model_path, "r", encoding="utf-8") as f:
    predictions_data = json.load(f)

# Build lookup dictionary
pred_lookup = {
    (item["sample_id"], item["original_sentence_id"]): item["simplified_sentence"]
    for item in predictions_data
}

sari = evaluate.load("sari")

batch_size = 32
per_sample_results = []

sources_batch = []
predictions_batch = []
references_batch = []
meta_batch = []

for item in sources_data:
    key = (item["sample_id"], item["original_sentence_id"])
    if key not in pred_lookup:
        continue

    source = item["original_sentence"]
    reference = item["simplified_sentence"]
    prediction = pred_lookup[key]

    sources_batch.append(source)
    predictions_batch.append(prediction)
    references_batch.append([reference])
    meta_batch.append(item)

    if len(sources_batch) == batch_size:
        result = sari.compute(
            sources=sources_batch,
            predictions=predictions_batch,
            references=references_batch
        )

        print(f"Evaluated {len(per_sample_results) + batch_size} / {len(sources_data)}")

        batch_sari = result["sari"]

        for m in meta_batch:
            per_sample_results.append({
                "sample_id": m["sample_id"],
                "original_sentence_id": m["original_sentence_id"],
                "original_sentence": m["original_sentence"],
                "prediction": pred_lookup[(m["sample_id"], m["original_sentence_id"])],
                "reference": m["simplified_sentence"],
                "sari": batch_sari
            })

        sources_batch = []
        predictions_batch = []
        references_batch = []
        meta_batch = []

# Process remaining samples
if sources_batch:
    result = sari.compute(
        sources=sources_batch,
        predictions=predictions_batch,
        references=references_batch
    )

    batch_sari = result["sari"]

    for m in meta_batch:
        per_sample_results.append({
            "sample_id": m["sample_id"],
            "original_sentence_id": m["original_sentence_id"],
            "original_sentence": m["original_sentence"],
            "prediction": pred_lookup[(m["sample_id"], m["original_sentence_id"])],
            "reference": m["simplified_sentence"],
            "sari": batch_sari
        })

for r in per_sample_results:
    print(f"Sample {r['sample_id']}-{r['original_sentence_id']}: SARI = {r['sari']}")

avg = np.mean([r["sari"] for r in per_sample_results])
std = np.std([r["sari"] for r in per_sample_results])
print(f"{MODEL_NAME} Avg SARI: {avg:.3f}, Std Dev: {std:.3f}")

save_results_to_json(per_sample_results, f"results/{MODEL_NAME}.json", average_score=avg, std_score=std)