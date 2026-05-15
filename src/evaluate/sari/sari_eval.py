import json
import evaluate
import numpy as np
from collections import defaultdict

MODEL_NAME = "iDEM-CA-Iberian-LLM-ES2"

# MODEL_NAME is the generations file
GENERATIONS_FILE = f"../../../data/inference/iDEM/{MODEL_NAME}.json"
REFERENCE_FILE = "../../../data/filtered/iDEM/iDEM-semantic.json"

def load_data():
    with open(GENERATIONS_FILE, "r", encoding="utf-8") as f:
        generations = json.load(f)

    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        references = json.load(f)

    return generations, references


def build_reference_lookup(references):
    return {
        item["id"]: {
            "original": item["original"],
            "reference": item["simplification"]
        }
        for item in references
    }


def compute_per_generation_results(sari, generations, ref_lookup):
    per_generation_results = []
    per_sample_aggregated = defaultdict(list)

    total = len(generations)
    processed = 0

    for item in generations:
        sample_id = item["id"]

        if sample_id not in ref_lookup:
            continue

        source = item["original"]
        reference = ref_lookup[sample_id]["reference"]
        predictions = item["simplification"]  # list

        for pred in predictions:
            result = sari.compute(
                sources=[source],
                predictions=[pred],
                references=[[reference]]
            )

            sari_score = result["sari"]

            per_generation_results.append({
                "id": sample_id,
                "original": source,
                "prediction": pred,
                "reference": reference,
                "sari": sari_score
            })

            per_sample_aggregated[sample_id].append(sari_score)

        processed += 1
        print(f"Processed {processed} / {total}")

    return per_generation_results, per_sample_aggregated


def compute_per_sample_results(per_sample_aggregated):
    per_sample_results = []

    for sample_id, scores in per_sample_aggregated.items():
        per_sample_results.append({
            "id": sample_id,
            "avg_sari": float(np.mean(scores)),
            "num_generations": len(scores)
        })

    return per_sample_results


def compute_global_statistics(per_generation_results):
    all_scores = [r["sari"] for r in per_generation_results]

    return float(np.mean(all_scores)), float(np.std(all_scores))


def save_results(per_generation_results, per_sample_results, global_avg, global_std):
    print(f"\n{MODEL_NAME} Global Avg SARI: {global_avg:.3f}")
    print(f"{MODEL_NAME} Std Dev: {global_std:.3f}")

    output = {
        "per_generation_results": per_generation_results,
        "per_sample_average": per_sample_results,
        "statistics": {
            "global_average_sari": global_avg,
            "global_std_sari": global_std
        }
    }

    with open(f"../../../data/eval/sari/{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def main():
    generations, references = load_data()
    ref_lookup = build_reference_lookup(references)

    sari = evaluate.load("sari")

    per_gen, per_sample = compute_per_generation_results(
        sari, generations, ref_lookup
    )

    per_sample_results = compute_per_sample_results(per_sample)
    global_avg, global_std = compute_global_statistics(per_gen)

    save_results(per_gen, per_sample_results, global_avg, global_std)


if __name__ == "__main__":
    main()