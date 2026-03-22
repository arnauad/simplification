import json
import evaluate
import numpy as np
from collections import defaultdict

MODEL_NAME = "CAT_iberian_best"
DATASET = "../data/CAT_semantic.json"
MODEL = f"../data/{MODEL_NAME}.json"



def load_data():
    # Load data
    with open(DATASET, "r", encoding="utf-8") as f:
        sources_data = json.load(f)

    with open(MODEL, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    return sources_data, predictions_data


def build_lookup(predictions_data):
    # Build lookup: now storing list of generations
    pred_lookup = {
        (item["sample_id"], item["original_sentence_id"]): item["simplified_sentences"]
        for item in predictions_data
    }

    return pred_lookup


def compute_per_generation_results(sari, sources_data, pred_lookup):
    per_generation_results = []
    per_sample_aggregated = defaultdict(list)

    total = len(sources_data)
    processed = 0

    for item in sources_data:
        key = (item["sample_id"], item["original_sentence_id"])

        if key not in pred_lookup or key in [(7, 42), (10, 76), (13, 21)]:
            continue

        source = item["original_sentence"]
        reference = item["simplified_sentence"]
        predictions = pred_lookup[key]

        for pred in predictions:
            result = sari.compute(
                sources=[source],
                predictions=[pred],
                references=[[reference]]
            )

            sari_score = result["sari"]


            per_generation_results.append({
                "sample_id": item["sample_id"],
                "original_sentence_id": item["original_sentence_id"],
                "original_sentence": source,
                "prediction": pred,
                "reference": reference,
                "sari": sari_score
            })

            per_sample_aggregated[key].append(sari_score)

        processed += 1
        print(f"Processed {processed} / {total}")

    return per_generation_results, per_sample_aggregated


def compute_per_sample_results(per_sample_aggregated):
    per_sample_results = []

    for (sample_id, sent_id), scores in per_sample_aggregated.items():
        avg_sample_sari = np.mean(scores)

        per_sample_results.append({
            "sample_id": sample_id,
            "original_sentence_id": sent_id,
            "avg_sari": avg_sample_sari,
            "num_generations": len(scores)
        })

    return per_sample_results


def compute_global_statistics(per_generation_results):
    all_sari_scores = [r["sari"] for r in per_generation_results]

    global_avg = np.mean(all_sari_scores)
    global_std = np.std(all_sari_scores)

    return global_avg, global_std


def save_full_results(per_generation_results, per_sample_results, global_avg, global_std):
    print(f"\n{MODEL_NAME} Global Avg SARI (all generations): {global_avg:.3f}")
    print(f"{MODEL_NAME} Global Std Dev: {global_std:.3f}")

    output = {
        "per_generation_results": per_generation_results,
        "per_sample_average": per_sample_results,
        "statistics": {
            "global_average_sari": global_avg,
            "global_std_sari": global_std
        }
    }

    with open(f"results/{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def eval(sari, sources_data, pred_lookup):
    per_generation_results, per_sample_aggregated = compute_per_generation_results(
        sari, sources_data, pred_lookup
    )

    per_sample_results = compute_per_sample_results(per_sample_aggregated)

    global_avg, global_std = compute_global_statistics(per_generation_results)

    save_full_results(
        per_generation_results,
        per_sample_results,
        global_avg,
        global_std
    )


if __name__ == '__main__':
    sources_data, predictions_data = load_data()

    pred_lookup = build_lookup(predictions_data)

    sari = evaluate.load("sari")

    eval(sari, sources_data, pred_lookup)


