# pip install git+https://github.com/google-research/bleurt.git
from bleurt import score
import json
import numpy as np
import os
import tensorflow as tf


DATAPATH = "../data/CAT_salamandra2B_base_shot.json"
OUTPUT_PATH = "results/CAT_salamandra2B_base_shot.json"
CHECKPOINT = "./bleurt-20"

def load_data():

    with open(DATAPATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


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


def bleurt_inference(data, batch_size=32):
    scorer = score.BleurtScorer(CHECKPOINT)
    results = []

    references = []
    candidates = []
    meta = []

    for item in data:
        references.append(item["original_sentence"])
        candidates.append(item["simplified_sentence"])
        meta.append((item["sample_id"], item["original_sentence_id"]))

    for i in range(0, len(data), batch_size):
        batch_refs = references[i:i + batch_size]
        batch_cands = candidates[i:i + batch_size]

        scores = scorer.score(
            references=batch_refs,
            candidates=batch_cands,
            batch_size=batch_size
        )

        for j, s in enumerate(scores):
            results.append({
                "sample_id": meta[i + j][0],
                "original_sentence_id": meta[i + j][1],
                "score": float(s)
            })

        
        print(f"Evaluated {i + batch_size} / {len(data)}")

    return results


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print("Checkpoint exists:", os.path.exists(CHECKPOINT))
    print("Contents:", os.listdir(CHECKPOINT))

    data = load_data()

    """target = [
        item for item in data
        if item["sample_id"] == 3
        and item["original_sentence_id"] == 8
    ]"""

    results = bleurt_inference(data)

    average_score = np.mean([item['score'] for item in results])
    std_score = np.std([item['score'] for item in results])
    print(f"Average Bleurt Score: {average_score}, Std Dev: {std_score}")

    save_results_to_json(results, OUTPUT_PATH, average_score=average_score, std_score=std_score)
    
