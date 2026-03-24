# pip install git+https://github.com/google-research/bleurt.git
from bleurt import score
import json
import numpy as np
import os
import tensorflow as tf
import uuid  # for unique IDs

DATAPATH = "../data/asset.json"
OUTPUT_PATH = "results/asset.json"
CHECKPOINT = "./bleurt-20"


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def assign_ids(data):
    for idx, item in enumerate(data):
        if "id" not in item:
            item["id"] = idx
    return data


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def bleurt_inference(data, checkpoint, batch_size=128):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading BLEURT model...")
    scorer = score.BleurtScorer(checkpoint)

    references = [item["original"] for item in data]
    candidates = [item["simplification"] for item in data]
    ids = [item["id"] for item in data]

    print(f"Scoring {len(data)} samples (batch_size={batch_size})...")

    scores = scorer.score(
        references=references,
        candidates=candidates,
        batch_size=batch_size
    )

    results = [
        {"id": ids[i], "score": float(s)}
        for i, s in enumerate(scores)
    ]

    return results


def compute_stats(results):
    scores = [item["score"] for item in results]
    return float(np.mean(scores)), float(np.std(scores))


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception as e:
            print("GPU setup warning:", e)


if __name__ == "__main__":
    setup_gpu()

    data = load_data(DATAPATH)

    data = assign_ids(data)

    save_json(data, DATAPATH)

    results = bleurt_inference(data, CHECKPOINT)

    avg, std = compute_stats(results)
    print(f"Average BLEURT Score: {avg:.6f}")
    print(f"Std Dev: {std:.6f}")

    output = {
        "results": results,
        "statistics": {
            "average_score": avg,
            "std_dev": std
        }
    }

    save_json(output, OUTPUT_PATH)