# pip install git+https://github.com/google-research/bleurt.git
from bleurt import score
import json
import numpy as np
import os

DATAPATH = "../data/CAT_processed.json"
CHECKPOINT = "./bleurt-20"

def load_data():

    with open(DATAPATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


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

        if i % (batch_size * 10) == 0:
            print(f"Evaluated {min(i + batch_size, len(data))} / {len(data)}")

    return results


if __name__ == "__main__":
    print("Checkpoint exists:", os.path.exists(CHECKPOINT))
    print("Contents:", os.listdir(CHECKPOINT))

    data = load_data()

    results = bleurt_inference(data)

    average_score = np.mean([item['score'] for item in results])
    std_score = np.std([item['score'] for item in results])
    print(f"Average Bleurt Score: {average_score}, Std Dev: {std_score}")

    # English translation results
    # Average Bleurt Score: 0.558201331684464, Std Dev: 0.13276575498748552

    # Catalan original results
    # Average Bleurt Score: 0.6252419832505678, Std Dev: 0.14472353956661285
