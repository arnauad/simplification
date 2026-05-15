import json

DATASET_SCORES = "../data/asset_es_score_filtered.json"
DATASET_SENTENCES = "../data/asset_es.json"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    scores = load_json(DATASET_SCORES)
    sentences = load_json(DATASET_SENTENCES)

    score_ids = {item["id"] for item in scores}

    filtered_sentences = [
        sentence for sentence in sentences
        if sentence["id"] in score_ids
    ]

    with open("../data/asset_es_filtered.json", "w", encoding="utf-8") as f:
        json.dump(filtered_sentences, f, indent=4, ensure_ascii=False)