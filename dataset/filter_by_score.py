import json

SCORE_PATH = "../data/asset_es_score.json"
OUTPUT_PATH = "../data/asset_es_score_filtered.json"

if __name__ == "__main__":
    with open(SCORE_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = dataset["results"]
    
    new_results = [item for item in results if item.get("score", 0) > 0.7 and item.get("score", 0) < 0.9]
    
    output = {"results": new_results}
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)