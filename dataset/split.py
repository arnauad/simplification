import json
import random

INPUT_FILE = "../data/asset_ca_filtered.json"
TRAIN_FILE = "../data/train_ca.json"
TEST_FILE = "../data/test_ca.json"

SPLIT_RATIO = 0.9
SEED = 42

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    data = load_json(INPUT_FILE)

    random.seed(SEED)
    random.shuffle(data)

    split_idx = int(len(data) * SPLIT_RATIO)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    save_json(train_data, TRAIN_FILE)
    save_json(test_data, TEST_FILE)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")