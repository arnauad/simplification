import json

INPUT = "../data/CAT_catallama_shot.json"
OUTPUT = "../data/CAT_catallama_shot.json" # Keep the exact same name to overwrite the original file

# Load your dataset
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for new_id, sample in enumerate(data):
    new_sample = {}

    # New ID
    new_sample["id"] = new_id

    # Rename fields
    new_sample["original"] = sample.get("original_sentence")
    new_sample["simplification"] = sample.get("simplified_sentence")

    new_data.append(new_sample)

# Save result
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)