import json

# Load your JSON file
with open("../data/asset_ratings_ca.json", "r", encoding="utf-8") as f:
    data = json.load(f)

processed_data = []

for entry in data:
    # Take only the part before the first newline
    original = entry["original"].split("\n")[0]
    simplification = entry["simplification"].split("\n")[0]
    label = entry["label"]

    # Create new entry with renamed keys
    new_entry = {
        "original": original,
        "simplification": simplification,
        "label": label
    }
    processed_data.append(new_entry)

# Save back to JSON
with open("../data/asset_ratings_ca.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print("Processing complete")