from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("facebook/asset", "ratings")

dataset = ds["full"]

new_data = []

count = 0

for example in dataset:
    original = example["original"]
    simplification = example["simplification"]
    rating = example["rating"]
    
    # for simpl in simplifications:
    #    new_data.append({
    #         "original": original,
    #         "simplification": simpl
    #     })
    if rating > 30:
        label = 1
        count += 1
    else:
        label = 0

    new_data.append({
        "original": original,
        "simplification": simplification,
        "label": label 
    })

with open("../data/asset_ratings.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f" {count} out of {len(new_data)} are accepted")
