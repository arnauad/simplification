from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("facebook/asset", "simplification")

dataset = ds["validation"]

new_data = []

for example in dataset:
    original = example["original"]
    simplifications = example["simplifications"]
    
    for simpl in simplifications:
        new_data.append({
            "original": original,
            "simplification": simpl
        })

with open("../data/asset.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(new_data)} samples.")