import json

with open('../data/asset_ratings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pairs = set()

for item in data:
    original = item.get("original", "").strip()
    simplification = item.get("simplification", "").strip()
    pairs.add((original, simplification))

print("Number of unique pairs:", len(pairs))