import re
import json

NAME = "CAT_salamandra_shot"

def clean_simplifications(dataset):
    for sample in dataset:
        sample["simplified_sentences"] = [
            re.split(r'\n\s*\n', s)[0].strip()
            for s in sample.get("simplified_sentences", [])
        ]
    return dataset


with open(f'../data/{NAME}.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

clean_dataset = clean_simplifications(dataset)

with open(f'../data/{NAME}.json', 'w', encoding='utf-8') as f:
    json.dump(clean_dataset, f, ensure_ascii=False, indent=4)