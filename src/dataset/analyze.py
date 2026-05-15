import json

def build_original_lookup(original_data):
    """
    Build a dict keyed by (sample_id, original_sentence_id)
    """
    lookup = {}
    for item in original_data:
        key = (item['sample_id'], item['original_sentence_id'])
        lookup[key] = item
    return lookup


def filter_high_scoring_samples(results, original_lookup, threshold=0.5):
    """
    Keep only samples with score >= threshold
    """
    filtered_data = []

    for r in results:
        if r['score'] >= threshold:
            key = (r['sample_id'], r['original_sentence_id'])
            if key in original_lookup:
                filtered_data.append(original_lookup[key])

    print(f"Kept {len(filtered_data)} samples with score >= {threshold}")
    return filtered_data


if __name__ == '__main__':
    RESULTS_PATH = 'results/CAT_processed.json'
    ORIGINAL_PATH = '../data/CAT_processed.json'
    OUTPUT_PATH = '../data/CAT_semantic.json'

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        r = json.load(f)
    results = r['results']

    with open(ORIGINAL_PATH, 'r', encoding='utf-8') as f:
        original = json.load(f)

    original_lookup = build_original_lookup(original)

    filtered_dataset = filter_high_scoring_samples(
        results,
        original_lookup,
        threshold=0.5
    )

    # Save filtered dataset
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)

    print(f"Filtered dataset saved to {OUTPUT_PATH}")