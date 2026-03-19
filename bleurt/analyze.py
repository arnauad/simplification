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


def print_low_scoring_samples(results, original_lookup, threshold=-2.0):
    low_samples = [r for r in results if r['score'] < threshold]

    print(f'Found {len(low_samples)} samples with score < {threshold}\n')

    for r in low_samples:
        key = (r['sample_id'], r['original_sentence_id'])

        original_item = original_lookup[key]

        print(
            f"Sample ID: {r['sample_id']}, Original Sentence ID: {r['original_sentence_id']} |\n "
            f"Original Sentence (original):   {original_item['original_sentence']}\n"
            f"Simplified Sentence (original):   {original_item['simplified_sentence']}\n"
            f"Score: {r['score']:.4f}\n\n"
        )

    return low_samples


def equal(results):
    count = 0
    for r in results:
        if r['simplified_sentence'] == r['original_sentence']:
            count += 1
    return count


if __name__ == '__main__':
    RESULTS_PATH = 'results/CAT_processed.json'
    ORIGINAL_PATH = '../data/CAT_processed.json'

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        r = json.load(f)
    results = r['results']

    with open(ORIGINAL_PATH, 'r', encoding='utf-8') as f:
        original = json.load(f)

    original_lookup = build_original_lookup(original)

    low_samples = print_low_scoring_samples(
        results,
        original_lookup,
        threshold= 0.5
    )
    """num_equal = equal(original)
    print(f'Number of equal sentences: {num_equal}, out of {len(original)} total samples.')"""

