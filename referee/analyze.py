import json
import os


def load_results(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def print_low_scoring_samples(results, threshold=-2.0):
    low_samples = [r for r in results if r['score'] < threshold]

    print(f'Found {len(low_samples)} samples with score < {threshold}\n')

    for r in low_samples:
        print(
            #f"Sample ID: {r['sample_id']} | "
            #f"Original Sentence ID: {r['original_sentence_id']} | "
            f"Original Sentence: {r['original_sentence']} \n "
            f"Simplified Sentence: {r['simplified_sentence']} \n "
            f"Score: {r['score']:.4f} \n\n"
        )

    return low_samples


if __name__ == '__main__':
    RESULTS_PATH = 'results/deberta_scores.json'

    results = load_results(RESULTS_PATH)
    low_samples = print_low_scoring_samples(results, threshold=-1.0)

