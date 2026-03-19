import json
from scipy.stats import spearmanr


def load_scores(path):
    """
    Loads a result file and returns a dict:
    (sample_id, original_sentence_id) -> score
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}
    for item in data["results"]:
        key = (item["sample_id"], item["original_sentence_id"])
        scores[key] = item["score"]

    return scores


def spearman_between(result_a, result_b):
    """
    Computes Spearman correlation between two result dicts
    """
    common_keys = sorted(set(result_a.keys()) & set(result_b.keys()))

    if not common_keys:
        raise ValueError("No overlapping samples found.")

    scores_a = [result_a[k] for k in common_keys]
    scores_b = [result_b[k] for k in common_keys]

    rho, p = spearmanr(scores_a, scores_b)
    return rho, p


def spearman_between_filtered(result_a, result_b, threshold=-1.0):
    """
    Computes Spearman correlation between two result dicts,
    keeping only samples where at least one score < threshold.
    """
    common_keys = set(result_a.keys()) & set(result_b.keys())

    filtered_keys = [
        k for k in common_keys
        if (result_a[k] < threshold or result_b[k] < threshold)
    ]

    if len(filtered_keys) < 2:
        raise ValueError("Not enough samples after filtering.")

    filtered_keys.sort()

    scores_a = [result_a[k] for k in filtered_keys]
    scores_b = [result_b[k] for k in filtered_keys]

    rho, p = spearmanr(scores_a, scores_b)
    return rho, p


if __name__ == "__main__":
    path_referee_en = "results/CAT_processed(opus).json"
    path_bleurt_cat = "../bleurt/results/CAT_processed.json"
    path_bleurt_en = "../bleurt/results/CAT_processed(opus).json"

    referee_en = load_scores(path_referee_en)
    bleurt = load_scores(path_bleurt_cat)
    bleurt_en = load_scores(path_bleurt_en)
    
    comparisons = [
        ("REFeREE (en) vs Bleurt", referee_en, bleurt),
        ("REFeREE (en) vs Bleurt (en)", referee_en, bleurt_en),
        ("Bleurt vs Bleurt (en)", bleurt, bleurt_en)
    ]

    for name, a, b in comparisons:
        rho, p = spearman_between(a, b)
        print(f"{name}: ρ = {rho:.4f}, p = {p:.2e}")

    """rho, p = spearman_between_filtered(cs, salamandra, threshold=-1.0)
    print(f"Filtered comparison: ρ = {rho:.4f}, p = {p:.2e}")"""