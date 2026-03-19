import json
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def plot_score_distribution(results, bins=50, title='Distribution of SARI Scores for IberianLLM-7B-Instruct EtR Few Shot'):
    scores = [item['sari'] for item in results]

    mean = np.mean(scores)
    std = np.std(scores)
    
    plt.figure()
    plt.hist(scores, bins=bins)
    plt.axvline(mean, linestyle='--', linewidth=2, label='Mean')
    plt.axvline(mean + std, linestyle=':', linewidth=2, label='+1 Std')
    plt.axvline(mean - std, linestyle=':', linewidth=2, label='-1 Std')
    plt.legend()
    plt.xlabel('SARI Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    RESULTS_PATH = 'results/CAT_iberian_etr_shot.json'

    results = load_results(RESULTS_PATH)
    plot_score_distribution(results)
