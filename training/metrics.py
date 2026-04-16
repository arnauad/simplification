import evaluate
import random
import math
import re
import os
import time
import json
#from sentence_transformers import SentenceTransformer
#import torch.nn.functional as F

REWARDS = os.environ.get("REWARDS")

sari_metric = evaluate.load("/home/aayguade/simplification/training/utils/sari", keep_in_memory=True)
#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

def extract_text(c):
        if isinstance(c, dict):
            return c.get("content", "")
        return c

def clean_text(x):
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

def is_copy(pred, src, threshold=0.98):
    pred_n = clean_text(pred)
    src_n = clean_text(src)

    overlap = sum(p == s for p, s in zip(pred_n, src_n))
    ratio = overlap / max(len(src_n), 1)

    return ratio > threshold

def length_penalty(predictions, references, alpha=4.0, tolerance=5):
    penalties = []

    for pred, ref in zip(predictions, references):
        pred_len = len(pred.split())
        ref_len = len(ref.split())

        if ref_len == 0:
            penalties.append(1.0)
            continue

        excess = pred_len - ref_len

        if excess <= tolerance:
            penalties.append(1.0)
        else:
            penalty = math.exp(-alpha * (excess / ref_len))
            penalties.append(penalty)

    return penalties

def compute_sari(sources, predictions, references):
    scores = []

    for src, pred, ref in zip(sources, predictions, references):
        try:
            score = sari_metric.compute(
                sources=[src],
                predictions=[pred],
                references=[[ref]],
            )["sari"] / 100.0  # normalize to [0,1]

            # Penalize copying input when simplification is expected
            if is_copy(pred, src) and not is_copy(src, ref):
                score = max(score - 0.5, 0.0)
                print("(SARI penalization)")

            # Penalize empty outputs
            if len(pred.strip()) == 0:
                score = 0.0

            scores.append(float(score))

        except Exception as e:
            # Robust fallback
            print(f"Robust fallback on sample src: {src}, pred: {pred}, ref: {ref}")
            print(f"Exception: {type(e).__name__}: {e}")
            scores.append(0.0)

    return scores


def compute_reward(prompts, completions, completion_ids=None, **kwargs):
    predictions = [clean_text(extract_text(c)) for c in completions]
    sources = [clean_text(s) for s in kwargs["source"]]
    references = [clean_text(r) for r in kwargs["reference"]]

    sari_scores = compute_sari(sources, predictions, references)
    length_penalties = length_penalty(predictions, references)

    final_scores = [s * lp for s, lp in zip(sari_scores, length_penalties)]
    avg = sum(final_scores) / len(final_scores)

    # save per-sample rewards here
    
    os.makedirs(os.path.dirname(REWARDS), exist_ok=True)
    with open(REWARDS, "a", encoding="utf-8") as f:
        for i, score in enumerate(final_scores):
            rec = {
                "time": time.time(),
                "reward": float(score),
                "sari": float(sari_scores[i]),
                "length_penalty": float(length_penalties[i]),
                "source": sources[i],
                "prediction": predictions[i],
                "reference": references[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if "log_metric" in kwargs:
        kwargs["log_metric"]("sari", sum(sari_scores) / len(sari_scores))
        kwargs["log_metric"]("reward", avg)

    print(f"[REWARD RAW] {final_scores}")
    return [float(s) for s in final_scores]