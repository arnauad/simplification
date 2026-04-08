import evaluate
import random
import math
#from sentence_transformers import SentenceTransformer
#import torch.nn.functional as F

sari_metric = evaluate.load("/home/aayguade/simplification/training/sari")
#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

def extract_text(c):
        if isinstance(c, dict):
            return c.get("content", "")
        return c

def compute_similarity(predictions, references):
    emb1 = model.encode(predictions, batch_size=32, convert_to_tensor=True)
    emb2 = model.encode(references, batch_size=32, convert_to_tensor=True)

    cos = F.cosine_similarity(emb1, emb2).cpu().tolist()

    cos = [(c + 1) / 2 for c in cos]

    return cos

def length_penalty(predictions, references, alpha=2.0, tolerance=3):
    penalties = []

    for pred, ref in zip(predictions, references):
        pred_len = len(pred.split())
        ref_len = len(ref.split())

        if ref_len == 0:
            penalties.append(1.0)
            continue

        # Only penalize if exceeding tolerance
        excess = pred_len - ref_len

        if excess <= tolerance:
            penalties.append(1.0)
        else:
            # Normalize excess relative to reference length
            ratio_excess = (excess - tolerance) / ref_len
            penalty = math.exp(-alpha * ratio_excess)
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
            if pred == src and src != ref:
                score = max(score - 0.5, 0.0)

            # Penalize empty outputs
            if len(pred.strip()) == 0:
                score = 0.0

            scores.append(float(score))

        except Exception:
            # Robust fallback
            scores.append(0.0)

    return scores


def compute_reward(prompts, completions, completion_ids=None, **kwargs):

    predictions = [extract_text(c) for c in completions]

    sources = kwargs["source"]
    references = kwargs["reference"]

    sari_scores = compute_sari(sources, predictions, references)
    length_penalties = length_penalty(predictions, references)

    # Combine multiplicatively
    final_scores = [
        s * lp for s, lp in zip(sari_scores, length_penalties)
    ]

    avg = sum(final_scores) / len(final_scores)

    if random.random() < 0.05:
        print(f"[REWARD] sari={sum(sari_scores)/len(sari_scores):.3f} | final={avg:.3f}")

        # also randomly pick one example to print
        idx = random.randint(0, len(predictions) - 1)
        print("------------")
        print(f"Original : {sources[idx]}")
        print(f"Reference: {references[idx]}")
        print(f"Output   : {predictions[idx]}")
        print("------------")


    if "log_metric" in kwargs:
        kwargs["log_metric"]("sari", sum(sari_scores) / len(sari_scores))
        kwargs["log_metric"]("reward", avg)

    return [float(s) for s in final_scores]