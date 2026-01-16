import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from Levenshtein import distance


LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
    "ita_Latn",
    "por_Latn",
    "nld_Latn",
    "ron_Latn",
    "swe_Latn",
]


def mark_og_sentences(data):
    for item in data:
        item["aug"] = "og"
    return data


def sentence_key(item):
    """Unique identifier for a sentence pair"""
    return (item["sample_id"], item["original_sentence_id"])


# Levenshtein distance
def compare_sentences_lev(sent1, sent2):
    dist = distance(sent1, sent2) 
    return dist == 0


def compare_sentences(sent1, sent2):
    return sent1 == sent2


def translate_sentences(
    data, model, tokenizer, src_lang, tgt_lang, device, batch_size=16
):
    translated_data = []

    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(data), batch_size):
        print(f"Batch {i}")
        batch = data[i:i + batch_size]

        complex_texts = [x["original_sentence"] for x in batch]
        simplified_texts = [x["simplified_sentence"] for x in batch]

        complex_inputs = tokenizer(
            complex_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        simplified_inputs = tokenizer(
            simplified_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            complex_out = model.generate(
                **complex_inputs,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )

            simplified_out = model.generate(
                **simplified_inputs,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )

        complex_decoded = tokenizer.batch_decode(
            complex_out, skip_special_tokens=True
        )
        simplified_decoded = tokenizer.batch_decode(
            simplified_out, skip_special_tokens=True
        )

        for j, item in enumerate(batch):
            translated_data.append({
                "sample_id": item["sample_id"],
                "original_sentence_id": item["original_sentence_id"],
                "original_sentence": complex_decoded[j],
                "simplified_sentence": simplified_decoded[j],
                "aug": tgt_lang
            })

    return translated_data


if __name__ == "__main__":

    with open("../data/CAT_processed.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data = mark_og_sentences(data)

    # Index original sentences by (sample_id, original_sentence_id)
    original_index = {
        sentence_key(item): item
        for item in data
    }

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    augmented_data = []

    for lang in LANGUAGES:
        print(f"Translating to {lang}...")

        # cat → X
        forward = translate_sentences(
            data, model, tokenizer,
            src_lang="cat_Latn",
            tgt_lang=lang,
            device=device
        )

        # X → cat
        backward = translate_sentences(
            forward, model, tokenizer,
            src_lang=lang,
            tgt_lang="cat_Latn",
            device=device
        )

        for item in backward:
            key = sentence_key(item)
            original_item = original_index.get(key)

            if original_item is None:
                continue

            # Skip exact copies
            if (
                compare_sentences(
                    original_item["original_sentence"],
                    item["original_sentence"]
                )
                and
                compare_sentences(
                    original_item["simplified_sentence"],
                    item["simplified_sentence"]
                )
            ):
                continue

            item["aug"] = f"bt_{lang}"
            augmented_data.append(item)

    # Append augmentations to dataset
    data.extend(augmented_data)

    # Save
    with open("../data/CAT_augmented.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
