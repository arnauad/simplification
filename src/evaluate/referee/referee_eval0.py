import torch
import json
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from metrics.deberta_for_eval import DebertaForEval
import src.evaluate.referee.globals as uglobals


def translate_data(data, src_lang='ca', tgt_lang='en'):

    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated_data = []

    for i, item in enumerate(data):
        complex_sent = item['original_sentence']
        simplified_sent = item['simplified_sentence']

        complex_tokens = tokenizer(complex_sent, return_tensors='pt')
        simplified_tokens = tokenizer(simplified_sent, return_tensors='pt')

        complex_translated = model.generate(**complex_tokens)
        simplified_translated = model.generate(**simplified_tokens)

        complex_translated_sent = tokenizer.decode(complex_translated[0], skip_special_tokens=True)
        simplified_translated_sent = tokenizer.decode(simplified_translated[0], skip_special_tokens=True)

        translated_data.append({
            'sample_id': item['sample_id'],
            'original_sentence_id': item['original_sentence_id'],
            'original_sentence': complex_translated_sent,
            'simplified_sentence': simplified_translated_sent
        })

        if (i) % 50 == 0:
            print(f'Translated {i + 1} / {len(data)} samples')

    return translated_data

def deberta_inference(data):
    results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DebertaForEval(uglobals.DERBERTA_DIR, uglobals.DERBERTA_DIR, device, head_type='linear')
    model.load_state_dict(torch.load(f'{uglobals.CHECKPOINTS_DIR}/pretrained_deberta.bin', map_location=device)['model_state_dict'], strict=False)
    model.eval()

    for i, item in enumerate(data):

        complex_sent = item['original_sentence']
        simplified_sent = item['simplified_sentence']

        model_input = [str(complex_sent) + ' ' + model.tokenizer.sep_token + ' ' + str(simplified_sent)]
        model_out = model(model_input)
        score = model_out[:, -1].item()

        results.append({'sample_id': item['sample_id'], 'original_sentence_id': item['original_sentence_id'], 'score': score})

        if (i) % 50 == 0:
            print(f'Evaluated {i + 1} / {len(data)} samples')

    return results


if __name__ == '__main__':
    translated_data = [{"sample_id": 0, "original_sentence_id": 0, "original_sentence": "A research and supervision of the municipal performance is started regarding a problem in particular.", "simplified_sentence": "The union starts an investigation. To see that the city council is doing a specific problem."},]

    #translated_data = translate_data(data, src_lang='ca', tgt_lang='en')


    results = deberta_inference(translated_data)

    #print(f"Original: {data[0]['original_sentence']}, Simplified: {data[0]['simplified_sentence']}")
    print(f"Translated: {translated_data[0]['original_sentence']}, Simplified: {translated_data[0]['simplified_sentence']}")
    print(f"Deberta Score: {results[0]['score']}")

    
    
    