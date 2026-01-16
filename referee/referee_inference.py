import torch
import json
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

from models.deberta_for_eval import DebertaForEval
import utils.globals as uglobals


def load_data():

<<<<<<< HEAD
    with open(uglobals.DATA_DIR_ENGLISH, 'r', encoding='utf-8') as f:
=======
    with open(uglobals.DATA_DIR, 'r', encoding='utf-8') as f:
>>>>>>> f64d09b27b422c05b55fbba9df02f464027f50dc
        data = json.load(f)
    
    return data


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
<<<<<<< HEAD
    translated_data = load_data()
    #data = [{"sample_id": 0, "original_sentence_id": 0, "original_sentence": "Àpats a domicili El servei d'àpats a domicili dona suport alimentari a aquelles persones que, per la seva situació personal, no poden preparar-se els àpats, necessiten ajuda per fer-ho o no tenen un habitatge en condicions per cuinar aliments.", "simplified_sentence": "Menjars a domicili Hi ha un servei d'ajuda que porta el menjar a casa. Aquest servei és per a persones que tenen problemes per cuinar. Aquests problemes poden ser: -  per la seva situació personal no poden preparar el menjar. - Necessiten ajuda per cuinar. - La cuina de casa seva està en mal estat."}]

    """translated_data = translate_data(data)"""

    # Store translated data
    """with open(f'../data/CAT_processed_translated.json', 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)"""
    
    results = deberta_inference(translated_data)

    #print(f"Original: {data[0]['original_sentence']}, Simplified: {data[0]['simplified_sentence']}")
=======
    data = load_data()
    #data = [{"sample_id": 0, "original_sentence_id": 0, "original_sentence": "Àpats a domicili El servei d'àpats a domicili dona suport alimentari a aquelles persones que, per la seva situació personal, no poden preparar-se els àpats, necessiten ajuda per fer-ho o no tenen un habitatge en condicions per cuinar aliments.", "simplified_sentence": "Menjars a domicili Hi ha un servei d'ajuda que porta el menjar a casa. Aquest servei és per a persones que tenen problemes per cuinar. Aquests problemes poden ser: -  per la seva situació personal no poden preparar el menjar. - Necessiten ajuda per cuinar. - La cuina de casa seva està en mal estat."}]

    translated_data = translate_data(data)
    
    results = deberta_inference(translated_data)

    print(f"Original: {data[0]['original_sentence']}, Simplified: {data[0]['simplified_sentence']}")
>>>>>>> f64d09b27b422c05b55fbba9df02f464027f50dc
    print(f"Translated Original: {translated_data[0]['original_sentence']}, Translated Simplified: {translated_data[0]['simplified_sentence']}")
    print(f"Deberta Score: {results[0]['score']}")

    average_score = np.mean([item['score'] for item in results])
    std_score = np.std([item['score'] for item in results])
    print(f"Average Deberta Score: {average_score}, Std Dev: {std_score}")
<<<<<<< HEAD

    with open(f'./results/results_referee.json', 'w') as r:
        json.dump(results)

    # Average Deberta Score: -0.115781185131422, Std Dev: 0.7842857257657601
=======
>>>>>>> f64d09b27b422c05b55fbba9df02f464027f50dc
    
    
    