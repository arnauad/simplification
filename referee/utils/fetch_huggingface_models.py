from transformers import AutoTokenizer, AutoModel
import globals as uglobals
import os

pretrained_dir = f'{uglobals.PRETRAINED_DIR}'

model_name = 'microsoft/deberta-v3-base'
save_name = 'deberta'


path = f'{pretrained_dir}/{save_name}'

print("Saving to:", os.path.abspath(path))
print("Exists before?", os.path.exists(path))

AutoTokenizer.from_pretrained(model_name, force_download=True).save_pretrained(path)
AutoModel.from_pretrained(model_name, force_download=True).save_pretrained(path)