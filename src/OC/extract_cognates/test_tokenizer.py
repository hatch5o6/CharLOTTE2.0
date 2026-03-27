from transformers import AutoTokenizer
from sloth_hatch.sloth import read_lines

# ARAGONESE
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", model_max_length=512)
data_file = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/Aragonese/Comprised/an.7k.txt"
sents = read_lines(data_file)
print("SENTS:", len(sents))
encodings = tokenizer(sents, add_special_tokens=False)
total = sum(len(ids) for ids in encodings["input_ids"]) 
print("TOTAL TOKENS:", total)

# MFE
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base", model_max_length=512)
data_file = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/MauritianCreole/Comprised/mfe.7k.txt"
sents = read_lines(data_file)
print("SENTS:", len(sents))
encodings = tokenizer(sents, add_special_tokens=False)
total = sum(len(ids) for ids in encodings["input_ids"]) 
print("TOTAL TOKENS:", total)
