import csv

def read_data_csv(f):
    data = {}
    with open(f, newline='') as inf:
        for i, row in enumerate(csv.reader(inf)):
            if i == 0:
                if row != ["src_lang", "tgt_lang", "src_path", "tgt_path"]:
                    raise ValueError(f"Data file `{f}` does not have the header `src_lang,tgt_lang,src_path,tgt_path`.")
                continue
            src_lang, tgt_lang, src_path, tgt_path = row
            lang_pair = src_lang, tgt_lang
            if lang_pair in data:
                raise ValueError(f"Duplicate {lang_pair} in data file {f}.")
            data[lang_pair] = src_path, tgt_path
    return data