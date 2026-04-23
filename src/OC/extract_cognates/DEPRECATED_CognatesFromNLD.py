from collections import Counter
from tqdm import tqdm
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
from sloth_hatch.sloth import read_lines, read_content, write_content, write_lines, log_function_call

from OC.utilities.word_tokenizers import get_tokenizer
from OC.utilities.word_preprocessing import clean, is_only_punct
from OC.utilities.utilities import write_oc_data

import time

def time_function(f):
    def wrapper(*args, **kwargs):
        print(f"------ TIMING FUNCTION {f.__name__} ------")
        start = time.time()
        result = f(*args, **kwargs)
        total_time = time.time() - start
        print(f"------ FINISHED {f.__name__} IN {total_time} seconds ------\n\n")
        return result
    return wrapper

@time_function
@log_function_call
def extract_cognates(
    src_file,
    tgt_file,
    src_lang,
    tgt_lang,
    cognate_list_out,
    theta=0.5,
    top_k=None
):
    print("--EXTRACT COGNATES FROM NLD--\n\n")
    print(f"Getting {src_lang} words from {src_file}")
    src_words, src_cts = get_words(src_file, src_lang, top_k)
    print(f"Getting {tgt_lang} words from {tgt_file}")
    tgt_words, tgt_cts = get_words(tgt_file, tgt_lang, top_k)

    print(f"SRC_WORDS {src_lang}: {len(src_words)}")
    print(f"TGT_WORDS {tgt_lang}: {len(tgt_words)}")

    print("Calculating matrix")
    start = time.time()
    matrix = cdist(src_words, tgt_words, scorer=Levenshtein.normalized_distance, workers=-1)
    print("TOOK:", time.time() - start, "seconds")

    print("Getting cognates")
    #TODO only get the best pairs for a given src_word and a given tgt_word. i.e. a src_word and a tgt_word should each only occur once.
    dists = []
    for s in range(len(src_words)):
        for t in range(len(tgt_words)):
            dists.append((matrix[s][t], s, t))
    dists.sort()
    cognate_list = set()
    used_src = set()
    used_tgt = set()
    for dist, s, t in dists:
        src_word = src_words[s]
        src_ct = src_cts[s]
        tgt_word = tgt_words[t]
        tgt_ct = tgt_cts[t]
        if src_word not in used_src and tgt_word not in used_tgt:
            if dist <= theta:
                cognate_list.add((src_ct, tgt_ct, src_word, tgt_word, dist))
                used_src.add(src_word)
                used_tgt.add(tgt_word)

    cognate_list = sorted(cognate_list, key=lambda x: x[4])

    print("Writing cognates to", cognate_list_out)
    write_oc_data(cognate_list, cognate_list_out)
    return cognate_list

@time_function
def get_words(file_path, lang, top_k=None):
    tokenizer = get_tokenizer(lang)
    lines = read_lines(file_path)
    words = Counter()
    for line in tqdm(lines):
        line_words = [clean(w) for w in tokenizer(line)]
        line_words = [w for w in line_words if not is_only_punct(w)]
        for w in line_words:
            words[w] += 1
    words = [(ct, w) for w, ct in words.items()]
    words.sort(reverse=True)
    words = words[:top_k]
    cts = [ct for ct, w in words]
    ws = [w for ct, w in words]
    return ws, cts


if __name__ == "__main__":
    print("#####################################################################")
    print("EXTRACTING ES-AN COGNATES")
    extract_cognates(
        src_file="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/es-an/train.es.txt",
        tgt_file="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/es-an/train.an.txt",
        src_lang="es",
        tgt_lang="an",
        cognate_list_out="/home/hatch5o6/nobackup/archive/CharLOTTE2.0/test_NLD_an/es_an_cognates.txt",
        theta=0.5,
        top_k=20000
    )

    # print("\n\n\n#####################################################################")
    # print("EXTRACTING FR-MFE COGNATES")
    # extract_cognates(
    #     src_file="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/fr-en/train.fr.txt",
    #     tgt_file="/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/MauritianCreole/Comprised/mfe.txt",
    #     src_lang="fr",
    #     tgt_lang="mfe",
    #     cognate_list_out="/home/hatch5o6/nobackup/archive/CharLOTTE2.0/test_NLD_mfe/fr_mfe_cognates.txt",
    #     theta=0.5,
    #     top_k=20000
    # )
