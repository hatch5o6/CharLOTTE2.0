from collections import Counter
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
import time
from sloth_hatch.sloth import read_lines, write_lines, log_function_call

from OC.utilities.word_tokenizers import get_tokenizer
from OC.utilities.word_preprocessing import clean

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
def extract_candidates(
    src_file,
    tgt_file,
    src_lang,
    tgt_lang,
    word_list_out,
    long_enough,
    top_k=None
):
    print("--FUZZY CANDIDATES--\n\n")
    print(f"Getting {src_lang} words from {src_file}")
    src_words, src_cts = _get_words(src_file, src_lang, long_enough, top_k)
    print(f"Getting {tgt_lang} words from {tgt_file}")
    tgt_words, tgt_cts = _get_words(tgt_file, tgt_lang, long_enough, top_k)

    print(f"SRC_WORDS {src_lang}: {len(src_words)}")
    print(f"TGT_WORDS {tgt_lang}: {len(tgt_words)}")

    print("Calculating matrix")
    matrix = cdist(src_words, tgt_words, scorer=Levenshtein.normalized_distance, workers=-1)

    print("Getting word pairs")
    #Only get the best pairs for a given src_word and a given tgt_word. i.e. a src_word and a tgt_word should each only occur once.
    dists = []
    for s in range(len(src_words)):
        for t in range(len(tgt_words)):
            dists.append((matrix[s][t], src_words[s], src_cts[s], tgt_words[t], tgt_cts[t]))
    dists.sort()
    word_list = set()
    used_src = set()
    used_tgt = set()
    for dist, src_w, src_ct, tgt_w, tgt_ct in dists:
        if src_w not in used_src and tgt_w not in used_tgt:
            word_list.add((src_ct, tgt_ct, src_w, tgt_w, dist))
            used_src.add(src_w)
            used_tgt.add(tgt_w)
    
    word_list = sorted(word_list, reverse=True)

    write_lines([str(item) for item in word_list], word_list_out + ".word_pairs")

    return word_list


@time_function
def _get_words(file_path, lang, long_enough, top_k=None):
    tokenizer = get_tokenizer(lang)
    lines = read_lines(file_path)
    words = Counter()
    for line in lines:
        line_words = tokenizer(line)
        for w in line_words:
            w = clean(w, long_enough=long_enough)
            if w:
                words[w] += 1
    words = [(ct, w) for w, ct in words.items()]
    words.sort(reverse=True)
    words = words[:top_k]
    cts = [ct for ct, w in words]
    ws = [w for ct, w in words]
    return ws, cts

