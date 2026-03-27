from tqdm import tqdm
import subprocess
from collections import Counter
from tempfile import NamedTemporaryFile
from sloth_hatch.sloth import read_lines, read_content, write_content, write_lines, log_function_call
from eflomal import Aligner

from OC.utilities.word_tokenizers import get_tokenizer
from OC.utilities.word_preprocessing import clean, is_only_punct
from OC.utilities.utilities import NLD

aligner = Aligner()

def extract_cognates(
    src_file, 
    tgt_file, 
    src_lang, 
    tgt_lang,
    cognate_list_out,
    theta=0.5
):
    # Read
    src_sents, tgt_sents = read_parallel_sents(src_file, tgt_file)

    # Tokenize
    src_tokenizer = get_tokenizer(src_lang)
    tgt_tokenizer = get_tokenizer(tgt_lang)
    src_sents = tokenize(src_sents, src_tokenizer)
    tgt_sents = tokenize(tgt_sents, tgt_tokenizer)

    # Get alignments
    alignments = align(src_sents, tgt_sents, output_file=cognate_list_out)
    
    # Get word pairs
    sent_pairs = list(zip(src_sents, tgt_sents))
    word_pairs = sort_word_pairs(get_word_pairs(sent_pairs, alignments))
    #TODO for testing 
    write_lines([str(w) for w in word_pairs], cognate_list_out + ".word_pairs")

    # Filter by NLD
    cognates = get_cognates(word_pairs, theta=theta)
    #TODO for testing
    write_lines([str(c) for c in cognates], cognate_list_out + ".cognate_list")

    return cognates


def read_parallel_sents(src_file, tgt_file):
    src_sents = read_lines(src_file)
    tgt_sents = read_lines(tgt_file)
    if len(src_sents) != len(tgt_sents):
        raise ValueError(f"Length src_file `{src_file}` ({len(src_sents)}) != length tgt_file `{tgt_file}` ({len(tgt_sents)}).")
    return src_sents, tgt_sents

def tokenize(sentences, tokenizer):
    if not isinstance(sentences, list):
        raise ValueError("sentences must be a list of strings!")
    return [" ".join(tokenizer(sent)) for sent in sentences]

def align(src_sents, tgt_sents, output_file):
    # TODO For testing. Can delete later.
    write_lines(src_sents, output_file + ".src_sents")
    write_lines(tgt_sents, output_file + ".tgt_sents")

    with NamedTemporaryFile('w') as fwd_f, NamedTemporaryFile('w') as rev_f:
        aligner.align(
            src_input=src_sents,
            trg_input=tgt_sents,
            links_filename_fwd=fwd_f.name,
            links_filename_rev=rev_f.name
        )
        # TODO For testing. Can delete later.
        forward = read_content(fwd_f.name)
        reverse = read_content(rev_f.name)
        write_content(forward, output_file + ".fwd")
        write_content(reverse, output_file + ".rev")

        result = subprocess.run(
            ['src/fast_align/build/atools', 
             '-i', fwd_f.name, 
             '-j', rev_f.name,
             '-c', 'grow-diag-final-and'],
            capture_output=True,
            text=True,
            check=True
        )
        symmetrized = result.stdout.splitlines()

        # TODO For testing. Can delete later.
        write_lines(symmetrized, output_file + ".sym")
        # ^^^ DELETE ^^^
    return symmetrized

def get_word_pairs(sent_pairs, alignments, VERBOSE=False, STOP=None):
    word_list = Counter()

    data = list(zip(sent_pairs, alignments))
    ct_nbsp = 0
    for idx, ((src_sent, tgt_sent), word_alignments) in tqdm(enumerate(data), total=len(data)):
        if idx == STOP:
            break
        if VERBOSE:
            print(f"\n\n--------------------- ({idx}) ------------------------")
            print(src_sent)
            print(tgt_sent)
            print(word_alignments)
        # should already be tokenized and joined on whitespace, so no need for word_tokenize function
            
        # Found a NBSP in a fon (or ewe, but I think fon) sent, so we need to handle that.
        FOUND_NBSP = False
        if " " in src_sent:
            src_sent = src_sent.replace(" ", "<NBSP>")
            FOUND_NBSP = True
        if " " in tgt_sent:
            tgt_sent = tgt_sent.replace(" ", "<NBSP>")
            FOUND_NBSP = True

        if FOUND_NBSP:
            ct_nbsp += 1

        src_words = src_sent.split()
        src_words = replace_word_in_sent(src_words, "<NBSP>", " ")
        tgt_words = tgt_sent.split()
        tgt_words = replace_word_in_sent(tgt_words, "<NBSP>", " ")
        max_len = max(len(src_words), len(tgt_words))
        word_alignments = word_alignments.split()

        for word_alignment in word_alignments:
            w1, w2 = tuple(word_alignment.split("-"))
            w1, w2 = int(w1), int(w2)
            
            try:
                word_pair_x = (src_words[w1], tgt_words[w2])

                # don't add cognate pairs with the NBSP
                if " " not in word_pair_x:
                    word_pair_x = (clean(word_pair_x[0]), clean(word_pair_x[1]))
                    if not is_only_punct(word_pair_x[0]) and not is_only_punct(word_pair_x[1]):
                        word_list[word_pair_x] += 1
            except:
                print("CODE BROKEN:")
                print(f"\n\n--------------------- ({idx}) ------------------------")
                print("SRC_SENT:", src_sent)
                print("TGT_SENT:", tgt_sent)
                print(word_alignments)
                print(f"--Alignment {idx}--")
                print(word_alignment)
                print("SRC_WORDS:", len(src_words), src_words)
                print("TGT_WORDS:", len(tgt_words), tgt_words)
                exit()

    print(f"{ct_nbsp} sentence pairs have a NBSP")

    return word_list

def replace_word_in_sent(sent, word, replace_word):
    for idx, w in enumerate(sent):
        if w == word:
            sent[idx] = replace_word
    return sent

def sort_word_pairs(word_pairs):
    word_list_ordered = []
    for word_pair, ct in word_pairs.items():
        assert isinstance(word_pair, tuple)
        assert len(word_pair) == 2
        word_a, word_b = word_pair
        word_list_ordered.append((ct, word_a, word_b))
    word_list_ordered.sort(reverse=True)
    return word_list_ordered

def get_cognates(word_list, theta=0.5):
    cognate_list = set()
    for ct, word1, word2 in word_list:
        # I cleaned them in get_word_list instead. Makes more sense there.
        # word1 = clean(word1)
        # word2 = clean(word2)
        passed, distance = are_cognates(word1, word2, theta=theta)
        if passed:
            cognate_list.add((ct, word1, word2, distance))
    return sorted(cognate_list, reverse=True, key=lambda x: x[3])

def are_cognates(word1, word2, theta=0.5):
    if len(word1) == 0 or len(word2) == 0:
        return False, None
    distance = NLD(word1, word2)
    if distance <= theta:
        # I now only add word pairs that meet this condition to the word list to begin with
        # if not is_only_punct(word1) and not is_only_punct(word2):
        return True, distance
    return False, distance

