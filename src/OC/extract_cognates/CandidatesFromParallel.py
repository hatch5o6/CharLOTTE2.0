import subprocess
from collections import Counter
from sloth_hatch.sloth import read_lines, write_lines, log_function_call

from OC.utilities.word_tokenizers import get_tokenizer
from OC.utilities.word_preprocessing import clean

@log_function_call
def extract_candidates(
    src_file,
    tgt_file,
    src_lang,
    tgt_lang,
    word_list_out,
    long_enough
):
    print("--CANDIDATES FROM PARALLEL--\n\n")
    # Read
    src_sents, tgt_sents = _read_parallel_sents(src_file, tgt_file)

    # Tokenize
    src_tokenizer = get_tokenizer(src_lang)
    tgt_tokenizer = get_tokenizer(tgt_lang)
    src_sents = _tokenize(src_sents, src_tokenizer)
    tgt_sents = _tokenize(tgt_sents, tgt_tokenizer)

    # Get alignments
    alignments = _fast_align(src_sents, tgt_sents, output_file=word_list_out)
    
    # Get word pairs
    sent_pairs = list(zip(src_sents, tgt_sents))
    word_pairs = _sort_word_pairs(_get_word_pairs(sent_pairs, alignments, long_enough=long_enough))

    # write file for visibility
    write_lines([str(w) for w in word_pairs], word_list_out + ".word_pairs")

    return word_pairs


def _read_parallel_sents(src_file, tgt_file):
    src_sents = read_lines(src_file)
    tgt_sents = read_lines(tgt_file)
    if len(src_sents) != len(tgt_sents):
        raise ValueError(f"Length src_file `{src_file}` ({len(src_sents)}) != length tgt_file `{tgt_file}` ({len(tgt_sents)}).")
    return src_sents, tgt_sents

def _tokenize(sentences, tokenizer):
    if not isinstance(sentences, list):
        raise ValueError("sentences must be a list of strings!")
    return [" ".join(tokenizer(sent)) for sent in sentences]

def _fast_align(src_sents, tgt_sents, output_file):
    sents_file = output_file + ".sents"
    _write_fast_align_sents(src_sents, tgt_sents, sents_file)

    # forward alignment
    result = subprocess.run(
        ['src/fast_align/build/fast_align',
         '-i', sents_file,
         '-d', '-o', '-v'],
         capture_output=True,
         text=True,
         check=True
    )
    fwd_alignment = result.stdout.splitlines()
    fwd_file = output_file + ".fwd"
    write_lines(fwd_alignment, fwd_file)

    # reverse alignment
    result = subprocess.run(
        ['src/fast_align/build/fast_align',
         '-i', sents_file,
         '-d', '-o', '-v', '-r'],
         capture_output=True,
         text=True,
         check=True
    )
    rev_alignment = result.stdout.splitlines()
    rev_file = output_file + ".rev"
    write_lines(rev_alignment, rev_file)

    # symmetric alignment
    result = subprocess.run(
        ['src/fast_align/build/atools',
         '-i', fwd_file,
         '-j', rev_file,
         '-c', 'grow-diag-final-and'],
         capture_output=True,
         text=True,
         check=True
    )
    sym_alignment = result.stdout.splitlines()
    sym_file = output_file + ".sym"
    write_lines(sym_alignment, sym_file)

    return sym_alignment

def _write_fast_align_sents(src_sents, tgt_sents, path):
    assert len(src_sents) == len(tgt_sents)
    with open(path, "w") as outf:
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            outf.write(f"{src_sent} ||| {tgt_sent}\n")

def _get_word_pairs(sent_pairs, alignments, long_enough):
    assert len(sent_pairs) == len(alignments)

    word_list = Counter()
    data = zip(sent_pairs, alignments)
    for ((src_sent, tgt_sent), word_alignments) in data:
        src_words = src_sent.split()
        tgt_words = tgt_sent.split()
        word_alignments = word_alignments.split()

        for word_alignment in word_alignments:
            w1, w2 = word_alignment.split("-")
            w1 = int(w1)
            w2 = int(w2)
            word_pair = (clean(src_words[w1], long_enough=long_enough), 
                         clean(tgt_words[w2], long_enough=long_enough))
            if None in word_pair:
                continue
            word_list[word_pair] += 1
    return word_list

def _sort_word_pairs(word_pairs):
    word_list_ordered = []
    for word_pair, ct in word_pairs.items():
        assert isinstance(word_pair, tuple)
        assert len(word_pair) == 2
        word_a, word_b = word_pair
        word_list_ordered.append((ct, word_a, word_b))
    word_list_ordered.sort(reverse=True)
    return word_list_ordered
