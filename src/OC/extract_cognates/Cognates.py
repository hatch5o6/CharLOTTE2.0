import os

from OC.utilities.word_preprocessing import clean
from OC.utilities.utilities import NLD, write_oc_data, read_oc_data

def make_cognates(
    src_path:str,
    tgt_path:str,
    src_lang:str,
    tgt_lang:str,
    out_stem:str,
    long_enough:int,
    theta:float,
    extract_candidates,
    return_cognates:bool=True
):
    """
    src_path: path to source corpus
    tgt_path: path to target corpus
    src_lang: source language iso code
    tgt_lang: target language iso code
    out_stem: file stem
    long_enough: minimum length of words in cognate pair
    theta: NLD threshold
    extract_candidates: Function to extract candidates. Either CandidatesFromParallel.extract_candidates or FuzzyCandidates.extract_candidates
    return_cognates: If True, return list of cognate pairs. If False, return path to file of cognate pairs.
    """

    # get candidate word pairs
    word_list_out = out_stem + ".word_list"
    candidates = extract_candidates(
        src_file=src_path,
        tgt_file=tgt_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        word_list_out=word_list_out,
        long_enough=long_enough
    )

    # filter down to cognates
    cognates = _filter_cognate_pairs(
        word_pairs=candidates,
        theta=theta,
        long_enough=long_enough
    )
    cognate_list_out = out_stem + ".cognates"
    if os.path.exists(cognate_list_out):
        # if this file already exists, make sure it matches the cognates we just go
        assert read_oc_data(cognate_list_out) == cognates
    else:
        # otherwise, write the cognates to the file
        write_oc_data(cognates, cognate_list_out)
    
    if return_cognates:
        return cognates
    else:
        return cognate_list_out


def _filter_cognate_pairs(word_pairs, theta, long_enough):
    cognate_list = []
    decimal_pair_list = []
    for word_pair_item in word_pairs:
        assert len(word_pair_item) in [3, 5]
        if len(word_pair_item) == 3:
            word1, word2 = word_pair_item[1:]
            freq = word_pair_item[0],
        else:
            word1, word2, _ = word_pair_item[2:]
            freq = word_pair_item[:2]
        
        assert isinstance(freq, tuple)
        assert len(freq) in [1, 2]

        # Cleaning should be redundant
        word1 = clean(word1, long_enough=long_enough)
        word2 = clean(word2, long_enough=long_enough)
        if not word1:
            continue
        if not word2:
            continue
        
        decimal_pairs = _get_decimal_pairs(freq, word1, word2)
        if decimal_pairs:
            decimal_pair_list += decimal_pairs
        else:
            distance = NLD(word1, word2)
            if distance <= theta:
                cognate_list.append(freq + (word1, word2, distance))
    cognate_list += _consolidate_decimal_pairs(decimal_pair_list)
    return cognate_list

def _get_decimal_pairs(freq, word1, word2):
    assert len(freq) in [1, 2]
    if len(freq) == 1:
        freq1 = freq[0],
        freq2 = freq[0],
    else:
        freq1, freq2 = freq
        freq1 = freq1, freq1
        freq2 = freq2, freq2
    decimal_pairs = []
    if word1.isdecimal():
        decimal_pairs.append(freq1 + (word1, word1, 0.0))
    if word2.isdecimal():
        decimal_pairs.append(freq2 + (word2, word2, 0.0))
    return decimal_pairs

def _consolidate_decimal_pairs(decimal_pairs):
    decimal_freqs = {}
    for item in decimal_pairs:
        word1, word2, nld = item[-3:]
        freq = item[:-3]
        assert word1 == word2
        assert nld == 0.0
        assert len(freq) in [1, 2]

        if (word1, word2) not in decimal_freqs:
            decimal_freqs[(word1, word2)] = [0] if len(freq) == 1 else [0, 0]

        if len(freq) == 1:
            decimal_freqs[(word1, word2)][0] += freq[0]
        else:
            decimal_freqs[(word1, word2)][0] += freq[0]
            decimal_freqs[(word1, word2)][1] += freq[1]
            assert decimal_freqs[(word1, word2)][0] == decimal_freqs[(word1, word2)][1]
    new_decimal_pairs = [
        tuple(total_freq) + (w1, w2, 0.0)
        for (w1, w2), total_freq in decimal_freqs.items()
    ]
    return new_decimal_pairs
