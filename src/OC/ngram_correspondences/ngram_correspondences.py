from tqdm import tqdm
import argparse
import Levenshtein
from collections import defaultdict, Counter
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import ast
from sklearn.mixture import GaussianMixture
import subprocess
from scipy.optimize import brentq
import random
import os
from functools import lru_cache
from utilities.utilities import set_env

set_env()
ngram_dir = "src/OC/ngram_correspondences"

def main(lang_pair, training_data=False, applied_counts=None, frequency_threshold=10):
    # get data
    word_pairs, word_counts = get_data(training_data, applied_counts)
    ngram_pairs, ngram_counts = get_counts(word_pairs, word_counts)

    # filter identity and frequency
    kept_pairs, d_top = filter_ident_freq(ngram_pairs, frequency_threshold)

    # calculate entropy threshold
    entropy_threshold, best_k = get_entropy_threshold(lang_pair, training_data, kept_pairs, frequency_threshold)

    # filter by entropy and redundancy
    filtered_ngrams = filter_ent_red(ngram_pairs, kept_pairs, d_top, entropy_threshold, frequency_threshold)

    # print to a file
    filtered_dir = f'{lang_pair}/filtered'
    if training_data:
        filtered_dir = 'rule_based/' + filtered_dir
    filtered_dir = ngram_dir + '/' + filtered_dir
    filtered_f = f'{filtered_dir}/fr_{frequency_threshold}.gm_{best_k}.ent_{entropy_threshold:.2f}.csv'
    subprocess.call(['mkdir', '-p', filtered_dir])
    with open(filtered_f, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["entropy", "ngram", "alignments"])
        for ngram in sorted(filtered_ngrams, key=lambda x: (entropy(ngram_pairs[x]), -sum(ngram_pairs[x].values()))): # sorted by entropy, frequency
            alignments = dict(ngram_pairs[ngram])
            ent = entropy(ngram_pairs[ngram])
            if ent <= entropy_threshold:
                writer.writerow([round(ent, 2), ngram, dict(sorted(alignments.items(), key=lambda x: -x[1]))])  


def get_counts(word_pairs, word_counts):
    """Get all ngram transformations based on Levenshtein Alignment,
    while properly combining insertions and deletions"""
    ngram_pairs = defaultdict(lambda: defaultdict(int))
    ngram_counts = defaultdict(lambda: defaultdict(int))

    for pl_word in word_pairs:
        for n in range(1, len(pl_word) + 1):
            a1, a2 = get_alignments(f"${pl_word}$", f"${word_pairs[pl_word]}$")
            for c1, c2 in zip(ngrams(a1, n), ngrams(a2, n)):
                # check insertion (if so, will be accounted for at a higher context)
                idx = a1.find(c1)
                left = idx - 1
                right = idx + n
                if left in range(len(a1)):
                    if a1[left] == "_":
                        continue
                if right in range(len(a1)):
                    if a1[right] == "_":
                        continue
                # check deletion (if so, will be accounted for at a higher context)
                idx = a2.find(c2)
                left = idx - 1
                right = idx + n 
                if left in range(len(a2)):
                    if a2[left] == "_":
                        continue
                if right in range(len(a2)):
                    if a2[right] == "_":
                        continue
                ngram_pairs[c1.replace("_", "")][c2.replace("_", "")] += 1

                if word_counts is not None:
                    try:
                        ngram_counts[c1.replace("_", "")][c2.replace("_", "")] += word_counts[pl_word]
                    except KeyError:
                        pass

    return ngram_pairs, ngram_counts

def filter_ident_freq(ngram_pairs, frequency_threshold=10):
    kept = []
    d_top = []
    rows = sorted((len(i), entropy(ngram_pairs[i]), -sum(ngram_pairs[i].values()), i, dict(ngram_pairs[i])) for i in ngram_pairs) # sort by len, ent, freq
    for _, ent, freq, ngram, freq_distribution in rows:
        top = max(freq_distribution, key=freq_distribution.get)
        d_top.append((ngram, top))
        if -freq < frequency_threshold:
            continue
        if top == ngram:
            if (freq_distribution[top] / -freq) < .5:
                pass
            else:
                continue
        kept.append((ngram, top, ent))
    return kept, d_top

def get_entropy_threshold(lang_pair, training_data, kept_pairs, frequency_threshold):
    entropies = [ent for (ngram, top, ent) in kept_pairs if ent != 0]

    X = np.array(entropies).reshape(-1, 1)

    k_range = range(2, 20)
    bics = []
    models = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=0, reg_covar=1e-3, n_init=20)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)

    best_idx = np.argmin(np.array(bics))
    best_k = k_range[best_idx]
    gmm = models[best_idx]

    sort_idx = np.argsort(gmm.means_.flatten())
    sorted_means = gmm.means_.flatten()[sort_idx]
    sorted_stds = gmm.covariances_.flatten()[sort_idx]**0.5
    sorted_weights = gmm.weights_[sort_idx]

    x = np.linspace(min(entropies), max(entropies), 300)

    # find intersection of first two smaller components
    m1, m2 = sorted_means[0], sorted_means[1]
    s1, s2 = sorted_stds[0], sorted_stds[1]
    w1, w2 = sorted_weights[0], sorted_weights[1]

    def intersection(x):
        return (w1 * norm.pdf(x, m1, s1)) - (w2 * norm.pdf(x, m2, s2))
    
    if np.sign(intersection(m1)) == np.sign(intersection(m2)):
        x_search = np.linspace(m1, m2, 5000)
        diffs = np.abs(w1 * norm.pdf(x_search, m1, s1) - w2 * norm.pdf(x_search, m2, s2))
        entropy_threshold = x_search[np.argmin(diffs)]
    else:
        entropy_threshold = brentq(intersection, m1, m2)

    print(f"Entropy Threshold: {entropy_threshold}")

    threshold_density = w1 * norm.pdf(entropy_threshold, m1, s1)

    # Plot BIC scores
    print(f"Optimal compenets by BIC: {best_k}")
    plt.figure(figsize=(7, 4))
    plt.plot(k_range, bics, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    # plt.title('GMM component selection')
    plt.tight_layout()
    ent_bic_dir = f'{lang_pair}/entropy_distributions'
    if training_data:
        ent_bic_dir = 'rule_based/' + ent_bic_dir
    ent_bic_dir = ngram_dir + '/' + ent_bic_dir
    ent_bic_f = f'{ent_bic_dir}/fr_{frequency_threshold}.gm_{best_k}.ent_{entropy_threshold:.2f}.BIC.png'
    subprocess.call(['mkdir', '-p', ent_bic_dir])
    plt.savefig(ent_bic_f)
    plt.clf()

    # plot entropy distribution
    plt.figure(figsize=(7, 4))
    plt.hist(entropies, bins=100, density=True)
    for i, idx in enumerate(sort_idx):
        mean = sorted_means[i]
        std = sorted_stds[i]
        weight = sorted_weights[i]
        plt.plot(x, weight * norm.pdf(x, mean, std), label=f'Component {i+1}: μ={mean:.2f}, σ={std:.2f}')
    plt.plot(entropy_threshold, threshold_density, 'ro', markersize=8, label=f'Threshold: {entropy_threshold:.2f}')
    plt.axvline(x=entropy_threshold, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Density')
    # plt.title('Distribution of NGram Transformation Entropies')
    plt.legend()
    plt.tight_layout()
    ent_dist_dir = f'{lang_pair}/entropy_distributions'
    if training_data:
        ent_dist_dir = 'rule_based/' + ent_dist_dir
    ent_dist_dir = ngram_dir + '/' + ent_dist_dir
    ent_dist_f = f'{ent_dist_dir}/fr_{frequency_threshold}.gm_{best_k}.ent_{entropy_threshold:.2f}.png'
    subprocess.call(['mkdir', '-p', ent_dist_dir])
    plt.savefig(ent_dist_f)
    plt.show()
    plt.clf()

    return entropy_threshold, best_k

def filter_ent_red(ngram_pairs, kept_pairs, d_top, entropy_threshold, frequency_threshold):
    filtered_ngrams = []
    for (ngram, top, ent) in tqdm(kept_pairs, desc=f"Checking Redundancy"):
        if should_keep(ngram_pairs, d_top, ngram, ent, top, entropy_threshold, frequency_threshold):
            filtered_ngrams.append(ngram)
    
    return filtered_ngrams


def should_keep(ngram_pairs, d_top, ngram, ent, top, entropy_threshold, frequency_threshold):
    l = len(ngram)
    if ent <= entropy_threshold:
        explainers = [(ng, tgt) for (ng, tgt) in d_top if (entropy(ngram_pairs[ng]) <= entropy_threshold and len(ng) < l and sum(ngram_pairs[ng].values()) >= frequency_threshold)] # any smaller ngram with entropy below threshold and high enough frequency
        if explained_by_dp(ngram, top, explainers):
            return False
    return True

def explained_by_dp(ngram, target, explainers):
    result, relevant_explainers = is_explained_by(ngram, target, explainers)
    if result == False:
            macros = get_macro_rules(relevant_explainers, len(ngram))
            new_rules = [(src, (macros | relevant_explainers)[src]) for src in (macros | relevant_explainers)]
            result, _ = is_explained_by(ngram, target, new_rules)
    return result

def is_explained_by(ngram, target, explainers):
    """Check if ngram -> target is fully explained by any combination of explainer mappings."""
    relevant_explainers = {}
    @lru_cache(maxsize=None)
    def dp(i, j):
        if i == len(ngram) and j == len(target): # reached end of both strings, base case
            return True
        for short_ng, short_tgt in explainers: # each possible starting point for the transformation
            ni, nj = i + len(short_ng), j + len(short_tgt) # new indices if transformation is applied
            if ngram[i:i+len(short_ng)] == short_ng and target[j:j+len(short_tgt)] == short_tgt: # see if that transformation didn't actually change the source string
                relevant_explainers[short_ng] = short_tgt
                if dp(ni, nj): # check the rest of the string
                    return True
        return False
    
    result = dp(0, 0)
    dp.cache_clear()
    return result, relevant_explainers

def get_alignments(src, tgt, gap='_'):
    """Align two words with Levenshtein character alignment"""
    ops = Levenshtein.editops(src, tgt)
    a1, a2 = list(src), list(tgt)
    offset1 = offset2 = 0

    for op, i, j in ops:
        if op == 'insert':        # gap in s1
            a1.insert(i + offset1, gap)
            offset1 += 1
        elif op == 'delete':      # gap in s2
            a2.insert(j + offset2, gap)
            offset2 += 1
        # 'replace': no gap needed, characters already paired

    return ''.join(a1), ''.join(a2)

def ngrams(s, n):
    """Split a word into all possible ngrams of size n"""
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def entropy(freq_distribution):
    """Calculate entropy for a dictionary"""
    total = sum(freq_distribution.values())
    n = len(freq_distribution)
    if n <= 1:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in freq_distribution.values())

def get_macro_rules(mappings, length):
    """If two or more rules overlap on context somewhere, make a macro rule that fits all"""
    macros = {}
    
    def build_macros(src1, tgt1):
        for src2, tgt2 in mappings.items():
            macro = check_macro(src1, tgt1, src2, tgt2)
            if macro is not None:
                macro_src, macro_tgt = macro
                if macro_src not in macros and macro_src not in mappings:
                    macros[macro_src] = macro_tgt
                    if len(macro_src) < length: # word size
                        build_macros(macro_src, macro_tgt)  # recurse
    
    for src_ngram, tgt_ngram in mappings.items():
        build_macros(src_ngram, tgt_ngram)

    return macros

def check_macro(src_ngram1, tgt_ngram1, src_ngram2, tgt_ngram2):
    if src_ngram1[-1] == "$" or src_ngram2[0] == "$":
        return None
    s_align1, t_align1 = get_alignments(src_ngram1, tgt_ngram1)
    s_align2, t_align2 = get_alignments(src_ngram2, tgt_ngram2)
    max_overlap = min(len(s_align1), len(s_align2)) - 1 # both mappings need to contribute
    for overlap_len in range(max_overlap, 0, -1):
        src_suffix = s_align1[-overlap_len:]
        src_prefix = s_align2[:overlap_len]
        if src_suffix != src_prefix: # find first largest overlap
            continue
        
        tgt_suffix = t_align1[-overlap_len:]
        tgt_prefix = t_align2[:overlap_len]
        if tgt_suffix != tgt_prefix:
            continue
        
        macro_src = (s_align1 + s_align2[overlap_len:]).replace("_", "")
        macro_tgt = (t_align1 + t_align2[overlap_len:]).replace("_", "")
        return macro_src, macro_tgt

    return None

def get_data(training_data, applied_counts):
    # TODO
    word_pairs = {}
    # data from fastalign to build rule-based model
    if training_data:
        # get data
        if applied_counts is not None:
            pass
        
    # data from neural OC to characterize neural OC
    else:
        with open("src/OC/ngram_correspondences/fr_mfe_test_mappings.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                src, _, tgt = line.strip().split(' ')
                word_pairs[src] = tgt
        return word_pairs, None


def get_args():
    parser = argparse.ArgumentParser(description="Filter Meaningful Mappings")

    parser.add_argument('--training_data', '-t', action='store_true', help='Use training data instead')

    parser.add_argument('--oc_output_path', '-p', type=str, default="es-an-213.output.txt", help="Path to OC output")

    parser.add_argument('--counts_path', '-c', type=str, default=None, help='file containing counts for all words in SMT training data')

    parser.add_argument('--language_pair', '-l', type=str, default="es-an", help="Language Pair")

    parser.add_argument('--ngram_size', '-n', type=int, default=5, help="Check ngrams of size up until this number")

    parser.add_argument('--frequency', '-f', type=int, default=10, help="Filter out all ngrams that appear less than this number")

    parser.add_argument('--all_ngrams', '-a', action='store_true', help="Print out list of all ngrams with their mappings and entropy to a separate file")

    parser.add_argument('--compile_results', '-r', action='store_true', help='only compile the results, nothing else')

    parser.add_argument('--mappings', '-m', action='store_true', help="Print out list of all word mappings to a separate file")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # print(args.training_data)
    main(args.language_pair, args.training_data, args.counts_path, args.frequency)
