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
import optuna
from optuna.samplers import GridSampler
import os
from functools import lru_cache
from utilities.utilities import set_env
from utilities.metrics import *

import cProfile
import pstats
import io

set_env()
ngram_dir = "src/OC/ngram_correspondences"
EXP_HOME = os.environ["EXP_HOME"]

def main(lang_pair, training_data=False, applied_counts=None, frequency_threshold=10, return_scores=False):
    # get data
    word_pairs, word_counts = get_data(training_data, applied_counts, lang_pair)
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
    filtered_f = f'fr_{frequency_threshold}.gm_{best_k}.ent_{entropy_threshold:.2f}'
    subprocess.call(['mkdir', '-p', filtered_dir])
    with open(f"{filtered_dir}/{filtered_f}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["entropy", "ngram", "freq_dist"])
        for ngram in sorted(filtered_ngrams, key=lambda x: (entropy(ngram_pairs[x]), -sum(ngram_pairs[x].values()))): # sorted by entropy, frequency
            freq_dist = dict(ngram_pairs[ngram])
            ent = entropy(ngram_pairs[ngram])
            if ent <= entropy_threshold:
                writer.writerow([round(ent, 2), ngram, dict(sorted(freq_dist.items(), key=lambda x: -x[1]))])  

    # can add characterize noise here if we want it as part of the pipeline
    # rules = {}
    # for ngram in filtered_ngrams:
    #     rules[ngram] = d_top[ngram]
    # character_coverage(rules, word_pairs, lang_pair, filtered_f, training_data, word_counts, filename=filtered_f)
    

    # chrf, bleu = evaluate_rule_based(filtered_f, lang_pair)
    # return chrf, bleu


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
    d_top = {}
    rows = sorted((len(i), entropy(ngram_pairs[i]), -sum(ngram_pairs[i].values()), i, dict(ngram_pairs[i])) for i in ngram_pairs) # sort by len, ent, freq
    for _, ent, freq, ngram, freq_distribution in rows:
        top = max(freq_distribution, key=freq_distribution.get)
        d_top[ngram] = top
        if -freq < frequency_threshold:
            continue
        if top == ngram:
            if (freq_distribution[top] / -freq) < .5:
                pass
            else:
                continue
        kept.append((ngram, ent))
    return kept, d_top

def get_entropy_threshold(lang_pair, training_data, kept_pairs, frequency_threshold):
    entropies = [ent for (ngram, ent) in kept_pairs if ent != 0]

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

    # entropy_threshold = .6
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

def filter_ent_red(ngram_pairs, kept_pairs, d_top, entropy_threshold, frequency_threshold, return_max=False):
    filtered_ngrams = []
    max_len, max_freq, max_ent = 0, 0, 0
    potential_explainers = [(ng, d_top[ng]) for ng in d_top if (entropy(ngram_pairs[ng]) <= entropy_threshold and sum(ngram_pairs[ng].values()) >= frequency_threshold)]
    for (ngram, ent) in tqdm(kept_pairs, desc=f"Checking Redundancy"):
        if ent < entropy_threshold and should_keep(potential_explainers, ngram, d_top[ngram]):
            filtered_ngrams.append(ngram)
            if ent > max_ent:
                max_ent = ent
            if sum(ngram_pairs[ngram].values()) > max_freq:
                max_freq = sum(ngram_pairs[ngram].values())
            if len(ngram) > max_len:
                max_len = len(ngram)

    if return_max == True:
        return filtered_ngrams, max_len, max_freq, max_ent
    return filtered_ngrams

def should_keep(potential_explainers, ngram, top):
    l = len(ngram)
    explainers = [(ng, tgt) for (ng, tgt) in potential_explainers if len(ng) < l] # any smaller ngram with entropy below threshold and high enough frequency
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

@lru_cache(maxsize=None)
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

def get_macro_rules(mappings, length, scores=None, ngram_pairs=None, entropy_weight=None, max_len=None, frequency_averages=None, context_w=None, frequency_w=None, stats_cache=None):
    """If two or more rules overlap on context somewhere, make a macro rule that fits all"""
    macros = {}
    
    mapping_items = []
    for src2, tgt2 in mappings.items():
        if len(src2) == 0 or src2[0] == "$":
            continue
        s2, t2 = get_alignments(src2, tgt2)
        mapping_items.append((src2, tgt2, s2, t2, len(s2)))

    def build_macros(src1, tgt1):
        if len(src1) == 0 or src1[-1] == "$":
            return
        s_align1, t_align1 = get_alignments(src1, tgt1)
        len1 = len(s_align1)
        for src2, tgt2, s_align2, t_align2, len2 in mapping_items:
            macro = check_macro(s_align1, t_align1, len1, s_align2, t_align2, len2)
            if macro is not None:
                macro_src, macro_tgt = macro
                if macro_src not in macros and macro_src not in mappings:
                    if scores is not None:
                        scores[macro_src] = get_macro_score(src1, src2, macro_src, stats_cache, ngram_pairs, entropy_weight, max_len, frequency_averages, context_w, frequency_w)
                    macros[macro_src] = macro_tgt
                    # if len(macro_src) < length: # word size
                    if len(macro_src) < min(10, length):
                        build_macros(macro_src, macro_tgt)  # recurse
    
    for src_ngram, tgt_ngram in mappings.items():
        build_macros(src_ngram, tgt_ngram)

    if scores is not None:
        return macros, scores
    else:
        return macros

def check_macro(s_align1, t_align1, len1, s_align2, t_align2, len2):
    max_overlap = min(len1, len2) - 1 # both mappings need to contribute
    for overlap_len in range(max_overlap, 0, -1):
        if s_align1[-overlap_len:] != s_align2[:overlap_len]:
            continue
        if t_align1[-overlap_len:] != t_align2[:overlap_len]:
            continue
        macro_src = (s_align1 + s_align2[overlap_len:]).replace("_", "")
        macro_tgt = (t_align1 + t_align2[overlap_len:]).replace("_", "")
        return macro_src, macro_tgt
    return None

def character_coverage(meaningful_rules, word_pairs, lang_pair, filtered_f, training_data=False, word_counts=None, filename='removed_noise'):
    total_unchanged, total_explained, total_unexplained = 0, 0, 0

    if word_counts is not None:
        total_unchanged_counts, total_explained_counts, total_unexplained_counts = 0, 0, 0

    removed_noise_dir = f"{lang_pair}/removed_noise"
    if training_data:
        removed_noise_dir = 'rule_based/' + removed_noise_dir
    removed_noise_dir = ngram_dir + '/' + removed_noise_dir
    subprocess.call(['mkdir', '-p', removed_noise_dir])
    with open(f"{removed_noise_dir}/{filename}.txt", 'w') as out_f:
        for source in tqdm(word_pairs, desc=f"Checking {lang_pair} character coverage"):
            unchanged, explained, unexplained, no_noise = characterize_transformation(source, word_pairs[source], meaningful_rules)
            out_f.write(f"{source} {no_noise}\n")
            total_unchanged += unchanged
            total_explained += explained
            total_unexplained += unexplained
        
            if word_counts is not None:
                count = word_counts[source]
                total_unchanged_counts += unchanged * count
                total_explained_counts += explained * count
                total_unexplained_counts += unexplained * count
    
    with open(f"{removed_noise_dir}/{filename}-results.txt", "w") as results_f:
        results_f.write(f"Character Coverage for {lang_pair}\n")
        results_f.write(f"Total unchanged characters: {total_unchanged}\n")
        results_f.write(f"Total explained character changes: {total_explained}\n")
        results_f.write(f"Total unexplained character changes: {total_unexplained}\n")
        results_f.write(f"Meaningful Change / Total Characters: {total_explained / (total_unchanged + total_explained + total_unexplained):.4f}\n")
        results_f.write(f"Noise / Total Characters: {total_unexplained / (total_unchanged + total_explained + total_unexplained):.4f}\n")

        if word_counts is not None:
            results_f.write(f"\n\nIN NMT TRAINING DATA\n")
            results_f.write(f"Total unchanged characters: {total_unchanged_counts}\n")
            results_f.write(f"Total explained character changes: {total_explained_counts}\n")
            results_f.write(f"Total unexplained character changes: {total_unexplained_counts}\n")
            results_f.write(f"Meaningful Change / Total Characters: {total_explained_counts / (total_unchanged_counts + total_explained_counts + total_unexplained_counts):.4f}\n")
            results_f.write(f"Noise / Total Characters: {total_unexplained_counts / (total_unchanged_counts + total_explained_counts + total_unexplained_counts):.4f}\n")

    return


def characterize_transformation(source, target, meaningful_mappings):
    if source == target: # no transformation happened
        return len(source), 0, 0, source

    s, t = f"${source}$", f"${target}$"
    n = len(s)
    m = len(t)
    
    # update the dp at the correct index if it is better
    def update(dp, i, j, next_i, next_j, metrics, mapping):
        if next_i > n or next_j > m:
            return
        unchanged, explained, unexplained, trace = dp[i][j]
        d_unchanged, d_explained, d_unexplained = metrics

        new_state = (unchanged + d_unchanged, explained + d_explained, unexplained + d_unexplained, trace + [(i, j, mapping)])
        target_cell = dp[next_i][next_j]

        if target_cell is None:
            dp[next_i][next_j] = new_state
        else:
            if new_state[2] < target_cell[2]:
                dp[next_i][next_j] = new_state
            elif new_state[2] == target_cell[2]:
                if (new_state[0] + new_state[1] > target_cell[0] + target_cell[1]): # this prioritizes using more mappings so that the trace is more interpretable. See test case 1 trace without this line for why we need it
                    dp[next_i][next_j] = new_state

    def run_dp(mappings):
        # dp[i][j] = (unchanged, explained, unexplained, mapping trace)
        dp = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = (0, 0, 0, [])  

        relevant_mappings = {} # the mappings that fit somewhere in this word for building macro rules

        for i in range(n + 1):
            for j in range(m + 1):
                if dp[i][j] is None:
                    continue
                
                # Try meaningful mappings
                for src_ngram, tgt_ngram in mappings.items():
                    # mapping fits
                    if s.startswith(src_ngram, i) and t.startswith(tgt_ngram, j):
                        relevant_mappings[src_ngram] = tgt_ngram
                        changed, unchanged = count_changed_unchanged(src_ngram, tgt_ngram)
                        update(dp, i, j, i + len(src_ngram), j + len(tgt_ngram), (unchanged, changed, 0), (src_ngram, tgt_ngram))

                # Fallthrough: single position unexplained
                if i < n and j < m:
                    if s[i] == t[j]:
                        update(dp, i, j, i+1, j+1, (1, 0, 0), (s[i], '<uc>')) # unchanged
                    else:
                        update(dp, i, j, i+1, j+1, (0, 0, 1), (s[i], '<ue>'))
                
                # deletion
                if i < n:
                    update(dp, i, j, i+1, j, (0, 0, 1), (s[i], '<del>'))
                # insertion
                if j < m:
                    update(dp, i, j, i, j+1, (0, 0, 1), (t[j], '<ins>'))

        unchanged, explained, unexplained, trace = dp[n][m]
        return unchanged - 2, explained, unexplained, trace, relevant_mappings # subtract 2 from unchanged because of word boundary $ characters

    unchanged, explained, unexplained, trace, relevant_mappings = run_dp(meaningful_mappings) 

    if unexplained > 0:
        macros = get_macro_rules(relevant_mappings, n)
        unchanged, explained, unexplained, trace, relevant_mappings = run_dp(relevant_mappings | macros)


    transform_no_noise = transform_without_noise(trace)
    if unexplained == 0:
        assert target == transform_no_noise 
    return unchanged, explained, unexplained, transform_no_noise 

# count the number of changed vs unchanged characters that are in the meaningful mapping that was applied
def count_changed_unchanged(src_ngram, tgt_ngram):
    src_ngram_aligned, tgt_ngram_aligned = get_alignments(src_ngram, tgt_ngram)
    changed = sum(1 for sc, tc in zip(src_ngram_aligned, tgt_ngram_aligned) if sc != tc)
    unchanged = len(src_ngram_aligned) - changed
    return changed, unchanged

def transform_without_noise(trace):
    """Reconstruct the words by only using the explained transformations,
    and undoing any noise from the neural OC model. Requires trace from 
    check_word_transformation.
    """
    new_word = ''
    for item in trace:
        i, j, map = item
        src, tgt = map
        if tgt == '<uc>' or tgt == '<ue>':
            new_word += src
        elif tgt == '<del>' or tgt == '<ins>':
            pass
        else:
            new_word += tgt
    new_word = new_word.replace('$', '')
    return new_word

def get_data(training_data, applied_counts, lang_pair, method='charlotte'):
    # TODO
    pl, cl = lang_pair.split('-')
    word_pairs = {}
    # data from fastalign to build rule-based model
    if training_data:
        # get data
        with open(f'{EXP_HOME}/{pl}_{cl}-->en/OC/{method}/{pl}-{cl}/data/train.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if method != 'fuzz':
                    _, src, tgt, theta = line.strip().split(' ||| ')
                else:
                    _, _, src, tgt, theta = line.strip().split(' ||| ')
                word_pairs[src] = tgt
        
        return word_pairs, None
        if applied_counts is not None:
            pass
        
    # data from neural OC to characterize neural OC
    else:
        with open(f"src/OC/ngram_correspondences/{pl}-{cl}_test_mappings.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                src, _, tgt = line.strip().split(' ')
                word_pairs[src] = tgt
        return word_pairs, None
    

### Rule Based Transformation ###

# def rule_based_oc(filtered_f, words):
#     ### No longer used ###
#     scores, rules = get_scores_rules(filtered_f)

#     transformed_words = {}
#     for word in tqdm(words, desc="Rule Based Transformation"):
#         transformed_words[word] = transform_word(rules, scores, word)
    
#     return transformed_words

def transform_word(rules, scores, word, ngram_pairs, max_len, entropy_weight, frequency_averages, frequency_w, context_w, tiebreaker, stats_cache):
    word = f"${word}$"
    n = len(word)

    def reconstruct_from_trace(trace):
        result = ""
        prev = 0
        for i, src, tgt in sorted(trace):
            result += word[prev:i]
            result += tgt
            prev = i + len(src)
        result += word[prev:]
        return result

    relevant_mappings = {}
    relevant_mappings_scores = {}
    for src_ngram, tgt_ngram in rules.items():
        if src_ngram in word:
            relevant_mappings[src_ngram] = tgt_ngram
            relevant_mappings_scores[src_ngram] = scores[src_ngram]

    macros, macros_scores = get_macro_rules(relevant_mappings, n, relevant_mappings_scores, ngram_pairs, entropy_weight, max_len, frequency_averages, context_w, frequency_w, stats_cache)

    relevant_mappings = relevant_mappings | macros
    relevant_mappings_scores = relevant_mappings_scores | macros_scores


    best_words = set()
    dp = [None] * (n + 1)
    dp[0] = [0, []] # score, mapping trace

    for i in range(n):
        if dp[i] is None:
            continue
        s, trace = dp[i]
        # try meaningful mappings
        for src_ngram, tgt_ngram in relevant_mappings.items():
            if word.startswith(src_ngram, i):
                n_s = s + relevant_mappings_scores[src_ngram]
                n_pos = i + len(src_ngram)
                n_trace = trace + [(i, src_ngram, tgt_ngram)]
                if dp[n_pos] is None or n_s > dp[n_pos][0]:
                    dp[n_pos] = [n_s, n_trace]
                if n_pos == n:
                    if dp[n_pos] is None or n_s > dp[n_pos][0]:
                        best_words = set()
                        best_words.add(reconstruct_from_trace(n_trace))
                    elif n_s == dp[n_pos][0]:
                        best_words.add(reconstruct_from_trace(n_trace))

        # heavy penalty since only should be used if character in val is not in training data
        # try no mappings at current pos
        n_pos = i + 1
        n_s = -1e9
        if dp[n_pos] is None or s > dp[n_pos][0]:
            dp[n_pos] = [s, trace]
        if n_pos == n:
            if dp[n_pos] is None or s > dp[n_pos][0]:
                best_words = set()
                best_words.add(reconstruct_from_trace(trace))

            elif s == dp[n_pos][0]:
                best_words.add(reconstruct_from_trace(trace))


    # Tiebreakers
    if len(best_words) > 1:
        distances = {}
        for best_word in best_words:
            distances[best_word] = Levenshtein.distance(word, best_word) / max(n, len(best_word), 1)

        sorted_best = sorted(distances, key=lambda item:(distances[item], item))
        # print(sorted_best)
        if tiebreaker == 'least_edit_dist':
            return sorted_best[0].replace('$', '')
        elif tiebreaker == "most_edit_dist":
            return sorted_best[-1].replace('$', '')


    else:
        # print(best_words)
        return next(iter(best_words)).replace('$', '')



# def read_ngrams_from_csv(filtered_f):
#     ### OUTDATED, no longer useful ###
#     """Get top ngram for each ngram in a filtered file"""
#     max_freq, max_len, max_ent = 0, 0, 0

#     filtered_ngrams = defaultdict(lambda: defaultdict(int))
#     with open(filtered_f, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             ngram = row["ngram"]
#             if len(ngram) > max_len:
#                 max_len = len(ngram)
#             ent = float(row["entropy"])
#             if ent > max_ent:
#                 max_ent = ent
#             freq_dist = ast.literal_eval(row["freq_dist"])
#             if sum(freq_dist.values()) > max_freq:
#                 max_freq = sum(freq_dist.values())
#             for target, count in freq_dist.items():
#                 filtered_ngrams[ngram][target] = count

#     return filtered_ngrams, max_len, max_freq, max_ent
    

def get_score(ent, length, frequency_component, entropy_weight, max_len, context_w, frequency_w):
    length_component = length / max_len

    return (length_component ** context_w) + (frequency_w * frequency_component) - (ent * entropy_weight)


def get_macro_score(src1, src2, macro_src, stats_cache, ngram_pairs, entropy_weight, max_len, frequency_averages, context_w, frequency_w):
    length_component = len(macro_src) / max_len
    freq_comp1, ent1 = stats_cache[src1]
    freq_comp2, ent2 = stats_cache[src2]
    frequency_component = (freq_comp1 + freq_comp2) / 2
    entropy_component = (ent1 + ent2) / 2
    stats_cache[macro_src] = (frequency_component, entropy_component)


    return ((length_component) ** context_w) + (frequency_w * frequency_component) - (entropy_weight * entropy_component)


# def evaluate_rule_based(filtered_f, lang_pair):
#     ### No longer useful since the rules-based system has been reworked ###
#     pl, cl = lang_pair.split('-')
#     val_pairs = {}
#     with open(f'{EXP_HOME}/{pl}_{cl}-->en/OC/charlotte/{pl}-{cl}/data/val.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             _, src, tgt, theta = line.strip().split(' ||| ')
#             val_pairs[src] = tgt
    
#     transformed_val = rule_based_oc(f"src/OC/ngram_correspondences/rule_based/{lang_pair}/filtered/{filtered_f}.csv", val_pairs)

#     hyp = []
#     ref = []
#     for src in val_pairs:
#         hyp.append(val_pairs[src])
#         ref.append(transformed_val[src])

#     chrf = calc_chrF(hyp, ref).score
#     # bleu = calc_charBLEU(hyp, ref)
    

#     return chrf, bleu


######### HYPERPARAMETER SEARCH ###############

def hyperparameter_search(lang_pair, training_data, metric='chrf', method='charlotte', n_trials=100):
    word_pairs, word_counts = get_data(training_data, None, lang_pair, method)
    ngram_pairs, ngram_counts = get_counts(word_pairs, word_counts)
    kept_pairs, max_ent, frequency_averages, max_len = filter_ident(ngram_pairs, keep_ident=True)


    pl, cl = lang_pair.split('-')
    val_pairs = {}
    with open(f'{EXP_HOME}/{pl}_{cl}-->en/OC/{method}/{pl}-{cl}/data/val.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if method != 'fuzz':
                _, src, tgt, theta = line.strip().split(' ||| ')
            else:
                _, _, src, tgt, theta = line.strip().split(' ||| ')
            val_pairs[src] = tgt


    if metric in ['chrf', 'mdelta', 'charBLEU']:
        direction = 'maximize'
    elif metric == 'nld':
        direction = 'minimize'
    

    ### Baseline ###
    hyp = []
    ref = []
    
    for src in val_pairs:
        hyp.append(src)
        ref.append(val_pairs[src])

    print("BASELINE:")
    if metric == 'chrf':
        print(calc_chrF(hyp, ref).score)
    elif metric == 'charBLEU':
        print(calc_charBLEU(hyp, ref).score)
    elif metric == 'nld':
        print(calc_edit_distance(hyp, ref))
    print('\n\n')


    ### grid search ###
    # search_space = {
    # 'entropy_w':   [0, 1, 2, 3],
    # 'frequency_w': [0, 1, 2, 3],
    # 'context_w':   [0, 1, 2, 3],
    #                 }

    # study = optuna.create_study(direction=direction,
    #                             storage=f"sqlite:///src/OC/ngram_correspondences/rule_based/scores_db.sqlite3",
    #                             study_name=f"{lang_pair}-{metric}-superlin-float",
    #                             load_if_exists=True,
    #                             sampler=GridSampler(search_space))


    ### TPE optimizer ###
    study = optuna.create_study(direction=direction,
                                storage=f"sqlite:///src/OC/ngram_correspondences/rule_based/scores_db.sqlite3",
                                study_name=f"{lang_pair}-{metric}-0.2", load_if_exists=True)

    study.optimize(lambda trial: objective_no_switches(trial, val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len), n_trials=n_trials)

    ## boundary grid search ###
    n_queued = 0
    for con_w in [1, 5]:
        for freq_off in [True, False]:
            for ent_off in [True, False]:
                params = {"context_w": con_w, "frequency_off": freq_off, "entropy_off": ent_off}
                if not freq_off:
                    params["frequency_w"] = 100
                if not ent_off:
                    params["entropy_w"] = 100
                study.enqueue_trial(params)
                n_queued += 1

    study.optimize(lambda trial: objective_with_switches(trial, val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len), n_trials=n_queued)
    

    study.best_params


def objective_with_switches(trial, val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len):
    # grid search of boundaries
    entropy_w = 0.0 if trial.suggest_categorical('entropy_off', [True, False]) \
        else trial.suggest_float('entropy_w', 0.001, 100, log=True)
    frequency_w = 0.0 if trial.suggest_categorical('frequency_off', [True, False]) \
        else trial.suggest_float('frequency_w', 0.001, 100, log=True)
    context_w = trial.suggest_float('context_w', 1, 5)
    return objective(val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len, entropy_w, frequency_w, context_w)

def objective_no_switches(trial, val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len):
    # optimizer search without boundaries
    entropy_w = trial.suggest_float('entropy_w', 0.001, 100, log=True)
    frequency_w = trial.suggest_float('frequency_w', 0.001, 100, log=True)
    context_w = trial.suggest_float('context_w', 1, 5)
    return objective(val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len, entropy_w, frequency_w, context_w)

def objective(val_pairs, metric, ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len, entropy_w, frequency_w, context_w):
    tiebreaker = 'least_edit_dist'

    
    if metric in ['chrf', 'nld', 'charBLEU']:
        entropy_weight = entropy_w / max_ent
        stats_cache = make_ngram_stats_cache()
        scores, rules = get_scores_rules_hyperparam(kept_pairs, max_len, frequency_averages, entropy_weight, context_w, frequency_w, stats_cache)


        transformed_val = {}
        for word in tqdm(val_pairs, desc="Rule Based Transformation"):
            transformed_val[word] = transform_word(rules, scores, word, ngram_pairs, max_len, entropy_weight, frequency_averages, frequency_w, context_w, tiebreaker, stats_cache)

        hyp = []
        ref = []
        for src in val_pairs:
            hyp.append(val_pairs[src])
            ref.append(transformed_val[src])

        if metric == 'chrf':
            metric_score = calc_chrF(hyp, ref).score
        elif metric == 'nld':
            metric_score = calc_edit_distance(hyp, ref)
        elif metric == 'charBLEU':
            metric_score = calc_charBLEU(hyp, ref).score
    
    # reimplement with scoring system hyperparam search -- doesn't really work anymore, since we are not filtering anymore, so everything is 'explained now'
    # elif metric == 'mdelta':
    #     meaningful_rules = {}
    #     for ngram in filtered_ngrams:
    #         meaningful_rules[ngram] = d_top[ngram]

    #     total_unchanged, total_explained, total_unexplained = 0, 0, 0
    #     for src in tqdm(val_pairs, desc="Characterizing Transformation"):
    #         unchanged, explained, unexplained, no_noise = characterize_transformation(src, val_pairs[src], meaningful_rules)
    #         total_unchanged += unchanged
    #         total_explained += explained
    #         total_unexplained += unexplained

    #     metric_score = total_explained / (total_unchanged + total_explained + total_unexplained)

    return metric_score


def filter_ident(ngram_pairs, keep_ident=True):
    kept = []
    frequency_bins = Counter()
    frequency_counts = Counter()
    max_ent, max_len = 0, 0
    rows = sorted((len(i), entropy(ngram_pairs[i]), sum(ngram_pairs[i].values()), i, dict(ngram_pairs[i])) for i in ngram_pairs) # sort by len, ent, freq
    for length, ent, freq, ngram, freq_distribution in rows:
        top = max(freq_distribution, key=freq_distribution.get)
        if top == ngram:
            if (freq_distribution[top] / freq) < .5:
                pass
            else:
                if keep_ident==True:
                    pass
                else:
                    continue
        
        kept.append((ngram, top, ent, freq, length))
        if ent > max_ent:
            max_ent = ent
        if length > max_len:
            max_len = length
        frequency_counts[length] += freq
        frequency_bins[length] += 1

    frequency_averages = {}
    for length_val in frequency_bins:
        frequency_averages[length_val] = frequency_counts[length_val] / frequency_bins[length_val]

    return kept, max_ent, frequency_averages, max_len


def get_scores_rules_hyperparam(kept_pairs, max_len, frequency_averages, entropy_weight, context_w, frequency_w, stats_cache):
    scores = {}
    rules = {}
    
    for (ngram, top, ent, freq, length) in kept_pairs:
        frequency_component = math.log2(max(freq / frequency_averages[length], .2))
        scores[ngram] = get_score(ent, length, frequency_component, entropy_weight, max_len, context_w, frequency_w)
        rules[ngram] = top
        stats_cache[ngram] = (frequency_component, ent)

    return scores, rules

def calc_edit_distance(hyp, ref):
    distances = [
        Levenshtein.distance(a, b) / max(len(a), len(b), 1)
        for a, b in zip(hyp, ref)
    ]
    return sum(distances) / len(distances)

def make_ngram_stats_cache():
    return {}

def ngram_stats(src, cache, ngram_pairs, frequency_averages):
    if src not in cache:
        freq_comp = math.log2(max(sum(ngram_pairs[src].values()) / frequency_averages[len(src)], .2))
        ent = entropy(ngram_pairs[src])
        cache[src] = (freq_comp, ent)
    return cache[src]


def get_args():
    parser = argparse.ArgumentParser(description="Filter Meaningful Mappings")

    parser.add_argument('--training_data', '-t', action='store_true', help='Use training data instead')

    parser.add_argument('--oc_output_path', '-p', type=str, default="es-an-213.output.txt", help="Path to OC output")

    parser.add_argument('--counts_path', '-c', type=str, default=None, help='file containing counts for all words in SMT training data')

    parser.add_argument('--language_pair', '-l', type=str, default="es-an", help="Language Pair")

    parser.add_argument('--frequency', '-f', type=int, default=10, help="Filter out all ngrams that appear less than this number")

    parser.add_argument('--hyperparam_metric', '-m', type=str, default='chrf', choices=['chrf', 'mdelta', 'nld', 'charBLEU'], help="Metric to optimize in hyperparameter search")

    parser.add_argument('--num_trials', '-n', type=int, default=100)

    parser.add_argument('--cognate_method', '-cm', type=str, default="charlotte", choices=["charlotte", "web", "fuzz"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # hyperparameter_search(args.language_pair, args.training_data, args.hyperparam_metric, args.cognate_method, args.num_trials)
    # 

    # frequency_ablation(args.language_pair, args.training_data)
    # main(args.language_pair, args.training_data, args.counts_path, args.frequency)
    # test_check_word_transformation()

    # print(evaluate_rule_based("fr_10.gm_6.ent_0.62", args.language_pair))

    
    # PROFILING
    # pr = cProfile.Profile()
    # pr.enable()

    # hyperparameter_search(args.language_pair, args.training_data, args.hyperparam_metric, args.cognate_method, args.num_trials)
    
    # pr.disable()
    # stream = io.StringIO()
    # ps = pstats.Stats(pr, stream=stream).sort_stats('cumulative')
    # ps.print_stats(50)  # top 50 functions by cumulative time
    # print(stream.getvalue())





    # CHAR1.0 COMPARISON TEST
    # word_pairs = {}
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/es-an_ES-AN-RNN-0_RNN-0_S-0/fastalign/word_list.es-an.NG.cognates.0.5.parallel-es.train-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/es-an_ES-AN-RNN-0_RNN-0_S-0/fastalign/word_list.es-an.NG.cognates.0.5.parallel-an.train-s=0.txt", "r") as tgt_f:
    # # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-mfe_FR-MFE-RNN-0_RNN-0_S-0/fastalign/word_list.fr-mfe.NG.cognates.0.5.parallel-fr.train-s=0.txt", "r") as src_f, \
    # #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-mfe_FR-MFE-RNN-0_RNN-0_S-0/fastalign/word_list.fr-mfe.NG.cognates.0.5.parallel-mfe.train-s=0.txt", "r") as tgt_f:
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/uz-kaa_UZ-KAA-RNN-0_RNN-0_S-0/fastalign/word_list.uz-kaa.NG.cognates.0.5.parallel-uz.train-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/uz-kaa_UZ-KAA-RNN-0_RNN-0_S-0/fastalign/word_list.uz-kaa.NG.cognates.0.5.parallel-kaa.train-s=0.txt", "r") as tgt_f:
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-oc_FR-OC-RNN-0_RNN-0_S-0/fastalign/word_list.fr-oc.NG.cognates.0.5.parallel-fr.train-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-oc_FR-OC-RNN-0_RNN-0_S-0/fastalign/word_list.fr-oc.NG.cognates.0.5.parallel-oc.train-s=0.txt", "r") as tgt_f:


    #     src_lines = src_f.readlines()
    #     tgt_lines = tgt_f.readlines()
    #     print(len(src_lines), len(tgt_lines))
    #     for src, tgt in zip(src_lines, tgt_lines):
    #         word_pairs[src.strip()] = tgt.strip()

    # ngram_pairs, _ = get_counts(word_pairs, None)


    # kept_pairs, max_ent, frequency_averages, max_len = filter_ident(ngram_pairs, keep_ident=True)
    
    val_pairs = {}
    with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/es-an_ES-AN-RNN-0_RNN-0_S-0/fastalign/word_list.es-an.NG.cognates.0.5.parallel-es.val-s=0.txt", "r") as src_f, \
         open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/es-an_ES-AN-RNN-0_RNN-0_S-0/fastalign/word_list.es-an.NG.cognates.0.5.parallel-an.val-s=0.txt", "r") as tgt_f:
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-mfe_FR-MFE-RNN-0_RNN-0_S-0/fastalign/word_list.fr-mfe.NG.cognates.0.5.parallel-fr.val-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-mfe_FR-MFE-RNN-0_RNN-0_S-0/fastalign/word_list.fr-mfe.NG.cognates.0.5.parallel-mfe.val-s=0.txt", "r") as tgt_f:
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/uz-kaa_UZ-KAA-RNN-0_RNN-0_S-0/fastalign/word_list.uz-kaa.NG.cognates.0.5.parallel-uz.val-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/uz-kaa_UZ-KAA-RNN-0_RNN-0_S-0/fastalign/word_list.uz-kaa.NG.cognates.0.5.parallel-kaa.val-s=0.txt", "r") as tgt_f:
    # with open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-oc_FR-OC-RNN-0_RNN-0_S-0/fastalign/word_list.fr-oc.NG.cognates.0.5.parallel-fr.val-s=0.txt", "r") as src_f, \
    #      open("../nobackup/archive/data_and_models/data/COGNATE_TRAIN/fr-oc_FR-OC-RNN-0_RNN-0_S-0/fastalign/word_list.fr-oc.NG.cognates.0.5.parallel-oc.val-s=0.txt", "r") as tgt_f:
        
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        for src, tgt in zip(src_lines, tgt_lines):
            val_pairs[src.strip()] = tgt.strip()

    # print(len(word_pairs))
    print(len(val_pairs))
    
    hyp = []
    ref = []
    
    for src in val_pairs:
        hyp.append(src)
        ref.append(val_pairs[src])
    print("BASELINE")
    print(calc_charBLEU(hyp, ref).score)

    # print(objective(val_pairs, 'charBLEU', ngram_pairs, kept_pairs, max_ent, frequency_averages, max_len, 0, 0, 5))












# def test_check_word_transformation():
#     # some of everything
#     source = "abcde"
#     target = "axydef"
#     maps = {'a':'ax', "cd":"yd"}
#     assert characterize_transformation(source, target, maps) == (3, 2, 2)

#     source = "ab"
#     target = "abab"
#     maps = {'a':'aba'}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     source = "abc"
#     target = "xby"
#     maps = {"ab":"xb", "bc":"by"}
    
#     assert characterize_transformation(source, target, maps) == (1, 2, 0)

#     # # overlapping mappings (macros)
#     source = "abcd"
#     target = "xbyd"
#     maps = {"ab": "xb", "bc":'by'}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     source = "abedy"
#     target = "bdx"
#     maps = {"abed":"bd", "bedy":"dx"}
#     assert characterize_transformation(source, target, maps) == (1, 3, 1)

#     source = "abedy"
#     target = "bdx"
#     maps = {"abed":"bd", "edy":"dx"}
#     assert characterize_transformation(source, target, maps) == (2, 3, 0)

#     # # Basic substitution
#     source = "abc"
#     target = "axc"
#     maps = {"b": "x"}
#     assert characterize_transformation(source, target, maps) == (2, 1, 0)

#     # # Deletion
#     source = "abbc"
#     target = "ac"
#     maps = {"bb": ""}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     # Insertion
#     source = "ac"
#     target = "abc"
#     maps = {"": "b"}
#     assert characterize_transformation(source, target, maps) == (2, 1, 0)

#     # No mappings apply
#     source = "abc"
#     target = "xyz"
#     maps = {}
#     assert characterize_transformation(source, target, maps) == (0, 0, 3)

#     # All unchanged
#     source = "abc"
#     target = "abc"
#     maps = {"x": "y"}
#     assert characterize_transformation(source, target, maps) == (3, 0, 0)

#     # Overlapping possible mappings, best coverage wins
#     source = "abcd"
#     target = "axcd"
#     maps = {"ab": "ax", "b": "x"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 0)

#     # Length increasing mapping
#     source = "ac"
#     target = "abbc"
#     maps = {"": "bb"}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     # Length decreasing mapping
#     source = "abbc"
#     target = "ac"
#     maps = {"bb": ""}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     # Multiple mappings in sequence
#     source = "abcd"
#     target = "xyzw"
#     maps = {"a": "x", "b": "y", "c": "z", "d": "w"}
#     assert characterize_transformation(source, target, maps) == (0, 4, 0)

#     # Mapping at start
#     source = "abcd"
#     target = "xbcd"
#     maps = {"$a": "$x"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 0)

#     # Mapping at end
#     source = "abcd"
#     target = "abcx"
#     maps = {"d$": "x$"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 0)

#     source = "abcd"
#     target = "abxd"
#     maps = {"cd$": "xd$"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 0)

#     # Unexplained in middle
#     source = "abcd"
#     target = "axxd"
#     maps = {"a": "a"}
#     assert characterize_transformation(source, target, maps) == (2, 0, 2)

#     # Competing mappings, longer wins
#     source = "abcd"
#     target = "xycd"
#     maps = {"a": "x", "ab": "xy"}
#     assert characterize_transformation(source, target, maps) == (2, 2, 0)

#     # Mixed explained and unexplained
#     source = "abcde"
#     target = "axcye"
#     maps = {"b": "x"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 1)

#     # All unexplained
#     source = "abc"
#     target = "de"
#     maps = {}
#     assert characterize_transformation(source, target, maps) == (0, 0, 3)

#     # Mapping produces empty at word boundary
#     source = "abcs"
#     target = "abc"
#     maps = {"s$": "$"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 0)

#     # Longer word, multiple mappings, some overlapping
#     source = "abcdefgh"
#     target = "axcdyfgh"
#     maps = {"ab": "ax", "b": "x", "e": "y", "de": "dy"}
#     assert characterize_transformation(source, target, maps) == (6, 2, 0) 

#     # # Longer word, best path requires skipping a shorter mapping
#     source = "abcdef"
#     target = "xycdef"
#     maps = {"a": "x", "ab": "xy", "b": "y"}
#     assert characterize_transformation(source, target, maps) == (4, 2, 0) 

#     # # # Mixture of mappings and unexplained
#     source = "abcdefgh"
#     target = "axcdeyzh"
#     maps = {"b": "x", "fg": "yz"}
#     assert characterize_transformation(source, target, maps) == (5, 3, 0)

#     # # Long word, competing paths, longer mapping wins
#     source = "abcdefgh"
#     target = "abxyzfgh"
#     maps = {"c": "x", "cd": "xy", "cde": "xyz"}
#     assert characterize_transformation(source, target, maps) == (5, 3, 0) 

#     # # Mappings at both boundaries
#     source = "abcdef"
#     target = "xbcdey"
#     maps = {"$a": "$x", "f$": "y$"}
#     assert characterize_transformation(source, target, maps) == (4, 2, 0)

#     # # Some unexplained in middle, mappings at edges
#     source = "abcdef"
#     target = "xqqdef"
#     maps = {"$a": "$x", "f$": "f$"}
#     assert characterize_transformation(source, target, maps) == (3, 1, 2) 

#     # # Deletion and insertion in same word
#     source = "abcde"
#     target = "axbcf"
#     maps = {"": "x", "de": "f"}
#     characterize_transformation(source, target, maps) == (3, 3, 0)

#     # Multiple deletions
#     source = "aaabccdd"
#     target = "abcd"
#     maps = {"aaa": "a", "ab": "b", "cc": "c", "cdd": "d"}
#     # print(characterize_transformation(source, target, maps))
#     assert characterize_transformation(source, target, maps) == (4, 4, 0)

#     # Multiple Insertions
#     source = "abcd"
#     target = "aaabccdd"
#     maps = {"a":"aaa", "b":"ab", "c":"cc", "d":"cdd"}
#     assert characterize_transformation(source, target, maps) == (4, 4, 0)