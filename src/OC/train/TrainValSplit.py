from statistics import geometric_mean
import random

def get_train_val_split(pairs, theta, size=1000, n_buckets=10, max_fraction=0.3, seed=42):
    random.seed(seed)

    # Sort the word pairs first by frequency
    if len(pairs[0]) == 4:
        # if from parallel data (charlotte or web), sort by frequency of the word pair
        pairs = _sort_by_pair_freq(pairs)
    else:
        # if from fuzz, sort by the geometric mean of the (source word freq, target word freq).
        assert len(pairs[0]) == 5
        pairs = _sort_by_geo_freq(pairs)
    
    # Make buckets based on NLD
    bucket_range = theta / n_buckets
    buckets = [[] for i in range(n_buckets)]
    for item in pairs:
        nld = item[-1]
        bucket_index = int(nld / bucket_range)
        assert bucket_index <= n_buckets
        if bucket_index == n_buckets:
            bucket_index -= 1 # Make the last bucket inclusive on the upper bound
        buckets[bucket_index].append(item)
    
    default_quota = int(size / n_buckets)
    print(f"Default quota: {default_quota}")
    bucket_quotas = [default_quota for i in range(n_buckets)]
    defecit = 0
    capped = set()
    for b, quota in enumerate(bucket_quotas):
        if quota >= max_fraction * len(buckets[b]):
            bucket_quotas[b] = int(max_fraction * len(buckets[b]))
            print(f"\tBucket {b} capped at {bucket_quotas[b]}")
            capped.add(b)
            defecit += quota - bucket_quotas[b]
    
    print(f"Defecit of {defecit}. Will increase size of other buckets until defecit is gone or all buckets are capped.")
    b = 0
    while defecit > 0 and len(capped) < n_buckets:
        b = (b + 1) % n_buckets
        if b not in capped:
            bucket_quotas[b] += 1
            defecit -= 1
            if bucket_quotas[b] >= max_fraction * len(buckets[b]):
                print(f"\tBucket {b} capped at {bucket_quotas[b]}")
                capped.add(b)
    
    print(f"Defecit is now {defecit}")
    if len(capped) == n_buckets:
        print("All buckets were capped.")
    
    # Make val and training sets
    # Val will take its quota from the top of each bucket
        # This means val will have the most frequent pairs from each bucket to have reasonably high quality pairs
    val = []
    train = []
    for b, bucket in enumerate(buckets):
        quota = bucket_quotas[b]
        b_range = f"[{b * bucket_range}, {(b + 1) * bucket_range}"
        if b == len(buckets) - 1:
            b_range += "]"
        else:
            b_range += ")"
        print(f"Adding {quota} from bucket {b} {b_range} to val.")
        val += bucket[:quota]
        train += bucket[quota:]

    # Now make sure that the validation set does not have the same source or target word more than once.
    print("Ensuring validation set does not have duplicate source or duplicate target words:")
    val = _ensure_unique_words(val)
    
    print("VAL SIZE:", len(val))
    print("TRAIN SIZE:", len(train))
    assert set(train).intersection(set(val)) == set()

    random.shuffle(train)
    random.shuffle(val)

    return train, val
    
def _ensure_unique_words(dataset):
    """
    This returns a version of the dataset where a source word occurs no more than once 
    and a target word occurs no more than once.
    """
    source_words = set()
    target_words = set()
    new_dataset = []
    print("TOTAL PAIRS BEFORE:", len(dataset))
    removed = 0
    for pair in dataset:
        src_word, tgt_word = pair[-3:-1]
        if src_word in source_words or tgt_word in target_words:
            removed += 1
            print(f"\nPair ('{src_word}', '{tgt_word}') has a duplicate src or tgt word")
            continue
        source_words.add(src_word)
        target_words.add(tgt_word)
        new_dataset.append(pair)
    print("Removed", removed)
    assert len(dataset) - removed == len(new_dataset)
    print("TOTAL PAIRS AFTER:", len(new_dataset))
    return new_dataset

def get_train_split(pairs, val_pairs, seed=42):
    random.seed(seed)

    val_word_pairs = _get_just_words(val_pairs)
    train_pairs = []
    for item in pairs:
        if len(item) not in [4, 5]:
            raise ValueError(f"Items must be of len 4 or 5: (freq), freq, word1, word2, distance")
        word1, word2 = item[-3:-1]
        if (word1, word2) not in val_word_pairs:
            train_pairs.append(item)
    
    random.shuffle(train_pairs)

    return train_pairs

def _get_just_words(pairs):
    word_pairs = set()
    for item in pairs:
        if len(item) not in [4, 5]:
            raise ValueError(f"Items must be of len 4 or 5: (freq), freq, word1, word2, distance")
        word1, word2 = item[-3:-1]
        word_pairs.add((word1, word2))
    return word_pairs

def _sort_by_pair_freq(pairs):
    for item in pairs:
        assert len(item) == 4
    pairs.sort(reverse=True)
    return pairs

def _sort_by_geo_freq(pairs):
    new_pairs = []
    for item in pairs:
        assert len(item) == 5
        freq1, freq2, word1, word2, nld = item
        new_pairs.append(
            (geometric_mean([freq1, freq2]),
            freq1,freq2,
            word1,word2,
            nld)
        )
    new_pairs.sort(reverse=True)
    new_pairs = [
        (freq1, freq2, word1, word2, nld)
        for geo_mean, freq1, freq2, word1, word2, nld in new_pairs
    ]
    return new_pairs

