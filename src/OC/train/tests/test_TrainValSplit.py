import pytest

from OC.train import TrainValSplit as TVS
#######################
# get_train_val_split #
#######################

def test_get_train_val_split_inconsistent():
    pairs = [
        (10, 4, "musica", "music", 0.1),
        (3, "pizza", "pisa?", 0.3),
        (5, "hello", "hecko", 0.3),
        (3, "pizza", "pie", 0.3),
        (1, "hungry", "hippo", 0.5),
        (3, "what?", "pisa?", 0.3),
        (5, "zebra", "zebra", 0.3),
        (1, "hungry", "hippo", 0.5)
    ]
    with pytest.raises(ValueError, match=r'Inconsistent pair length: 4\. First item is of length 5\.'):
        TVS.get_train_val_split(pairs, theta=0.5)

def test_get_train_val_split_invalid_length():
    pairs = [
        (10, 4, 2, "musica", "music", 0.1),
        (3, 2, 3, "pizza", "pisa?", 0.3),
        (5, 1, 2, "hello", "hecko", 0.3),
        (3, 1, 1, "pizza", "pie", 0.3),
        (1, 3, 4, "hungry", "hippo", 0.5),
        (3, 3, 9, "what?", "pisa?", 0.3),
        (5, 10, 3, "zebra", "zebra", 0.3),
        (1, 5, 8, "hungry", "hippo", 0.5)
    ]
    with pytest.raises(ValueError, match=r'Pair length is 6\. Pairs must be of length 4 or 5: \(freq\), freq, word1, word2, distance'):
        TVS.get_train_val_split(pairs, theta=0.5)


# This should have 6 pairs removed becuase of duplicates
TEST_PAIRS = [
    (10, "musicalness", "music", 0.4), # 3 -- DELETED
    (3, "pizza", "pie", 0.29), # 2 -- DELETED
    (5, "hello", "hecko", 0.31), # 3 -- DELETED
    (5, "hello", "heko", 0.21), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (1, "hungry", "hippo", 0.37), # 3 -- DELETED
    (3, "what?", "pisa?", 0.39), # 3 -- DELETED
    (5, "zebra", "zebra", 0.31), # 3
    (10, "musical", "music", 0.16), # 1
    (1, "hungry", "hippo", 0.07), # 0
    (6, "legion", "lechion", 0.05), # 0
    (7, "happee", "happy", 0.21), # 2
    (9, "aabb", "aaaa", 0.4), # 3
    (10, "cccd", "cccc", 0.25), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (7, "orange", "naranja", 0.4), # 3
    (4, "miracle", "milagro", 0.33), # 3
    (2, "ocean", "oceano", 0.17), # 1
    (8, "elephant", "elefante", 0.21), # 2
    (9, "lion", "leon", 0.25), # 2
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "animals", "animales", 0.13), # 1
    (3, "mucho", "much", 0.21), # 2
    (4, "horse", "herse", 0.04), # 0
    (5, "party", "fiesta", 0.4), # 3
    (7, "fridge", "freza", 0.16), # 1
    (8, "giraffe", "jirafa", 0.03), # 0
    (1, "smoke", "stove", 0.31), # 3
    (2, "cookies", "biscuits", 0.4), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (6, "faint", "fanta", 0.05), # 0
    (8, "snake", "serpiente", 0.4), # 3
    (5, "ten", "dies", 0.4), # 3
    (3, "stomach", "estomago", 0.28), # 2
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (12, "stop", "pop", 0.31), # 3
    (9, "treet", "meet", 0.31) # 3 -- DELETED
]
# 3s: 11
# 2s: 9
# 1s: 6
# 0s: 6

def test_get_train_val_split_basic():
    size=12
    train, val = TVS.get_train_val_split(
        pairs=TEST_PAIRS,
        theta=0.4,
        size=size,
        n_buckets=4,
        max_fraction=0.5,
        seed=42
    )
    # assert correct pairs were deleted
    assert len(TEST_PAIRS) == 38
    unique_test_pairs = TVS._ensure_unique_words(TVS._sort_by_NLD(TEST_PAIRS))
    assert len(unique_test_pairs) == 32
    for item in [
        (10, "musicalness", "music", 0.4), # 3 -- DELETED
        (3, "pizza", "pie", 0.29), # 2 -- DELETED
        (5, "hello", "hecko", 0.31), # 3 -- DELETED
        (1, "hungry", "hippo", 0.37), # 3 -- DELETED
        (3, "what?", "pisa?", 0.39), # 3 -- DELETED
        (9, "treet", "meet", 0.31) # 3 -- DELETED
    ]:
        assert item in TEST_PAIRS
        assert item not in unique_test_pairs

    # assert total length of val + train is correct
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len( set(train).union(val) ) == len(unique_test_pairs) == 32

    # assert train and val content come from original data
    assert sorted( set(train).union(val) ) == sorted(unique_test_pairs)

    # assert len of val is correct (more thorough testing later)
    assert len(val) == size

    # assert no overlap of val and train
    assert set(train).intersection(val) == set()

    # assert precise val content
    assert set(val) == {
        # bucket 3:
        (12, "stop", "pop", 0.31), # 3
        (9, "aabb", "aaaa", 0.4), # 3
        (8, "snake", "serpiente", 0.4), # 3

        # bucket 2:
        (14, "meat", "meet", 0.24), # 2
        (13, "xxxy", "yyyy", 0.22), # 2
        (10, "cccd", "cccc", 0.25), # 2

        # bucket 1:
        (10, "musical", "music", 0.16), # 1
        (7, "fridge", "freza", 0.16), # 1
        (4, "mercantile", "domicile", 0.12), # 1

        # bucket 0:
        (15, "sandwich", "sandwitch", 0.03), # 0
        (8, "giraffe", "jirafa", 0.03), # 0
        (6, "legion", "lechion", 0.05) # 0
    }


def test_get_train_val_split_floor_division():
    # tests case where val < size due to floor division of theta / n_buckets
    size=13
    train, val = TVS.get_train_val_split(
        pairs=TEST_PAIRS,
        theta=0.4,
        size=size,
        n_buckets=4,
        max_fraction=0.5,
        seed=42
    )
    # assert correct pairs were deleted
    assert len(TEST_PAIRS) == 38
    unique_test_pairs = TVS._ensure_unique_words(TVS._sort_by_NLD(TEST_PAIRS))
    assert len(unique_test_pairs) == 32
    for item in [
        (10, "musicalness", "music", 0.4), # 3 -- DELETED
        (3, "pizza", "pie", 0.29), # 2 -- DELETED
        (5, "hello", "hecko", 0.31), # 3 -- DELETED
        (1, "hungry", "hippo", 0.37), # 3 -- DELETED
        (3, "what?", "pisa?", 0.39), # 3 -- DELETED
        (9, "treet", "meet", 0.31) # 3 -- DELETED
    ]:
        assert item in TEST_PAIRS
        assert item not in unique_test_pairs

    # assert total length of val + train is correct
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len( set(train).union(val) ) == len(unique_test_pairs) == 32

    # assert train and val content come from original data
    assert sorted( set(train).union(val) ) == sorted(unique_test_pairs)

    # assert len of val is correct, i.e. size (13) - 1 because of floor division when calculating bucket default quota:
    # 13 (size) // 4 (n_buckets) == 3 (default_quota)
    # 3 (default_quota) * 4 (n_buckets) == 12
    assert len(val) == 12

    # assert no overlap of val and train
    assert set(train).intersection(val) == set()

    # assert precise val content
    assert set(val) == {
        # bucket 3:
        (12, "stop", "pop", 0.31), # 3
        (9, "aabb", "aaaa", 0.4), # 3
        (8, "snake", "serpiente", 0.4), # 3

        # bucket 2:
        (14, "meat", "meet", 0.24), # 2
        (13, "xxxy", "yyyy", 0.22), # 2
        (10, "cccd", "cccc", 0.25), # 2

        # bucket 1:
        (10, "musical", "music", 0.16), # 1
        (7, "fridge", "freza", 0.16), # 1
        (4, "mercantile", "domicile", 0.12), # 1

        # bucket 0:
        (15, "sandwich", "sandwitch", 0.03), # 0
        (8, "giraffe", "jirafa", 0.03), # 0
        (6, "legion", "lechion", 0.05) # 0
    }
    



# This should have 6 pairs removed becuase of duplicates
TEST_PAIRS_CAP = [
    (10, "musicalness", "music", 0.4), # 3 -- DELETED
    (3, "pizza", "pie", 0.29), # 2 -- DELETED
    (5, "hello", "hecko", 0.31), # 3 -- DELETED
    (5, "hello", "heko", 0.21), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (1, "hungry", "hippo", 0.37), # 3 -- DELETED
    (3, "what?", "pisa?", 0.39), # 3 -- DELETED
    (5, "zebra", "zebra", 0.31), # 3
    (10, "musical", "music", 0.16), # 1
    (1, "hungry", "hippo", 0.07), # 0
    (6, "legion", "lechion", 0.25), # 2
    (7, "happee", "happy", 0.21), # 2
    (9, "aabb", "aaaa", 0.4), # 3
    (10, "cccd", "cccc", 0.25), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (7, "orange", "naranja", 0.4), # 3
    (4, "miracle", "milagro", 0.23), # 2
    (2, "ocean", "oceano", 0.17), # 1
    (8, "elephant", "elefante", 0.21), # 2
    (9, "lion", "leon", 0.25), # 2
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "animals", "animales", 0.23), # 2
    (3, "mucho", "much", 0.21), # 2
    (4, "horse", "herse", 0.04), # 0
    (5, "party", "fiesta", 0.4), # 3
    (7, "fridge", "freza", 0.16), # 1
    (8, "giraffe", "jirafa", 0.26), # 2
    (1, "smoke", "stove", 0.21), # 2
    (2, "cookies", "biscuits", 0.4), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (6, "faint", "fanta", 0.26), # 2
    (8, "snake", "serpiente", 0.4), # 3
    (5, "ten", "dies", 0.4), # 3
    (3, "stomach", "estomago", 0.28), # 2
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (12, "stop", "pop", 0.31), # 3
    (9, "treet", "meet", 0.31) # 3 -- DELETED
]
# 3s: 9
# 2s: 15
# 1s: 5
# 0s: 3

def test_get_train_val_split_buckets_capped():
    size=12
    train, val = TVS.get_train_val_split(
        pairs=TEST_PAIRS_CAP,
        theta=0.4,
        size=size,
        n_buckets=4,
        max_fraction=0.5,
        seed=42
    )
    # assert correct pairs were deleted
    assert len(TEST_PAIRS_CAP) == 38
    unique_test_pairs = TVS._ensure_unique_words(TVS._sort_by_NLD(TEST_PAIRS_CAP))
    assert len(unique_test_pairs) == 32
    for item in [
        (10, "musicalness", "music", 0.4), # 3 -- DELETED
        (3, "pizza", "pie", 0.29), # 2 -- DELETED
        (5, "hello", "hecko", 0.31), # 3 -- DELETED
        (1, "hungry", "hippo", 0.37), # 3 -- DELETED
        (3, "what?", "pisa?", 0.39), # 3 -- DELETED
        (9, "treet", "meet", 0.31) # 3 -- DELETED
    ]:
        assert item in TEST_PAIRS_CAP
        assert item not in unique_test_pairs

    # assert total length of val + train is correct
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len( set(train).union(val) ) == len(unique_test_pairs) == 32

    # assert train and val content come from original data
    assert sorted( set(train).union(val) ) == sorted(unique_test_pairs)

    # assert len of val is correct (more thorough testing later)
    assert len(val) == size

    # assert no overlap of val and train
    assert set(train).intersection(val) == set()

    # assert precise val content
    assert set(val) == {
        # bucket 3 (capped at 4):
        (12, 'stop', 'pop', 0.31), # 3
        (9, 'aabb', 'aaaa', 0.4), # 3
        (8, 'snake', 'serpiente', 0.4), # 3
        (7, 'orange', 'naranja', 0.4), # 3
        
        # bucket 2 (capped at 7):
        (14, 'meat', 'meet', 0.24), # 2
        (13, 'xxxy', 'yyyy', 0.22), # 2
        (10, 'cccd', 'cccc', 0.25), # 2
        (9, 'lion', 'leon', 0.25), # 2
        (8, 'giraffe', 'jirafa', 0.26), # 2
        
        # bucket 1 (capped at 2):
        (10, 'musical', 'music', 0.16), # 1
        (7, 'fridge', 'freza', 0.16), # 1

        # bucket 0 (capped at 1):
        (15, 'sandwich', 'sandwitch', 0.03), # 0
    }


# This should have 6 pairs removed becuase of duplicates
TEST_PAIRS_CAP_DEFECIT = [
    (10, "musicalness", "music", 0.4), # 3 -- DELETED
    (3, "pizza", "pie", 0.29), # 2 -- DELETED
    (5, "hello", "hecko", 0.31), # 3 -- DELETED
    (5, "hello", "heko", 0.21), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (1, "hungry", "hippo", 0.37), # 3 -- DELETED
    (3, "what?", "pisa?", 0.39), # 3 -- DELETED
    (5, "zebra", "zebra", 0.31), # 3
    (10, "musical", "music", 0.16), # 1
    (1, "hungry", "hippo", 0.07), # 0
    (9, "aabb", "aaaa", 0.4), # 3
    (10, "cccd", "cccc", 0.25), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (7, "orange", "naranja", 0.4), # 3
    (4, "miracle", "milagro", 0.23), # 2
    (2, "ocean", "oceano", 0.17), # 1
    (9, "lion", "leon", 0.25), # 2
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "animals", "animales", 0.23), # 2
    (3, "mucho", "much", 0.21), # 2
    (4, "horse", "herse", 0.04), # 0
    (7, "fridge", "freza", 0.16), # 1
    (1, "smoke", "stove", 0.21), # 2
    (2, "cookies", "biscuits", 0.4), # 3
    (8, "snake", "serpiente", 0.4), # 3
    (5, "ten", "dies", 0.4), # 3
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (9, "treet", "meet", 0.31) # 3 -- DELETED
]
# 3s: 6
# 2s: 9
# 1s: 5
# 0s: 3

def test_get_train_val_split_buckets_capped_deficit():
    size=12
    train, val = TVS.get_train_val_split(
        pairs=TEST_PAIRS_CAP_DEFECIT,
        theta=0.4,
        size=size,
        n_buckets=4,
        max_fraction=0.5,
        seed=42
    )
    # assert correct pairs were deleted
    assert len(TEST_PAIRS_CAP_DEFECIT) == 29
    unique_test_pairs = TVS._ensure_unique_words(TVS._sort_by_NLD(TEST_PAIRS_CAP_DEFECIT))
    assert len(unique_test_pairs) == 23
    for item in [
        (10, "musicalness", "music", 0.4), # 3 -- DELETED
        (3, "pizza", "pie", 0.29), # 2 -- DELETED
        (5, "hello", "hecko", 0.31), # 3 -- DELETED
        (1, "hungry", "hippo", 0.37), # 3 -- DELETED
        (3, "what?", "pisa?", 0.39), # 3 -- DELETED
        (9, "treet", "meet", 0.31) # 3 -- DELETED
    ]:
        assert item in TEST_PAIRS_CAP_DEFECIT
        assert item not in unique_test_pairs

    # assert total length of val + train is correct
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len( set(train).union(val) ) == len(unique_test_pairs) == 23

    # assert train and val content come from original data
    assert sorted( set(train).union(val) ) == sorted(unique_test_pairs)

    # assert len of val is correct (more thorough testing later)
    assert len(val) == 10

    # assert no overlap of val and train
    assert set(train).intersection(val) == set()

    # assert precise val content
    assert set(val) == {
        # bucket 3 (capped at 3):
        (9, 'aabb', 'aaaa', 0.4), # 3
        (8, 'snake', 'serpiente', 0.4), # 3
        (7, 'orange', 'naranja', 0.4), # 3
        
        # bucket 2 (capped at 4):
        (14, 'meat', 'meet', 0.24), # 2
        (13, 'xxxy', 'yyyy', 0.22), # 2
        (10, 'cccd', 'cccc', 0.25), # 2
        (9, 'lion', 'leon', 0.25), # 2
        
        # bucket 1 (capped at 2):
        (10, 'musical', 'music', 0.16), # 1
        (7, 'fridge', 'freza', 0.16), # 1
        
        # bucket 0 (capped at 1):
        (15, 'sandwich', 'sandwitch', 0.03), # 0
    }



# This should have 6 pairs removed becuase of duplicates
TEST_PAIRS_FUZZ = [
    (10, 4, "musicalness", "music", 0.4), # 3 -- DELETED
    (3, 5, "pizza", "pie", 0.29), # 2 -- DELETED
    (5, 8, "hello", "hecko", 0.31), # 3 -- DELETED
    (5, 2, "hello", "heko", 0.21), # 2
    (3, 4, "pizza", "pisa?", 0.17), # 1
    (1, 10, "hungry", "hippo", 0.37), # 3 -- DELETED
    (3, 4, "what?", "pisa?", 0.39), # 3 -- DELETED
    (5, 6, "zebra", "zebra", 0.31), # 3
    (10, 9, "musical", "music", 0.16), # 1
    (1, 4, "hungry", "hippo", 0.07), # 0
    (6, 7, "legion", "lechion", 0.05), # 0
    (7, 4, "happee", "happy", 0.21), # 2
    (9, 5, "aabb", "aaaa", 0.4), # 3
    (10, 7, "cccd", "cccc", 0.25), # 2
    (13, 10, "xxxy", "yyyy", 0.22), # 2
    (7, 6, "orange", "naranja", 0.4), # 3
    (4, 3, "miracle", "milagro", 0.33), # 3
    (2, 4, "ocean", "oceano", 0.17), # 1
    (8, 6, "elephant", "elefante", 0.21), # 2
    (9, 8, "lion", "leon", 0.25), # 2
    (4, 5, "mercantile", "domicile", 0.12), # 1
    (4, 5, "animals", "animales", 0.13), # 1
    (3, 6, "mucho", "much", 0.21), # 2
    (4, 3, "horse", "herse", 0.04), # 0
    (5, 8, "party", "fiesta", 0.4), # 3
    (7, 2, "fridge", "freza", 0.16), # 1
    (8, 1, "giraffe", "jirafa", 0.03), # 0
    (1, 5, "smoke", "stove", 0.31), # 3
    (2, 8, "cookies", "biscuits", 0.4), # 3
    (4, 6, "mermaid", "merman", 0.4), # 3
    (6, 6, "faint", "fanta", 0.05), # 0
    (8, 7, "snake", "serpiente", 0.4), # 3
    (5, 4, "ten", "dies", 0.4), # 3
    (3, 2, "stomach", "estomago", 0.28), # 2
    (15, 8, "sandwich", "sandwitch", 0.03), # 0
    (14, 15, "meat", "meet", 0.24), # 2
    (12, 10, "stop", "pop", 0.31), # 3
    (9, 6, "treet", "meet", 0.31) # 3 -- DELETED
]
# 3s: 11
# 2s: 9
# 1s: 6
# 0s: 6

def test_get_train_val_split_basic_monolingual():
    size=12
    train, val = TVS.get_train_val_split(
        pairs=TEST_PAIRS_FUZZ,
        theta=0.4,
        size=size,
        n_buckets=4,
        max_fraction=0.5,
        seed=42
    )
    # assert correct pairs were deleted
    assert len(TEST_PAIRS_FUZZ) == 38
    unique_test_pairs = TVS._ensure_unique_words(TVS._sort_by_NLD(TEST_PAIRS_FUZZ))
    assert len(unique_test_pairs) == 32
    for item in [
        (10, 4, "musicalness", "music", 0.4), # 3 -- DELETED
        (3, 5, "pizza", "pie", 0.29), # 2 -- DELETED
        (5, 8, "hello", "hecko", 0.31), # 3 -- DELETED
        (1, 10, "hungry", "hippo", 0.37), # 3 -- DELETED
        (3, 4, "what?", "pisa?", 0.39), # 3 -- DELETED
        (9, 6, "treet", "meet", 0.31) # 3 -- DELETED
    ]:
        assert item in TEST_PAIRS_FUZZ
        assert item not in unique_test_pairs

    # assert total length of val + train is correct
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len( set(train).union(val) ) == len(unique_test_pairs) == 32

    # assert train and val content come from original data
    assert sorted( set(train).union(val) ) == sorted(unique_test_pairs)

    # assert len of val is correct (more thorough testing later)
    assert len(val) == size

    # assert no overlap of val and train
    assert set(train).intersection(val) == set()

    # assert precise val content
    assert set(val) == {
        # bucket 3:
        (12, 10, 'stop', 'pop', 0.31), # 3
        (8, 7, 'snake', 'serpiente', 0.4), # 3
        (9, 5, 'aabb', 'aaaa', 0.4), # 3

        # bucket 2:
        (14, 15, 'meat', 'meet', 0.24), # 2
        (13, 10, 'xxxy', 'yyyy', 0.22), # 2
        (9, 8, 'lion', 'leon', 0.25), # 2

        # bucket 1:
        (10, 9, 'musical', 'music', 0.16), # 1
        (4, 5, 'mercantile', 'domicile', 0.12), # 1
        (4, 5, 'animals', 'animales', 0.13), # 1

        # bucket 0:
        (15, 8, 'sandwich', 'sandwitch', 0.03), # 0
        (6, 7, 'legion', 'lechion', 0.05), # 0
        (6, 6, 'faint', 'fanta', 0.05) # 0
    }

########################
# _ensure_unique_words #
########################

def test_ensure_unique_words_parallel():
    pairs = [
        (10, "musica", "music", 0.1),
        (3, "pizza", "pisa?", 0.3),
        (5, "hello", "hecko", 0.3),
        (3, "pizza", "pie", 0.3),
        (1, "hungry", "hippo", 0.5),
        (3, "what?", "pisa?", 0.3),
        (5, "zebra", "zebra", 0.3),
        (1, "hungry", "hippo", 0.5)
    ]
    assert TVS._ensure_unique_words(pairs) == [
        (10, "musica", "music", 0.1),
        (3, "pizza", "pisa?", 0.3),
        (5, "hello", "hecko", 0.3),
        (1, "hungry", "hippo", 0.5),
        (5, "zebra", "zebra", 0.3)
    ]

def test_ensure_unique_words_monolingual():
    pairs = [
        (10, 5, "musica", "music", 0.1),
        (3, 21, "pizza", "pisa?", 0.3),
        (5, 4, "hello", "hecko", 0.3),
        (3, 2, "pizza", "pie", 0.3),
        (1, 16, "hungry", "hippo", 0.5),
        (3, 2, "what?", "pisa?", 0.3),
        (5, 4, "zebra", "zebra", 0.3),
        (1, 3, "hungry", "hippo", 0.5)
    ]
    assert TVS._ensure_unique_words(pairs) == [
        (10, 5, "musica", "music", 0.1),
        (3, 21, "pizza", "pisa?", 0.3),
        (5, 4, "hello", "hecko", 0.3),
        (1, 16, "hungry", "hippo", 0.5),
        (5, 4, "zebra", "zebra", 0.3)
    ]

def test_ensure_unique_words_invalid():
    pairs = [
        (10, 3, 4, "musica", "music", 0.1),
        (3, "pizza", "pisa?", 0.3),
        (5, "hello", "hecko", 0.3),
        (3, "pizza", "pie", 0.3),
        (1, "hungry", "hippo", 0.5),
        (3, "what?", "pisa?", 0.3),
        (5, "zebra", "zebra", 0.3),
        (1, "hungry", "hippo", 0.5)
    ]
    with pytest.raises(ValueError, match=r"Items must be of len 4 or 5: \(freq\), freq, word1, word2, distance\. Got length 6\."):
        TVS._ensure_unique_words(pairs)

###################
# get_train_split #
###################

# def test_get_train_split():
#     pass

###################
# _get_just_words #
###################

# def test_get_just_words_parallel():
#     pairs = [
#         (3, "pizza", "pisa?", 0.3),
#         (10, "musica", "music", 0.1),
#         (1, "hungry", "hippo", 0.5),
#         (5, "hello", "hecko", 0.3),
#         (5, "zebra", "zebra", 0.3)
#     ]
#     assert TVS._get_just_words(pairs) == {
#         ("pizza", "pisa?"),
#         ("musica", "music"),
#         ("hungry", "hippo"),
#         ("hello", "hecko"),
#         ("zebra", "zebra")
#     }

# def test_get_just_words_monolingual():
#     pairs = [
#         (3, 2, "pizza", "pisa?", 0.3),
#         (10, 100, "musica", "music", 0.1),
#         (1, 76, "hungry", "hippo", 0.5),
#         (5, 3, "hello", "hecko", 0.3),
#         (5, 2, "zebra", "zebra", 0.3)
#     ]
#     assert TVS._get_just_words(pairs) == {
#         ("pizza", "pisa?"),
#         ("musica", "music"),
#         ("hungry", "hippo"),
#         ("hello", "hecko"),
#         ("zebra", "zebra")
#     }

# def test_get_just_words_invalid():
#     pairs = [
#         (3, 2, 6, "pizza", "pisa?", 0.3),
#         (10, 100, "musica", "music", 0.1),
#         (1, 76, "hungry", "hippo", 0.5),
#         (5, 3, "hello", "hecko", 0.3),
#         (5, 2, "zebra", "zebra", 0.3)
#     ]
#     with pytest.raises(ValueError, match=r"Items must be of len 4 or 5: \(freq\), freq, word1, word2, distance"):
#         TVS._get_just_words(pairs)

######################
# _sort_by_pair_freq #
######################

def test_sort_by_pair_freq():
    pairs = [
        (3, "pizza", "pisa?", 0.3),
        (10, "musica", "music", 0.1),
        (1, "hungry", "hippo", 0.5),
        (5, "hello", "hecko", 0.3),
        (5, "zebra", "zebra", 0.3)
    ]
    assert TVS._sort_by_pair_freq(pairs) == [
        (10, "musica", "music", 0.1),
        (5, "zebra", "zebra", 0.3),
        (5, "hello", "hecko", 0.3),
        (3, "pizza", "pisa?", 0.3),
        (1, "hungry", "hippo", 0.5)
    ]

def test_sort_by_pair_freq_bad_item():
    pairs = [
        (3, 2, "pizza", "pisa?", 0.3),
        (10, "musica", "music", 0.1),
        (1, "hungry", "hippo", 0.5),
        (5, "hello", "hecko", 0.3),
        (5, "zebra", "zebra", 0.3)
    ]
    with pytest.raises(AssertionError):
        TVS._sort_by_pair_freq(pairs)


#####################
# _sort_by_geo_freq #
#####################

def test_sort_by_geo_freq():
    pairs = [
        (3, 2, "pizza", "pisa?", 0.3),
        (10, 1, "musica", "music", 0.1),
        (1, 3, "hungry", "hippo", 0.5),
        (5, 1, "hello", "hecko", 0.3),
        (5, 9, "zebra", "zebra", 0.3)
    ]
    sorted_pairs = TVS._sort_by_geo_freq(pairs) 
    assert sorted_pairs == [
        (5, 9, "zebra", "zebra", 0.3),
        (10, 1, "musica", "music", 0.1),
        (3, 2, "pizza", "pisa?", 0.3),
        (5, 1, "hello", "hecko", 0.3),
        (1, 3, "hungry", "hippo", 0.5)
    ]
    assert sorted(sorted_pairs) == sorted(pairs)

def test_sort_by_geo_freq_bad_item():
    pairs = [
        (3, 2, "pizza", "pisa?", 0.3),
        (10, 4, "musica", "music", 0.1),
        (1, "hungry", "hippo", 0.5),
        (5, 1, "hello", "hecko", 0.3),
        (5, 9, "zebra", "zebra", 0.3)
    ]
    with pytest.raises(AssertionError):
        TVS._sort_by_geo_freq(pairs)


################
# _sort_by_NLD #
################
def test_sort_by_NLD():
    pairs = [
        (10, 5, "musica", "music", 0.8),
        (3, 21, "pizza", "pisa?", 0.3),
        (5, 4, "hello", "hecko", 0.4),
        (3, 2, "pizza", "pie", 0.1),
        (1, 16, "hungry", "hippo", 0.5),
        (3, 2, "what?", "pisa?", 0.2),
        (5, 4, "zebra", "zebra", 0.9),
        (1, 3, "hungry", "hippo", 0.5)
    ]
    pairs = TVS._sort_by_NLD(pairs)
    assert pairs == [
        (3, 2, "pizza", "pie", 0.1),
        (3, 2, "what?", "pisa?", 0.2),
        (3, 21, "pizza", "pisa?", 0.3),
        (5, 4, "hello", "hecko", 0.4),
        (1, 16, "hungry", "hippo", 0.5),
        (1, 3, "hungry", "hippo", 0.5),
        (10, 5, "musica", "music", 0.8),
        (5, 4, "zebra", "zebra", 0.9)
    ]

