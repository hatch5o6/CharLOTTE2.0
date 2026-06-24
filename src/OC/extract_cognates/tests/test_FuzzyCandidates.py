import os
from sloth_hatch import sloth

from OC.extract_cognates import FuzzyCandidates as FC

######################
# extract_candidates #
######################

def test_extract_candidates():
    parent_dir = "/home/hatch5o6/CharLOTTE2.0/src/OC/extract_cognates/tests/fixtures/fast_align"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_dir = os.path.join(parent_dir, "outputs")
    sloth.create_directory(output_dir, destroy=True)
    assert os.path.exists(output_dir)
    assert os.listdir(output_dir) == []

    src_file = "src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt"
    tgt_file = "src/OC/extract_cognates/tests/fixtures/parallel/tgt_big.txt"
    
    expected_output = [
        (3, 2, "loves", "estás", 0.8),
        (2, 2, "animals", "animales", 0.125),
        (2, 1, "what", "tomará", 0.833),
        (1, 2, "very", "pero", 0.5),
        (1, 1, "will", "variedades", 0.9),
    ]
    results = FC.extract_candidates(
        src_file=src_file,
        tgt_file=tgt_file,
        src_lang="en",
        tgt_lang="es",
        word_list_out=os.path.join(output_dir, "testing_fuzz"),
        long_enough=4,
        top_k=5
    )
    for i in range(len(results)):
        item = results[i]
        assert isinstance(item, tuple)
        item = list(item)
        assert len(item) == 5
        assert isinstance(item[0], int)
        assert isinstance(item[1], int)
        assert isinstance(item[2], str)
        assert isinstance(item[3], str)
        item[4] = round(float(item[4]), 3)
        results[i] = tuple(item)
    
    assert results == expected_output

    assert os.listdir(output_dir) == ["testing_fuzz.word_pairs"]

    word_pairs = sloth.read_lines(os.path.join(output_dir, "testing_fuzz.word_pairs"))
    word_pairs = [eval(item) for item in word_pairs]
    for j in range(len(word_pairs)):
        item = word_pairs[j]
        assert isinstance(item, tuple)
        item = list(item)
        item[-1] = round(item[-1], 3)
        word_pairs[j] = tuple(item)
    assert word_pairs == expected_output



##############
# _get_words #
##############

def test_get_words_long_enough_1():
    assert FC._get_words(
        file_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt",
        long_enough=1
    ) == (
        [
            "you",
            "loves",
            "he",
            "what",
            "to",
            "not",
            "do",
            "but",
            "are",
            "animals",
            "will",
            "very",
            "tell", 
            "take",
            "sorts",
            "people",
            "much",
            "miracle",
            "me",
            "love",
            "it",
            "him",
            "doing",
            "crazy",
            "convince",
            "and",
            "all",
            "a"
        ],
        [3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    )

def test_get_words_long_enough_4():
    assert FC._get_words(
        file_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt",
        long_enough=4
    ) == (
        [
            "loves",
            "what",
            "animals",
            "will",
            "very",
            "tell",
            "take",
            "sorts",
            "people",
            "much",
            "miracle",
            "love",
            "doing",
            "crazy",
            "convince"
        ],
        [3,2,2,1,1,1,1,1,1,1,1,1,1,1,1]
    )

def test_get_words_long_enough_4_top_k():
    assert FC._get_words(
        file_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt",
        long_enough=4,
        top_k=5
    ) == (
        [
            "loves",
            "what",
            "animals",
            "will",
            "very"
        ],
        [3,2,2,1,1]
    )

    assert FC._get_words(
        file_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt",
        long_enough=4,
        top_k=3
    ) == (
        [
            "loves",
            "what",
            "animals"
        ],
        [3,2,2]
    )