import pytest
from OC.extract_cognates import Cognates
import os
import shutil
from sloth_hatch import sloth

#################
# make_cognates #
#################

def test_make_cognates_from_parallel():
    from OC.extract_cognates.CandidatesFromParallel import extract_candidates
    out_dir = "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []

    cognates = Cognates.make_cognates(
        src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
        tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
        src_lang="en",
        tgt_lang="es",
        out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates",
        long_enough=4,
        theta=0.5,
        extract_candidates=extract_candidates
    )

    # test directory structure
    assert set(os.listdir(out_dir)) == {"test_make_cognates.sents",
                                        "test_make_cognates.fwd",
                                        "test_make_cognates.rev",
                                        "test_make_cognates.sym",
                                        "test_make_cognates.word_pairs",
                                        "test_make_cognates.cognates"}
    
    # test cognates
    assert isinstance(cognates, list)
    for item in cognates:
        assert isinstance(item, tuple)
        assert len(item) == 4
        freq, w1, w2, dist = item
        assert isinstance(freq, int)
        assert isinstance(w1, str)
        assert isinstance(w2, str)
        assert isinstance(dist, float)
    
    # test that cognates and cognates file have the same content
    cognates_out_f = os.path.join(out_dir, "test_make_cognates.cognates")
    cognates_out_f_lines = sloth.read_lines(cognates_out_f)
    cognates_out_f_data = [tuple(line.split(" ||| ")) for line in cognates_out_f_lines]
    cognates_out_f_data = [(int(freq), w1, w2, float(dist)) for freq, w1, w2, dist in cognates_out_f_data]
    assert cognates_out_f_data == cognates

    shutil.rmtree(out_dir)


def test_make_cognates_from_parallel_return_file():
    from OC.extract_cognates.CandidatesFromParallel import extract_candidates
    out_dir = "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []

    cognates = Cognates.make_cognates(
        src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
        tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
        src_lang="en",
        tgt_lang="es",
        out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates",
        long_enough=4,
        theta=0.5,
        extract_candidates=extract_candidates
    )

    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []

    cognates_file = Cognates.make_cognates(
        src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
        tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
        src_lang="en",
        tgt_lang="es",
        out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates",
        long_enough=4,
        theta=0.5,
        extract_candidates=extract_candidates,
        return_cognates=False
    )

    # test cognates
    assert isinstance(cognates, list)
    for item in cognates:
        assert isinstance(item, tuple)
        assert len(item) == 4
        freq, w1, w2, dist = item
        assert isinstance(freq, int)
        assert isinstance(w1, str)
        assert isinstance(w2, str)
        assert isinstance(dist, float)
    
    # test file has same content as cognates list
    assert isinstance(cognates_file, str)
    assert cognates_file == "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates.cognates" == os.path.join(out_dir, "test_make_cognates.cognates")
    cognates_file_content = [tuple(line.split(" ||| ")) for line in sloth.read_lines(cognates_file)]
    cognates_file_content = [(int(freq), w1, w2, float(dist)) for freq, w1, w2, dist in cognates_file_content]
    assert cognates_file_content == cognates
    
    shutil.rmtree(out_dir)

def test_make_cognates_from_parallel_already_exists_valid(capsys):
    from OC.extract_cognates.CandidatesFromParallel import extract_candidates
    out_dir = "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []
    shutil.copy(
        "src/OC/extract_cognates/tests/fixtures/make_cognates/test_make_cognates.cognates",
        out_dir
    )
    assert os.listdir(out_dir) == ["test_make_cognates.cognates"]

    cognates = Cognates.make_cognates(
        src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
        tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
        src_lang="en",
        tgt_lang="es",
        out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates",
        long_enough=4,
        theta=0.5,
        extract_candidates=extract_candidates
    )

    captured = capsys.readouterr()
    assert "cognates outputs file `src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates.cognates` already exists, but the contents match what was extracted." in captured.out
    shutil.rmtree(out_dir)

def test_make_cognates_from_parallel_already_exists_invalid():
    from OC.extract_cognates.CandidatesFromParallel import extract_candidates
    out_dir = "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []
    shutil.copy(
        "src/OC/extract_cognates/tests/fixtures/make_cognates/test_make_cognates.cognates.invalid",
        os.path.join(out_dir, "test_make_cognates.cognates")
    )
    assert os.listdir(out_dir) == ["test_make_cognates.cognates"]
    
    with pytest.raises(ValueError, match=r'cognates outputs file `src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates\.cognates` exists already, but does not match cognate list just created!'):
        Cognates.make_cognates(
            src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
            tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
            src_lang="en",
            tgt_lang="es",
            out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_cognates",
            long_enough=4,
            theta=0.5,
            extract_candidates=extract_candidates
        )
    shutil.rmtree(out_dir)


#TODO test make_cognates (from Fuzzy Candidates)
def test_make_fuzzy_cognates():
    from OC.extract_cognates.FuzzyCandidates import extract_candidates
    out_dir = "src/OC/extract_cognates/tests/fixtures/make_cognates_outputs"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []
    
    cognates = Cognates.make_cognates(
        src_path="src/OC/extract_cognates/tests/fixtures/parallel/src_big2.txt",
        tgt_path="src/OC/extract_cognates/tests/fixtures/parallel/tgt_big2.txt",
        src_lang="en",
        tgt_lang="es",
        out_stem="src/OC/extract_cognates/tests/fixtures/make_cognates_outputs/test_make_fuzzy_cognates",
        long_enough=4,
        theta=0.25,
        extract_candidates=extract_candidates
    )

    assert set(os.listdir(out_dir)) == {"test_make_fuzzy_cognates.word_pairs",
                                        "test_make_fuzzy_cognates.cognates"}
    
    assert isinstance(cognates, list)
    for item in cognates:
        assert isinstance(item, tuple)
        assert len(item) == 5
        src_freq, tgt_freq, src_word, tgt_word, dist = item
        assert isinstance(src_freq, int)
        assert isinstance(tgt_freq, int)
        assert isinstance(src_word, str)
        assert isinstance(tgt_word, str)
        assert isinstance(dist, float)
    
    cognates_out_f = os.path.join(out_dir, "test_make_fuzzy_cognates.cognates")
    cognates_out_f_content = [l.split(" ||| ") for l in sloth.read_lines(cognates_out_f)]
    cognates_out_f_content = [(int(freq1), int(freq2), w1, w2, float(nld))
                              for freq1, freq2, w1, w2, nld in cognates_out_f_content]
    assert cognates_out_f_content == cognates

    shutil.rmtree(out_dir)


#########################
# _filter_cognate_pairs #
#########################

def test_filter_cognate_pairs_parallel():
    candidates = [
        (5, "legion", "lechion"),
        (4, "1,000", "2,000"),
        (3, "music", "sonido"),
        (5, "aaaaaa", "aaaaaa"),
        (5, "aaaaab", "aaaaaa"),
        (5, "aaaabb", "aaaaaa"),
        (5, "aaabbb", "aaaaaa"),
        (5, "aabbbb", "aaaaaa"),
        (5, "abbbbb", "aaaaaa"),
        (5, "bbbbbb", "aaaaaa"),
        (8, "general", "cheneral")
    ]
    cognates = Cognates._filter_cognate_pairs(
        word_pairs=candidates,
        theta=0.5,
        long_enough=4
    )
    assert cognates == [
        (5, "legion", "lechion", 0.2857142857142857),
        (5, "aaaaaa", "aaaaaa", 0.0),
        (5, "aaaaab", "aaaaaa", 0.16666666666666666),
        (5, "aaaabb", "aaaaaa", 0.3333333333333333),
        (5, "aaabbb", "aaaaaa", 0.5),
        (8, "general", "cheneral", 0.25),
        (4, "1,000", "1,000", 0.0),
        (4, "2,000", "2,000", 0.0),
    ]

def test_filter_cognate_pairs_monolingual():
    candidates = [
        (5, 6, "legion", "lechion", 0.2857142857142857),
        (4, 2, "1,000", "2,000", 0.1),
        (3, 2, "music", "sonido", 0.8333333333333334),
        (5, 1, "aaaaaa", "aaaaaa", 0.0),
        (5, 4, "aaaaab", "aaaaaa", 0.16666666666666666),
        (5, 1, "aaaabb", "aaaaaa", 0.3333333333333333),
        (5, 3, "aaabbb", "aaaaaa", 0.5),
        (5, 6, "aabbbb", "aaaaaa", 0.6666666666666666),
        (5, 7, "abbbbb", "aaaaaa", 0.8333333333333334),
        (5, 2, "bbbbbb", "aaaaaa", 1.0),
        (8, 3, "general", "cheneral", 0.25)
    ]
    cognates = Cognates._filter_cognate_pairs(
        word_pairs=candidates,
        theta=0.5,
        long_enough=4
    )
    assert cognates == [
        (5, 6, "legion", "lechion", 0.2857142857142857),
        (5, 1, "aaaaaa", "aaaaaa", 0.0),
        (5, 4, "aaaaab", "aaaaaa", 0.16666666666666666),
        (5, 1, "aaaabb", "aaaaaa", 0.3333333333333333),
        (5, 3, "aaabbb", "aaaaaa", 0.5),
        (8, 3, "general", "cheneral", 0.25),
        (4, 4, "1,000", "1,000", 0.0),
        (2, 2, "2,000", "2,000", 0.0),
    ]

def test_filter_cognate_pairs_mixed():
    candidates = [
        (5, 6, "legion", "lechion", 0.2857142857142857),
        (4, "1,000", "2,000", 0.1),
        (3, 2, "music", "sonido", 0.8333333333333334),
        (5, 1, "aaaaaa", "aaaaaa", 0.0),
        (5, "aaaaab", "aaaaaa", 0.16666666666666666),
        (5, 1, "aaaabb", "aaaaaa", 0.3333333333333333),
        (5, 3, "aaabbb", "aaaaaa", 0.5),
        (5, 6, "aabbbb", "aaaaaa", 0.6666666666666666),
        (5, "abbbbb", "aaaaaa", 0.8333333333333334),
        (5, 2, "bbbbbb", "aaaaaa", 1.0),
        (8, 3, "general", "cheneral", 0.25)
    ]
    with pytest.raises(ValueError, match="cannot combine parallel and monolingual frequencies!"):
        Cognates._filter_cognate_pairs(
            word_pairs=candidates,
            theta=0.5,
            long_enough=4
        )

def test_filter_cognate_pairs_failed_redundant_cleaning():
    candidates = [
        (5, "legion", "lechion"),
        (4, "1,000", "2,000"),
        (3, "music", "sonido"),
        (5, "aaaaaa", "aaaaaa"),
        (5, "aaaaab.", "aaaaaa"),
        (5, "aaaabb", "aaaaaa")
    ]
    with pytest.raises(AssertionError, match=r'word1 \(`aaaaab\.`\) cleaning shouldn\'t change it but it does: `aaaaab`'):
        Cognates._filter_cognate_pairs(
            word_pairs=candidates,
            theta=0.5,
            long_enough=4
        )
    
    candidates = [
        (5, "legion", "lechion"),
        (4, "1,000", "2,000"),
        (3, "music", "sonido"),
        (5, "aaaaaa", "aaaaaa"),
        (5, "aaaaab", ";;aaaaaa?"),
        (5, "aaaabb", "aaaaaa")
    ]
    with pytest.raises(AssertionError, match=r'word2 \(`;;aaaaaa\?`\) cleaning shouldn\'t change it but it does: `aaaaaa`'):
        Cognates._filter_cognate_pairs(
            word_pairs=candidates,
            theta=0.5,
            long_enough=4
        )
    


######################
# _get_decimal_pairs #
######################
def test_get_decimal_pairs_errors():
    with pytest.raises(ValueError, match=r'freq must be a tuple!'):
        Cognates._get_decimal_pairs(100, "1001", "10002")
    with pytest.raises(ValueError, match=r'freq must be of length 1 or 2!'):
        Cognates._get_decimal_pairs((100, 2, 4), "1001", "10002")

def test_get_decimal_pairs_parallel_freq():
    assert Cognates._get_decimal_pairs((57,), "102", "56alphabet") == [
        (57, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((57,), "102", "alphabet") == [
        (57, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((57,), "alphabet", "102") == [
        (57, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((62,), "342", "1234") == [
        (62, "342", "342", 0.0),
        (62, "1234", "1234", 0.0)
    ]

def test_get_decimal_pairs_monolingual_freqs():
    assert Cognates._get_decimal_pairs((12, 43), "alp123habet", "102") == [
        (43, 43, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((12, 43), "alphabet", "102") == [
        (43, 43, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((12, 43), "102", "alphabet") == [
        (12, 12, "102", "102", 0.0)
    ]
    assert Cognates._get_decimal_pairs((50, 91), "342", "1234") == [
        (50, 50, "342", "342", 0.0),
        (91, 91, "1234", "1234", 0.0)
    ]

def test_get_decimal_pairs_numbers():
    assert Cognates._get_decimal_pairs((32,), "102", "304") == [
        (32, "102", "102", 0.0),
        (32, "304", "304", 0.0)
    ]
    assert Cognates._get_decimal_pairs((32,), "1-800-555-5555", "$35.00") == [
        (32, "1-800-555-5555", "1-800-555-5555", 0.0),
        (32, "$35.00", "$35.00", 0.0)
    ]
    assert Cognates._get_decimal_pairs((32,), "1.800.555.5555", "45$35") == [
        (32, "1.800.555.5555", "1.800.555.5555", 0.0),
        (32, "45$35", "45$35", 0.0)
    ]
    assert Cognates._get_decimal_pairs((32,), "1,800,555,5555", "45$35$64") == [
        (32, "1,800,555,5555", "1,800,555,5555", 0.0),
        (32, "45$35$64", "45$35$64", 0.0)
    ]
    assert Cognates._get_decimal_pairs((32,), "1,800,555,5555", "45$35$64asdf") == [
        (32, "1,800,555,5555", "1,800,555,5555", 0.0)
    ]


##############################
# _consolidate_decimal_pairs #
##############################

def test_consolidate_decimal_pairs_parallel_freq():
    assert Cognates._consolidate_decimal_pairs([
        (12, "a", "a", 0.0),
        (9, "b", "b", 0.0),
        (5, "a", "a", 0.0),
        (2, "b", "b", 0.0),
        (1, "a", "a", 0.0),
        (7, "c", "c", 0.0),
    ]) == [
        (18, "a", "a", 0.0),
        (11, "b", "b", 0.0),
        (7, "c", "c", 0.0)
    ]

def test_consolidate_decimal_pairs_monolingual_freq():
    assert Cognates._consolidate_decimal_pairs([
        (3, 3, "a", "a", 0.0),
        (4, 4, "b", "b", 0.0),
        (8, 8, "a", "a", 0.0),
        (6, 6, "b", "b", 0.0),
        (4, 4, "a", "a", 0.0),
        (2, 2, "c", "c", 0.0),
    ]) == [
        (15, 15, "a", "a", 0.0),
        (10, 10, "b", "b", 0.0),
        (2, 2, "c", "c", 0.0)
    ]

def test_consolidate_decimal_pairs_monolingual_freq_not_same():
    with pytest.raises(AssertionError, match="monolingual frequencies should be the same!"):
        Cognates._consolidate_decimal_pairs([
            (3, 3, "a", "a", 0.0),
            (4, 5, "b", "b", 0.0),
            (8, 8, "a", "a", 0.0),
            (6, 6, "b", "b", 0.0),
            (4, 4, "a", "a", 0.0),
            (2, 2, "c", "c", 0.0),
        ])

def test_consolidate_decimal_pairs_mixed_freq():
    with pytest.raises(ValueError, match="cannot combine parallel and monolingual frequencies!"):
        Cognates._consolidate_decimal_pairs([
            (12, "a", "a", 0.0),
            (9, "b", "b", 0.0),
            (5, 5, "a", "a", 0.0),
            (2, 2, "b", "b", 0.0),
            (1, "a", "a", 0.0),
            (7, "c", "c", 0.0),
        ])

def test_consolidate_decimal_pairs_words_not_match():
    with pytest.raises(ValueError, match=r"word1 != word2, a != c"):
        Cognates._consolidate_decimal_pairs([
            (12, "a", "a", 0.0),
            (9, "b", "b", 0.0),
            (5, "a", "c", 0.0),
            (2, "b", "b", 0.0),
            (1, "a", "a", 0.0),
            (7, "c", "c", 0.0),
        ])

def test_consolidate_decimal_pairs_empty():
    assert Cognates._consolidate_decimal_pairs([]) == []

