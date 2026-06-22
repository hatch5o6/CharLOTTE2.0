import pytest
import os
import shutil
from sloth_hatch import sloth

from OC.extract_cognates import CandidatesFromParallel as CFP
from OC.utilities.word_tokenizers import get_tokenizer

######################
# extract_candidates #
######################
def test_extract_candidates():
    parent_dir = "src/OC/extract_cognates/tests/fixtures/fast_align"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_dir = "src/OC/extract_cognates/tests/fixtures/fast_align/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.path.exists(output_dir)
    assert os.listdir(output_dir) == []

    src_file = "src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt"
    tgt_file = "src/OC/extract_cognates/tests/fixtures/parallel/tgt_big.txt"
    expected_result = [
        (3, 'loves', 'ama'),
        (3, 'he', 'él'),
        (2, 'you', 'estás'),
        (2, 'what', 'qué'),
        (2, 'but', 'pero'),
        (2, 'are', 'estás'),
        (2, 'animals', 'animales'),
        (1, 'you', 'convencerte'),
        (1, 'will', 'tomará'),
        (1, 'very', 'personas'),
        (1, 'to', 'para'),
        (1, 'to', 'debería'),
        (1, 'tell', 'digas'),
        (1, 'take', 'un'),
        (1, 'sorts', 'variedades'),
        (1, 'people', 'las'),
        (1, 'people', 'a'),
        (1, 'not', 'no'),
        (1, 'not', 'me'),
        (1, 'not', 'los'),
        (1, 'much', 'mucho'),
        (1, 'miracle', 'milagro'),
        (1, 'me', 'lo'),
        (1, 'love', 'lo'),
        (1, 'it', 'tomará'),
        (1, 'him', 'aman'),
        (1, 'doing', 'haciendo'),
        (1, 'do', 'no'),
        (1, 'do', 'hacer'),
        (1, 'crazy', 'loco'),
        (1, 'convince', 'para'),
        (1, 'animals', 'los'),
        (1, 'and', 'y'),
        (1, 'all', 'todos'),
        (1, 'a', 'milagro')
    ]
    assert CFP.extract_candidates(
        src_file=src_file,
        tgt_file=tgt_file,
        src_lang="en",
        tgt_lang="es",
        word_list_out=os.path.join(output_dir, "extract_cogs_parallel"),
        long_enough=1
    ) == expected_result

    assert sorted(os.listdir(output_dir)) == sorted([
        "extract_cogs_parallel.sents",
        "extract_cogs_parallel.fwd",
        "extract_cogs_parallel.rev",
        "extract_cogs_parallel.sym",
        "extract_cogs_parallel.word_pairs"
    ])

    word_pairs = sloth.read_lines(os.path.join(output_dir, "extract_cogs_parallel.word_pairs"))
    word_pairs = [eval(item) for item in word_pairs]
    assert word_pairs == expected_result

    shutil.rmtree(output_dir)


def test_extract_candidates_filter_short_words():
    parent_dir = "src/OC/extract_cognates/tests/fixtures/fast_align"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_dir = "src/OC/extract_cognates/tests/fixtures/fast_align/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.path.exists(output_dir)
    assert os.listdir(output_dir) == []

    src_file = "src/OC/extract_cognates/tests/fixtures/parallel/src_big.txt"
    tgt_file = "src/OC/extract_cognates/tests/fixtures/parallel/tgt_big.txt"
    expected_result = [
        (3, 'loves', 'ama'),
        (2, 'you', 'estás'),
        (2, 'what', 'qué'),
        (2, 'but', 'pero'),
        (2, 'are', 'estás'),
        (2, 'animals', 'animales'),
        (1, 'you', 'convencerte'),
        (1, 'will', 'tomará'),
        (1, 'very', 'personas'),
        (1, 'tell', 'digas'),
        (1, 'sorts', 'variedades'),
        (1, 'people', 'las'),
        (1, 'not', 'los'),
        (1, 'much', 'mucho'),
        (1, 'miracle', 'milagro'),
        (1, 'him', 'aman'),
        (1, 'doing', 'haciendo'),
        (1, 'crazy', 'loco'),
        (1, 'convince', 'para'),
        (1, 'animals', 'los'),
        (1, 'all', 'todos')
    ]
    assert CFP.extract_candidates(
        src_file=src_file,
        tgt_file=tgt_file,
        src_lang="en",
        tgt_lang="es",
        word_list_out=os.path.join(output_dir, "extract_cogs_parallel"),
        long_enough=3
    ) == expected_result

    assert sorted(os.listdir(output_dir)) == sorted([
        "extract_cogs_parallel.sents",
        "extract_cogs_parallel.fwd",
        "extract_cogs_parallel.rev",
        "extract_cogs_parallel.sym",
        "extract_cogs_parallel.word_pairs"
    ])

    word_pairs = sloth.read_lines(os.path.join(output_dir, "extract_cogs_parallel.word_pairs"))
    word_pairs = [eval(item) for item in word_pairs]
    assert word_pairs == expected_result

    shutil.rmtree(output_dir)

########################
# _read_parallel_sents #
########################

def test_read_parallel_sents():
    src_file = "src/OC/extract_cognates/tests/fixtures/parallel/src.txt"
    tgt_file = "src/OC/extract_cognates/tests/fixtures/parallel/tgt.txt"
    assert CFP._read_parallel_sents(src_file, tgt_file) == (
        ["hello",
         "look, an elephant",
         "a big dog",
         "what are you doing?"],

         ["hola",
          "mira, un elefante",
          "un perro grande",
          "qué estás haciendo?"]
    )

def test_read_parallel_sents_different_lengths():
    src_file = "src/OC/extract_cognates/tests/fixtures/parallel/src.txt"
    tgt_file = "src/OC/extract_cognates/tests/fixtures/parallel/bad_tgt.txt"
    with pytest.raises(ValueError, match=rf"Length src_file `{src_file}` \(4\) != length tgt_file `{tgt_file}` \(5\)\."):
        CFP._read_parallel_sents(src_file, tgt_file)

#############
# _tokenize #
#############

def test_tokenize():
    tokenizer = get_tokenizer('ws')
    sentences = [
        "hello...",
        "look, an elephant!",
        "a big, ... big ...dog",
        "what ...are you doing?",
        "aren't you gonna give me that?"
    ]
    assert CFP._tokenize(sentences, tokenizer) == [
        "hello",
        "look an elephant",
        "a big big dog",
        "what are you doing",
        "aren't you gonna give me that"
    ]

def test_tokenize_not_a_list():
    tokenizer = get_tokenizer('ws')
    sentences = (
        "hello...",
        "look, an elephant!",
        "a big, ... big ...dog",
        "what ...are you doing?",
        "aren't you gonna give me that?"
    )
    with pytest.raises(ValueError, match="sentences must be a list of strings!"):
        CFP._tokenize(sentences, tokenizer)

def test_tokenize_not_strings():
    tokenizer = get_tokenizer('ws')
    sentences = [
        "hello...",
        "look, an elephant!",
        "a big, ... big ...dog",
        34,
        "aren't you gonna give me that?"
    ]
    with pytest.raises(ValueError, match="each sentence in sentences must be a string!"):
        CFP._tokenize(sentences, tokenizer)

###############
# _fast_align #
###############
def test_fast_align():
    parent_dir = "src/OC/extract_cognates/tests/fixtures/fast_align"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_dir = "src/OC/extract_cognates/tests/fixtures/fast_align/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.path.exists(output_dir)
    assert os.listdir(output_dir) == []

    output_template = os.path.join(output_dir, "fast_align_test")

    src_sents = [
        "hello",
        "look an elephant",
        "a big big dog",
        "what are you doing",
        "aren't you gonna give me that"
    ]
    tgt_sents = [
        "hola",
        "mira un elefante",
        "un perro bien grande",
        "qué estás haciendo",
        "no vas a darme eso"
    ]

    assert CFP._fast_align(src_sents, tgt_sents, output_template) == [
        "0-0",
        "0-0 1-1 2-2",
        "0-0 1-1 2-2 3-3",
        "0-0 1-1 2-1 3-2",
        "0-0 1-1 2-1 3-2 4-3 5-4"
    ]

    assert sorted(os.listdir(output_dir)) == sorted([
        "fast_align_test.sents",
        "fast_align_test.fwd",
        "fast_align_test.rev",
        "fast_align_test.sym"
    ])

    assert sloth.read_lines(os.path.join(output_dir, "fast_align_test.sents")) == [
        "hello ||| hola",
        "look an elephant ||| mira un elefante",
        "a big big dog ||| un perro bien grande",
        "what are you doing ||| qué estás haciendo",
        "aren't you gonna give me that ||| no vas a darme eso"
    ]

    shutil.rmtree(output_dir)

def test_fast_align_not_same_lengths():
    output_dir = "src/OC/extract_cognates/tests/fixtures/fast_align/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.path.exists(output_dir)
    assert os.listdir(output_dir) == []

    output_template = os.path.join(output_dir, "fast_align_test")

    src_sents = [
        "hello",
        "look an elephant",
        "a big big dog",
        "what are you doing",
        "aren't you gonna give me that",
        "an extra sentence"
    ]
    tgt_sents = [
        "hola",
        "mira un elefante",
        "un perro bien grande",
        "qué estás haciendo",
        "no vas a darme eso"
    ]

    with pytest.raises(AssertionError, match=r"length src_sents \(6\) != length tgt_sents \(5\)"):
        CFP._fast_align(src_sents, tgt_sents, output_template)
    
    assert os.listdir(output_dir) == []
    shutil.rmtree(output_dir)


###########################
# _write_fast_align_sents #
###########################
# Tested as part of _fast_align tests

###################
# _get_word_pairs #
###################
def test_get_word_pairs():
    sent_pairs = [
        ("what are you doing?", "¿qué estás haciendo?"),
        ("are you crazy?", "¿estás loco?"),
        ("it will take a miracle to convince you.", "tomará un milagro para convencerte."),
        ("do not tell me what to do.", "no me digas lo qué debería hacer."),
        ("he loves people very much.", "él ama a las personas mucho."),
        ("and he loves all sorts.", "y él ama todos variedades."),
        ("but he loves not animals.", "pero él no ama los animales."),
        ("but animals love him.", "pero los animales lo aman.")
    ]
    alignments = [
        "0-0 1-1 2-1 3-2",
        "0-0 1-0 2-1",
        "0-0 1-0 2-0 3-1 4-2 5-3 6-4 7-4",
        "0-0 1-0 2-2 3-1 4-3 4-4 5-6 6-6",
        "0-0 1-1 2-3 2-4 3-5 4-5",
        "0-0 1-1 2-2 3-3 4-4",
        "0-0 1-1 2-3 3-2 4-4 4-5",
        "0-0 1-1 1-2 2-4 3-3"
    ]
    assert CFP._get_word_pairs(
        sent_pairs=sent_pairs,
        alignments=alignments,
        long_enough=1
    ) == {
        ("what", "qué"): 2,
        ("what", "lo"): 1,
        ("are", "estás"): 2,
        ("you", "estás"): 2,
        ("crazy", "loco"): 1,
        ("you", "convencerte"): 1,
        ("doing", "haciendo"): 1,
        ("it", "tomará"): 1,
        ("will", "tomará"): 1,
        ("take", "tomará"): 1,
        ("a", "un"): 1,
        ("miracle", "milagro"): 1,
        ("to", "para"): 1,
        ("to", "hacer"): 1,
        ("convince", "convencerte"): 1,
        ("do", "no"): 1,
        ("do", "hacer"): 1,
        ("not", "no"): 2,
        ("tell", "digas"): 1,
        ("me", "me"): 1,
        ("he", "él"): 3,
        ("loves", "ama"): 3,
        ("people", "las"): 1,
        ("people", "personas"): 1,
        ("very", "mucho"): 1,
        ("much", "mucho"): 1,
        ("and", "y"): 1,
        ("all", "todos"): 1,
        ("sorts", "variedades"): 1,
        ("but", "pero"): 2,
        ("animals", "los"): 2,
        ("animals", "animales"): 2,
        ("love", "aman"): 1,
        ("him", "lo"): 1
    }

    assert CFP._get_word_pairs(
        sent_pairs=sent_pairs,
        alignments=alignments,
        long_enough=4
    ) == {
        ("crazy", "loco"): 1,
        ("doing", "haciendo"): 1,
        ("will", "tomará"): 1,
        ("take", "tomará"): 1,
        ("miracle", "milagro"): 1,
        ("convince", "convencerte"): 1,
        ("tell", "digas"): 1,
        ("people", "personas"): 1,
        ("very", "mucho"): 1,
        ("much", "mucho"): 1,
        ("sorts", "variedades"): 1,
        ("animals", "animales"): 2,
        ("love", "aman"): 1
    }

####################
# _sort_word_pairs #
####################

def test_sort_word_pairs():
    assert CFP._sort_word_pairs({
        ("crazy", "loco"): 1,
        ("doing", "haciendo"): 200,
        ("will", "tomará"): 5,
        ("take", "tomará"): 1,
        ("miracle", "milagro"): 32,
        ("convince", "convencerte"): 1,
        ("tell", "digas"): 101,
        ("much", "mucho"): 56,
        ("people", "personas"): 1,
        ("very", "mucho"): 1,
        ("sorts", "variedades"): 56,
        ("animals", "animales"): 2,
        ("love", "aman"): 1
    }) == [
        (200, "doing", "haciendo"),
        (101, "tell", "digas"),
        (56, "sorts", "variedades"),
        (56, "much", "mucho"),
        (32, "miracle", "milagro"),
        (5, "will", "tomará"),
        (2, "animals", "animales"),
        (1, "very", "mucho"),
        (1, "take", "tomará"),
        (1, "people", "personas"),
        (1, "love", "aman"),
        (1, "crazy", "loco"),
        (1, "convince", "convencerte")
    ]
