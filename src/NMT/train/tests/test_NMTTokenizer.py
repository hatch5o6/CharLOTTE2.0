import pytest
import os
import shutil
from transformers import PreTrainedTokenizerFast
from sloth_hatch.sloth import read_json, read_lines

from NMT.train.NMTTokenizer import *
from NMT.train.NMTTokenizer import (
    _make_scenario_data,
    _dedupe,
    _make_data_ratios,
    _upsample,
    _get_share_size,
    _validate_data_ratios
)
import utilities
from utilities.utilities import set_env, set_vars_in_path
from Pipeline.Pipeline.pipeline import get_scenario_directory

CONFIG_F = "src/configs/test.yaml"
MULTI_CONFIG_F = "src/configs/multi.test.yaml"



############## train_unigram / load_tokenizer ##############
class TestNMTTokenizerTraining:
    @classmethod
    def setup_class(cls):
        set_env()

        # Bilingual xx|xx-->xx tokenizer
        print("- Creating Bilingual xx|xx-->xx tokenizer -")
        cls.config_f = "src/configs/test.yaml"
        cls.config = utilities.read_data.read_config(cls.config_f)

        cls.save_dir = cls.config["save"]
        assert os.path.exists(cls.save_dir)

        cls.exp_dir = os.path.join(cls.save_dir, cls.config["experiment_name"])
        if os.path.exists(cls.exp_dir):
            print("REMOVING", cls.exp_dir)
            shutil.rmtree(cls.exp_dir)
        assert not os.path.exists(cls.exp_dir)
        print("CREATING", cls.exp_dir)
        os.mkdir(cls.exp_dir)

        os.mkdir(os.path.join(cls.exp_dir, "NMT"))

        print("\tMaking data")
        cls.tokenizer_train_data, cls.tokenizer_dir = make_tokenizer_data(cls.config)
        assert len(cls.config["data"]) == 1
        assert len(cls.tokenizer_train_data) == 1
        cls.scenario = tuple(cls.config["data"][0][1:])
        cls.data_files = cls.tokenizer_train_data[cls.scenario]

        print("\tTraining Unigram")
        cls.unigram_tokenizer_dir = train_unigram(files=cls.data_files,
                                                  save=cls.tokenizer_dir,
                                                  vocab_size=cls.config["nmt_vocab_size"],
                                                  seed=cls.config["seed"])
        cls.tokenizer = load_tokenizer(cls.unigram_tokenizer_dir)
        print("- Finished Creating tokenizer -")
        #TODO Multilingual tokenizer

    def setup_method(self):
        self.gt_save_dir = os.environ["EXP_HOME"]
        self.gt_exp_dir = os.path.join(self.gt_save_dir, "PYTEST_xx_xx-->xx")
        self.gt_tokenizer_dir = os.path.join(self.gt_exp_dir, "NMT/tokenizer")
        self.gt_data_files_dir = os.path.join(self.gt_tokenizer_dir, "data/es_an-->en")
    
    def test_paths(self):
        assert self.save_dir == self.gt_save_dir
        assert self.exp_dir == self.gt_exp_dir
        assert self.tokenizer_dir == self.gt_tokenizer_dir
        assert self.data_files == [
            os.path.join(self.gt_data_files_dir, "es.txt"),
            os.path.join(self.gt_data_files_dir, "an.txt"),
            os.path.join(self.gt_data_files_dir, "en.txt")
        ]
        assert self.unigram_tokenizer_dir == os.path.join(self.gt_tokenizer_dir, "UnigramTokenizer")

    def test_scenario(self):
        assert self.scenario == ("es", "an", "en")
    
    def test_load_tokenizer(self):
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast)
    
    def test_vocab_size(self):
        assert len(self.tokenizer) == 32000
    
    def test_special_toks(self):
        assert self.tokenizer.pad_token == "<pad>"
        assert self.tokenizer.unk_token == "<unk>"
        assert self.tokenizer.bos_token == "<bos>"
        assert self.tokenizer.eos_token == "<eos>"
        assert self.tokenizer.pad_token_id == 0
        assert self.tokenizer.unk_token_id == 1
        assert self.tokenizer.bos_token_id == 2
        assert self.tokenizer.eos_token_id == 3
    
    @pytest.mark.skip(reason="multilingual tokenizer not yet implemented")
    def test_lang_toks(self):
        #TODO For multilingual tokenizer, need to implement still
        pass

    def test_train_unigram_save_dne(self):
        with pytest.raises(FileNotFoundError, match="Directory dummy/directory does not exist!"):
            train_unigram(files=self.data_files,
                          save="dummy/directory",
                          vocab_size=self.config["nmt_vocab_size"],
                          seed=self.config["seed"])

    @pytest.mark.skip(reason="Unigram is not deterministic")
    def test_train_unigram_determinism(self):
        print("Training new tokenizer to test determinism")
        v2_tokenizer_dir = os.path.join(self.exp_dir, "NMT/tokenizer_v2")
        if os.path.exists(v2_tokenizer_dir):
            print("DELETING", v2_tokenizer_dir)
            shutil.rmtree(v2_tokenizer_dir)
        print("CREATING", v2_tokenizer_dir)
        os.mkdir(v2_tokenizer_dir)
        new_unigram_tokenizer_dir = train_unigram(files=self.data_files,
                                                  save=v2_tokenizer_dir,
                                                  vocab_size=self.config["nmt_vocab_size"],
                                                  seed=self.config["seed"])
        v1_json = os.path.join(self.unigram_tokenizer_dir, "tokenizer.json")
        v2_json = os.path.join(new_unigram_tokenizer_dir, "tokenizer.json")
        print(f"\nComparing:\n\t\t-`{v1_json}`\n\t\t-`{v2_json}`")
        assert read_json(v1_json) == read_json(v2_json)

def setup_function(function):
    print(f"Setting up for {function.__name__}")
    root_path = set_vars_in_path("${EXP_HOME}/PYTEST_xx_xx-->xx/")
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    path = os.path.join(root_path, "NMT")
    os.makedirs(path)
    
    if os.path.exists(TEST_SCEN_DATA):
        shutil.rmtree(TEST_SCEN_DATA)

def teardown_function(function):
    print(f"Cleaning up after {function.__name__}")
    path = set_vars_in_path("${EXP_HOME}/PYTEST_xx_xx-->xx/")
    shutil.rmtree(path)

    if os.path.exists(TEST_SCEN_DATA):
        shutil.rmtree(TEST_SCEN_DATA)



############## assemble_multilingual_tokenizer_data ##############
@pytest.mark.skip(reason="assemble_multilingual_tokenizer_data not yet implemented")
def test_assemble_multilingual_tokenizer_data():
    pass



############## make_tokenizer_data ##############
def test_make_tokenizer_data_dir_exists():
    gt_tokenizer_dir = set_vars_in_path("${EXP_HOME}/PYTEST_xx_xx-->xx/NMT/tokenizer")
    if os.path.exists(gt_tokenizer_dir):
        shutil.rmtree(gt_tokenizer_dir)

    os.makedirs(gt_tokenizer_dir)

    config = utilities.read_data.read_config(CONFIG_F)
    with pytest.raises(AssertionError, match=f"Tokenizer directory {gt_tokenizer_dir} already exists."):
        tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config)
    
    shutil.rmtree(gt_tokenizer_dir)

def test_make_tokenizer_data_train_data():
    gt_tokenizer_dir = set_vars_in_path("${EXP_HOME}/PYTEST_xx_xx-->xx/NMT/tokenizer")
    if os.path.exists(gt_tokenizer_dir):
        shutil.rmtree(gt_tokenizer_dir)

    assert not os.path.exists(gt_tokenizer_dir)

    config = utilities.read_data.read_config(CONFIG_F)
    tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config)

    assert gt_tokenizer_dir == tokenizer_dir
    assert tokenizer_train_data == {
        ('es', 'an', 'en'): [
            os.path.join(gt_tokenizer_dir, f"data/es_an-->en/es.txt"),
            os.path.join(gt_tokenizer_dir, f"data/es_an-->en/an.txt"),
            os.path.join(gt_tokenizer_dir, f"data/es_an-->en/en.txt")
        ]
    }

    shutil.rmtree(gt_tokenizer_dir)

@pytest.mark.skip(reason="Multilingual tokenizer not implemented yet")
def test_make_tokenizer_data_train_data_multilingual():
    pass
    #TODO

def test_make_tokenizer_data_directory():
    gt_tokenizer_dir = set_vars_in_path("${EXP_HOME}/PYTEST_xx_xx-->xx/NMT/tokenizer")
    if os.path.exists(gt_tokenizer_dir):
        shutil.rmtree(gt_tokenizer_dir)

    assert not os.path.exists(gt_tokenizer_dir)

    config = utilities.read_data.read_config(CONFIG_F)
    tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config)

    assert gt_tokenizer_dir == tokenizer_dir
    assert os.path.exists(os.path.join(gt_tokenizer_dir, f"data/es_an-->en/es.txt"))
    assert os.path.exists(os.path.join(gt_tokenizer_dir, f"data/es_an-->en/an.txt"))
    assert os.path.exists(os.path.join(gt_tokenizer_dir, f"data/es_an-->en/en.txt"))

    shutil.rmtree(gt_tokenizer_dir)

@pytest.mark.skip(reason="Multilingual tokenizer not implemented yet")
def test_make_tokenizer_data_directory_multilingual():
    pass
    #TODO



############## _make_data_ratios ##############
def test_make_data_ratios():
    data_folder = set_vars_in_path("${DATA_HOME}/data/CharLOTTE_data/")
    data = [[
        data_folder,
        "es",
        "an",
        "en"
    ]]

    # 1:1:2
    nmt_tokenizer_ratios = {
        "pl": 1,
        "cl": 1,
        "tl": 2
    }
    assert _make_data_ratios(data, nmt_tokenizer_ratios) == {
        ("es", "an", "en"): {
            "es": {"ratio": 1, "files": [os.path.join(data_folder, "es-en/train.es.txt")]},
            "an": {"ratio": 1, "files": [os.path.join(data_folder, "an-en/train.an.txt")]},
            "en": {"ratio": 2, "files": [os.path.join(data_folder, "es-en/train.en.txt"), 
                                         os.path.join(data_folder, "an-en/train.en.txt")]}
        }
    }
    
    # 3:1:7
    nmt_tokenizer_ratios = {
        "pl": 3,
        "cl": 1,
        "tl": 7
    }
    assert _make_data_ratios(data, nmt_tokenizer_ratios) == {
        ("es", "an", "en"): {
            "es": {"ratio": 3, "files": [os.path.join(data_folder, "es-en/train.es.txt")]},
            "an": {"ratio": 1, "files": [os.path.join(data_folder, "an-en/train.an.txt")]},
            "en": {"ratio": 7, "files": [os.path.join(data_folder, "es-en/train.en.txt"), 
                                         os.path.join(data_folder, "an-en/train.en.txt")]}
        }
    }

#TODO
@pytest.mark.skip(reason="Multilingual tokenizer not yet implemented.")
def test_make_data_ratios_multilingual():
    pass



############## _make_scenario_data ##############
invalid_data_ratios_message = r"""
data_ratios must be a dictionary like so:
{<lang>: {ratio: <ratio>, file: <original data file>}}
where <lang> is the language code, <ratio> is an int indicating 
the number of shares of data given, and <original data file> is 
the path to the source data.
""".strip()
TEST_SCEN_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/test_scen_data")
EN1_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/en1.txt")
EN2_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/en2.txt")
ES1_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/es1.txt")
ES2_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/es2.txt")
AN_DATA = set_vars_in_path("${CHARLOTTE_HOME}/src/NMT/train/tests/data/an.txt")
def test_make_scenario_data_invalid_data_ratios():
    data_ratios = {
        "en": {"ratio": 1.2, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    save_dir = TEST_SCEN_DATA
    os.mkdir(save_dir)
    with pytest.raises(ValueError, match=invalid_data_ratios_message):
        out_files = _make_scenario_data(data_ratios, 
                                        save=save_dir,
                                        training_data_size=1000,
                                        seed=1000)
    shutil.rmtree(save_dir)

def test_make_scenario_data_save_dne():
    data_ratios = {
        "en": {"ratio": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    save_dir = TEST_SCEN_DATA
    with pytest.raises(FileNotFoundError, match=f"Directory does not exist: {save_dir}"):
        out_files = _make_scenario_data(data_ratios, 
                                        save=save_dir,
                                        training_data_size=1000,
                                        seed=1000)

def _count_lines(files):
    data = []
    for f in files:
        data += read_lines(f)
    return len(data)

def _read_files(files):
    data = []
    for f in files:
        data += read_lines(f)
    return data

def test_make_scenario_data_notes():
    data_ratios = {
        "en": {"ratio": 1, "files": [EN1_DATA, EN2_DATA]},
        "es": {"ratio": 2, "files": [ES1_DATA, ES2_DATA]},
        "an": {"ratio": 3, "files": [AN_DATA]},
    }
    assert _count_lines(data_ratios["en"]["files"]) == 100000
    assert _count_lines(data_ratios["es"]["files"]) == 150000
    assert _count_lines(data_ratios["an"]["files"]) == 28594

    save_dir = TEST_SCEN_DATA
    os.mkdir(save_dir)

    out_files = _make_scenario_data(data_ratios,
                                    save=save_dir,
                                    training_data_size=300100,
                                    seed=1000)
    
    notes_file = os.path.join(save_dir, "notes.json")
    notes = read_json(notes_file)
    assert notes == [
        {"ratio": 1, "files": [EN1_DATA, EN2_DATA], "share_size": 50016, "quota": 50016},
        {"ratio": 2, "files": [ES1_DATA, ES2_DATA], "share_size": 50016, "quota": 100032},
        {"ratio": 3, "files": [AN_DATA], "share_size": 50016, "quota": 150048}
    ]

    shutil.rmtree(save_dir)

def test_make_scenario_data_downsamples():
    data_ratios = {
        "en": {"ratio": 1, "files": [EN1_DATA, EN2_DATA]},
        "es": {"ratio": 2, "files": [ES1_DATA, ES2_DATA]},
        "an": {"ratio": 3, "files": [AN_DATA]},
    }
    assert _count_lines(data_ratios["en"]["files"]) == 100000
    assert _count_lines(data_ratios["es"]["files"]) == 150000
    assert _count_lines(data_ratios["an"]["files"]) == 28594

    save_dir = TEST_SCEN_DATA
    os.mkdir(save_dir)

    out_files = _make_scenario_data(data_ratios,
                                    save=save_dir,
                                    training_data_size=300100,
                                    seed=1000)

    assert out_files == [
        os.path.join(TEST_SCEN_DATA, "en.txt"),
        os.path.join(TEST_SCEN_DATA, "es.txt"),
        os.path.join(TEST_SCEN_DATA, "an.txt")
    ]
    en_f, es_f, an_f = out_files

    assert len(read_lines(en_f)) == 50016
    assert len(read_lines(es_f)) == 100032

    shutil.rmtree(save_dir)

def test_make_scenario_data_upsamples():
    data_ratios = {
        "en": {"ratio": 1, "files": [EN1_DATA, EN2_DATA]},
        "es": {"ratio": 2, "files": [ES1_DATA, ES2_DATA]},
        "an": {"ratio": 3, "files": [AN_DATA]},
    }
    assert _count_lines(data_ratios["en"]["files"]) == 100000
    assert _count_lines(data_ratios["es"]["files"]) == 150000
    assert _count_lines(data_ratios["an"]["files"]) == 28594

    save_dir = TEST_SCEN_DATA
    os.mkdir(save_dir)

    out_files = _make_scenario_data(data_ratios,
                                    save=save_dir,
                                    training_data_size=300100,
                                    seed=1000)

    assert out_files == [
        os.path.join(TEST_SCEN_DATA, "en.txt"),
        os.path.join(TEST_SCEN_DATA, "es.txt"),
        os.path.join(TEST_SCEN_DATA, "an.txt")
    ]
    en_f, es_f, an_f = out_files

    assert len(read_lines(an_f)) == 150048

    shutil.rmtree(save_dir)

def test_make_scenario_data_out_file_contents():
    data_ratios = {
        "en": {"ratio": 1, "files": [EN1_DATA, EN2_DATA]},
        "es": {"ratio": 2, "files": [ES1_DATA, ES2_DATA]},
        "an": {"ratio": 3, "files": [AN_DATA]},
    }
    assert _count_lines(data_ratios["en"]["files"]) == 100000
    assert _count_lines(data_ratios["es"]["files"]) == 150000
    assert _count_lines(data_ratios["an"]["files"]) == 28594

    save_dir = TEST_SCEN_DATA
    os.mkdir(save_dir)

    out_files = _make_scenario_data(data_ratios,
                                    save=save_dir,
                                    training_data_size=300100,
                                    seed=1000)

    assert out_files == [
        os.path.join(TEST_SCEN_DATA, "en.txt"),
        os.path.join(TEST_SCEN_DATA, "es.txt"),
        os.path.join(TEST_SCEN_DATA, "an.txt")
    ]
    en_f, es_f, an_f = out_files

    en_in_data = set(_read_files(data_ratios["en"]["files"]))
    es_in_data = set(_read_files(data_ratios["es"]["files"]))
    an_in_data = set(_read_files(data_ratios["an"]["files"]))

    en_out_lines = set(read_lines(en_f))
    es_out_lines = set(read_lines(es_f))
    an_out_lines = set(read_lines(an_f))

    assert en_out_lines.difference(en_in_data) == set()
    assert es_out_lines.difference(es_in_data) == set()
    assert an_out_lines.difference(an_in_data) == set()



############## _dedupe ##############
def test_dedupe_removes_duplicates():
    data = ["a", "b", "c", "d", "a", "a", "e", "e", "f", "g", "h", "a", "i", "j", "k", "e", "l", "m", "a", "n"]
    assert _dedupe(data) == ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]

def test_dedupe_preserves_order():
    data = ["a", "b", "a", "b"]
    assert _dedupe(data) == ["a", "b"]

    data = ["b", "a", "b", "a"]
    assert _dedupe(data) == ["b", "a"]

def test_dedupe_empty():
    data = []
    assert _dedupe(data) == []

def test_dedup_no_duplicates():
    data = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    assert _dedupe(data) == ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]



############## _upsample ##############
def test_upsample_invalid_quota():
    data = ["b", "a", "d", "c", "g", "h", "i", "j"]
    quota = -10
    with pytest.raises(AssertionError, match="quota must be >= 0!"):
        upsampled_data = _upsample(data, quota)

def test_upsample_invalid_data():
    data = []
    quota = 100
    with pytest.raises(AssertionError, match="length of data must be > 0!"):
        upsampled_data = _upsample(data, quota)

def test_upsample_reaches_quota():
    data = ["b", "a", "d", "c", "g", "h", "i", "j"]
    quota = 100
    upsampled_data = _upsample(data, quota)
    assert len(upsampled_data) == 100

def test_upsample_has_proper_number_of_each_item():
    data = ["b", "a", "d", "c", "g", "h", "i", "j"]
    quota = 100
    upsampled_data = _upsample(data, quota)
    assert len(upsampled_data) == 100

    counts = _count_items(upsampled_data)
    assert counts["b"] == 13
    assert counts["a"] == 13
    assert counts["d"] == 13
    assert counts["c"] == 13
    assert counts["g"] == 12
    assert counts["h"] == 12
    assert counts["i"] == 12
    assert counts["j"] == 12

def _count_items(items):
    from collections import Counter
    counts = Counter()
    for item in items:
        counts[item] += 1
    return counts

def test_upsample_data_larger_than_quota():
    data = ["b", "a", "d", "c", "g", "h", "i", "j"]
    quota = 4
    assert _upsample(data, quota) == ["b", "a", "d", "c"]

def test_upsample_exact_multiple():
    data = ["b", "a", "d", "c", "g", "h", "m", "j", "z", "l"]
    quota = 100
    upsampled_data = _upsample(data, quota)
    assert len(upsampled_data) == 100

    counts = _count_items(upsampled_data)
    assert len(data) == 10
    for item in data:
        assert counts[item] == 10



############## _get_share_size ##############
def test_get_share_size_equal_ratios():
    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 1, "files": []}
    }
    assert _get_share_size(data_ratios, 30) == 10

def test_get_share_size_unequal_ratios():
    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 8) == 2

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 4, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 35) == 5

def test_get_share_size_uneven_division():
    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 1, "files": []}
    }
    assert _get_share_size(data_ratios, 32) == 10

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 1, "files": []}
    }
    assert _get_share_size(data_ratios, 31) == 10

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 10) == 2

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 1, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 9) == 2

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 4, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 41) == 5

    data_ratios = {
        "es": {"ratio": 1, "files": []},
        "an": {"ratio": 4, "files": []},
        "en": {"ratio": 2, "files": []}
    }
    assert _get_share_size(data_ratios, 38) == 5



############## _validate_data_ratios ##############
def test_validate_data_ratios_valid():
    #{<lang>: {ratio: <ratio>, files: [<original data file>, ...]}}
    data_ratios = {"en": {"ratio": 1, "files": ["path1", "path2"]}}
    assert _validate_data_ratios(data_ratios) == True
    data_ratios = {
        "en": {"ratio": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == True

def test_validate_data_ratios_not_dict():
    assert _validate_data_ratios([1]) == False
    assert _validate_data_ratios(set([1])) == False
    assert _validate_data_ratios((1,2)) == False
    assert _validate_data_ratios(1) == False
    assert _validate_data_ratios(1.1) == False
    assert _validate_data_ratios("a") == False
    assert _validate_data_ratios(range(10)) == False

def test_validate_data_ratios_not_lang_str():
    data_ratios = {
        1: {"ratio": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False
    data_ratios = {
        ("en",): {"ratio": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False
    data_ratios = {
        3.2: {"ratio": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

def test_validate_data_ratios_item_not_dict():
    data_ratios = {
        "en": '{"ratio": 1, "files": ["path1", "path2"]}',
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": ["ratio", "files"],
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": ("ratio", "files"),
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio", "files"},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": 2,
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": 2.3,
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

def test_validate_data_ratios_keys():
    data_ratios = {
        "en": {"files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"number": 1, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "fs": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"number": 1, "fs": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

def test_validate_data_ratios_files_type():
    data_ratios = {
        "en": {"ratio": 1, "files": ("path1", "path2")},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": {"path1", "path2"}},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": {"path1": "9", "path2": "0"}},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": 3},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": 2.1},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

def test_validate_data_ratios_file_item_type():
    data_ratios = {
        "en": {"ratio": 1, "files": [0]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": [1.2]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": [["path2", "path1"]]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": [{"a"}]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": [{"a": "3"}]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": 1, "files": [("a", "b")]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

def test_validate_data_ratios_ratio_type():
    data_ratios = {
        "en": {"ratio": 1.2, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": [1], "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": {1:2}, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": (1,), "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": {1}, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False

    data_ratios = {
        "en": {"ratio": "1", "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    assert _validate_data_ratios(data_ratios) == False
