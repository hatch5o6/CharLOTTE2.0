import pytest
import os
import shutil
from transformers import PreTrainedTokenizerFast
from sloth_hatch.sloth import read_json

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
        self.gt_save_dir = os.environ("EXP_HOME")
        self.gt_exp_dir = os.path.join(self.gt_save_dir, "TEST_xx_xx-->xx")
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
        with pytest.raises(FileNotFoundError):
            train_unigram(files=self.data_files,
                          save="dummy/directory",
                          vocab_size=self.config["nmt_vocab_size"],
                          seed=self.config["seed"])

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


############## assemble_multilingual_tokenizer_data ##############
@pytest.mark.skip(reason="assemble_multilingual_tokenizer_data not yet implemented")
def test_assemble_multilingual_tokenizer_data():
    pass



############## make_tokenizer_data ##############


invalid_data_ratios_message = r"""
data_ratios must be a dictionary like so:
{<lang>: {ratio: <ratio>, file: <original data file>}}
where <lang> is the language code, <ratio> is an int indicating 
the number of shares of data given, and <original data file> is 
the path to the source data.
""".strip()
############## _make_scenario_data ##############
def test_make_scenario_data_invalid_data_ratios():
    data_ratios = {
        "en": {"ratio": 1.2, "files": ["path1", "path2"]},
        "es": {"ratio": 2, "files": ["path"]},
        "an": {"ratio": 3, "files": ["path4"]},
    }
    save_dir = "${CHARLOTTE_HOME}/src/NMT/train/tests/test_scen_data"
    set_vars_in_path(save_dir)
    os.mkdir(save_dir)
    while pytest.raises(ValueError, match=invalid_data_ratios_message):
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
    save_dir = "${CHARLOTTE_HOME}/src/NMT/train/tests/test_scen_data"
    set_vars_in_path(save_dir)
    while pytest.raises(FileNotFoundError, message=f"Directory does not exist: {save_dir}"):
        out_files = _make_scenario_data(data_ratios, 
                                        save=save_dir,
                                        training_data_size=1000,
                                        seed=1000)




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
