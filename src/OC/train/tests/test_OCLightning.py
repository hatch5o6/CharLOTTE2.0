import pytest
import os
import shutil
import torch

from utilities.utilities import set_vars_in_path
from utilities import read_data
from OC.train.train import get_tokenizers
from OC.train.OCLightning import OCDataModule

# test the collate_fn function mainly

def test_get_parallel_tokenizers():
    train_f = "src/OC/train/tests/fixtures/train/train.en.ar.parallel.txt"
    en_tokenizer, ar_tokenizer = get_tokenizers(train_f)

    # check en tokenizer
    assert en_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3, 
        "a": 4, "b": 5, "h": 6, "k": 7, "l": 8, "m": 9, "n": 10, "r": 11, "u": 12
    }
    assert en_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "a", 5: "b", 6: "h", 7: "k", 8: "l", 9: "m", 10: "n", 11: "r", 12: "u"
    }

    # check ar tokenizer
    assert ar_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3,
        "أ": 4, "ا": 5, "ب": 6, "ح": 7, "ر": 8, "ك": 9, "ل": 10, "م": 11, "ه": 12, "و": 13
    }
    assert ar_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "أ", 5: "ا", 6: "ب", 7: "ح", 8: "ر", 9: "ك", 10: "ل", 11: "م", 12: "ه", 13: "و"
    }

def test_get_reverse_parallel_tokenizers():
    train_f = "src/OC/train/tests/fixtures/train/train.ar.en.parallel.txt"
    ar_tokenizer, en_tokenizer = get_tokenizers(train_f)

    # check en tokenizer
    assert en_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3, 
        "a": 4, "b": 5, "h": 6, "k": 7, "l": 8, "m": 9, "n": 10, "r": 11, "u": 12
    }
    assert en_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "a", 5: "b", 6: "h", 7: "k", 8: "l", 9: "m", 10: "n", 11: "r", 12: "u"
    }

    # check ar tokenizer
    assert ar_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3,
        "أ": 4, "ا": 5, "ب": 6, "ح": 7, "ر": 8, "ك": 9, "ل": 10, "م": 11, "ه": 12, "و": 13
    }
    assert ar_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "أ", 5: "ا", 6: "ب", 7: "ح", 8: "ر", 9: "ك", 10: "ل", 11: "م", 12: "ه", 13: "و"
    }

def test_get_fuzzy_tokenizers():
    train_f = "src/OC/train/tests/fixtures/train/train.en.ar.monolingual.txt"
    en_tokenizer, ar_tokenizer = get_tokenizers(train_f)

    # check en tokenizer
    assert en_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3, 
        "a": 4, "b": 5, "h": 6, "k": 7, "l": 8, "m": 9, "n": 10, "r": 11, "u": 12
    }
    assert en_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "a", 5: "b", 6: "h", 7: "k", 8: "l", 9: "m", 10: "n", 11: "r", 12: "u"
    }

    # check ar tokenizer
    assert ar_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3,
        "أ": 4, "ا": 5, "ب": 6, "ح": 7, "ر": 8, "ك": 9, "ل": 10, "م": 11, "ه": 12, "و": 13
    }
    assert ar_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "أ", 5: "ا", 6: "ب", 7: "ح", 8: "ر", 9: "ك", 10: "ل", 11: "م", 12: "ه", 13: "و"
    }

def test_get_reverse_fuzzy_tokenizers():
    train_f = "src/OC/train/tests/fixtures/train/train.ar.en.monolingual.txt"
    ar_tokenizer, en_tokenizer = get_tokenizers(train_f)

    # check en tokenizer
    assert en_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3, 
        "a": 4, "b": 5, "h": 6, "k": 7, "l": 8, "m": 9, "n": 10, "r": 11, "u": 12
    }
    assert en_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "a", 5: "b", 6: "h", 7: "k", 8: "l", 9: "m", 10: "n", 11: "r", 12: "u"
    }

    # check ar tokenizer
    assert ar_tokenizer.vocab == {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3,
        "أ": 4, "ا": 5, "ب": 6, "ح": 7, "ر": 8, "ك": 9, "ل": 10, "م": 11, "ه": 12, "و": 13
    }
    assert ar_tokenizer.id_to_char == {
        0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<unk>",
        4: "أ", 5: "ا", 6: "ب", 7: "ح", 8: "ر", 9: "ك", 10: "ل", 11: "م", 12: "ه", 13: "و"
    }

class TestOCDataModule:
    @classmethod
    def setup_class(cls):

        # config
        cls.config_f = "src/configs/test/test.xx_yy-->zz.oc.yaml"
        cls.config = read_data.read_config(cls.config_f)
        cls.config["oc_train"] = "src/OC/train/tests/fixtures/train/train.parallel.txt"
        cls.config["oc_val"] = "src/OC/train/tests/fixtures/train/val.parallel.txt"

        # # exp_dir
        # exp_dir = set_vars_in_path("${EXP_HOME}/OC_TEST_xx_yy-->zz/OC/charlotte/")

        # tokenizers
        cls.src_tokenizer, cls.tgt_tokenizer = get_tokenizers(cls.config["oc_train"])

        cls.dm = OCDataModule(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            train=cls.config["oc_train"],
            val=cls.config["oc_val"],
            batch_size=cls.config["oc_batch_size"],
            max_length=cls.config["oc_max_length"]
        )
        cls.dm.setup()
        