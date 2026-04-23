import pytest
import os
import shutil

from utilities.read_data import *
from utilities.read_data import (
    # _get_save_dir,
    _get_sc_model_ids,
    _validate_sets,
    _validate_item
)

#TODO update this to be a config local to the test module
CONFIG_F = "src/configs/test.yaml"
MULTI_CONFIG_F = "" # needs to include a pair where charlotte method doesn't apply because there is no pl, cl data.

#TODO Need to update all tests for multilingual scenarios

def setup_function():
    from utilities.utilities import set_env
    set_env(".env")

def test_read_config_invalid_nmt_corpus():
    with pytest.raises(ValueError, match="nmt_corpus must be 'parent' or 'child'"):
        config = read_config(CONFIG_F, nmt_corpus="oc")

def test_read_config_nmt_corpus():
    config = read_config(CONFIG_F, nmt_corpus="parent")
    assert config.get("nmt_corpus") == "parent"
    config = read_config(CONFIG_F, nmt_corpus="child")
    assert config.get("nmt_corpus") == "child"

def test_read_config_invalid_reverse():
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse="string")
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse=2)
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse=3.2)
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse=[])
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse=tuple())
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse={})
    with pytest.raises(ValueError, match="reverse must be True or False"):
        config = read_config(CONFIG_F, reverse=set())

def test_read_config_reverse():
    config = read_config(CONFIG_F, reverse=True)
    assert config.get("nmt_reverse") == True
    config = read_config(CONFIG_F, reverse=False)
    assert config.get("nmt_reverse") == False

def test_read_config_vars():
    config = read_config(CONFIG_F)
    assert config.get("save") == os.environ["EXP_HOME"]
    assert config.get("oc_warmup_steps") == 5000
    assert config.get("nmt_warmup_steps") == 17500
    assert config.get("sc_model_ids") == {("es", "an", "en"): "OC0_es_an"}
    assert config.get("nmt_corpus") == None
    assert config.get("nmt_reverse") == None

# def test_save_dir():
#     config = read_config(CONFIG_F)
#     gt_parent_dir = os.path.join(
#         os.environ["EXP_HOME"],
#         "UTILS_PYTEST_xx_xx-->xx"
#     )
#     gt_save_dir = os.path.join(
#         gt_parent_dir, "NMT"
#     )
#     if os.path.exists(gt_parent_dir):
#         shutil.rmtree(gt_parent_dir)

#     with pytest.raises(FileNotFoundError, match=f"Directory `{gt_save_dir}`"):
#         save_dir = _get_save_dir(config)
    
#     os.makedirs(gt_save_dir)
#     save_dir = _get_save_dir(config)
#     assert save_dir == gt_save_dir

#     shutil.rmtree(gt_save_dir)

def test_get_sc_model_ids():
    config = read_config(CONFIG_F)
    sc_model_ids = _get_sc_model_ids(
        config["data"],
        sc_model_id_prefix=config["sc_model_id_prefix"]
    )
    assert sc_model_ids == {("es", "an", "en"): "OC0_es_an"}

def test_get_pl_cl_pairs():
    config = read_config(CONFIG_F)
    pl_cl_pairs = get_pl_cl_pairs(config["data"])
    assert pl_cl_pairs == [("es", "an")]

def test_get_tl_pairs():
    config = read_config(CONFIG_F)
    tl_pairs = get_tl_pairs(config["data"])
    assert tl_pairs == [(("es", "en"), ("an", "en"))]

def test_read_pl_cl_paths():
    config = read_config(CONFIG_F)
    pl_cl_paths = read_pl_cl_paths(config["data"])
    data_home = os.environ["DATA_HOME"]
    assert pl_cl_paths == {
        ("es", "an"): (
            os.path.join(data_home, "data/CharLOTTE_data/es-an/train.es.txt"),
            os.path.join(data_home, "data/CharLOTTE_data/es-an/train.an.txt")
        )
    }
    #TODO when you make the multilingual version, do a case where you skip a pl-cl pair because the data can't exist
        # i.e. a case where the charlotte method is not available


def test_read_pl_cl_paths_duplicate():
    data = [
        ["path/to/data", "es", "an", "en"],
        ["path/to/data", "fr", "mfe", "en"],
        ["path/to/some/data", "es", "an", "en"],
    ]
    # with pytest.raises(ValueError, match="Duplicate OC \(pl-cl\) pair \('es', 'an'\) in data!"):
    with pytest.raises(ValueError, match=r'Provided duplicate scenario \(\'es\', \'an\', \'en\'\).'):
        read_pl_cl_paths(data)

def test_read_pl_cl_web_paths():
    config = read_config(CONFIG_F)
    pl_cl_web_paths = read_pl_cl_web_paths(config["data"])
    data_home = os.environ["DATA_HOME"]
    assert pl_cl_web_paths == {
        ("es", "an"): (
            os.path.join(data_home, "data/CharLOTTE_data/es-en"),
            os.path.join(data_home, "data/CharLOTTE_data/an-en"),
            "en"
        )
    }

def test_read_pl_cl_web_paths_duplicate():
    data = [
        ["path/to/data", "es", "an", "en"],
        ["path/to/data", "fr", "mfe", "en"],
        ["path/to/some/data", "es", "an", "en"],
    ]
    # with pytest.raises(ValueError, match="Duplicate OC \(pl-cl\) pair \('es', 'an'\) in data!"):
    with pytest.raises(ValueError, match=r'Provided duplicate scenario \(\'es\', \'an\', \'en\'\).'):
        read_pl_cl_web_paths(data)

def test_read_pl_cl_fuzz_paths():
    config = read_config(CONFIG_F)
    pl_cl_fuzz_paths = read_pl_cl_fuzz_paths(config["data"])
    data_home = os.environ["DATA_HOME"]
    assert pl_cl_fuzz_paths == {
        ("es", "an"): (
            os.path.join(data_home, "data/CharLOTTE_data/es-en/train.es.txt"),
            os.path.join(data_home, "data/CharLOTTE_data/an-en/train.an.txt")
        )
    }

def test_read_pl_cl_fuzz_paths_duplicate():
    data = [
        ["path/to/data", "es", "an", "en"],
        ["path/to/data", "fr", "mfe", "en"],
        ["path/to/some/data", "es", "an", "en"],
    ]
    # with pytest.raises(ValueError, match="Duplicate OC \(pl-cl\) pair \('es', 'an'\) in data!"):
    with pytest.raises(ValueError, match=r'Provided duplicate scenario \(\'es\', \'an\', \'en\'\).'):
        read_pl_cl_fuzz_paths(data)

def test_read_tokenizer_train_paths():
    config = read_config(CONFIG_F)
    tokenizer_train_paths = read_tokenizer_train_paths(config["data"])
    data_home = os.environ["DATA_HOME"]
    assert tokenizer_train_paths == {
        ("es", "an", "en"):{
            "es": [
                os.path.join(data_home, "data/CharLOTTE_data/es-en/train.es.txt")
            ],
            "an": [
                os.path.join(data_home, "data/CharLOTTE_data/an-en/train.an.txt")
            ],
            "en": [
                os.path.join(data_home, "data/CharLOTTE_data/es-en/train.en.txt"),
                os.path.join(data_home, "data/CharLOTTE_data/an-en/train.en.txt")
            ]
        }
    }

def test_get_set():
    config = read_config(CONFIG_F)
    data_home = os.environ["DATA_HOME"]
    assert get_set(
        config["data"],
        scenario=("es", "an", "en"),
        pair=("es", "en"),
        div="val"
    ) == (
        os.path.join(data_home, "data/CharLOTTE_data/es-en/val.es.txt"),
        os.path.join(data_home, "data/CharLOTTE_data/es-en/val.en.txt")
    )

    assert get_set(
        config["data"],
        scenario=("es", "an", "en"),
        pair=("es", "en"),
        div="train"
    ) == (
        os.path.join(data_home, "data/CharLOTTE_data/es-en/train.es.txt"),
        os.path.join(data_home, "data/CharLOTTE_data/es-en/train.en.txt")
    )

    assert get_set(
        config["data"],
        scenario=("es", "an", "en"),
        pair=("an", "en"),
        div="val"
    ) == (
        os.path.join(data_home, "data/CharLOTTE_data/an-en/val.an.txt"),
        os.path.join(data_home, "data/CharLOTTE_data/an-en/val.en.txt")
    )

    assert get_set(
        config["data"],
        scenario=("es", "an", "en"),
        pair=("an", "en"),
        div="test"
    ) == (
        os.path.join(data_home, "data/CharLOTTE_data/an-en/test.an.txt"),
        os.path.join(data_home, "data/CharLOTTE_data/an-en/test.en.txt")
    )

def test_get_set_invalid_pair():
    config = read_config(CONFIG_F)
    with pytest.raises(ValueError, match=r'Invalid pair \(\'en\', \'es\'\) for scenario \(\'es\', \'an\', \'en\'\)'):
        get_set(
            config["data"],
            scenario=("es", "an", "en"),
            pair=("en", "es"),
            div="train"
        )

    with pytest.raises(ValueError, match=r'Invalid pair \(\'xy\', \'qr\'\) for scenario \(\'es\', \'an\', \'en\'\)'):
        get_set(
            config["data"],
            scenario=("es", "an", "en"),
            pair=("xy", "qr"),
            div="train"
        )

def test_get_set_invalid_div():
    config = read_config(CONFIG_F)
    with pytest.raises(ValueError, match="div must be 'train', 'val', or 'test'."):
        get_set(
            config["data"],
            scenario=("es", "an", "en"),
            pair=("an", "en"),
            div="normal"
        )

def test_get_set_scenario_not_found():
    config = read_config(CONFIG_F)
    with pytest.raises(ValueError, match=r'Scenario \(\'xy\', \'qr\', \'st\'\) could not be found.'):
        get_set(
            config["data"],
            scenario=("xy", "qr", "st"),
            pair=("xy", "st"),
            div="val"
        )


def test_validate_sets():
    data = read_config(CONFIG_F)["data"]
    data.append(["${DATA_HOME}/new_data/CharLOTTE_data/", "es", "an", "en"])
    with pytest.raises(ValueError, match=r'Provided duplicate scenario \(\'es\', \'an\', \'en\'\).'):
        _validate_sets(data)
    
    data = [["path/to/data", "pl", "cl"]]
    with pytest.raises(ValueError, match="Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl"):
        _validate_sets(data)

    data = [["path/to/data", "pl", "cl", "tl", "oc"]]
    with pytest.raises(ValueError, match="Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl"):
        _validate_sets(data)
    
    data = [range(4)]
    with pytest.raises(ValueError, match="datasets must be a list of tuples/lists!"):
        _validate_sets(data)

    
def test_validate_item():
    with pytest.raises(ValueError, match="Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl"):
        _validate_item(["path/to/data", "pl", "cl"])
    with pytest.raises(ValueError, match="Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl"):
        _validate_item(["path/to/data", "pl", "cl", "tl", "oc"])
    with pytest.raises(ValueError, match="datasets must be a list of tuples/lists!"):
        _validate_item(range(4))
