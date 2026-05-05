import os
from sloth_hatch.sloth import read_content, read_lines

from utilities.model_names import get_new_name
from utilities.utilities import set_vars_in_path

MODEL_NAME_CACHE = "${DATA_HOME}/PYTEST_USED_MODEL_NAMES/cache.txt"

def setup_function():
    cache_path = set_vars_in_path(MODEL_NAME_CACHE)
    if os.path.exists(cache_path):
        print(f"REMOVING {cache_path}")
        os.remove(cache_path)

def test_get_new_name():
    # print("setting gt path")
    cache_path = set_vars_in_path(MODEL_NAME_CACHE)
    # print("asserting gt path empty")
    assert not os.path.exists(cache_path)

    # print("getting first name")
    new_model_name = get_new_name(MODEL_NAME_CACHE)
    print(new_model_name)
    # print("testing gt path")
    assert read_content(cache_path) == new_model_name + "\n"
    assert read_lines(cache_path) == [new_model_name]

    model_names = [new_model_name]
    for i in range(100):
        # print("getting and testing name", i)
        yet_another_model_name = get_new_name(MODEL_NAME_CACHE)
        model_names.append(yet_another_model_name)
        print(yet_another_model_name)
        assert read_lines(cache_path) == model_names

def test_get_new_name_uniqueness():
    # print("setting gt path")
    cache_path = set_vars_in_path(MODEL_NAME_CACHE)
    # print("asserting gt path empty")
    assert not os.path.exists(cache_path)

    model_names = []
    for i in range(10000):
        model_names.append(get_new_name(MODEL_NAME_CACHE))
    assert len(model_names) == len(set(model_names))