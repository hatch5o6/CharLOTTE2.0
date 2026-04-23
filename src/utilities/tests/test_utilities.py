import pytest
import os
from utilities.utilities import _set_env, set_vars_in_path

def test_set_env():
    env_path = "src/utilities/tests/test_env.env"
    _set_env(env_path)
    assert os.environ["CHARLOTTE_HOME"] == "/path/to/CharLOTTE2.0"
    assert os.environ["DATA_HOME"] == "/path/to/training/data"
    assert os.environ["EXP_HOME"] == "/path/to/experiments"
    assert os.environ["HF_TOKEN"] == "HUGGING_FACE_AUTHORIZATION_TOKEN"
    assert os.environ["PROCESSES"] == "8"
    with pytest.raises(KeyError):
        test_comment_value = os.environ["TEST_COMMENT"]
    with pytest.raises(KeyError):
        test_no_equal_sign_value = os.environ["TEST_NO_EQUAL_SIGN"]

def test_set_vars_in_path():
    env_path = "src/utilities/tests/test_env.env"
    data_folder = "${DATA_HOME}/data_folder"
    data_folder = set_vars_in_path(data_folder, env_path=env_path)
    print("DATA FOLDER", data_folder)
    assert data_folder == "/path/to/training/data/data_folder"
