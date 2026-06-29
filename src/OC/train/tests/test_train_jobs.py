import argparse
import subprocess
import os
import shutil
import json
from sloth_hatch import sloth

from utilities.utilities import set_vars_in_path
from utilities.read_data import read_config
from OC.train import train_jobs

HPC = True
EXP_HOME = set_vars_in_path("${EXP_HOME}")
CONFIG = "src/configs/test/test.xx_yy-->zz.oc.yaml"

def setup_function(function):
    print(f"\nSetting up state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz")
    print(f"Checking if {test_out_dir} exists already")
    if os.path.exists(test_out_dir):
        print("SETUP: Removing", test_out_dir)
        shutil.rmtree(test_out_dir)

def teardown_function(function):
    print(f"\nTearing down state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz")
    print(f"Will delete {test_out_dir}, if it was successfully built.")
    if os.path.exists(test_out_dir):
        print("SETUP: Removing", test_out_dir)
        shutil.rmtree(test_out_dir)

@sloth.log_function_call
def test_reproducibility():
    test_out_dir = os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz")
    assert not os.path.exists(test_out_dir)

    # language scenario for all models
    oc_scenario = ("xx", "yy", "zz")

    # ------- first_round -------

    # charlotte
    model_name = "this-is-a-charlotte-oc-model"
    oc_method = "charlotte"
    oc_train = "src/OC/train/tests/fixtures/train/train.parallel.txt"
    oc_val = "src/OC/train/tests/fixtures/train/val.parallel.txt"
    char1_results = _train_system(
        model_name=model_name,
        oc_method=oc_method,
        oc_train=oc_train,
        oc_val=oc_val,
        oc_scenario=oc_scenario
    )
    char1_scores_f = char1_results["eval"][0].result()

    assert os.path.exists(test_out_dir)
    assert os.listdir(test_out_dir) == ["OC"]
    assert os.listdir(os.path.join(test_out_dir, "OC")) == ["charlotte"]
    assert os.listdir(os.path.join(test_out_dir, "OC", "charlotte")) == ["this-is-a-charlotte-oc-model_xx-yy"]
    assert char1_scores_f == os.path.join(test_out_dir, f"OC/charlotte/this-is-a-charlotte-oc-model_xx-yy/predictions/scores.json")

    # fuzz
    model_name = "this-is-a-fuzz-oc-model"
    oc_method = "fuzz"
    oc_train = "src/OC/train/tests/fixtures/train/train.monolingual.txt"
    oc_val = "src/OC/train/tests/fixtures/train/val.monolingual.txt"
    fuzz1_results = _train_system(
        model_name=model_name,
        oc_method=oc_method,
        oc_train=oc_train,
        oc_val=oc_val,
        oc_scenario=oc_scenario
    )
    fuzz1_scores_f = fuzz1_results["eval"][0].result()

    assert os.listdir(test_out_dir) == ["OC"]
    assert os.listdir(os.path.join(test_out_dir, "OC")) == ["charlotte", "fuzz"]
    assert os.listdir(os.path.join(test_out_dir, "OC", "fuzz")) == ["this-is-a-fuzz-oc-model_xx-yy"]
    assert fuzz1_scores_f == os.path.join(test_out_dir, f"OC/fuzz/this-is-a-fuzz-oc-model_xx-yy/predictions/scores.json")



    # ------- second_round -------

    # charlotte
    model_name = "this-charlotte-oc-model-is-like-the-first"
    oc_method = "charlotte"
    oc_train = "src/OC/train/tests/fixtures/train/train.parallel.txt"
    oc_val = "src/OC/train/tests/fixtures/train/val.parallel.txt"
    char2_results = _train_system(
        model_name=model_name,
        oc_method=oc_method,
        oc_train=oc_train,
        oc_val=oc_val,
        oc_scenario=oc_scenario
    )
    char2_scores_f = char2_results["eval"][0].result()

    assert os.listdir(test_out_dir) == ["OC"]
    assert os.listdir(os.path.join(test_out_dir, "OC")) == ["charlotte", "fuzz"]
    assert set(os.listdir(os.path.join(test_out_dir, "OC", "charlotte"))) == {"this-is-a-charlotte-oc-model_xx-yy", "this-charlotte-oc-model-is-like-the-first_xx-yy"}
    assert char2_scores_f == os.path.join(test_out_dir, f"OC/charlotte/this-charlotte-oc-model-is-like-the-first_xx-yy/predictions/scores.json")

    # fuzz
    model_name = "this-fuzz-oc-model-is-like-the-first"
    oc_method = "fuzz"
    oc_train = "src/OC/train/tests/fixtures/train/train.monolingual.txt"
    oc_val = "src/OC/train/tests/fixtures/train/val.monolingual.txt"
    fuzz2_results = _train_system(
        model_name=model_name,
        oc_method=oc_method,
        oc_train=oc_train,
        oc_val=oc_val,
        oc_scenario=oc_scenario
    )
    fuzz2_scores_f = fuzz2_results["eval"][0].result()

    assert os.listdir(test_out_dir) == ["OC"]
    assert os.listdir(os.path.join(test_out_dir, "OC")) == ["charlotte", "fuzz"]
    assert set(os.listdir(os.path.join(test_out_dir, "OC", "fuzz"))) == {"this-is-a-fuzz-oc-model_xx-yy", "this-fuzz-oc-model-is-like-the-first_xx-yy"}
    assert fuzz2_scores_f == os.path.join(test_out_dir, f"OC/fuzz/this-fuzz-oc-model-is-like-the-first_xx-yy/predictions/scores.json")


    # Check scores
    # Will call _check_scores
    # char 1 should equal fuzz 1
    _check_scores(
        scores_f1=char1_scores_f,
        method1="charlotte",
        name1="this-is-a-charlotte-oc-model",

        scores_f2=fuzz1_scores_f,
        method2="fuzz",
        name2="this-is-a-fuzz-oc-model"
    )
    # char 2 should equal fuzz 2
    _check_scores(
        scores_f1=char2_scores_f,
        method1="charlotte",
        name1="this-charlotte-oc-model-is-like-the-first",

        scores_f2=fuzz2_scores_f,
        method2="fuzz",
        name2="this-fuzz-oc-model-is-like-the-first"
    )
    # char 1 should equal char 2
    _check_scores(
        scores_f1=char1_scores_f,
        method1="charlotte",
        name1="this-is-a-charlotte-oc-model",

        scores_f2=char2_scores_f,
        method2="charlotte",
        name2="this-charlotte-oc-model-is-like-the-first"
    )
    # fuzz 1 should equal fuzz 2
    _check_scores(
        scores_f1=fuzz1_scores_f,
        method1="fuzz",
        name1="this-is-a-fuzz-oc-model",

        scores_f2=fuzz2_scores_f,
        method2="fuzz",
        name2="this-fuzz-oc-model-is-like-the-first"
    )

    print("ALL OC REPRODUCIBILITY TESTS PASSED :)")

def _train_system(model_name, oc_method, oc_train, oc_val, oc_scenario):
    config = read_config(CONFIG,
                         add_sc_model_ids=True,
                         oc_model_id=model_name)
    config["oc_method"] = oc_method
    config["oc_train"] = oc_train
    config["oc_val"] = oc_val
    config["oc_scenario"] = oc_scenario
    pl, cl, tl = oc_scenario
    config["oc_lang_pair"] = (pl, cl)
    config["sc_model_id_prefix"] = config["sc_model_id_prefix"].replace("{method}", oc_method)
    config["sc_model_ids"][oc_scenario] = config["sc_model_ids"][oc_scenario].replace("{method}", oc_method)
    return train_jobs.train_and_eval(
        config=config,
        cognate_method=oc_method,
        on_hpc=HPC,
    )

def _check_scores(
        scores_f1,
        method1,
        name1,

        scores_f2,
        method2,
        name2
    ):
    scores1 = sloth.read_json(scores_f1)
    scores2 = sloth.read_json(scores_f2)

    assert isinstance(scores1, dict)
    assert isinstance(scores2, dict)
    assert len(scores1) == len(scores2)
    assert "BEST_VAL_chrF" in scores1
    assert "BEST_VAL_chrF" in scores2

    for chkpt1, chkpt1_scores in scores1.items():
        if chkpt1 == "BEST_VAL_chrF":
            continue

        chkpt1_dir = "/".join(chkpt1.split("/")[:-1])
        chkpt_fname = chkpt1.split("/")[-1]
        assert chkpt1_dir == os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz/OC/{method1}/{name1}_xx-yy/checkpoints")

        chkpt2_dir = os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz/OC/{method2}/{name2}_xx-yy/checkpoints")
        chkpt2 = os.path.join(chkpt2_dir, chkpt_fname)
        chkpt2_scores = scores2[chkpt2]
        
        print(f"Comparing:\n\t-`{chkpt1}`\n\t-`{chkpt2}`")
        assert isinstance(chkpt1_scores, dict)
        assert set(chkpt1_scores.keys()) == {"chrF", "charBLEU"}
        assert chkpt1_scores == chkpt2_scores
    
    best_scores1 = scores1["BEST_VAL_chrF"]
    best_scores2 = scores2["BEST_VAL_chrF"]
    
    assert isinstance(best_scores1, dict)
    assert set(best_scores1.keys()) == set(best_scores2.keys())
    for key in best_scores1.keys():
        if key == "checkpoint":
            best_chkpt1 = best_scores1["checkpoint"]
            best_chkpt2 = best_scores2["checkpoint"]
            best_chkpt_fname = best_chkpt1.split("/")[-1]
            assert best_chkpt2.split("/")[-1] == best_chkpt_fname
            
            best_chkptdir1 = "/".join(best_chkpt1.split("/")[:-1])
            best_chkptdir2 = "/".join(best_chkpt2.split("/")[:-1])

            assert best_chkptdir1 == os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz/OC/{method1}/{name1}_xx-yy/checkpoints")
            assert best_chkptdir2 == os.path.join(EXP_HOME, f"OC_TEST_xx_yy-->zz/OC/{method2}/{name2}_xx-yy/checkpoints")
        else:
            assert best_scores1[key] == best_scores2[key]
   
@sloth.log_parsed_args
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("function", default="reproduce", choices=["reproduce"])
    return parser.parse_args()

if __name__ == "__main__":
    sloth.log_script("OC.train.tests", __file__)
    args = _get_args()
    f = {
        "reproduce": test_reproducibility
    }[args.function]
    setup_function(f)
    f()
    # teardown_function(f)

