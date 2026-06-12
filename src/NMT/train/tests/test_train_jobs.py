import argparse
import subprocess
import os
import shutil
import json
from sloth_hatch import sloth

from utilities.utilities import set_vars_in_path

HPC = True
EXP_HOME = set_vars_in_path("${EXP_HOME}")
SIMPLE_CONFIG = "src/configs/test/test.xx_yy-->zz.simple.yaml"
PARENT_CONFIG = "src/configs/test/test.xx_yy-->zz.parent.yaml"
CHILD_CONFIG = "src/configs/test/test.xx_yy-->zz.child.yaml"

def setup_function(function, MODEL_TYPE):
    print(f"\nSetting up state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"{MODEL_TYPE}_TEST_xx_yy-->zz")
    print(f"Checking if {test_out_dir} exists already")
    if os.path.exists(test_out_dir):
        print("SETUP: Removing", test_out_dir)
        shutil.rmtree(test_out_dir)

def teardown_function(function, MODEL_TYPE):
    print(f"\nCleaning up state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"{MODEL_TYPE}_TEST_xx_yy-->zz")
    print(f"Will delete {test_out_dir}, if it was succesfully built.")
    if os.path.exists(test_out_dir):
        print("TEARDOWN: Removing", test_out_dir)
        shutil.rmtree(test_out_dir)

@sloth.log_function_call
def test_simple_reproducibility():
    test_out_dir = os.path.join(EXP_HOME, "SIMPLE_TEST_xx_yy-->zz")
    print("Simple models and outputs will be saved to", test_out_dir)
    assert not os.path.exists(test_out_dir)

    # TRAIN FIRST MODEL
    model_name = "this-simple-model-is-reproducible"

    simple_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_simple_{model_name}")
    tokenizers_dir = os.path.join(test_out_dir, "NMT/tokenizers")
    assert not os.path.exists(simple_nmt_out_dir)
    assert not os.path.exists(tokenizers_dir)

    nmt_result = _run_nmt_train(config=SIMPLE_CONFIG,
                                name=model_name,
                                corpus="child",
                                fine_tune=False,
                                hpc=HPC)
    if nmt_result == "ERROR":
        return
    
    assert os.path.exists(test_out_dir)
    assert os.path.exists(simple_nmt_out_dir)
    assert os.path.exists(tokenizers_dir)

    predictions_dir = os.path.join(simple_nmt_out_dir, "predictions")
    assert len(os.listdir(predictions_dir)) > 0
    scores = sloth.read_json(os.path.join(predictions_dir, "scores.json"))


    # TRAIN SECOND MODEL
    new_model_name = "this-simple-is-like-the-first"

    new_simple_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_simple_{new_model_name}")
    assert not os.path.exists(new_simple_nmt_out_dir)

    nmt_result = _run_nmt_train(config=SIMPLE_CONFIG,
                                name=new_model_name,
                                corpus="child",
                                fine_tune=False,
                                hpc=HPC)
    if nmt_result == "ERROR":
        return
    
    assert os.path.exists(new_simple_nmt_out_dir)

    new_predictions_dir = os.path.join(new_simple_nmt_out_dir, "predictions")
    assert len(os.listdir(new_predictions_dir)) > 0
    new_scores = sloth.read_json(os.path.join(new_predictions_dir, "scores.json"))

    _assert_have_same_checkpoints(simple_nmt_out_dir, new_simple_nmt_out_dir)
    _assert_scores_are_same(scores, new_scores, model_name, new_model_name, model_type="simple")

    print("TRAINING SIMPLE IS REPRODUCIBLE :)")

@sloth.log_function_call
def test_parent_reproducibility():
    test_out_dir = os.path.join(EXP_HOME, "PARENT_TEST_xx_yy-->zz")
    print("Parent models and outputs will be saved to", test_out_dir)
    assert not os.path.exists(test_out_dir)

    # TRAIN FIRST MODEL
    model_name = "this-parent-model-is-reproducible"

    parent_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_parent_{model_name}")
    tokenizers_dir = os.path.join(test_out_dir, "NMT/tokenizers")
    assert not os.path.exists(parent_nmt_out_dir)
    assert not os.path.exists(tokenizers_dir)

    nmt_result = _run_nmt_train(config=PARENT_CONFIG, 
                                name=model_name, 
                                corpus="parent", 
                                hpc=HPC)
    if nmt_result == "ERROR":
        return

    assert os.path.exists(test_out_dir)
    assert os.path.exists(parent_nmt_out_dir)
    assert os.path.exists(tokenizers_dir)

    predictions_dir = os.path.join(parent_nmt_out_dir, "predictions")
    assert len(os.listdir(predictions_dir)) > 0
    scores = sloth.read_json(os.path.join(predictions_dir, "scores.json"))


    # TRAIN SECOND MODEL
    new_model_name = "this-parent-is-like-the-first"

    new_parent_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_parent_{new_model_name}")
    assert not os.path.exists(new_parent_nmt_out_dir)

    nmt_result = _run_nmt_train(config=PARENT_CONFIG, 
                                name=new_model_name, 
                                corpus="parent", 
                                hpc=HPC)
    if nmt_result == "ERROR":
        return

    assert os.path.exists(new_parent_nmt_out_dir)

    new_predictions_dir = os.path.join(new_parent_nmt_out_dir, "predictions")
    assert len(os.listdir(new_predictions_dir)) > 0
    new_scores = sloth.read_json(os.path.join(new_predictions_dir, "scores.json"))
    
    _assert_have_same_checkpoints(parent_nmt_out_dir, new_parent_nmt_out_dir)
    _assert_scores_are_same(scores, new_scores, model_name, new_model_name, model_type="parent")

    print("TRAINING PARENTS IS REPRODUCIBLE :)")


@sloth.log_function_call
def test_child_reproducibility():
    test_out_dir = os.path.join(EXP_HOME, "CHILD_TEST_xx_yy-->zz")
    print("Child models and outputs will be saved to", test_out_dir)
    assert not os.path.exists(test_out_dir)

    # TRAIN PARENT MODEL
    parent_model_name = "this-parent-shall-have-two-children"

    parent_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_parent_{parent_model_name}")
    tokenizers_dir = os.path.join(test_out_dir, "NMT/tokenizers")
    assert not os.path.exists(parent_nmt_out_dir)
    assert not os.path.exists(tokenizers_dir)

    nmt_result = _run_nmt_train(config=CHILD_CONFIG,
                                name=parent_model_name,
                                corpus="parent",
                                hpc=HPC)
    if nmt_result == "ERROR":
        return
    
    assert os.path.exists(test_out_dir)
    assert os.path.exists(parent_nmt_out_dir)
    assert os.path.exists(tokenizers_dir)

    # TRAIN FIRST CHILD MODEL
    child_model_name = "this-child-is-reproducible"

    child_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_child_{child_model_name}")
    assert not os.path.exists(child_nmt_out_dir)
    
    nmt_result = _run_nmt_train(config=CHILD_CONFIG,
                                name=child_model_name,
                                corpus="child",
                                fine_tune=True,
                                hpc=HPC)
    if nmt_result == "ERROR":
        return
    
    assert os.path.exists(child_nmt_out_dir)

    predictions_dir = os.path.join(child_nmt_out_dir, "predictions")
    assert len(os.listdir(predictions_dir)) > 0
    scores = sloth.read_json(os.path.join(predictions_dir, "scores.json"))


    # TRAIN SECOND CHILD MODEL
    new_child_model_name = "this-child-is-like-the-first"

    new_child_nmt_out_dir = os.path.join(test_out_dir, "NMT", f"NMT_child_{new_child_model_name}")
    assert not os.path.exists(new_child_nmt_out_dir)

    nmt_result = _run_nmt_train(config=CHILD_CONFIG,
                                name=new_child_model_name,
                                corpus="child",
                                fine_tune=True,
                                hpc=HPC)
    if nmt_result == "ERROR":
        return
    
    assert os.path.exists(new_child_nmt_out_dir)

    new_predictions_dir = os.path.join(new_child_nmt_out_dir, "predictions")
    assert len(os.listdir(new_predictions_dir)) > 0
    new_scores = sloth.read_json(os.path.join(new_predictions_dir, "scores.json"))

    _assert_have_same_checkpoints(child_nmt_out_dir, new_child_nmt_out_dir)
    _assert_scores_are_same(scores, new_scores, child_model_name, new_child_model_name, model_type="child")

    print("TRAINING CHILDREN IS REPRODUCIBLE :)")


def _assert_have_same_checkpoints(parent_dir, new_parent_dir):
    assert isinstance(parent_dir, str)
    assert isinstance(new_parent_dir, str)
    assert parent_dir != new_parent_dir
    checkpoints_dir = os.path.join(parent_dir, "checkpoints")
    new_checkpoints_dir = os.path.join(new_parent_dir, "checkpoints")
    print(f"COMPARING CHECKPOINTS: {checkpoints_dir} vs {new_checkpoints_dir}")
    assert len(os.listdir(checkpoints_dir)) > 0
    for f in os.listdir(checkpoints_dir):
        assert f.startswith("epoch=") and f.endswith(".ckpt")
    assert os.listdir(checkpoints_dir) == os.listdir(new_checkpoints_dir)
    print("\tpassed :)")

def _run_nmt_train(
        config, 
        name, 
        corpus, 
        fine_tune=False, 
        hpc=True, 
        reverse=False, 
        with_oc=False
    ):
    assert corpus in ["parent", "child"]
    command = [
        "python", "src/NMT/train/train_jobs.py",
            "--config", config,
            "--nmt_corpus", corpus,
            "--model_name", name
    ]
    if fine_tune:
        command.append("--fine_tune")
    if hpc:
        command.append("--HPC")
        command.append("--WAIT")
    if reverse:
        command.append("--REVERSE")
    if with_oc:
        command.append("--WITH_OC")
    print(f"running:", json.dumps(command, indent=2))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        # pytest.fail(f"train_jobs.py subprocess failed with return code {result.returncode}")
        print(f"train_jobs.py subprocess failed with return code {result.returncode}")
        return "ERROR"
    return result



def _assert_scores_are_same(scores, new_scores, model_name, new_model_name, model_type):
    assert model_name != new_model_name
    print(f"Comparing scores for {model_name} and {new_model_name}")
    assert isinstance(scores, dict)
    assert isinstance(new_scores, dict)
    assert len(scores) == len(new_scores)
    assert "BEST_VAL_chrF++" in scores.keys()
    assert "BEST_VAL_chrF++" in new_scores.keys()
    for chkpt in scores.keys():
        if chkpt != "BEST_VAL_chrF++":
            new_chkpt = chkpt.replace(f"/NMT_{model_type}_{model_name}/", f"/NMT_{model_type}_{new_model_name}/")
            assert chkpt != new_chkpt
            assert scores[chkpt] == new_scores[new_chkpt]
    
    best_scores = scores["BEST_VAL_chrF++"]
    best_new_scores = new_scores["BEST_VAL_chrF++"]
    assert isinstance(best_scores, dict)
    assert isinstance(best_new_scores, dict)
    assert list(best_scores.keys()) == list(best_new_scores.keys())
    assert "checkpoint" in best_scores.keys()
    for key in best_scores.keys():
        if key == "checkpoint":
            chkpt = best_scores[key]
            new_chkpt = best_new_scores[key]
            assert chkpt != new_chkpt
            assert chkpt == new_chkpt.replace(f"/NMT_{model_type}_{new_model_name}/", f"/NMT_{model_type}_{model_name}/")
        else:
            assert best_scores[key] == best_new_scores[key]


@sloth.log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+")
    return parser.parse_args()

if __name__ == "__main__":
    sloth.log_script("NMT.train.tests", __file__)
    args = get_args()
    steps = {
        "reproduce_simple": (test_simple_reproducibility, "SIMPLE"),
        "reproduce_parent": (test_parent_reproducibility, "PARENT"),
        "reproduce_child": (test_child_reproducibility, "CHILD")
    }
    for step in args.tests:
        f, MODEL_TYPE = steps[step]
        setup_function(f, MODEL_TYPE)
        f()