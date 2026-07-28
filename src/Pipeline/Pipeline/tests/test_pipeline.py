import pytest
import argparse
import subprocess
import os
import shutil
import json
import re
from sloth_hatch import sloth

from Pipeline.Pipeline import pipeline
from utilities.utilities import set_vars_in_path

# HPC = True
EXP_HOME = set_vars_in_path("${EXP_HOME}")
DATA_HOME = set_vars_in_path("${DATA_HOME}")
CONFIG = "src/configs/test/test.xx_yy-->zz.pipeline.yaml"

def setup_function(function, MODEL_TYPE):
    print(f"\nSetting up state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"{MODEL_TYPE}_TEST_xx_yy-->zz")
    print(f"Checking if {test_out_dir} exists already")
    if os.path.exists(test_out_dir):
        print("\tRemoving", test_out_dir)
        shutil.rmtree(test_out_dir)

def teardown_function(function, MODEL_TYPE):
    print(f"\nCleaning up state for {function.__name__}")
    test_out_dir = os.path.join(EXP_HOME, f"{MODEL_TYPE}_TEST_xx_yy-->zz")
    print(f"Will delete {test_out_dir}, if it was succesfully built.")
    if os.path.exists(test_out_dir):
        print("\tRemoving", test_out_dir)
        shutil.rmtree(test_out_dir)

def test_baselines():
    test_out_dir = os.path.join(EXP_HOME, "PIPELINE_TEST_xx_yy-->zz")
    print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)
    assert not os.path.exists(test_out_dir)

    pl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/xx-zz'
    cl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/yy-zz'

    assert set(os.listdir(pl_tl_dir)) == {
        "test.xx.txt",
        "test.zz.txt",
        "train.xx.txt",
        "train.zz.txt",
        "val.xx.txt",
        "val.zz.txt"
    }
    assert set(os.listdir(cl_tl_dir)) == {
        "test.yy.txt",
        "test.zz.txt",
        "train.yy.txt",
        "train.zz.txt",
        "val.yy.txt",
        "val.zz.txt"
    }

    pipeline_results, oc_model_name, nmt_model_name = pipeline.main(
        config_f=CONFIG,
        pipeline=["baselines"],
        nmt_models=['parent', 'child', 'simple'],
        apply_methods=['charlotte', 'web', 'fuzz'],
        lang_filters=None,
        nmt_directions=['-->TL', 'SL-->']
    )

    print("PIPELINE RESULTS:\n")
    print(pipeline_results)
    print("OC_MODEL_NAME:\n")
    print(oc_model_name)
    print("NMT_MODEL_NAME:\n")
    print(nmt_model_name)


def test_baselines_after_training():
    test_out_dir = os.path.join(EXP_HOME, "PIPELINE_TEST_xx_yy-->zz/NMT")

    tokenizers_dir = os.path.join(test_out_dir, 'tokenizers')
    assert os.path.isdir(tokenizers_dir), f"{tokenizers_dir} does not exist"

    tokenizer_dir = os.path.join(tokenizers_dir, f"std|xx-yy_zz|tokenizer/UnigramTokenizer")
    assert os.path.isdir(tokenizer_dir), f"{tokenizer_dir} does not exist"

    tokenizer = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer_config = os.path.join(tokenizer_dir, "tokenizer_config.json")
    assert os.path.isfile(tokenizer), f"Missing Tokenizer File for {test_out_dir}"
    assert os.path.isfile(tokenizer_config), f"Missing Tokenizer Config File for {test_out_dir}"

    sub_dirs = os.listdir(test_out_dir)
    nmt_dirs = [os.path.join(test_out_dir, d) for d in sub_dirs if d.startswith("NMT") and os.path.isdir(os.path.join(test_out_dir, d))]
    assert len(nmt_dirs) == 6, f"{test_out_dir} did not submit {6 - len(nmt_dirs)} jobs that it should have" 
    for nmt_dir in nmt_dirs:
        nmt_sub_dirs = os.listdir(nmt_dir)
        assert len(nmt_sub_dirs) == 7, f"{nmt_dir} does not have the correct number of output folders"

        chkpts_dir = os.path.join(nmt_dir, "checkpoints")
        pred_dir = os.path.join(nmt_dir, 'predictions')
        assert os.path.isdir(chkpts_dir), f"{chkpts_dir} does not exist"
        assert os.path.isdir(pred_dir), f"{pred_dir} does not exist"

        scores_path = os.path.join(pred_dir, 'scores.json')
        assert os.path.isfile(scores_path), f"{pred_dir} is missing scores.json file"

        chkpts = os.listdir(chkpts_dir)
        assert len(chkpts) == 10, f"incorrect number of checkpoints in {chkpts_dir}: {len(chkpts)} instead of 10"
        for chkpt in chkpts:
            assert re.fullmatch(r'epoch=\d+?-step=\d+?-val_loss=\d+?\.\d+?\.ckpt', chkpt) != None

        preds = [d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
        assert len(preds) == 10, f"incorrect number of predictions in {pred_dir}: {len(preds)} instead of 10"
        assert set(preds) == set(chkpts)

        for pred in preds:
                pred_dir_path = os.path.join(pred_dir, pred)
                assert set(os.listdir(pred_dir_path)) == {
                    "test.preds.txt",
                    "validation.preds.txt"
                }
                assert len(sloth.read_lines(os.path.join(pred_dir_path, "test.preds.txt"))) == 1000
                assert len(sloth.read_lines(os.path.join(pred_dir_path, "validation.preds.txt"))) == 997


    

def test_translate_tl_to_pl_only():
    test_out_dir = os.path.join(EXP_HOME, "PIPELINE_TEST_xx_yy-->zz")
    print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)
    assert not os.path.exists(test_out_dir)
    
    cl_tl_dir = "CharLOTTE_data/yy-zz"
    assert set(os.listdir(cl_tl_dir)) == {
        "test.yy.txt",
        "test.zz.txt",
        "train.yy.txt",
        "train.zz.txt",
        "val.yy.txt",
        "val.zz.txt"
    }

    pipeline_results, oc_model_name, nmt_model_name = pipeline.main(
        config_f=CONFIG,
        pipeline=['TL-->PL'],
        nmt_models=['parent', 'child', 'simple'],
        apply_methods=['charlotte', 'web', 'fuzz'],
        lang_filters=None,
        nmt_directions=['-->TL', 'SL-->']
    )
    tl_pl_config, jobs = pipeline_results['tl_pl_translation'][('xx', 'yy', 'zz')]
    if tl_pl_config["use_hpc"]:
        infer_job, infer_job_name = jobs['infer']
        print("WAITING ON INFERENCE JOB FROM HPC CLUSTER")
        output_file, output_tag = infer_job.result()
    else:
        LOCAL_JOB, infer_job_name, output_file, output_tag = jobs['infer']
    
    assert output_file == f"/home/hatch5o6/groups/grp_charlotte/nobackup/archive/char2.0_data/CharLOTTE_data/yy-zz/train.zz.txt.zz-->xx.{nmt_model_name}"
    assert output_tag == f".zz-->xx.{nmt_model_name}"
    assert set(os.listdir(cl_tl_dir)) == {
        "test.yy.txt",
        "test.zz.txt",
        "train.yy.txt",
        "train.zz.txt",
        f"train.zz.txt.zz-->xx.{nmt_model_name}",
        "val.yy.txt",
        "val.zz.txt"
    }
    assert len(sloth.read_lines(output_file)) == 1000

    model_dir = f"NMT_parent_reverse_{nmt_model_name}"
    model_dir_path = os.path.join(test_out_dir, "NMT", model_dir)

    assert set(os.listdir(model_dir_path)) == {
        "checkpoints",
        "data",
        "predictions",
        "logs",
        "tb"
    }

    def get_dir(dirname):
        return os.path.join(model_dir_path, dirname)
    chkpt_dir = get_dir("checkpoints")
    predictions_dir = get_dir("predictions")

    chkpts = os.listdir(chkpt_dir)
    assert len(chkpts) == 10
    for chkpt in chkpts:
        assert re.fullmatch(r'epoch=\d+?-step=\d+?-val_loss=\d+?\.\d+?\.ckpt', chkpt) != None
    
    preds = os.listdir(predictions_dir)
    assert len(preds) == 10
    assert set(preds) == set(chkpts)

    for pred_dir in preds:
        pred_dir_path = os.path.join(predictions_dir, pred_dir)
        assert set(os.listdir(pred_dir_path)) == {
            "test.preds.txt",
            "validation.preds.txt"
        }
        assert len(sloth.read_lines(os.path.join(pred_dir_path, "test.preds.txt"))) == 1000
        assert len(sloth.read_lines(os.path.join(pred_dir_path, "validation.preds.txt"))) == 997
    



def _run_nmt_train_jobs(
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

@sloth.log_parsed_args
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+")
    return parser.parse_args()

if __name__ == "__main__":
    sloth.log_script("Pipeline.Pipeline.tests", __file__)
    args = _get_args()
    steps = {
        "test_translate_tl_to_pl_only": (test_translate_tl_to_pl_only, "PIPELINE"),
        "test_baselines": (test_baselines, "PIPELINE"),
        "test_baselines_after_training": (test_baselines_after_training, "PIPELINE")
    }
    for step in args.tests:
        f, MODEL_TYPE = steps[step]
        if f == test_baselines:
            setup_function(f, MODEL_TYPE)
        f()

