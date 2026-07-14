import pytest
import argparse
import subprocess
import os
import shutil
import json
from sloth_hatch import sloth

from Pipeline.Pipeline import pipeline
from utilities.utilities import set_vars_in_path

# HPC = True
EXP_HOME = set_vars_in_path("${EXP_HOME}")
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

def test_translate_tl_to_pl_only():
    test_out_dir = os.path.join(EXP_HOME, "PIPELINE_TEST_xx_yy-->zz")
    print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)
    assert not os.path.exists(test_out_dir)
    
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
    

    model_dir = f"NMT_parent_{nmt_model_name}_reverse"
    model_dir_path = os.path.join(test_out_dir, "NMT", model_dir)




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
        "test_translate_tl_to_pl_only": (test_translate_tl_to_pl_only, "PIPELINE")
    }
    for step in args.tests:
        f, MODEL_TYPE = steps[step]
        setup_function(f, MODEL_TYPE)
        f()

