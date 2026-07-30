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
from utilities.read_data import read_config

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

    #TODO - remove pieces of the data folder that got added (web inference, OC rewrites)



# def test_translate_tl_to_pl():
#     exp_name = "PIPELINE_TEST_xx_yy-->zz"
#     test_out_dir = os.path.join(EXP_HOME, exp_name)
#     print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)
#     # assert not os.path.exists(test_out_dir) # should exist from previous test
    
#     cl_tl_dir = os.path.join(DATA_HOME, "CharLOTTE_data/yy-zz")
#     assert set(os.listdir(cl_tl_dir)) == {
#         "test.yy.txt",
#         "test.zz.txt",
#         "train.yy.txt",
#         "train.zz.txt",
#         "val.yy.txt",
#         "val.zz.txt"
#     }

#     pipeline_results, oc_model_name, nmt_model_name = pipeline.main(
#         config_f=CONFIG,
#         pipeline=['TL-->PL'],
#         nmt_models=['parent', 'child', 'simple'],
#         apply_methods=['charlotte', 'web', 'fuzz'],
#         lang_filters=None,
#         nmt_directions=['-->TL', 'SL-->']
#     )

#     tl_pl_config, jobs = pipeline_results['tl_pl_translation'][('xx', 'yy', 'zz')]
#     if tl_pl_config["use_hpc"]:
#         infer_job, infer_job_name = jobs['infer']
#         print("WAITING ON INFERENCE JOB FROM HPC CLUSTER")
#         output_file, output_tag = infer_job.result()
#     else:
#         LOCAL_JOB, infer_job_name, output_file, output_tag = jobs['infer']


#     assert output_file == f"/home/pbickel/groups/grp_charlotte/nobackup/archive/char2.0_data/CharLOTTE_data/yy-zz/train.zz.txt.zz-->xx.{nmt_model_name}"
#     assert output_tag == f".zz-->xx.{nmt_model_name}"
#     assert set(os.listdir(cl_tl_dir)) == {
#         "test.yy.txt",
#         "test.zz.txt",
#         "train.yy.txt",
#         "train.zz.txt",
#         f"train.zz.txt.zz-->xx.{nmt_model_name}",
#         "val.yy.txt",
#         "val.zz.txt"
#     }
#     assert len(sloth.read_lines(output_file)) == 1000

#     model_dir = f"NMT_parent_reverse_{nmt_model_name}"
#     model_dir_path = os.path.join(test_out_dir, "NMT", model_dir)

#     if tl_pl_config["use_hpc"]:
#         assert set(os.listdir(model_dir_path)) == {
#             "checkpoints",
#             "data",
#             f"EVAL|NMT|{exp_name}|NMT_parent_reverse",
#             "predictions",
#             "SLURM",
#             "logs",
#             "tb",
#             f"TRAIN|NMT|{exp_name}|NMT_parent_reverse"
#             }
#     else:
#         assert set(os.listdir(model_dir_path)) == {
#             "checkpoints",
#             "data",
#             "predictions",
#             "logs",
#             "tb"
#             }


#     def get_dir(dirname):
#         return os.path.join(model_dir_path, dirname)

#     chkpts_dir = get_dir("checkpoints")
#     pred_dir = get_dir('predictions')
#     assert os.path.isdir(chkpts_dir), f"{chkpts_dir} does not exist"
#     assert os.path.isdir(pred_dir), f"{pred_dir} does not exist"

#     chkpts = os.listdir(chkpts_dir)
#     assert len(chkpts) == 10, f"incorrect number of checkpoints in {chkpts_dir}: {len(chkpts)} instead of 10"
#     for chkpt in chkpts:
#         assert re.fullmatch(r'epoch=\d+?-step=\d+?-val_loss=\d+?\.\d+?\.ckpt', chkpt) != None
    
#     preds = [d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
#     assert len(preds) == 10, f"incorrect number of predictions in {pred_dir}: {len(preds)} instead of 10"
#     assert set(preds) == set(chkpts)

#     for pred in preds:
#                     pred_dir_path = os.path.join(pred_dir, pred)
#                     assert set(os.listdir(pred_dir_path)) == {
#                         "test.preds.txt",
#                         "validation.preds.txt"
#                     }
#                     assert len(sloth.read_lines(os.path.join(pred_dir_path, "test.preds.txt"))) == 1000
#                     assert len(sloth.read_lines(os.path.join(pred_dir_path, "validation.preds.txt"))) == 997
    

def test_prepare_OC():
    test_out_dir = os.path.join(EXP_HOME, "PIPELINE_TEST_xx_yy-->zz")
    print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)

    pipeline_results, oc_model_name, nmt_model_name = pipeline.main(
            config_f=CONFIG,
            pipeline=["prepare_OC"],
            nmt_models=['parent', 'child', 'simple'],
            apply_methods=['charlotte', 'web', 'fuzz'],
            lang_filters=None,
            nmt_directions=['-->TL', 'SL-->']
        )

    print(pipeline_results)
    print(oc_model_name)
    print(nmt_model_name)

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


def test_pipeline_subset(pipeline_subset=["baselines", "TL-->PL"]):
    """specify a subset of the pipeline to test"""

    exp_name = "PIPELINE_TEST_xx_yy-->zz"
    test_out_dir = os.path.join(EXP_HOME, exp_name)
    print("EXPERIMENTS MODELS AND OUTPUTS WILL BE SAVED TO", test_out_dir)
    # assert not os.path.exists(test_out_dir)

    pl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/xx-zz'
    cl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/yy-zz'

    if "baselines" or "TL-->PL" in pipeline_subset:
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
            pipeline=pipeline_subset,
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


    if read_config(CONFIG)["use_hpc"]:
            print("WAITING ON ALL PIPELINE JOBS FROM HPC CLUSTER...")
            # Unpack all nested structures into a flat list of items
            stack = list(pipeline_results.values())
            jobs_to_wait = []
    
            while stack:
                item = stack.pop()
                if hasattr(item, "result"):
                    jobs_to_wait.append(item)
                elif isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
    
            # Block until every job completes
            for job in jobs_to_wait:
                job.result()
    
            print("ALL HPC JOBS COMPLETED.")


    test_pipeline_subset_after_training(pipeline_subset, oc_model_name, nmt_model_name)
    


def test_pipeline_subset_after_training(pipeline_subset=["baselines", "TL-->PL"], oc_model_name=None, nmt_model_name=None):
    exp_name = "PIPELINE_TEST_xx_yy-->zz"
    test_out_dir = os.path.join(EXP_HOME, exp_name)
    hpc = read_config(CONFIG)["use_hpc"]
    pl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/xx-zz'
    cl_tl_dir = f'{DATA_HOME}/CharLOTTE_data/yy-zz'

    if "baselines" in pipeline_subset:
        # tokenizer
        test_NMT_tokenizer(test_out_dir, tokenizer_type="std")

        main_nmt_dir = os.path.join(test_out_dir, "NMT")
        sub_dirs = os.listdir(main_nmt_dir)
        nmt_dirs = [d for d in sub_dirs if d.startswith("NMT") and os.path.isdir(os.path.join(main_nmt_dir, d))]

        # NMT Folders
        assert len(nmt_dirs) == 6, f"{main_nmt_dir} did not submit {6 - len(nmt_dirs)} jobs that it should have" 
        
        for nmt_dir in nmt_dirs:
            check_in_tl_pl = ("TL-->PL" in pipeline_subset) and (nmt_dir == f"NMT_parent_reverse_{nmt_model_name}")
            if not check_in_tl_pl:
                test_NMT_subfolder(test_out_dir, nmt_dir, exp_name, nmt_model_name, hpc=hpc)

    if "TL-->PL" in pipeline_subset:
        # test tokenizer if the baselines didn't already
        if "baselines" not in pipeline_subset: 
            test_NMT_tokenizer(test_out_dir, tokenizer_type="std")

        # test output of the parent_reverse NMT folder
        test_NMT_subfolder(test_out_dir, f"NMT_parent_reverse_{nmt_model_name}", exp_name, nmt_model_name, hpc=hpc, inference=True)

        # test inf file output (testing proper alignment is part of the inference script)
        assert set(os.listdir(cl_tl_dir)) == {
                    "test.yy.txt",
                    "test.zz.txt",
                    "train.yy.txt",
                    "train.zz.txt",
                    f"train.zz.txt.zz-->xx.{nmt_model_name}",
                    "val.yy.txt",
                    "val.zz.txt"
                }
        web_inference_file = f"{DATA_HOME}/CharLOTTE_data/yy-zz/train.zz.txt.zz-->xx.{nmt_model_name}"
        assert len(sloth.read_lines(web_inference_file)) == 1000

    if "prepare_OC" in pipeline_subset:
        pass


    print("TESTS PASSED")

        







def test_NMT_tokenizer(test_out_dir, tokenizer_type="std"):
    tokenizers_dir = os.path.join(test_out_dir, "NMT/tokenizers")
    assert os.path.isdir(tokenizers_dir), f"{tokenizers_dir} does not exist"

    tokenizer_dir = os.path.join(tokenizers_dir, f"{tokenizer_type}|xx-yy_zz|tokenizer/UnigramTokenizer")
    assert os.path.isdir(tokenizer_dir), f"{tokenizer_dir} does not exist"

    tokenizer = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer_config = os.path.join(tokenizer_dir, "tokenizer_config.json")
    assert os.path.isfile(tokenizer), f"Missing {tokenizer_type} Tokenizer File for {test_out_dir}"
    assert os.path.isfile(tokenizer_config), f"Missing Tokenizer Config File for {test_out_dir}"


def test_NMT_subfolder(test_out_dir, nmt_dir, exp_name, nmt_model_name, hpc=True, inference=False):
    nmt_dir_path = os.path.join(test_out_dir, "NMT", nmt_dir)
    nmt_sub_dirs = os.listdir(nmt_dir_path)

    correct_set = {
            "checkpoints",
            "data",
            "predictions",
            "logs",
            "tb",
            }   

    if hpc:
        correct_set.add(f"EVAL|NMT|{exp_name}|{nmt_dir.removesuffix(f'_{nmt_model_name}')}")
        correct_set.add(f"TRAIN|NMT|{exp_name}|{nmt_dir.removesuffix(f'_{nmt_model_name}')}")
        if inference:
            correct_set.add("SLURM")

    assert set(nmt_sub_dirs) == correct_set

    chkpts_dir = os.path.join(nmt_dir_path, "checkpoints")
    pred_dir = os.path.join(nmt_dir_path, 'predictions')
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




@sloth.log_parsed_args
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+")
    return parser.parse_args()

if __name__ == "__main__":
    sloth.log_script("Pipeline.Pipeline.tests", __file__)
    args = _get_args()

    # test_pipeline_subset_after_training(["baselines", "TL-->PL"], oc_model_name="knowing-goldfish-of-exotic-maturity", nmt_model_name="terrestrial-meticulous-pigeon-of-finesse")
    steps = {
        "test_baselines": (test_pipeline_subset, "PIPELINE", ["baselines"]),
        "test_translate_tl_to_pl": (test_pipeline_subset, "PIPELINE", ["TL-->PL"]),
        "test_prepare_OC": (test_pipeline_subset, "PIPELINE", ["prepare_OC"]),
        "test_all": (test_pipeline_subset, "PIPELINE", ["baselines", "TL-->PL"])
    }
    for step in args.tests:
        f, MODEL_TYPE, pipeline_subset = steps[step]
        # setup_function(f, MODEL_TYPE)       
        f(pipeline_subset)
