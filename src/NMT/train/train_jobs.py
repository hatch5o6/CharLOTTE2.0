"""
This script will take a config and train parent, child, and simple models, all in one go.
"""
import argparse
import os
from copy import deepcopy
from sloth_hatch.sloth import log_parsed_args, log_script

import utilities
from utilities import model_names
from NMT.train.train import train_model, eval_models, inference, _nmt_config_key
from NMT.train.train_tokenizer import train_tokenizer
from utilities.hpc import submit_slurm

LOCAL_JOB = "performed locally"

def _get_nmt_config(config, model_type, oc_method=None, reverse=False):
    if not isinstance(reverse, bool):
        raise ValueError(f"reverse must be True or False!")
    if oc_method not in ["charlotte", "web", "fuzz", None]:
        raise ValueError("oc_method must be 'charlotte', 'web', 'fuzz', or None!")
    if model_type not in ["parent", "child", "simple"]:
        raise ValueError(f"model_type must be 'parent', 'child', or 'simple'!")
    if model_type == 'simple':
        if oc_method is not None:
            raise ValueError("oc_method must be None for model_type='simple'!")

    nmt_config = deepcopy(config)
    if model_type == "parent":
        nmt_config["nmt_corpus"] = "parent"
    else:
        nmt_config["nmt_corpus"] = "child"

    if oc_method is not None:
        nmt_config["sc_model_id_prefix"] = nmt_config["sc_model_id_prefix"].replace("{method}", oc_method)
        nmt_config["sc_model_ids"] = {s: s_id.replace("{method}", oc_method) for s, s_id in nmt_config["sc_model_ids"].items()}
    else:
        nmt_config["sc_model_id_prefix"] = None
        nmt_config["sc_model_ids"] = None
    
    nmt_config["oc_method"] = oc_method
    nmt_config["nmt_reverse"] = reverse
    return nmt_config

def train_and_eval(config, fine_tune=False, on_hpc=False, afterok=None, oc_tag="OC", wait=False):
    nmt_config_key = _nmt_config_key(config, fine_tune=fine_tune)
    reverse_tag = "_reverse" if config["nmt_reverse"] else ""
    oc_tag = oc_tag + "_" if config["sc_model_ids"] != None else ""
    
    tok_job_name = "STD_TOK" if config["sc_model_ids"] == None else "OC_TOK"

    job_suffix = "|NMT|" + config["experiment_name"] + f"|{oc_tag}NMT_{nmt_config_key}{reverse_tag}"
    train_job_name = "TRAIN" + job_suffix
    eval_job_name = "EVAL" + job_suffix

    tok_output_folder = os.path.join(config["save"], config["experiment_name"], f"NMT/tokenizers/{tok_job_name}_SLURM")
    train_output_folder = os.path.join(config["save"], config["experiment_name"], f"NMT/{oc_tag}NMT_{nmt_config_key}{reverse_tag}_{config['nmt_model_id']}/{train_job_name}/SLURM")
    eval_output_folder = os.path.join(config["save"], config["experiment_name"], f"NMT/{oc_tag}NMT_{nmt_config_key}{reverse_tag}_{config['nmt_model_id']}/{eval_job_name}/SLURM")
    if on_hpc:
        for output_folder in [tok_output_folder, train_output_folder, eval_output_folder]:
            os.makedirs(output_folder, exist_ok=True)
    
    assert "tokenizer" not in config
    
    jobs = {}

    # Tokenizer
    print(f"Training tokenizer {tok_job_name}")
    tok_function = lambda: train_tokenizer(config, train_with_oc=config["sc_model_ids"] != None)
    if on_hpc:
        tok_job = submit_slurm(
            function=tok_function,
            job_name=tok_job_name,
            output_folder=tok_output_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            mem_gb=config["basic_mem"],
            n_gpus=0,
            qos=config[f"{nmt_config_key}_nmt_qos"],
            afterok=afterok
        )
        jobs["tok"] = tok_job, tok_job_name
        config["tokenizer"] = tok_job.result() # should block until tok_job finishes, may not need afterok=tok_job.job_id below
    else:
        config["tokenizer"] = tok_function()
        jobs["tok"] = LOCAL_JOB, tok_job_name

    # Train + Eval
    train_function = lambda: train_model(config, fine_tune=fine_tune)
    eval_function = lambda: eval_models(config, fine_tune=fine_tune)
    print(f"Training and Evaluating {job_suffix}")
    if on_hpc:
        train_job = submit_slurm(
            function=train_function,
            job_name=train_job_name,
            output_folder=train_output_folder,
            mail_user=config["email"],
            timeout=config[f"{nmt_config_key}_nmt_timeout"],
            ntasks_per_node=config[f"{nmt_config_key}_nmt_n_gpus"],
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=config[f"{nmt_config_key}_nmt_n_gpus"],
            gpu_type=config["gpu_type"],
            qos=config[f"{nmt_config_key}_nmt_qos"],
            afterok=tok_job.job_id
        )
        jobs["train"] = train_job, train_job_name
        eval_job = submit_slurm(
            function=eval_function,
            job_name=eval_job_name,
            output_folder=eval_output_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=1,
            gpu_type=config["gpu_type"],
            qos=config[f"{nmt_config_key}_nmt_qos"],
            afterok=train_job.job_id
        )
        if wait:
            eval_job.result() # block until eval job finishes
        jobs["eval"] = eval_job, eval_job_name
    else:
        train_function()
        jobs["train"] = LOCAL_JOB, train_job_name
        eval_function()
        jobs["eval"] = LOCAL_JOB, eval_job_name
    
    return jobs

def train_simple(
    config,
    afterok=None,
    reverse=False
):
    if not isinstance(reverse, bool):
        raise ValueError(f"reverse must be True or False!")
    simple_config = _get_nmt_config(config,
                                   model_type='simple',
                                   reverse=reverse)
    return train_and_eval(
        simple_config,
        fine_tune=False,
        on_hpc=config["on_hpc"],
        afterok=afterok
    )

def _train_parent(
    config,
    afterok=None,
    oc_method=None, 
    reverse=False
):
    if not isinstance(reverse, bool):
        raise ValueError(f"reverse must be True or False!")
    if oc_method not in ["charlotte", "web", "fuzz", None]:
        raise ValueError("oc_method must be 'charlotte', 'web', 'fuzz', or None!")
    
    parent_config = _get_nmt_config(config,
                                    model_type='parent',
                                    oc_method=oc_method,
                                    reverse=reverse)
    oc_tag = oc_method if oc_method else ""
    return train_and_eval(parent_config,
                          fine_tune=False,
                          on_hpc=config["on_hpc"],
                          afterok=afterok,
                          oc_tag=oc_tag)

def _train_child(
    config,
    afterok=None,
    oc_method=None, 
    reverse=False
):
    if not isinstance(reverse, bool):
        raise ValueError(f"reverse must be True or False!")
    if oc_method not in ["charlotte", "web", "fuzz", None]:
        raise ValueError("oc_method must be 'charlotte', 'web', 'fuzz', or None!")

    child_config = _get_nmt_config(config,
                                  model_type='child',
                                  oc_method=oc_method,
                                  reverse=reverse)
    oc_tag = oc_method if oc_method else ""
    return train_and_eval(child_config,
                          fine_tune=True,
                          on_hpc=config["on_hpc"],
                          afterok=afterok,
                          oc_tag=oc_tag)

def train_parent_child(
    config,
    afterok=None,
    oc_method=None, 
    reverse=False,
    do_train_parent=True,
    do_train_child=True
):
    if not isinstance(reverse, bool):
        raise ValueError(f"reverse must be True or False!")
    if oc_method not in ["charlotte", "web", "fuzz", None]:
        raise ValueError("oc_method must be 'charlotte', 'web', 'fuzz', or None!")
    
    jobs = {}
    parent_eval_job_id = None
    if do_train_parent:
        # train parent
        jobs["parent"] = _train_parent(config=config,
                                       afterok=afterok,
                                       oc_method=oc_method,
                                       reverse=reverse)
        if config["on_hpc"]:
            parent_eval_job_id = jobs["parent"]["eval"][0].job_id

    # train child
    if do_train_child:
        jobs["child"] = _train_child(config=config,
                                     afterok=parent_eval_job_id,
                                     oc_method=oc_method,
                                     reverse=reverse)
    return jobs


def infer(
    config, 
    inference_file, 
    src_lang, 
    tgt_lang, 
    fine_tune=False, 
    on_hpc=False, 
    afterok=None, 
    oc_tag="OC_"
):
    nmt_config_key = _nmt_config_key(config, fine_tune=fine_tune)
    reverse_tag = "_reverse" if config["nmt_reverse"] else ""
    oc_tag = oc_tag if config["sc_model_ids"] != None else ""
    inf_output_folder = os.path.join(config["save"], config["experiment_name"], f"NMT/{oc_tag}NMT_{nmt_config_key}{reverse_tag}_{config['nmt_model_id']}/SLURM")
    if on_hpc:
        os.makedirs(inf_output_folder, exist_ok=True)
    
    job_suffix = "|" + config["experiment_name"] + f"|{oc_tag}NMT_{nmt_config_key}{reverse_tag}"
    inf_job_name = "INFER" + job_suffix

    inference_function = lambda: inference(config=config,
                                           inference_file=inference_file,
                                           src_lang=src_lang,
                                           tgt_lang=tgt_lang,
                                           fine_tune=fine_tune)
    
    jobs = {}
    if on_hpc:
        inf_job = submit_slurm(
            function=inference_function,
            job_name=inf_job_name,
            output_folder=inf_output_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=1,
            gpu_type=config[f"gpu_type"],
            qos=config["{nmt_config_key}_nmt_qos"],
            afterok=afterok
        )
        jobs["infer"] = inf_job, inf_job_name
    else:
        output_file, output_tag = inference_function()
        jobs["infer"] = LOCAL_JOB, inf_job_name, output_file, output_tag
        
    return jobs


@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-C", "--nmt_corpus", choices=["parent", "child"])
    parser.add_argument("-n", "--model_name")
    parser.add_argument("-f", "--fine_tune", action="store_true")
    parser.add_argument("-HPC", "--HPC", action="store_true")
    parser.add_argument("-w", "--WAIT", action="store_true", default=False)
    parser.add_argument("--REVERSE", action="store_true", default=False)
    parser.add_argument("--WITH_OC", action="store_true", default=False)
    args = parser.parse_args()
    if args.model_name == None:
        args.model_name = model_names.get_new_name()
    return args

if __name__ == "__main__":
    log_script("NMT.train", __file__)
    args = get_args()
    config = utilities.read_data.read_config(args.config, 
                                             nmt_corpus=args.nmt_corpus,
                                             reverse=args.REVERSE,
                                             add_sc_model_ids=args.WITH_OC,
                                             nmt_model_id=args.model_name)
    train_and_eval(config, fine_tune=args.fine_tune, on_hpc=args.HPC, wait=args.WAIT)
