"""
This script will take a config and train parent, child, and simple models, all in one go.
"""
import argparse
import os
from sloth_hatch.sloth import log_parsed_args, log_script

import utilities
from utilities import model_names
from NMT.train.train import train_model, eval_models, inference, _nmt_config_key
from NMT.train.train_tokenizer import train_tokenizer
from utilities.hpc import submit_slurm

LOCAL_JOB = "performed locally"

def train_and_eval(config, fine_tune=False, on_hpc=False, afterok=None):
    nmt_config_key = _nmt_config_key(config, fine_tune=fine_tune)
    reverse_tag = "_reverse" if config["nmt_reverse"] else ""
    oc_tag = "OC_" if config["sc_model_ids"] != None else ""
    
    tok_job_name = "STD_TOK" if config["sc_model_ids"] == None else "OC_TOK"

    job_suffix = "NMT|" + config["experiment_name"] + f"|{oc_tag}NMT_{nmt_config_key}{reverse_tag}"
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
            timeout=1,
            ntasks_per_node=1,
            mem_gb=5,
            n_gpus=0,
            qos=config[f"{nmt_config_key}_nmt_qos"],
            afterok=afterok
        )
        jobs["tok"] = tok_job, tok_job_name
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
            timeout=2,
            ntasks_per_node=1,
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=1,
            gpu_type=config["gpu_type"],
            qos=config[f"{nmt_config_key}_nmt_qos"],
            afterok=train_job.job_id
        )
        jobs["eval"] = eval_job, eval_job_name
    else:
        train_function()
        jobs["train"] = LOCAL_JOB, train_job_name
        eval_function()
        jobs["eval"] = LOCAL_JOB, eval_job_name
    
    return jobs

def infer(config, inference_file, src_lang, tgt_lang, fine_tune=False, on_hpc=False, afterok=None):
    nmt_config_key = _nmt_config_key(config, fine_tune=fine_tune)
    reverse_tag = "_reverse" if config["nmt_reverse"] else ""
    oc_tag = "OC_" if config["sc_model_ids"] != None else ""
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
            timeout=2,
            ntasks_per_node=1,
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=1,
            gpu_type=config[f"gpu_type"],
            qos=config["{nmt_config_key}_nmt_qos"],
            afterok=afterok
        )
        jobs["infer"] = inf_job
    else:
        output_file, output_tag = inference_function()
        jobs["infer"] = LOCAL_JOB, inf_job_name, output_file, output_tag
        
    return jobs


@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-C", "--nmt_corpus", choices=["parent", "child"])
    parser.add_argument("-f", "--fine_tune", action="store_true")
    parser.add_argument("-HPC", "--HPC", action="store_true")
    parser.add_argument("--REVERSE", action="store_true", default=False)
    parser.add_argument("--WITH_OC", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    log_script("NMT.train", __file__)
    args = get_args()
    config = utilities.read_data.read_config(args.config, 
                                             nmt_corpus=args.nmt_corpus,
                                             reverse=args.REVERSE,
                                             add_sc_model_ids=args.WITH_OC,
                                             nmt_model_id=model_names.get_new_name())
    train_and_eval(config, fine_tune=args.fine_tune, on_hpc=args.HPC)

