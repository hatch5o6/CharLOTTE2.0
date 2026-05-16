import argparse
import os
from copy import deepcopy
from sloth_hatch.sloth import log_parsed_args, log_script

import utilities
from OC.train.train import train_model, eval_models, inference
from utilities.hpc import submit_slurm

LOCAL_JOB = "performed locally"

def train_and_eval(config, cognate_method, on_hpc=False, afterok=None):
    scenario = config["oc_scenario"]
    pl, cl, tl = scenario

    assert cognate_method in config["sc_model_ids"][scenario]
    assert f"{pl}-{cl}" in config["sc_model_ids"][scenario]
    job_suffix = f"|{config['experiment_name']}|{config['sc_model_ids'][scenario]}"
    
    train_job_name = f"TRAIN{job_suffix}"
    eval_job_name = f"EVAL{job_suffix}"

    train_output_folder = os.path.join(config["save"], config["experiment_name"], f"OC/{cognate_method}/{pl}-{cl}/{train_job_name}/SLURM")
    eval_output_folder = os.path.join(config["save"], config["experiment_name"], f"OC/{cognate_method}/{pl}-{cl}/{eval_job_name}/SLURM")

    jobs = {}
    train_function = lambda: train_model(config)
    eval_function = lambda: eval_models(config)
    print(f"Training and Evaluating {job_suffix}")
    if on_hpc:
        train_job = submit_slurm(
            function=train_function,
            job_name=train_job_name,
            output_folder=train_output_folder,
            mail_user=config["email"],
            timeout=config["oc_timeout"],
            ntasks_per_node=config["oc_n_gpus"],
            mem_gb=config["oc_mem"],
            n_gpus=config["oc_n_gpus"],
            gpu_type=config["gpu_type"],
            qos=config["oc_qos"],
            afterok=afterok
        )
        jobs["train"] = train_job, train_job_name
        eval_job = submit_slurm(
            function=eval_function,
            job_name=eval_job_name,
            output_folder=eval_output_folder,
            mail_user=config["email"],
            timeout=2,
            ntasks_per_node=1,
            mem_gb=config["oc_mem"],
            n_gpus=1,
            gpu_type=config["gpu_type"],
            qos=config["oc_qos"],
            afterok=train_job.job_id
        )
        jobs["eval"] = eval_job, eval_job_name
    else:
        train_function()
        jobs["train"] = LOCAL_JOB, train_job_name
        eval_function()
        jobs["eval"] = LOCAL_JOB, eval_job_name

    return jobs


def infer(config, cognate_method, source_words_f, on_hpc=False, afterok=None):
    scenario = config["oc_scenario"]
    pl, cl, tl = scenario

    assert cognate_method in config["sc_model_ids"][scenario]
    assert f"{pl}-{cl}" in config["sc_model_ids"][scenario]
    job_suffix = f"OC|{config['experiment_name']}|{config['sc_model_ids'][scenario]}"

    infer_job_name = f"INFER{job_suffix}"
    infer_output_folder = os.path.join(config["save"], config["experiment_name"], f"OC/{cognate_method}/{pl}-{cl}/{infer_job_name}/SLURM")

    jobs = {}
    infer_function = lambda: inference(config=config, source_words_f=source_words_f)
    print(f"Infering {infer_job_name}")
    if on_hpc:
        infer_job = submit_slurm(
            function=infer_function,
            job_name=infer_job_name,
            output_folder=infer_output_folder,
            mail_user=config["email"],
            timeout=2,
            ntasks_per_node=1,
            mem_gb=config["oc_mem"],
            n_gpus=1,
            gpu_type=config["gpu_type"],
            qos=config["oc_qos"],
            afterok=afterok
        )
        jobs["infer"] = infer_job
    else:
        output_file, output_tag = infer_function()
        jobs["infer"] = LOCAL_JOB, infer_job_name, output_file, output_tag
    
    return jobs

