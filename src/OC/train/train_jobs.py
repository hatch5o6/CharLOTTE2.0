import argparse
import os
import json
import optuna
from copy import deepcopy
from sloth_hatch.sloth import log_parsed_args, log_script

import utilities
from OC.train.train import train_model, eval_models, inference
from utilities.hpc import submit_slurm

LOCAL_JOB = "performed locally"

# def hyper_param_search(config, cognate_method, on_hpc=False, afterok=None):
#     search_space = {
#         "oc_num_layers": config["oc_enc_num_layers"],
#         "oc_embed_dim": config["oc_enc_embed_dim"],
#         "oc_hidden_dim": config["oc_enc_hidden_dim"],
#         "oc_batch_size": config["oc_batch_size"],
#     }
    
#     param_lens = []
#     for param in search_space:
#         param_lens.append(len(search_space[param]))
#     len_search_space = 1
#     for n in param_lens:
#         len_search_space *= n

#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, config, cognate_method, search_space, on_hpc, afterok), n_trials=len_search_space)

#     # return the optimal job (have to rerun it to make this work in this way)
#     best = study.best_params
#     config["oc_enc_num_layers"] = best["layer"]
#     config["oc_dec_num_layers"] = best["layer"]
#     config["oc_enc_embed_dim"] = best["embed_dim"]
#     config["oc_dec_embed_dim"] = best["embed_dim"]
#     config["oc_enc_hidden_dim"] = best["hidden_dim"]
#     config["oc_dec_hidden_dim"] = best["hidden_dim"]
#     config["oc_batch_size"] = best["batch_size"]
#     config["hyp_param_name"] = make_hyp_param_name(best)

#     best_jobs = train_and_eval(config, cognate_method, on_hpc, afterok)
#     return best_jobs

# def make_hyp_param_name(params):
#     return "_".join(f"{k}={v}" for k, v in params.items())

# def objective(trial, config, cognate_method, search_space, on_hpc=False, afterok=None):
#     """Runs train_and_eval and returns chrf from the output file"""

#     # search space
#     layer = trial.suggest_categorical("layer", search_space['oc_num_layers'])
#     embed_dim = trial.suggest_categorical("embed_dim", search_space['oc_embed_dim'])
#     hidden_dim = trial.suggest_categorical("hidden_dim", search_space['oc_hidden_dim'])
#     batch_size = trial.suggest_categorical("batch_size", search_space['oc_batch_size'])

#     # set params
#     config["oc_enc_num_layers"] = layer
#     config["oc_dec_num_layers"] = layer
#     config["oc_enc_embed_dim"] = embed_dim
#     config["oc_dec_embed_dim"] = embed_dim
#     config["oc_enc_hidden_dim"] = hidden_dim
#     config["oc_dec_hidden_dim"] = hidden_dim
#     config["oc_batch_size"] = batch_size

#     config["hyp_param_name"] = make_hyp_param_name(trial.params)

#     jobs = train_and_eval(config, cognate_method, on_hpc, afterok)
#     pl, cl, tl = config["oc_scenario"]
#     json_f = os.path.join(config["save"], config["experiment_name"], f"OC/{cognate_method}/{pl}-{cl}/{config["hyp_param_name"]}/predictions/scores.json")
#     with open(json_f) as f:
#         json_data = json.load(f)
#     best_chrF = json_data["BEST_VAL_chrF"]["TEST_chrF"]
#     return best_chrF

def train_and_eval(config, cognate_method, on_hpc=False, afterok=None):
    scenario = config["oc_scenario"]
    pl, cl, tl = scenario

    assert cognate_method == config["oc_method"]
    assert cognate_method in config["sc_model_ids"][scenario]
    assert config["oc_model_id"] in config["sc_model_ids"][scenario]
    assert f"{pl}-{cl}" in config["sc_model_ids"][scenario]
    job_suffix = f"_OC_{cognate_method}|{config['experiment_name']}|{config['sc_model_ids'][scenario]}"
    
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
            timeout=config["basic_timeout"],
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
    job_suffix = f"OC_{cognate_method}|{config['experiment_name']}|{config['sc_model_ids'][scenario]}"

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
            timeout=config["basic_timeout"],
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

