import os
from copy import deepcopy
from sloth_hatch.sloth import log_function_call

from utilities.read_data import read_pl_cl_web_paths
from NMT.train.train import (
    train_model as train_nmt_model,
    eval_models as eval_nmt_models,
    inference as nmt_inference
)
from NMT.train.train_tokenizer import train_tokenizer
from OC.extract_cognates import CandidatesFromParallel
from utilities.hpc import submit_slurm

@log_function_call
def extract_candidates(
    config,
    word_list_out,
    long_enough
):
    print("--CANDIDATES FROM SYNTHETIC PARALLEL--\n\n")
    config_tl_pl = deepcopy(config)
    config_tl_pl["nmt_reverse"] = True
    config_tl_pl["nmt_corpus"] = "parent"

    jobs, eval_jobs, inf_jobs = _nmt_pipeline(config_tl_pl)

#TODO Should implement this without calling submitit. The submitit part should be a wrapper around this.
def _nmt_pipeline(config_tl_pl):
    jobs = {}
    eval_jobs = {}
    inf_jobs = {}
    web_paths = read_pl_cl_web_paths(config_tl_pl["data"])

    for data_folder, pl, cl, tl in list(config_tl_pl["data"]):
        config_tl_pl["data"] = [data_folder, pl, cl, tl] # Doing this to only train a monolingual system, not multilingual
        config_tl_pl["tokenizer"] = train_tokenizer(config_tl_pl, train_with_oc=False) # Train/Get the corresponding monolingual tokenizer

        parent_data, child_data, web_tl = web_paths[(pl, cl)]
        assert web_tl == tl
        child_pair_tl_path = os.path.join(child_data, f"train.{tl}.txt")
        translated_path = child_pair_tl_path + ".{tl}->{pl}.txt", #TODO May need to tag it with some id. This implies we need to id every model we train.

        train_tl_pl = lambda: train_nmt_model(config_tl_pl, fine_tune=False)
        eval_tl_pl = lambda: eval_nmt_models(config_tl_pl, fine_tune=False)
        inf_tl_pl = lambda: nmt_inference(config=config_tl_pl, 
                                          inference_file=child_pair_tl_path,
                                          output_file=translated_path,
                                          src_lang=tl,
                                          tgt_lang=pl,
                                          fine_tune=False) # not a fine-tuned model
        
        assert (tl, pl) not in jobs
        assert (tl, pl) not in eval_jobs
        if config_tl_pl["use_hpc"]:
            job_name = f"TRAIN_TL->PL_{tl}->{pl}|" + config_tl_pl["experiment_name"]
            output_folder = os.path.join(config_tl_pl["save"], config_tl_pl["experiment_name"], f"NMT/{job_name}/SLURM")

            eval_job_name = f"EVAL_TL->PL_{tl}->{pl}|" + config_tl_pl["experiment_name"]
            eval_output_folder = os.path.join(config_tl_pl["save"], config_tl_pl["experiment_name"], f"NMT/{eval_job_name}/SLURM")

            inf_job_name = f"INFERENCE_TL->PL_{tl}->{pl}|" + config_tl_pl["experiment_name"]
            inf_output_folder = os.path.join(config_tl_pl["save"], config_tl_pl["experiment_name"], f"NMT/{inf_job_name}/SLURM")

            jobs[(tl, pl)] = submit_slurm(
                function=train_tl_pl,
                job_name=job_name,
                output_folder=output_folder,
                mail_user=config_tl_pl["email"],
                timeout=config_tl_pl["parent_nmt_timeout"],
                ntasks_per_node=config_tl_pl["parent_nmt_n_gpus"],
                mem_gb=config_tl_pl["parent_nmt_mem"],
                n_gpus=config_tl_pl["parent_nmt_n_gpus"],
                gpu_type=config_tl_pl["gpu_type"],
                qos=config_tl_pl["parent_nmt_qos"]
            )
            eval_jobs[(tl, pl)] = submit_slurm(
                function=eval_tl_pl,
                job_name=eval_job_name,
                output_folder=eval_output_folder,
                mail_user=config_tl_pl["email"],
                timeout=24,
                ntasks_per_node=1,
                mem_gb=config_tl_pl["parent_nmt_mem"],
                n_gpus=1,
                gpu_type=config_tl_pl["gpu_type"],
                qos=config_tl_pl["parent_nmt_qos"],
                afterok=jobs[(tl, pl)].job_id # Wait until training is done before evaluating
            )
            inf_jobs[(tl, pl)] = submit_slurm(
                function=inf_tl_pl,
                job_name=inf_job_name,
                output_folder=inf_output_folder,
                mail_user=config_tl_pl["email"],
                timeout=24,
                ntasks_per_node=1,
                mem_gb=config_tl_pl["parent_nmt_mem"],
                n_gpus=1,
                gpu_type=config_tl_pl["gpu_type"],
                qos=config_tl_pl["parent_nmt_qos"],
                afterok=eval_jobs[(tl, pl)].job_id # Wait until evaluating is done before inference
            )
        else:
            train_tl_pl()
            jobs[(tl, pl)] = "trained locally"

            eval_tl_pl()
            eval_jobs[(tl, pl)] = "evaluated locally"

            inf_tl_pl()
            inf_jobs[(tl, pl)] = "did inference locally"
    
    return jobs, eval_jobs, inf_jobs
