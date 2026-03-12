import argparse
import os
from copy import deepcopy
import submitit
from sloth_hatch.sloth import read_yaml, create_directory, log_parsed_args, log_script

from Pipeline.utilities.read_data_csv import read_data_csv
from OC.utilities.utilities import split, write_oc_data
from OC.train.train import (
    train_model as train_oc_model, 
    eval_models as eval_oc_models, 
    inference as oc_inference
)
from NMT.train.train import (
    train_model as train_nmt_model,
    eval_models as eval_nmt_models,
)

def main(config_f):
    config = read_yaml(config_f)

    pl_cl_mode = validate_config(config)
    config["experiment_name"] += "_" + pl_cl_mode

    pl_cl_files = read_data_csv(config[f"pl_cl_{pl_cl_mode}"])
    pl_cl_pairs = pl_cl_files.keys()

    train_dirs = make_scenario_directory(config["save"], config["experiment_name"], pl_cl_pairs)
    if pl_cl_mode == "mono":
        from OC.extract_cognates.CognatesFromMonolingual import extract_cognates
    else:
        from OC.extract_cognates.CognatesFromParallel import extract_cognates

    oc_data = get_oc_data(config, pl_cl_files, train_dirs, extract_cognates)
    pl_cl_configs = get_pl_cl_configs(config, oc_data)
    validate_pl_cl_configs(pl_cl_configs, base_config_f=config_f)

    ################ SUBMIT OC JOBS ################

    # oc train jobs
    pl_cl_train_jobs = submit_oc_train(config, pl_cl_configs, train_dirs)
    pl_cl_eval_jobs = submit_oc_eval(config, pl_cl_configs, train_dirs, pl_cl_train_jobs)
    
    # oc inference with the best models (one inference per pl/cl pair)
    # oc reshape
    
    # nmt train
    # nmt test
        
def submit_oc_train(config, pl_cl_configs, train_dirs):
    executor = get_oc_executor(config)
    pl_cl_train_jobs = {}
    for (pl, cl), pl_cl_config in pl_cl_configs.items():
        output_folder = os.path.join(train_dirs[(pl, cl)], "slurm_outputs/")
        output_file = os.path.join(output_folder, f"%j_%x.out")

        job_name = f"{pl}_{cl}_oc_train"
        clean_jobs(output_folder, job_name)
        executor.update_parameters(
            slurm_job_name=job_name,
            slurm_additional_parameters={
                "output": output_file
            }
        )
        train_job = executor.submit(train_oc_model, pl_cl_config)
        pl_cl_train_jobs[(pl, cl)] = train_job
    return pl_cl_train_jobs

def submit_oc_eval(config, pl_cl_configs, train_dirs, pl_cl_train_jobs):
    executor = get_oc_executor(config)
    pl_cl_eval_jobs = {}
    for (pl, cl), pl_cl_config in pl_cl_configs.items():
        output_folder = os.path.join(train_dirs[(pl, cl)], "slurm_outputs/")
        output_file = os.path.join(output_folder, f"%j_%x.out")
        
        job_name = f"{pl}_{cl}_oc_eval"
        clean_jobs(output_folder, job_name)
        train_job = pl_cl_train_jobs[(pl, cl)]
        executor.update_parameters(
            slurm_job_name=job_name,
            gpus_per_node=1,
            slurm_additional_parameters={
                "ntasks-per-node": 1,
                "output": output_file,
                "dependency": f"afterok:{train_job.job_id}"
            }
        )
        test_job = executor.submit(eval_oc_models, pl_cl_config)
        pl_cl_eval_jobs[(pl, cl)] = test_job

def get_oc_executor(config):
    executor = submitit.AutoExecutor(folder=os.path.join(config["save"], config["experiment_name"]))
    executor.update_parameters(
        timeout_min=config["oc_timeout"] * 60,
        nodes=1,
        mem_gb=config["oc_mem"],
        gpus_per_node=config["oc_n_gpus"],
        slurm_additional_parameters={
            "ntasks-per-node": config["oc_n_gpus"],
            "qos": config["oc_qos"],
            "mail-type": "BEGIN,END,FAIL",
            "mail-user": config["email"]
        }
    )
    return executor

def clean_jobs(slurm_out_folder, job_name):
    for f in os.listdir(slurm_out_folder):
        if f.endswith(f"_{job_name}.out"):
            f_path = os.path.join(slurm_out_folder, f)
            os.remove(f_path)

def get_pl_cl_configs(config, oc_data):
    all_configs = {}
    for pl_cl_pair, (oc_train, oc_val) in oc_data.items():
        pl_cl_config = deepcopy(config)
        pl_cl_config["oc_train"] = oc_train
        pl_cl_config["oc_val"] = oc_val
        assert pl_cl_pair not in all_configs
        all_configs[pl_cl_pair] = pl_cl_config
    return all_configs

def get_oc_data(config: dict, pl_cl_files: dict, train_dirs: dict, extract_cognates: function):    
    oc_data = {}
    for (src_lang, tgt_lang), (src_path, tgt_path) in pl_cl_files.items():
        out_stem = os.path.join(train_dirs[(src_lang, tgt_lang)]["data"], f"{src_lang}_{tgt_lang}")
        cognate_list_out = out_stem + ".cognates"
        oc_val_out = out_stem + ".val"
        oc_train_out = out_stem + ".train"
        
        cognates = extract_cognates(
            src_file=src_path,
            tgt_file=tgt_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            cognate_list_out=cognate_list_out,
            theta=config["theta"]
        )
        oc_train, oc_val = split(
            data=cognates,
            val_ratio=config["oc_val_ratio"],
            seed=config["seed"]
        )

        write_oc_data(oc_val, oc_val_out)
        write_oc_data(oc_train, oc_train_out)

        lang_pair = src_lang, tgt_lang
        assert lang_pair not in oc_data
        oc_data[lang_pair] = oc_train, oc_val
    return oc_data

def validate_config(config):
    if (config["pl_cl_mono"] and config["pl_cl_para"]) \
        or (not config["pl_cl_mono"] and not config["pl_cl_para"]):
        raise ValueError(f"Must specify either pl_cl_mono or pl_cl_para.")
    return "mono" if config["pl_cl_mono"] else "para"

def validate_pl_cl_configs(pl_cl_configs, base_config_f):
    print("Validating PL/CL configs")
    base_config = read_yaml(base_config_f)
    pl_cl_mode = validate_config(base_config)
    base_kvs = dict_to_tuples(base_config)
    for (pl, cl), pl_cl_cfg in pl_cl_configs.items():
        pl_cl_kvs = dict_to_tuples(pl_cl_cfg)
        assert base_kvs.intersection(pl_cl_kvs) == base_kvs
        assert len(base_kvs.difference(pl_cl_kvs)) == 0
        pl_cl_diff = pl_cl_kvs.difference(base_kvs)
        
        oc_data_folder = os.path.join(
            base_config["save"], 
            base_config["experiment_name"] + "_" + pl_cl_mode,
            "OC/data"
        )
        assert len(pl_cl_diff) == {
            ("oc_train", f"{oc_data_folder}/{pl}_{cl}.train"),
            ("oc_val", f"{oc_data_folder}/{pl}_{cl}.val")
        }
        print(f"\t-{pl}/{cl} oc_train + oc_val:\n\t\t", sorted(pl_cl_diff))

def dict_to_tuples(dictionary):
    return set((k, v) for k, v in dictionary.items())

def make_scenario_directory(save_d, experiment_name, lang_pairs):
    if not os.path.exists(save_d):
        raise FileExistsError(f"Directory `{save_d}` does not exist!")
    scen_dir = os.path.join(save_d, experiment_name)
    create_directory(scen_dir)
    train_dirs = {"NMT": make_train_dir(scen_dir, "NMT")}

    OC_scen_dir = os.path.join(scen_dir, "OC")
    create_directory(OC_scen_dir)
    for src_lang, tgt_lang in lang_pairs:
        assert (src_lang, tgt_lang) not in train_dirs
        train_dirs[(src_lang, tgt_lang)] = make_train_dir(OC_scen_dir, f"{src_lang}_{tgt_lang}")
    return train_dirs

def make_train_dir(scen_dir, name):
    train_dir = os.path.join(scen_dir, name)
    create_directory(train_dir)
    subdirs = get_train_subdirs(train_dir)
    for sub_d in subdirs.values():
        os.mkdir(sub_d)
    return subdirs

def get_train_subdirs(d):
    subdirs = {}
    for sub_d in ["checkpoints", "data", "predictions", "logs", "tb"]:
        assert sub_d not in subdirs
        subdirs[sub_d] = os.path.join(d, sub_d)
    return subdirs

@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="/home/hatch5o6/CharLOTTE2.0/src/configs/test.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    log_script(__package__, __file__)
    args = get_args()
    main(args.config)
