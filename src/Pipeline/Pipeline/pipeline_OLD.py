import argparse
import os
from copy import deepcopy
import submitit
from functools import partial
from sloth_hatch.sloth import read_yaml, create_directory, log_parsed_args, log_script, log_function_call

from utilities.utilities import set_vars_in_path
from utilities.read_data import read_config, get_pl_cl_pairs, read_pl_cl_paths, read_pl_cl_web_paths, read_pl_cl_fuzz_paths
from OC.extract_cognates.FilterCognatePairs import filter_cognate_pairs
from OC.utilities.utilities import write_oc_data, read_oc_data
from OC.extract_cognates.TrainValSplit import get_train_val_split, get_train_split
from OC.reshape.reshape import prepare_source_words, reshape_data
from OC.train.train import (
    train_model as train_oc_model, 
    eval_models as eval_oc_models, 
    inference as oc_inference
)
from NMT.train.train import (
    train_model as train_nmt_model,
    eval_models as eval_nmt_models,
)

COGNATE_METHODS = ['charlotte', 'web', 'fuzz']

@log_function_call
def main(config_f, cognate_method=None):
    config = read_config(config_f)
    if not os.path.exists(config["save"]):
        os.mkdir(config["save"])

    pl_cl_pairs = get_pl_cl_pairs(config["data"])
    pl_cl_files = read_pl_cl_paths(config["data"]) # returns {(pl, cl): (pl_path, cl_path)}

    for cm in COGNATE_METHODS:
        get_scenario_directory(config["save"], config["experiment_name"] + "_" + cm, pl_cl_pairs)

    validation_set_files = {}
    val_set_methods = {
        "charlotte": set(),
        "web": set()
    }
    for pl, cl in pl_cl_pairs:
        if (pl, cl) in pl_cl_files:
            val_set_method = "charlotte"
        else:
            val_set_method = "web"
        print(f"OC {pl},{cl} VALIDATION SET METHOD: {val_set_method}")
        val_set_methods[val_set_method].add((pl, cl))
        experiment_name = config["experiment_name"] + "_" + val_set_method
        val_file = os.path.join(config["save"], experiment_name, f"OC/{pl}_{cl}/data/{pl}_{cl}.val")
        if os.path.exists(val_file):
            print(f"Retrieving validation set from method: {val_set_method}")
            print(f"\tFile: {val_file}")
            validation_set_files[(pl, cl)] = val_file
    
    assert len(validation_set_files) in [0, len(pl_cl_pairs)]
    if len(validation_set_files) == 0:
        validation_set_files = None

    # Get validation_set_files
    if validation_set_files == None:
        charlotte_val_set_files = run_experiment(
            config,
            cognate_method="charlotte",
            validation_set_files=None,
            only_pl_cls=val_set_methods["charlotte"]
        )
        print("CharLOTTE VAL SETS:")
        _print_data_paths(charlotte_val_set_files)

        web_val_set_files = run_experiment(
            config,
            cognate_method="web",
            validation_set_files=None,
            only_pl_cls=val_set_methods["web"]
        )
        print("WEB VAL SETS:")
        _print_data_paths(web_val_set_files)

        assert set(charlotte_val_set_files.keys()).intersection( set(web_val_set_files.keys()) ) == set()

        # Concatenate them together
        validation_set_files = charlotte_val_set_files | web_val_set_files
        assert len(validation_set_files) == len(charlotte_val_set_files) + len(web_val_set_files)
    
    assert validation_set_files is not None

    if cognate_method:
        cognate_methods = [cognate_method]
    else:
        cognate_methods = COGNATE_METHODS

    for method in cognate_methods:
        run_experiment(config,
                       cognate_method=method,
                       validation_set_files=validation_set_files)

@log_function_call
def run_experiment(config: str, cognate_method: str, validation_set_files:dict=None, only_pl_cls:set=None):
    """
    config (dict): experiment configuration
    cognate_method (str): 'charlotte', 'web', or 'fuzz'
    validation_set_files (dict): dictionary of OC (pl, cl) tuples to the paths of the validation sets.

    To execute the full pipeline, and to train any models, validation_set_files MUST be passed.
    If not passed, then a validation set files will be created for each pl/cl OC direction and returned.
    """
    config = deepcopy(config)

    experiment_name = config["experiment_name"] + "_" + cognate_method

    # Get all pl, cl pairs
    pl_cl_pairs = get_pl_cl_pairs(config["data"]) # [(pl, cl)]
    if validation_set_files is not None:
        if set(validation_set_files.keys()) != set(pl_cl_pairs):
            raise ValueError(f"validation_set_files must contain validation files for each pl/cl pair: {sorted(pl_cl_pairs)}")

    # Get pl, cl data files from which cognates will be extracted
    if cognate_method == "charlotte":
        pl_cl_files = read_pl_cl_paths(config["data"]) # returns {(pl, cl): (pl_path, cl_path)}
    elif cognate_method == "web":
        pl_cl_files = { # These don't exist yet
            pair: (None, None)
            for pair in pl_cl_pairs
        }
    elif cognate_method == "fuzz":
        pl_cl_files = read_pl_cl_fuzz_paths(config["data"])
    else:
        raise ValueError("Cognate method must be 'charlotte', 'web', or 'fuzz'.")

    if cognate_method in ["web", "fuzz"]:
        assert sorted(pl_cl_files.keys()) == sorted(pl_cl_pairs)
    else:
        assert len(pl_cl_files.keys()) <= len(pl_cl_pairs)
        assert set(pl_cl_files.keys()).difference( set(pl_cl_pairs) ) == set()

    # When validation_set_files == None, i.e. when we're building the validation data,
        # we run this function for the val sets made from charlotte and those made from web, so we
        # need to filter the pl_cl_files to be only the charlotte or web datasets that we're extracting cognates from.
    if only_pl_cls is not None:
        assert validation_set_files is None, "Can only pass only_pl_cls when validation_set_files is None, i.e. only when we're building the validation set."
        pl_cl_files = {k: v for k, v in pl_cl_files.items() if k in only_pl_cls}
    print("PL CL FILES:")
    print(pl_cl_files)
    
    # get training data directories
    train_dirs = get_scenario_directory(config["save"], experiment_name, pl_cl_pairs, create=False)

    # Get the appropriate extract_candidates function, per the set cognate_method
    if cognate_method == "charlotte":
        from OC.extract_cognates.CandidatesFromParallel import extract_candidates
    elif cognate_method == "web":
        # the web extract_candidates uses a very different interface, so we create a higher-order function to handle this case
        from OC.extract_cognates.CandidatesFromSyntheticParallel import extract_candidates as _web_extract_candidates
        web_nmt_data = read_pl_cl_web_paths(config["data"])
        def extract_candidates(src_file, tgt_file, src_lang, tgt_lang, word_list_out, long_enough):
            assert src_file == tgt_file == None
            pl = src_lang
            cl = tgt_lang
            parent_data, child_data, tl = web_nmt_data[(pl, cl)]
            return _web_extract_candidates(
                parent_data=parent_data,
                child_data=child_data,
                pl=pl,
                cl=cl,
                tl=tl,
                word_list_out=word_list_out,
                long_enough=long_enough
            )
    elif cognate_method == "fuzz":
        from OC.extract_cognates.FuzzyCandidates import extract_candidates
        extract_candidates = partial(extract_candidates, top_k=config.get("fuzz_top_k"))
    else:
        raise ValueError("Cognate method must be 'charlotte', 'web', or 'fuzz'")

    # Get the OC training and validation data files per pl, cl pair
    # oc_data = {(pl, cl): (train file, validation file)}
    oc_data = get_oc_data(
        config=config, 
        pl_cl_files=pl_cl_files, 
        train_dirs=train_dirs, 
        extract_candidates=extract_candidates,
        validation_set_files=validation_set_files
    )
    if validation_set_files == None:
        # if we're only making the validation files, then we return them now instead of running the full pipeline
        print("Returning validation_set_files")
        return {(pl, cl): val_file for (pl, cl), (train_file, val_file) in oc_data.items()} 
    
    print("TRAINING AND VALIDATION DATA:")
    _print_data_paths(oc_data)

    # Set the experiment name in the config
    config["experiment_name"] = experiment_name

    # pl_cl_configs = {(pl, cl): config dictionary (same as the base config, but with the oc train and val data files added)}
    pl_cl_configs = get_pl_cl_configs(config, oc_data)
    validate_pl_cl_configs(pl_cl_configs, base_config=config)
    exit()

    ################ SUBMIT OC JOBS ################

    # oc train jobs
    pl_cl_train_jobs = submit_oc_train(config, pl_cl_configs, train_dirs)
    pl_cl_eval_jobs = submit_oc_eval(config, pl_cl_configs, train_dirs, pl_cl_train_jobs)
    
    # oc inference with the best models (one inference per pl/cl pair)
    # oc reshape
    
    # nmt train
    # nmt test

def _print_data_paths(dictionary):
    for (l1, l2), data in dictionary.items():
        if not isinstance(data, (list, tuple)):
            data = [data]
        print(f"({l1}, {l2}):")
        for path in data:
            print(f"\t-`{path}`")

def submit_oc_train(config, pl_cl_configs, train_dirs):
    executor = get_oc_executor(config)
    pl_cl_train_jobs = {}
    for (pl, cl), pl_cl_config in pl_cl_configs.items():
        output_folder = train_dirs[(pl, cl)]["slurm_outputs"]
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
        print(f"({(pl, cl)}) submitted train {train_job.job_id}")
        pl_cl_train_jobs[(pl, cl)] = train_job
    return pl_cl_train_jobs

def submit_oc_eval(config, pl_cl_configs, train_dirs, pl_cl_train_jobs):
    executor = get_oc_executor(config)
    pl_cl_eval_jobs = {}
    for (pl, cl), pl_cl_config in pl_cl_configs.items():
        output_folder = train_dirs[(pl, cl)]["slurm_outputs"]
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
        print(f"({(pl, cl)}) submitted eval {test_job.job_id} afterok:{train_job.job_id}")
        pl_cl_eval_jobs[(pl, cl)] = test_job

def submit_oc_inference(config, pl_cl_configs, train_dirs, pl_cl_eval_jobs):
    executor = get_oc_executor(config)
    pl_cl_infer_jobs = {}
    for (pl, cl), pl_cl_config in pl_cl_configs.items():
        output_folder = train_dirs[(pl, cl)]["slurm_outputs"]
        output_file = os.path.join(output_folder, f"%j_%x.out")

        # reshape: prepare_source_words
        # then submit inference

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
        pl_cl_config["oc_lang_pair"] = list(pl_cl_pair)
        pl_cl_config["oc_train"] = oc_train
        pl_cl_config["oc_val"] = oc_val
        assert pl_cl_pair not in all_configs
        all_configs[pl_cl_pair] = pl_cl_config
    return all_configs

def get_oc_data(
    config:dict, 
    pl_cl_files:dict, 
    train_dirs:dict, 
    extract_candidates,
    validation_set_files=None
):
    oc_data = {}
    for (src_lang, tgt_lang), (src_path, tgt_path) in pl_cl_files.items():
        out_stem = os.path.join(train_dirs[(src_lang, tgt_lang)]["data"], f"{src_lang}_{tgt_lang}")
        
        # get candidate word pairs
        word_list_out = out_stem + ".word_list"
        candidates = extract_candidates(
            src_file=src_path,
            tgt_file=tgt_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            word_list_out=word_list_out,
            long_enough=config["oc_min_word_len"]
        )

        # filter down to cognates
        cognates = filter_cognate_pairs(
            word_pairs=candidates,
            theta=config["theta"],
            long_enough=config["oc_min_word_len"]
        )
        cognate_list_out = out_stem + ".cognates"
        if os.path.exists(cognate_list_out):
            # if this file already exists, make sure it matches the cognates we just got
            assert cognates == read_oc_data(cognate_list_out)
        else:
            # otherwise, write the cognates to the file
            write_oc_data(cognates, cognate_list_out)

        # get train / validation split
        print("GET_OC_DATA: Get train / valiation split:")
        if validation_set_files:
            assert (src_lang, tgt_lang) in validation_set_files
            val_set_file = validation_set_files[(src_lang, tgt_lang)]
            print(f"\tGetting val set from {val_set_file}")
            validation_set = read_oc_data(val_set_file)
            train_set = get_train_split(cognates, validation_set, seed=config["seed"])
        else:
            print("\tMaking train / validation split from scratch")
            train_set, validation_set = get_train_val_split(cognates,
                                                            theta=config["theta"],
                                                            size=config["oc_val_size"],
                                                            n_buckets=config["oc_val_nld_buckets"],
                                                            max_fraction=config["oc_val_max_bucket_fraction"],
                                                            seed=config["seed"])

        assert isinstance(validation_set, list)
        assert isinstance(train_set, list)
        
        # Assert no contamination of the training set
        _assert_no_train_contamination(train_set, validation_set)

        val_file = out_stem + ".val"
        train_file = out_stem + ".train"

        print("GET_OC_DATA: Writing validation file:")
        if os.path.exists(val_file):
            # If the validation set file is already written, don't write it again
            assert validation_set_files is not None
            assert val_file == validation_set_files[(src_lang, tgt_lang)]
            assert read_oc_data(validation_set_files[(src_lang, tgt_lang)]) == validation_set
            print(f"\tActually, we read validation data from {val_file}. No need to write.")
        else:
            print(f"\tWriting validation data to {val_file}")
            write_oc_data(validation_set, val_file)

        # Only write the training data when validation_set_files is passed
        # That is, we don't need to make the training data when we're just running the method to make a validation set.
        print("GET_OC_DATA: Writing training data")
        if validation_set_files:
            print(f"\tWriting training data to {train_file}")
            write_oc_data(train_set, train_file)
        else:
            print("\tOnly creating validation set from scratch. No need to write training data yet.")

        assert (src_lang, tgt_lang) not in oc_data
        oc_data[(src_lang, tgt_lang)] = train_file, val_file
    return oc_data

def _get_word_pairs(dataset):
    return set(
        (pair[-3], pair[-2])
        for pair in dataset
    )

def _assert_no_train_contamination(train_set, val_set):
    assert isinstance(train_set, list)
    assert isinstance(val_set, list)

    train_word_pairs = _get_word_pairs(train_set)
    val_word_pairs = _get_word_pairs(val_set)

    assert len(train_set) == len(train_word_pairs)
    assert len(val_set) == len(val_word_pairs)

    assert train_word_pairs.intersection(val_word_pairs) == set()

def validate_pl_cl_configs(pl_cl_configs, base_config):
    print("Validating PL/CL configs")
    base_kvs = dict_to_tuples(base_config)
    for (pl, cl), pl_cl_cfg in pl_cl_configs.items():
        pl_cl_kvs = dict_to_tuples(pl_cl_cfg)
        assert base_kvs.intersection(pl_cl_kvs) == base_kvs
        assert len(base_kvs.difference(pl_cl_kvs)) == 0
        pl_cl_diff = pl_cl_kvs.difference(base_kvs)
        
        oc_data_folder = os.path.join(
            base_config["save"], 
            base_config["experiment_name"],
            f"OC/{pl}_{cl}/data"
        )
        assert pl_cl_diff == {
            ("oc_train", f"{oc_data_folder}/{pl}_{cl}.train"),
            ("oc_val", f"{oc_data_folder}/{pl}_{cl}.val")
        }
        print(f"\t-{pl}/{cl} oc_train + oc_val:\n\t\t", sorted(pl_cl_diff))

def dict_to_tuples(dictionary):
    set_of_tuples = set()
    for k, v in dictionary.items():
        if isinstance(v, list):
            v = str(v)
        set_of_tuples.add((k, v))
    return set_of_tuples

def get_scenario_directory(save_d, experiment_name, lang_pairs, create=True):
    if not os.path.exists(save_d):
        raise FileNotFoundError(f"Save directory `{save_d}` does not exist!")
    scen_dir = os.path.join(save_d, experiment_name)
    if create:
        create_directory(scen_dir)
    elif not os.path.exists(scen_dir):
        raise FileNotFoundError(f"Scenario directory `{scen_dir}` does not exist!")
    
    NMT_scen_dir = os.path.join(scen_dir, "NMT")
    if create:
        create_directory(NMT_scen_dir)
    elif not os.path.exists(NMT_scen_dir):
        raise FileNotFoundError(f"NMT directory `{NMT_scen_dir}` does not exist!")
    train_dirs = {
        "NMT_parent": get_train_dir(NMT_scen_dir, "NMT_parent", create=create),
        "NMT_child": get_train_dir(NMT_scen_dir, "NMT_child", create=create)
    }

    OC_scen_dir = os.path.join(scen_dir, "OC")
    if create:
        create_directory(OC_scen_dir)
    elif not os.path.exists(OC_scen_dir):
        raise FileNotFoundError(f"OC directory `{OC_scen_dir}` does not exist!")
    
    for src_lang, tgt_lang in lang_pairs:
        assert (src_lang, tgt_lang) not in train_dirs
        train_dirs[(src_lang, tgt_lang)] = get_train_dir(OC_scen_dir, f"{src_lang}_{tgt_lang}", create=create)
    return train_dirs

def get_train_dir(scen_dir, name, create=True):
    train_dir = os.path.join(scen_dir, name)
    if create:
        create_directory(train_dir)
    elif not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory `{train_dir}` does not exist!")
    
    subdirs = get_train_subdirs(train_dir)
    if create:
        for sub_d in subdirs.values():
            os.mkdir(sub_d)
    else:
        for sub_d in subdirs.values():
            if not os.path.exists(sub_d):
                raise FileNotFoundError(f"Sub train directory `{sub_d}` does not exist!")
    return subdirs

def get_train_subdirs(d):
    subdirs = {}
    for sub_d in ["checkpoints", "data", "predictions", "logs", "tb", "slurm_outputs"]:
        assert sub_d not in subdirs
        subdirs[sub_d] = os.path.join(d, sub_d)
    return subdirs

@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="/home/hatch5o6/CharLOTTE2.0/src/configs/test.yaml")
    parser.add_argument("-m", "--method", choices=COGNATE_METHODS, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    log_script(__package__, __file__)
    args = get_args()
    main(args.config, args.method)
