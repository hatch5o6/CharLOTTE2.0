import argparse
import os
from copy import deepcopy
import functools
from sloth_hatch.sloth import read_content, write_content, write_json, read_lines, read_yaml, create_directory, log_parsed_args, log_script, log_function_call, read_json

from NMT.train import train_jobs as NMT_train_jobs
from NMT.train.train import _nmt_config_key
from OC.train import train_jobs as OC_train_jobs
from OC.train.TrainValSplit import get_train_val_split, get_train_split
from OC.extract_cognates.CandidatesFromParallel import extract_candidates as extract_candidates_from_parallel
from OC.extract_cognates.FuzzyCandidates import extract_candidates as extract_fuzzy_candidates
from OC.extract_cognates.Cognates import make_cognates
from OC.reshape import reshape
from OC.utilities.utilities import write_oc_data, read_oc_data
import utilities
import utilities.model_names as model_names
from utilities.read_data import get_pl_cl_pairs, read_pl_cl_paths, read_pl_cl_web_paths, read_pl_cl_fuzz_paths, read_pl_cl_parent_child_paths, _validate_sets, _validate_cognate_methods, read_pl_tl_data
from utilities.experiment_file_system import get_exp_dir, get_task_dir, get_train_dir
from utilities import hpc

PREV_JOB = "preformed previously"
TO_TL = "-->TL"
FROM_SL = "SL-->"

@log_function_call
def main(
    config_f,
    pipeline,
    nmt_models,
    apply_methods,
    lang_filters,
    nmt_directions
):
    # lang_filters = [(pl, cl, tl), ...]
    # temp config to get the OC directory
    config = utilities.read_data.read_config(config_f, 
                                             add_sc_model_ids=True)
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    NMT_dir = get_task_dir(exp_dir, task="NMT")


    # Get the OC and NMT model names
    oc_model_name = _get_model_name(OC_dir, model_type="OC")
    nmt_model_name = _get_model_name(NMT_dir, model_type="NMT")

    # Get the config for real now
    config = utilities.read_data.read_config(config_f, 
                                             add_sc_model_ids=True, 
                                             oc_model_id=oc_model_name,
                                             nmt_model_id=nmt_model_name)
    # validate methods
    for method in config["methods"]:
        if method not in ["charlotte", "web", "fuzz"] or len(config["methods"]) > 3 or len(set(config["methods"])) != len(config["methods"]):
            raise ValueError("config['methods'] must be a unique-set list containing ONLY 'charlotte', 'web', and/or 'fuzz'!")
    
    # apply only apply_methods
    config["methods"] = sorted(
        set(config["methods"]).intersection(set(apply_methods)),
        key=functools.cmp_to_key(_method_comparator)
    )

    ####################################################################
    ######################## CharLOTTE PIPELINE ########################
    ####################################################################

    # ------------------------ Baselines ------------------------
    if 'baselines' in pipeline:
        for direction in nmt_directions:
            # Simple baseline
            if 'simple' in nmt_models:
                NMT_train_jobs.train_simple(
                    config=config,
                    reverse=direction==FROM_SL
                )

            # Transfer baseline
            NMT_train_jobs.train_parent_child(
                config=config,
                reverse=direction==FROM_SL,
                do_train_parent='parent' in nmt_models,
                do_train_child='child' in nmt_models
            )

    # ------------------------ TL-->PL ------------------------
    # if doing web, Train, Eval, and Infer TL --> PL translation
    tl_pl_translation_results = {}
    if "web" in config["methods"]:
        tl_pl_translation_results = tl_to_pl_translation(config,
                                                         do_translation=True if 'TL-->PL' in pipeline else False,
                                                         lang_filters=lang_filters)
    
    # ------------------------ prepare_OC ------------------------
    if "prepare_OC" in pipeline:
        # Should probably kick off each scenario-specific job after its respective tl-->pl is done, but
            # this should only matter in the multilingual case, which we're not focussing on, so move on for now
        # Prepare OC training data
        prepare_oc_data_folder = os.path.join(config["save"], config["experiment_name"], "OC/SLURM/prepare_OC_data")
        prepare_oc_after_ok = _get_all_scen_NMT_afterok(tl_pl_translation_results) \
            if tl_pl_translation_results and config["use_hpc"] and 'TL-->PL' in pipeline else None
        prepare_OC_data_results = hpc.abstract_function(
            function=lambda: prepare_OC_data(config, 
                                            tl_pl_translation_results,
                                            previous_oc_inference='TL-->PL' not in pipeline,
                                            lang_filters=lang_filters),
            job_name="prepare_OC_data_" + config["experiment_name"],
            output_folder=prepare_oc_data_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            nodes=1,
            mem_gb=config["basic_mem"],
            n_gpus=0,
            qos=config["basic_qos"],
            afterok=prepare_oc_after_ok,
            use_hpc=config["use_hpc"]
        )


    # ------------------------ OC ------------------------
    if "OC" in pipeline:
        # Likewise, could kick off each OC job after respective prepare_OC_data job is done (based on scenario), but will move on for now
        # Train, Eval, and Infer OC
        OC_folder = os.path.join(config["save"], config["experiment_name"], "OC/SLURM/OC")
        OC_afterok = prepare_OC_data_results.job_id if config["use_hpc"] and "prepare_OC" in pipeline else None
        OC_results = hpc.abstract_function(
            function=lambda: OC(config, lang_filters=lang_filters),
            job_name="OC_" + config["experiment_name"],
            output_folder=OC_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            nodes=1,
            mem_gb=config["basic_mem"],
            n_gpus=0,
            qos=config["basic_qos"],
            afterok=OC_afterok,
            use_hpc=config["use_hpc"]
        )

    # ------------------------ OC_reshape ------------------------
    if "OC_reshape" in pipeline:
        # Same, could rework this so we don't have to wait for all OC models to train, eval, and infer
        # OC reshape
        OC_reshape_folder = os.path.join(config["save"], config["experiment_name"], "OC/SLURM/OC_reshape")
        # OC_reshape_afterok = _get_all_scen_OC_afterok(OC_results) if config["use_hpc"] and "OC" in pipeline else None
        OC_reshape_afterok = OC_results.job_id if config["use_hpc"] and "OC" in pipeline else None
        OC_reshape_results = hpc.abstract_function(
            function=lambda: OC_reshape(config, lang_filters=lang_filters),
            job_name="OC_reshape_" + config["experiment_name"],
            output_folder=OC_reshape_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            nodes=1,
            mem_gb=config["basic_mem"],
            n_gpus=0,
            qos=config["basic_qos"],
            afterok=OC_reshape_afterok,
            use_hpc=config["use_hpc"]
        )
    
    # ------------------------ OC-augmented_NMT ------------------------
    if "OC_NMT" in pipeline:
        for direction in nmt_directions:
            for oc_method in config["methods"]:
                NMT_train_jobs.train_parent_child(
                    config=config,
                    afterok=OC_reshape_results.job_id if config["use_hpc"] and "OC_reshape" in pipeline else None,
                    oc_method=oc_method,
                    reverse=direction==FROM_SL,
                    do_train_parent='parent' in nmt_models,
                    do_train_child='child' in nmt_models
                )
    
    # Compile all results in experiment dir (whether just run or not)
    _compile_nmt_results(exp_dir, config["methods"])

def _method_comparator(x, y):
    assert x in ["charlotte", "web", "fuzz"]
    assert y in ["charlotte", "web", "fuzz"]
    assert x != y
    if x == "charlotte":
        return - 1
    elif x == "fuzz":
        return 1
    elif y == "charlotte":
        return 1
    elif y == "fuzz":
        return - 1

def _get_all_scen_OC_afterok(results):
    scen_afterok = []
    for cognate_method, scen_stuff in results.items():
        for scenario, scen_results in scen_stuff.items():
            scen_infer_job = scen_results["jobs"]["infer"]
            scen_afterok.append(str(scen_infer_job.id))
    scen_afterok = ":".join(scen_afterok)
    return scen_afterok

def _get_all_scen_NMT_afterok(results):
    scen_afterok = []
    for scenario, (scen_config, scen_jobs) in results.items():
        scen_afterok.append(_get_NMT_afterok(scen_jobs))
    scen_afterok = ":".join(scen_afterok)
    return scen_afterok

def _get_NMT_afterok(jobs):
    assert isinstance(jobs, dict)
    assert set(jobs.keys()) == {"train", "eval", "infer"}
    after_ok = []
    for funct, (job, job_name) in jobs.items():
        after_ok.append(str(job.job_id))
    after_ok = ":".join(after_ok)
    return after_ok

def _get_model_name(directory, model_type):
    if model_type not in ["NMT", "OC"]:
        raise ValueError(f"model_type must be 'NMT' or 'OC'!")
    model_name_f = os.path.join(directory, f"{model_type}_MODEL_NAME")
    if not os.path.exists(model_name_f):
        model_name = model_names.get_new_name()
        print(_wrap_in_pounds(f"Thy {model_type} models shall bear the name \"{model_name}\""))
        write_content(model_name.strip(), model_name_f)
    else:
        model_name = read_content(model_name_f).strip()
        print(_wrap_in_pounds(f"Thy {model_type} models doth already bear the name \"{model_name}\""))
    return model_name

def _wrap_in_pounds(name):
    boundary = "#" * (len(name) + 6)
    return f"\n\n{boundary}\n## {name} ##\n{boundary}\n\n"

def _validate_lang_filters(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) >= 2:
            filters = args[-1]
        else:
            filters = kwargs["lang_filters"]
        if not _lang_filters_are_valid(filters):
            raise ValueError(f"lang_filters must be a list/tuple of language tuples (pl, cl, tl), or None!")
        result = f(*args, **kwargs)
        return result
    return wrapper

def _lang_filters_are_valid(filters):
    if filters == None:
        return True
    if not isinstance(filters, (list, tuple)):
        return False
    for item in filters:
        if not isinstance(item, tuple):
            return False
        if len(item) != 3:
            return False
        for elem in item:
            if not isinstance(elem, str):
                return False
    return True

@_validate_lang_filters
def prepare_OC_data(config, tl_to_pl_results, previous_oc_inference=False, lang_filters=None):
    tl_to_pl_tags = _get_tl_to_pl_tags(tl_to_pl_results, previous_inference=previous_oc_inference, use_hpc=config["use_hpc"])
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    for cognate_method in config["methods"]:
        assert not os.path.exists( os.path.join(OC_dir, cognate_method) )

    # Run cognate methods for each pair, including getting the common validation set
    # Write oc data for all languages -- don't use filters
    # TODO write a check to see if oc data is already written. If it is, don't need to redo it.
    _run_all_cognates_methods(config, tl_to_pl_tags, lang_filters=lang_filters)

    for cognate_method in config["methods"]:
        assert os.path.exists( os.path.join(OC_dir, cognate_method) )

@_validate_lang_filters
def OC(config, lang_filters=None):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # Get the configs for each OC model
    oc_configs = {}
    for cognate_method in config["methods"]:
        assert cognate_method not in oc_configs
        oc_configs[cognate_method] = {}

        cognate_method_dir = os.path.join(OC_dir, cognate_method)
        assert os.path.exists(cognate_method_dir)
        for data_folder, pl, cl, tl in config["data"]:
            if lang_filters and (pl, cl, tl) not in lang_filters:
                continue

            train_dir, sub_dirs = get_train_dir(cognate_method_dir, name=f"{pl}-{cl}", create=False)
            train_data = os.path.join(sub_dirs["data"], "train.txt")
            val_data = os.path.join(sub_dirs["data"], f"val.txt")

            scenario = (pl, cl, tl)
            scen_oc_config = deepcopy(config)
            scen_oc_config["oc_train"] = train_data
            scen_oc_config["oc_val"] = val_data
            scen_oc_config["oc_scenario"] = scenario
            scen_oc_config["oc_method"] = cognate_method
            scen_oc_config["sc_model_id_prefix"] = scen_oc_config["sc_model_id_prefix"].replace("{method}", cognate_method)
            scen_oc_config["sc_model_ids"] = {s: sc_id.replace("{method}", cognate_method) for s, sc_id in scen_oc_config["sc_model_ids"].items()}

            assert scenario not in oc_configs[cognate_method]
            oc_configs[cognate_method][scenario] = scen_oc_config

            scen_oc_config_filepath = os.path.join(sub_dirs["logs"], "scen_of_config.json")
            write_json(scen_oc_config, scen_oc_config_filepath, indent=2, ensure_ascii=False)
    
    # Kick off OC training, evaluation, and inference jobs
    pl_tl_data = read_pl_tl_data(config["data"])
    scen_jobs = {}
    for cognate_method, scenario_configs in oc_configs.items():
        assert cognate_method not in scen_jobs
        scen_jobs[cognate_method] = {}
        for scen, scen_oc_config in scenario_configs.items():
            assert scen_oc_config["oc_method"] == cognate_method
            jobs = OC_train_jobs.train_and_eval(config=scen_oc_config,
                                                cognate_method=cognate_method,
                                                on_hpc=config["use_hpc"])
            if config["use_hpc"]:
                eval_job_id = jobs["eval"][0].job_id
            else:
                eval_job_id = None

            pl, cl, tl = scen
            pl_train = os.path.join(pl_tl_data[scen], f"train.{pl}.txt")
            pl_val = os.path.join(pl_tl_data[scen], f"val.{pl}.txt")
            pl_test = os.path.join(pl_tl_data[scen], f"test.{pl}.txt")
            pl_words_out_path = os.path.join(pl_tl_data[scen], f"words_for_inference.txt")
            reshape.prepare_source_words([pl_train, pl_val, pl_test],
                                         long_enough=config["oc_min_word_len_applied"],
                                         out_path=pl_words_out_path)

            jobs.update(OC_train_jobs.infer(config=scen_oc_config,
                                            cognate_method=cognate_method,
                                            source_words_f=pl_words_out_path,
                                            on_hpc=config["use_hpc"],
                                            afterok=eval_job_id))
            
            scen_jobs[cognate_method][scen] = {"jobs": jobs, "words_for_inference": pl_words_out_path}
    return scen_jobs
        

@_validate_lang_filters
def OC_reshape(config, lang_filters=None):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    pl_tl_data = read_pl_tl_data(config["data"])
    for cognate_method in config["methods"]:
        cognate_method_dir = os.path.join(OC_dir, cognate_method)
        assert os.path.exists(cognate_method_dir)

        for data_folder, pl, cl, tl in config["data"]:
            scenario = (pl, cl, tl)
            if lang_filters and scenario not in lang_filters:
                continue

            output_tag = "." + config["sc_model_ids"][scenario].replace("{method}", cognate_method)

            source_words_f = os.path.join(pl_tl_data[scenario], f"words_for_inference.txt")
            hyp_words_f = source_words_f + output_tag
            if not os.path.exists(source_words_f):
                raise FileNotFoundError(f"source_words_f does not exist: {source_words_f}")
            if not os.path.exists(hyp_words_f):
                raise FileNotFoundError(f"hyp_words_f does not exist: {hyp_words_f}")

            source_words = [
                word1 for _, word1, _, _
                in read_oc_data(source_words_f)
            ]
            hyp_words = read_lines(hyp_words_f)
            assert len(source_words) == len(hyp_words)
            mappings = {
                source_word: hyp_word
                for source_word, hyp_word in zip(source_words, hyp_words)
            }

            for pl_file in [os.path.join(pl_tl_data[scenario], f"train.{pl}.txt"),
                            os.path.join(pl_tl_data[scenario], f"val.{pl}.txt"),
                            os.path.join(pl_tl_data[scenario], f"test.{pl}.txt")]:
                reshape.reshape_data(pl_file,
                                     word_mappings=mappings,
                                     output_tag=output_tag,
                                     long_enough=config["oc_min_word_len_applied"])


def _get_val_method(methods):
    _validate_cognate_methods(methods)

    if "charlotte" in methods:
        val_method = "charlotte"
    elif "web" in methods:
        val_method = "web"
    else:
        assert "fuzz" in methods
        val_method = "fuzz"
    return val_method

@_validate_lang_filters
def _run_all_cognates_methods(config, tl_to_pl_tags:dict={}, lang_filters=None):
    # Get cognates
    oc_data = {}
    for cognate_method in config["methods"]:
        if cognate_method == "web":
            pl_cl_cognates = _run_cognate_method(config, 
                                                 cognate_method=cognate_method, 
                                                 tl_to_pl_tags=tl_to_pl_tags, 
                                                 lang_filters=lang_filters)
        else:
            pl_cl_cognates = _run_cognate_method(config, 
                                                 cognate_method=cognate_method, 
                                                 lang_filters=lang_filters)
        assert cognate_method not in oc_data
        oc_data[cognate_method] = pl_cl_cognates
    
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # NORMAL VAL - TRAIN SPLIT FOR EACH COGNATE METHOD
    for cognate_method in oc_data:
        for (pl, cl), pl_cl_cognate_data in oc_data[cognate_method].items():
            train_split, val_split = get_train_val_split(
                pairs=pl_cl_cognate_data["cognates"],
                theta=config["theta"],
                size=config["oc_val_size"],
                n_buckets=config["oc_val_nld_buckets"],
                max_fraction=config["oc_val_max_bucket_fraction"],
                seed=config["seed"]
            )

            _assert_no_train_contamination(train_split, val_split)
            data_dir = pl_cl_cognate_data["subdirs"]["data"]
            train_file = os.path.join(data_dir, "train.txt")
            val_file = os.path.join(data_dir, "val.txt")

            assert not os.path.exists(train_file)
            assert not os.path.exists(val_file)

            write_oc_data(train_split, train_file)
            write_oc_data(val_split, val_file)

    # VAL - TRAIN SPLIT USING COMMON VALIDATION SET BETWEEN METHODS
    # val_method = _get_val_method(config["methods"])
    # # Get pl-cl validation sets with chosen cognate method
    # val_method_train_splits_by_pair = {}
    # val_files_by_pair = {}
    # for (pl, cl), pl_cl_cognate_data in oc_data[val_method].items():
    #     val_method_train_split, validation_set = get_train_val_split(
    #         pairs=pl_cl_cognate_data["cognates"],
    #         theta=config["theta"],
    #         size=config["oc_val_size"],
    #         n_buckets=config["oc_val_nld_buckets"],
    #         max_fraction=config["oc_val_max_bucket_fraction"],
    #         seed=config["seed"]
    #     )

    #     val_dir = os.path.join(OC_dir, "validation_sets", f"{pl}-{cl}")
    #     assert not os.path.exists(val_dir)
    #     os.makedirs(val_dir)
    #     val_file = os.path.join(val_dir, "val.txt")
    #     write_oc_data(validation_set, val_file)

    #     assert (pl, cl) not in val_files_by_pair
    #     val_files_by_pair[(pl, cl)] = val_file

    #     assert (pl, cl) not in val_method_train_splits_by_pair
    #     val_method_train_splits_by_pair[(pl, cl)] = val_method_train_split
    
    # # Get train splits
    # for cognate_method in oc_data:
    #     for (pl, cl), pl_cl_cognate_data in oc_data[cognate_method].items():
    #         cognates = pl_cl_cognate_data["cognates"]
    #         validation_set = read_oc_data(val_files_by_pair[(pl, cl)])
    #         train_set = get_train_split(cognates, validation_set, seed=config["seed"])
    #         if cognate_method == val_method:
    #             assert sorted(train_set) == sorted(val_method_train_splits_by_pair[(pl, cl)])
    #         _assert_no_train_contamination(train_set, validation_set)

    #         data_dir = pl_cl_cognate_data["subdirs"]["data"]
    #         train_file = os.path.join(data_dir, "train.txt")
    #         assert not os.path.exists(train_file)
    #         write_oc_data(train_set, train_file)

@_validate_lang_filters
def _run_cognate_method(
    config:dict, 
    cognate_method:str, 
    # only_pl_cls:set=None, 
    tl_to_pl_tags:dict=None,
    lang_filters=None,
):
    # validate config, cognate_method, and only_pl_cls
    if not isinstance(config, dict) or "data" not in config.keys():
        raise ValueError("config must be dictionary with a 'data' key!")
    if cognate_method not in ["charlotte", "web", "fuzz"]:
        raise ValueError("Cognate method must be 'charlotte', 'web', or 'fuzz'.")
    # if not _validate_only_pl_cls(only_pl_cls):
    #     raise ValueError("only_pl_cls must be a list of language pair tuples (pl, cl) or None!")
    
    to_remove = set()
    if lang_filters:
        for (f_pl, f_cl, f_tl) in lang_filters:
            to_remove.add((f_pl, f_cl))

    # Create directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    cognate_method_dir = os.path.join(OC_dir, cognate_method)
    assert not os.path.exists(cognate_method_dir)
    os.mkdir(cognate_method_dir)

    # Get all pl, cl pairs, filter by lang_filters if needed
    pl_cl_pairs = get_pl_cl_pairs(config["data"]) # [(pl, cl)]
    for f_pair in to_remove:
        pl_cl_pairs.remove(f_pair)
        assert f_pair not in pl_cl_pairs

    # validate tl_to_pl_tags
    if cognate_method == "web":
        if not isinstance(tl_to_pl_tags, dict):
            raise ValueError("tl_to_pl_tags must be a dictionary!")
        for (x_pl, x_cl, x_tl) in tl_to_pl_tags.keys():
            if (x_pl, x_cl) not in pl_cl_pairs:
                raise ValueError(f"tl_to_pl_tags has irrelevant scenario {(x_pl, x_cl, x_tl)} where {(x_pl, x_cl)} is not in pl_cl_pairs!")
        
        scens = list(tl_to_pl_tags.keys())
        for (x_pl, x_cl) in pl_cl_pairs:
            cts = 0
            for scen in scens:
                if scen[:2] == (x_pl, x_cl):
                    cts += 1
            if cts != 1:
                raise ValueError(f"PL/CL Pair {(x_pl, x_cl)} should only occur once in tl_to_pl_tags, but occured {cts} times!")
    else:
        if tl_to_pl_tags != None:
            raise ValueError("Cannot pass tl_to_pl_tags when cognate_method is not 'web'!")

    # Get pl, cl data files from which cognates will be extracted
    if cognate_method == "charlotte":
        pl_cl_files = read_pl_cl_paths(config["data"]) # returns {(pl, cl): (pl_path, cl_path)}
    elif cognate_method == "web":
        pl_cl_files = read_pl_cl_web_paths(config["data"], tl_to_pl_tags)
    else:
        pl_cl_files = read_pl_cl_fuzz_paths(config["data"]) # returns {(pl, cl): (pl_path, cl_path)}
    
    # filter pl_cl_files based on lang_filters, if needed
    for r_pair in to_remove:
        pl_cl_files.pop(r_pair)
            
    # Assert cognate method applicability
    if cognate_method in ["web", "fuzz"]:
        # if web or fuzz, then the cognate method should be applicable to all pl, cl pairs
        assert sorted(pl_cl_files.keys()) == sorted(pl_cl_pairs)
    else:
        # if charlotte, then it should be applicable to at least a subset of the pl, cl pairs
        assert len(pl_cl_files.keys()) <= len(pl_cl_pairs)
        assert set(pl_cl_files.keys()).difference( set(pl_cl_pairs) ) == set()
    

    # # I don't think we need this
    # # if only_pl_cls is passed, filter the pl_cl_files down to only the pairs included in only_pl_cls
    # # this means we are only running the cognate_method on these pairs
    # if only_pl_cls is not None:
    #     print(f"FILTERING PL,CL FILES DOWN TO THOSE IN only_pl_cls:\n\t{only_pl_cls}")
    #     pl_cl_files = {k: v for k, v in pl_cl_files.items() if k in only_pl_cls}
    # print(f"FINAL PL, CL FILES:\n\t{pl_cl_files}")

    # Get the appropriate extract_candidates function, per the set cognate_method
    if cognate_method in ["charlotte", "web"]:
        extract_candidates = extract_candidates_from_parallel
    else:
        extract_candidates = functools.partial(extract_fuzzy_candidates, top_k=config.get("fuzz_top_k"))
    
    # Get cognates
    pl_cl_cognates = {}
    for (pl, cl), (pl_path, cl_path) in pl_cl_files.items():
        pl_cl_train_dir, pl_cl_subdirs = get_train_dir(task_dir=cognate_method_dir, name=f"{pl}-{cl}")
        out_stem = os.path.join(pl_cl_subdirs["data"], "oc")
        cognates = make_cognates(
            src_path=pl_path,
            tgt_path=cl_path,
            src_lang=pl,
            tgt_lang=cl,
            out_stem=out_stem,
            long_enough=config["oc_min_word_len"],
            theta=config["theta"],
            extract_candidates=extract_candidates
        )
        assert (pl, cl) not in pl_cl_cognates
        pl_cl_cognates[(pl, cl)] = {
            "train_dir": pl_cl_train_dir,
            "subdirs": pl_cl_subdirs,
            "cognates": cognates
        }
    
    return pl_cl_cognates

def _assert_no_train_contamination(train_set, val_set):
    assert isinstance(train_set, list)
    assert isinstance(val_set, list)

    train_word_pairs = _get_word_pairs(train_set)
    val_word_pairs = _get_word_pairs(val_set)

    assert len(train_set) == len(train_word_pairs)
    assert len(val_set) == len(val_word_pairs)

    assert train_word_pairs.intersection(val_word_pairs) == set()

def _get_word_pairs(dataset):
    return set(
        (pair[-3], pair[-2])
        for pair in dataset
    )

def _validate_only_pl_cls(only_pl_cls):
    if only_pl_cls == None:
        return True
    if not isinstance(only_pl_cls, list):
        return False
    for item in only_pl_cls:
        if not isinstance(item, tuple):
            return False
        if len(item) != 2:
            return False
        for elem in item:
            if not isinstance(elem, str):
                return False
    return True


@_validate_lang_filters
def tl_to_pl_translation(config, do_translation=True, lang_filters=None):
    results = {}
    for data_folder, pl, cl, tl in list(config["data"]):
        if lang_filters and (pl, cl, tl) not in lang_filters:
            continue
        # create config
        tl_pl_config = NMT_train_jobs._get_nmt_config(config,
                                                      model_type="parent",
                                                      reverse=True) # train on tl --> pl, not pl --> tl
        tl_pl_config["data"] = [[data_folder, pl, cl, tl]] # only train the relevant bilingual model
        
        # get data
        pl_cl_parent_child_paths = read_pl_cl_parent_child_paths(tl_pl_config["data"])
        assert len(pl_cl_parent_child_paths) == 1
        parent_data, child_data, pc_tl  = pl_cl_parent_child_paths[(pl, cl)]
        assert pc_tl == tl
        child_target_lines_path = os.path.join(child_data, f"train.{tl}.txt")

        if do_translation:
            # train and eval
            #TODO check to see if we created this model already when training baselines
            #If so, don't need to again :)
            tl_pl_jobs = NMT_train_jobs.train_and_eval(
                tl_pl_config,
                fine_tune=False,
                on_hpc=tl_pl_config["use_hpc"]
            )

            # get eval job id
            if tl_pl_config["use_hpc"]:
                eval_job_id = tl_pl_jobs["eval"][0].job_id
            else:
                eval_job_id = None

            # inference (after evaluation is done)
            tl_pl_jobs.update(NMT_train_jobs.infer(
                tl_pl_config,
                inference_file=child_target_lines_path,
                src_lang=tl,
                tgt_lang=pl,
                fine_tune=False,
                on_hpc=tl_pl_config["use_hpc"],
                afterok=eval_job_id
            ))
        else:
            oc_tag = "OC_" if tl_pl_config["sc_model_ids"] != None else ""
            nmt_config_key = _nmt_config_key(tl_pl_config, fine_tune=False)
            reverse_tag = "_reverse" if tl_pl_config["nmt_reverse"] else ""
            inf_job_name = "PREVIOUS_INFER|" + tl_pl_config["experiment_name"] + f"|{oc_tag}NMT_{nmt_config_key}{reverse_tag}"
            output_tag = f".{tl}-->{pl}." + tl_pl_config["nmt_model_id"]
            output_file = child_target_lines_path + output_tag
            if not os.path.exists(output_file):
                raise ValueError(f"Looking for previous inference, but the file does not exist: `{output_file}`")
            tl_pl_jobs = {"infer": (PREV_JOB, inf_job_name, output_file, output_tag)}

        scenario = pl, cl, tl

        # get results
        assert scenario not in results
        results[scenario] = tl_pl_config, tl_pl_jobs
    return results

def _get_tl_to_pl_tags(tl_pl_results, use_hpc=False, previous_inference=False):
    tl_to_pl_tags = {}
    for scenario, (tl_pl_config, tl_pl_jobs) in tl_pl_results.items():
        if (use_hpc == False) or (previous_inference == True):
            infer_job_type, inf_job_name, output_file, output_tag = tl_pl_jobs["infer"]
        else:
            assert use_hpc == True
            assert previous_inference == False
            output_file, output_tag = tl_pl_jobs["infer"].result()
        assert scenario not in tl_to_pl_tags
        tl_to_pl_tags[scenario] = output_tag
    return tl_to_pl_tags

def _compile_nmt_results(experiment_directory, oc_methods, use_metric=f"chrF++", get_metrics=["chrF++", "spBLEU", "BLEU"]):
    NMT_dir = os.path.join(experiment_directory, "NMT")
    all_scores = {}
    for d in os.listdir(NMT_dir):
        d_path = os.path.join(NMT_dir, d)
        assert os.path.isdir(d_path)
        if d == "tokenizers":
            continue

        scores_f = os.path.join(d_path, "predictions", "scores.json")
        scores = read_json(scores_f)[f"BEST_VAL_{use_metric}"]

        assert d not in all_scores.keys()
        all_scores[d] = [scores[f"TEST_{metric}"] for metric in get_metrics]
    
    scores_out = os.path.join(experiment_directory, "NMT_scores.txt")
    with open(scores_out, "w") as outf:
        header = "MODEL\t\t\t\t\t|" + " | ".join(get_metrics)
        underline = "-" * len(header)
        outf.write(f"{header}\n{underline}\n")
        _write_NMT_line(outf, all_scores, "NMT_simple", "Simple baselines")
        _write_NMT_line(outf, all_scores, "NMT_parent", "Transfer baselines - parent")
        _write_NMT_line(outf, all_scores, "NMT_child", "Transfer baselines - child")
        for method in oc_methods:
            _write_NMT_line(outf, all_scores, f"{method}_NMT_parent", f"{method.capitalize()} - parent")
            _write_NMT_line(outf, all_scores, f"{method}_NMT_child", f"{method.capitalize()} - child")

def _write_NMT_line(outf, scores, model_name_prefix, title):
    outf.write(title + "\n")
    for model, model_scores in {m: k for m, k in scores.items() if m.startswith(model_name_prefix)}.items():
        outf.write(" | ".join([model] + model_scores) + "\n")

@log_parsed_args
def get_args():
    PIPELINE = ['baselines', 'TL-->PL', 'prepare_OC','OC', "OC_reshape", 'OC_NMT']
    NMT_MODELS = ['parent', 'child', 'simple']
    METHODS = ['charlotte', 'web', 'fuzz']
    NMT_DIRECTIONS = [TO_TL, FROM_SL]
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--pipeline", nargs='+', default=PIPELINE, choices=PIPELINE)
    parser.add_argument("-N", "--nmt_models", nargs="+", default=NMT_MODELS, choices=NMT_MODELS)
    parser.add_argument("-m", "--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("-d", "--directions", nargs="+", default=NMT_DIRECTIONS, choices=NMT_DIRECTIONS)
    parser.add_argument("-i", "--include_pairs", nargs='+', default=None, help="If None, applied to all pl, cl, tl pairs. Otherwise pass a list of pairs in the format 'pl,cl,tl' (only relevant to multilingual scenarios).")
    args = parser.parse_args()
    if args.include_pairs:
        if len(args.include_pairs) != len(set(args.include_pairs)):
            raise ValueError("--include_pairs must be list of unique language pairs, formatted pl,cl,tl")
        args.include_pairs = [tuple(item.split(',')) for item in args.include_pairs]

    if len(args.pipeline) != len(set(args.pipeline)):
        raise ValueError(f"--pipeline must only contain a list of unique pipeline steps: {PIPELINE}")
    # Make sure that the steps included in args.pipeline follow a logical order
    # This means that args.pipeline must match some slice of PIPELINE
    start_idx = PIPELINE.index(args.pipeline[0])
    if args.pipeline != PIPELINE[start_idx: start_idx + len(args.pipeline)]:
        raise ValueError(f"--pipeline must follow a logical order: {PIPELINE}")
    
    if len(args.nmt_models) != len(set(args.nmt_models)):
        raise ValueError(f"--nmt_models must only contain a list of unique nmt_models to train: {NMT_MODELS}")
    
    if len(args.methods) != len(set(args.methods)):
        raise ValueError(f"--methods must only contain a list of unique methods: {METHODS}")
    
    if len(args.directions) != len(set(args.directions)):
        raise ValueError(f"--directions must only contain a list of unique NMT directions: {NMT_DIRECTIONS}")

    return args

if __name__ == "__main__":
    log_script("Pipeline.Pipeline", __file__)
    args = get_args()
    main(
        config_f=args.config,
        pipeline=args.pipeline,
        nmt_models=args.nmt_models,
        apply_methods=args.methods,
        lang_filters=args.include_pairs,
        nmt_directions=args.directions
    )
    