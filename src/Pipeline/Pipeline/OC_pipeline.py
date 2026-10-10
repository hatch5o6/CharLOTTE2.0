import argparse
import os
from copy import deepcopy
import functools
from sloth_hatch.sloth import read_content, write_content, write_json, read_lines, read_yaml, create_directory, log_parsed_args, log_script, log_function_call, read_json

from NMT.train import train_jobs as NMT_train_jobs
from NMT.train.train import _nmt_config_key
from OC.train import train_jobs as OC_train_jobs
# from OC.train.TrainValSplit import get_train_val_split, get_train_split
from OC.train.TrainValSplit import get_train_val_split
from OC.extract_cognates.CandidatesFromParallel import extract_candidates as extract_candidates_from_parallel
from OC.extract_cognates.FuzzyCandidates import extract_candidates as extract_fuzzy_candidates
from OC.extract_cognates.Cognates import make_cognates
from OC.reshape import reshape
from OC.utilities.utilities import write_oc_data, read_oc_data
import utilities
import utilities.model_names as model_names
from utilities.utilities import set_vars_in_path
from utilities.read_data import get_pl_cl_pairs, read_pl_cl_paths, read_pl_cl_web_paths, read_pl_cl_fuzz_paths, read_pl_cl_parent_child_paths, _validate_sets, _validate_cognate_methods, read_pl_tl_data
from utilities.experiment_file_system import get_exp_dir, get_task_dir, get_train_dir
from utilities import hpc


@log_function_call
def main(
    config_f,
    pipeline,
):

    config = read_yaml(config_f)
    warmup_divisor = 20
    config["oc_warmup_steps"] = config["oc_max_steps"] // warmup_divisor
    
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # Get the OC model name
    oc_model_name = _get_model_name(OC_dir, model_type="OC")

    config["oc_model_id"] = oc_model_name


    ####################################################################
    ######################## OC PIPELINE ########################
    ####################################################################
    pipeline_results = {}

    # ------------------------ prepare_OC ------------------------
    if "prepare_OC" in pipeline:
        # Prepare OC training data
        prepare_oc_data_folder = os.path.join(config["save"], config["experiment_name"], "OC/SLURM/prepare_OC_data")

        prepare_OC_data(config)

        # prepare_OC_data_results = hpc.abstract_function(
        #     function=lambda: prepare_OC_data(config), 
        #     job_name="prepare_OC_data_" + config["experiment_name"],
        #     output_folder=prepare_oc_data_folder,
        #     mail_user=config["email"],
        #     timeout=config["basic_timeout"],
        #     ntasks_per_node=1,
        #     nodes=1,
        #     mem_gb=config["basic_mem"],
        #     n_gpus=0,
        #     qos=config["basic_qos"],
        #     use_hpc=config["use_hpc"]
        # )
        # pipeline_results["prepare_OC"] = prepare_OC_data_results


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
        pipeline_results["OC"] = OC_results

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
        pipeline_results["OC_reshape"] = OC_reshape_results

    return pipeline_results, oc_model_name 


def _get_all_scen_OC_afterok(results):
    """Build slurm job dependency string for OC scenarios"""
    scen_afterok = []
    for cognate_method, scen_stuff in results.items():
        for scenario, scen_results in scen_stuff.items():
            scen_infer_job = scen_results["jobs"]["infer"]
            scen_afterok.append(str(scen_infer_job.id))
    scen_afterok = ":".join(scen_afterok)
    return scen_afterok

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


def get_cognates(config, OC_dir, data_folder, xx, yy):
    lang_pair = xx, yy
    directory = os.path.join(data_folder, f"{xx}-{yy}")

    xx_path = os.path.join(directory, f"train.{xx}.txt")
    yy_path = os.path.join(directory, f"train.{yy}.txt")

    os.makedirs(os.path.join(OC_dir, "cognates", f"{xx}_{yy}"), exist_ok=True)
    out_stem = os.path.join(OC_dir, f"cognates/{xx}_{yy}/{xx}_{yy}")
    
    cognates = make_cognates(
                src_path=xx_path,
                tgt_path=yy_path,
                src_lang=xx,
                tgt_lang=yy,
                out_stem=out_stem, # word list out
                long_enough=config["oc_min_word_len"],
                theta=config["theta"],
                extract_candidates=extract_candidates_from_parallel
            )

    return cognates

def fast_pivot(pl_tl_cognates, cl_tl_cognates):
    pass

def prepare_OC_data(config):
    # Directory structure
    exp_dir = set_vars_in_path(get_exp_dir(config))
    OC_dir = get_task_dir(exp_dir, task="OC")

    data_folder, pl, cl, tl = config["data"]
    data_folder = set_vars_in_path(data_folder)

    if config["fast_pivot"]:
        # get pl_tl_pairs and cl_tl_pairs
        pl_tl_cognates = get_cognates(config, OC_dir, data_folder, pl, tl)
        cl_tl_cognates = get_cognates(config, OC_dir, data_folder, cl, tl)

        # run pivot
        pl_cl_cognates = fast_pivot(pl_tl_cognates, cl_tl_cognates)


    elif not config["fast_pivot"]:
        pl_cl_cognates = get_cognates(config, OC_dir, data_folder, pl, cl)

        # print(cognates) # [(86, 'como', 'como', 0.0), (25, 'sobre', 'sobre', 0.0), (23, 'tres', 'tres', 0.0)]
        

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

            def json_safe(obj):
                """write_json breaks if there is a tuple key (['sc_model_ids']), but we can't change the config itself"""
                if isinstance(obj, dict):
                    return {("-".join(k) if isinstance(k, tuple) else k): json_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [json_safe(v) for v in obj]
                return obj
            
            scen_oc_config_filepath = os.path.join(sub_dirs["logs"], "scen_of_config.json")
            write_json(json_safe(scen_oc_config), scen_oc_config_filepath, indent=2, ensure_ascii=False)

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


def _run_all_cognates_methods(config, tl_to_pl_tags:dict={}, lang_filters=None):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # Get cognates
    oc_data = {}
    for cognate_method in config["methods"]:
        # don't overwrite data if it already exists
        if not os.path.exists(os.path.join(OC_dir, cognate_method)):
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


def _run_cognate_method(
    config:dict, 
    cognate_method:str, 
    # only_pl_cls:set=None, 
    tl_to_pl_tags:dict=None,
    lang_filters=None,
):

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


@log_parsed_args
def get_args():
    PIPELINE = ['prepare_OC', 'OC', "OC_reshape"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--pipeline", nargs='+', default=PIPELINE, choices=PIPELINE)
    args = parser.parse_args()

    if len(args.pipeline) != len(set(args.pipeline)):
        raise ValueError(f"--pipeline must only contain a list of unique pipeline steps: {PIPELINE}")
    
    # Make sure that the steps included in args.pipeline follow a logical order
    # This means that args.pipeline must match some slice of PIPELINE
    start_idx = PIPELINE.index(args.pipeline[0])
    if args.pipeline != PIPELINE[start_idx: start_idx + len(args.pipeline)]:
        raise ValueError(f"--pipeline must follow a logical order: {PIPELINE}")

    return args

if __name__ == "__main__":
    log_script("Pipeline.Pipeline", __file__)
    args = get_args()
    main(
        config_f=args.config,
        pipeline=args.pipeline,
    )