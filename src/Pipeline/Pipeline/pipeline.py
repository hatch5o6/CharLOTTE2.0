import argparse
import os
from copy import deepcopy
import submitit
import functools
from sloth_hatch.sloth import read_content, write_content, read_lines, read_yaml, create_directory, log_parsed_args, log_script, log_function_call

from NMT.train import train_jobs as NMT_train_jobs
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


def main(config_f):
    # temp config to get the OC directory
    config = utilities.read_data.read_config(config_f, 
                                             add_sc_model_ids=True)
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # Get the OC model name
    oc_model_name = _get_oc_model_name(OC_dir)

    # Get the config for real now
    config = utilities.read_data.read_config(config_f, 
                                             add_sc_model_ids=True, 
                                             oc_model_name=oc_model_name)
    




def _get_oc_model_name(OC_dir):
    oc_model_name_f = os.path.join(OC_dir, "OC_MODEL_NAME")
    if not os.path.exists(oc_model_name_f):
        oc_model_name = model_names.get_new_name()
        print(_wrap_in_pounds(f"Thy OC models shall bear the name \"{oc_model_name}\""))
        write_content(oc_model_name.strip(), oc_model_name_f)
    else:
        oc_model_name = read_content(oc_model_name_f).strip()
        print(_wrap_in_pounds(f"Thy OC models doth already bear the name \"{oc_model_name}\""))
    return oc_model_name

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
def tl_to_pl_translation(config, lang_filters=None):
    # If running "web", train and apply TL -> PL
    tl_pl_translation_results = {}
    tl_to_pl_tags = {}
    if "web" in config["methods"]:
        # Run TL -> PL train, eval, and inference
        tl_pl_translation_results = _tl_pl_translation(config, lang_filters=lang_filters)
        # Get tags on inference files. If on HPC, this will also wait until inference is complete.
        tl_to_pl_tags = _get_tl_to_pl_tags(tl_pl_translation_results, use_hpc=config["use_hpc"])
    return tl_to_pl_tags

def prepare_OC_data(config, tl_to_pl_tags):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    for cognate_method in config["methods"]:
        assert not os.path.exists( os.path.join(OC_dir, cognate_method) )
    val_dir = os.path.join(OC_dir, "validation_sets")
    assert not os.path.exists(val_dir)

    # Run cognate methods for each pair, including getting the common validation set
    # Write oc data for all languages -- don't use filters
    # TODO write a check to see if oc data is already written. If it is, don't need to redo it.
    _write_oc_data(config, tl_to_pl_tags)

    for cognate_method in config["methods"]:
        assert os.path.exists( os.path.join(OC_dir, cognate_method) )
    assert os.path.exists(val_dir)

@_validate_lang_filters
def OC(config, lang_filters=None):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    val_dir = os.path.join(OC_dir, "validation_sets")
    assert os.path.exists(val_dir)

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
            val_data = os.path.join(val_dir, f"{pl}-{cl}/val.txt")

            scenario = (pl, cl, tl)
            scen_oc_config = deepcopy(config)
            scen_oc_config["oc_train"] = train_data
            scen_oc_config["oc_val"] = val_data
            scen_oc_config["oc_scenario"] = scenario

            assert scenario not in oc_configs[cognate_method]
            oc_configs[cognate_method][scenario] = scen_oc_config
    
    # Kick off OC training, evaluation, and inference jobs
    pl_tl_data = read_pl_tl_data(config["data"])
    scen_jobs = {}
    for cognate_method, scenario_configs in oc_configs.items():
        for scen, scen_oc_config in scenario_configs.items():
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
                                         lang=pl,
                                         long_enough=config["oc_min_word_len_applied"],
                                         out_path=pl_words_out_path)

            jobs.update(OC_train_jobs.infer(config=scen_oc_config,
                                            cognate_method=cognate_method,
                                            source_words_f=pl_words_out_path,
                                            on_hpc=config["use_hpc"],
                                            afterok=eval_job_id))
            
            scen_jobs[scen] = {"jobs": jobs, "words_for_inference": pl_words_out_path}
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

            output_tag = "." + config["sc_model_ids"][scenario]

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
                                     lang=pl,
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

def _write_oc_data(config, tl_to_pl_tags:dict={}):
    val_method = _get_val_method(config["methods"])

    # Get cognates
    oc_data = {}
    for cognate_method in config["methods"]:
        if cognate_method == "web":
            pl_cl_cognates = _run_cognate_method(config, cognate_method=cognate_method, tl_to_pl_tags=tl_to_pl_tags)
        else:
            pl_cl_cognates = _run_cognate_method(config, cognate_method=cognate_method)
        assert cognate_method not in oc_data
        oc_data[cognate_method] = pl_cl_cognates
    
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    # Get pl-cl validation sets with chosen cognate method
    val_method_train_splits_by_pair = {}
    val_files_by_pair = {}
    for (pl, cl), pl_cl_cognate_data in oc_data[val_method].items():
        val_method_train_split, validation_set = get_train_val_split(
            pairs=pl_cl_cognate_data["cognates"],
            theta=config["theta"],
            size=config["oc_val_size"],
            n_buckets=config["oc_val_nld_buckets"],
            max_fraction=config["oc_val_max_bucket_fraction"],
            seed=config["seed"]
        )

        val_dir = os.path.join(OC_dir, "validation_sets", f"{pl}-{cl}")
        assert not os.path.exists(val_dir)
        os.makedirs(val_dir)
        val_file = os.path.join(val_dir, "val.txt")
        write_oc_data(validation_set, val_file)

        assert (pl, cl) not in val_files_by_pair
        val_files_by_pair[(pl, cl)] = val_file

        assert (pl, cl) not in val_method_train_splits_by_pair
        val_method_train_splits_by_pair[(pl, cl)] = val_method_train_split
    
    # Get train splits
    for cognate_method in oc_data:
        for (pl, cl), pl_cl_cognate_data in oc_data[cognate_method].items():
            cognates = pl_cl_cognate_data["cognates"]
            validation_set = read_oc_data(val_files_by_pair[(pl, cl)])
            train_set = get_train_split(cognates, validation_set, seed=config["seed"])
            if cognate_method == val_method:
                assert sorted(train_set) == sorted(val_method_train_splits_by_pair[(pl, cl)])
            _assert_no_train_contamination(train_set, validation_set)

            data_dir = pl_cl_cognate_data["subdirs"]["data"]
            train_file = os.path.join(data_dir, "train.txt")
            assert not os.path.exists(train_file)
            write_oc_data(train_set, train_file)


def _run_cognate_method(
    config:dict, 
    cognate_method:str, 
    # only_pl_cls:set=None, 
    tl_to_pl_tags:dict=None
):
    # validate config, cognate_method, and only_pl_cls
    if not isinstance(config, dict) or "data" not in config.keys():
        raise ValueError("config must be dictionary with a 'data' key!")
    if cognate_method not in ["charlotte", "web", "fuzz"]:
        raise ValueError("Cognate method must be 'charlotte', 'web', or 'fuzz'.")
    # if not _validate_only_pl_cls(only_pl_cls):
    #     raise ValueError("only_pl_cls must be a list of language pair tuples (pl, cl) or None!")
    
    # Create directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    cognate_method_dir = os.path.join(OC_dir, cognate_method)
    assert not os.path.exists(cognate_method_dir)
    os.mkdir(cognate_method_dir)

    # Get all pl, cl pairs
    pl_cl_pairs = get_pl_cl_pairs(config["data"]) # [(pl, cl)]

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

def _tl_pl_translation(config, lang_filters=None):
    # pl_cl_parent_child_paths = read_pl_cl_parent_child_paths(config["data"])
    # for (pl, cl), (parent_data, child_data, tl) in pl_cl_parent_child_paths.items():
    results = {}
    for data_folder, pl, cl, tl in list(config["data"]):
        if lang_filters and (pl, cl, tl) not in lang_filters:
            continue
        # create config
        tl_pl_config = deepcopy(config)
        tl_pl_config["nmt_corpus"] = "parent"
        tl_pl_config["nmt_reverse"] = True # translate tl -> pl (not pl -> tl)
        tl_pl_config["sc_model_ids"] = None # don't train on oc-reshaped data
        tl_pl_config["data"] = [[data_folder, pl, cl, tl]] # only train the relevant bilingual model
        tl_pl_config["nmt_model_id"] = model_names.get_new_name()
        
        # get data
        pl_cl_parent_child_paths = read_pl_cl_parent_child_paths(tl_pl_config["data"])
        assert len(pl_cl_parent_child_paths) == 1
        parent_data, child_data, pc_tl  = pl_cl_parent_child_paths[(pl, cl)]
        assert pc_tl == tl
        child_target_lines_path = os.path.join(child_data, f"train.{tl}.txt")

        # train and eval
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

        scenario = pl, cl, tl

        # get results
        assert scenario not in results
        results[scenario] = tl_pl_config, tl_pl_jobs
    return results

def _get_tl_to_pl_tags(tl_pl_results, use_hpc):
    tl_to_pl_tags = {}
    for scenario, (tl_pl_config, tl_pl_jobs) in tl_pl_results.items():
        if use_hpc:
            output_file, output_tag = tl_pl_jobs["infer"].result()
        else:
            LOCAL_JOB, inf_job_name, output_file, output_tag = tl_pl_jobs["infer"]
        assert scenario not in tl_to_pl_tags
        tl_to_pl_tags[scenario] = output_tag
    return tl_to_pl_tags
