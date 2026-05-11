import argparse
import os
from copy import deepcopy
import submitit
from functools import partial
from sloth_hatch.sloth import read_yaml, create_directory, log_parsed_args, log_script, log_function_call

from NMT.train import train_jobs
import utilities
from utilities.read_data import read_config, get_pl_cl_pairs, read_pl_cl_paths, read_pl_cl_web_paths, read_pl_cl_fuzz_paths

def main(config_f):
    config = utilities.read_data.read_config(config_f, add_sc_model_ids=True)
    val_method = _get_val_method(config["methods"])

def _get_val_method(methods):
    if not (0 > len(methods) < 4):
        raise ValueError("Must specify 1-3 cognate methods: 'charlotte', 'web', and/or 'fuzz'.")
    for method in methods:
        if method not in ["charlotte", "web", "fuzz"]:
            raise ValueError(f"Cognate methods must only include 'charlotte', 'web', and 'fuzz'.")
    
    if "charlotte" in methods:
        val_method = "charlotte"
    elif "web" in methods:
        val_method = "web"
    else:
        assert "fuzz" in methods
        val_method = "fuzz"
    return val_method
    
def _get_validation_set(config, val_method):
    pass

def _run_cognate_method(config:dict, cognate_method:str, only_pl_cls:set=None):
    # Get all pl, cl pairs
    pl_cl_pairs = get_pl_cl_pairs(config["data"]) # [(pl, cl)]

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
        # if web or fuzz, then the cognate method is applicable to all pl, cl pairs
        assert sorted(pl_cl_files.keys()) == sorted(pl_cl_pairs)
    else:
        # if charlotte, then it should be applicable to a subset of the pl, cl pairs
        assert len(pl_cl_files.keys()) <= len(pl_cl_pairs)
        assert set(pl_cl_files.keys()).difference( set(pl_cl_pairs) ) == set()
    

    # if only_pl_cls is passed, filter the pl_cl_files down to only the pairs included in only_pl_cls
    # this means we are only running the cognate_method on these pairs
    if only_pl_cls is not None:
        print(f"FILTERING PL,CL FILES DOWN TO THOSE IN only_pl_cls:\n\t{only_pl_cls}")
        pl_cl_files = {k: v for k, v in pl_cl_files.items() if k in only_pl_cls}
    print(f"FINAL PL, CL FILES:\n\t{pl_cl_files}")


    # Get the appropriate extract_candidates function, per the set cognate_method
    # i.e. import the right extract candidates method. In the case of charlotte's web, we need to run the TL->PL translation

    


