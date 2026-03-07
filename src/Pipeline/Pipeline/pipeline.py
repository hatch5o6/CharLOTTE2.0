import argparse
import os
from sloth_hatch.sloth import read_yaml, create_directory

from Pipeline.utilities.read_data_csv import read_data_csv

TRAIN_DIRS = ["OC", "NMT"]

def main(config):
    pl_cl_mode = validate_config(config)
    config["experiment_name"] += "_" + pl_cl_mode
    scen_dir, train_dirs = make_scenario_directory(config["save"], config["experiment_name"])
    if pl_cl_mode == "mono":
        from OC.extract_cognates.CognatesFromMonolingual import extract_cognates
    else:
        from OC.extract_cognates.CognatesFromParallel import extract_cognates
    
    pl_cl_files = read_data_csv(config[f"pl_cl_{pl_cl_mode}"])
    cognate_lists = {}
    for (src_lang, tgt_lang), (src_path, tgt_path) in pl_cl_files.items():
        cognate_list_out = os.path.join(train_dirs["OC"]["data"], f"{src_lang}_{tgt_lang}.cognates.txt")
        
        lang_pair = src_lang, tgt_lang
        assert lang_pair not in cognate_lists
        cognate_lists[lang_pair] = extract_cognates(
            src_file=src_path,
            tgt_file=tgt_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            cognate_list_out=cognate_list_out,
            theta=config["theta"]
        )


def validate_config(config):
    if (config["pl_cl_mono"] and config["pl_cl_para"]) \
        or (not config["pl_cl_mono"] and not config["pl_cl_para"]):
        raise ValueError(f"Must specify either pl_cl_mono or pl_cl_para.")
    return "mono" if config["pl_cl_mono"] else "para"

def make_scenario_directory(save_d, experiment_name):
    if not os.path.exists(save_d):
        raise FileExistsError(f"Directory `{save_d}` does not exist!")
    scen_dir = os.path.join(save_d, experiment_name)
    create_directory(scen_dir)
    train_dirs = {}
    for name in TRAIN_DIRS:
        assert name not in train_dirs
        train_dirs[name] = make_train_dir(scen_dir, name)
    return scen_dir, train_dirs

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="/home/hatch5o6/CharLOTTE2.0/src/configs/test.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    print("#################################")
    print("# Pipeline/Pipeline/pipeline.py #")
    print("#################################")
    args = get_args()
    config = read_yaml(args.config)
    main(config)
