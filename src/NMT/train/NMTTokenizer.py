import argparse
import random
import os
from copy import copy
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import UnigramTrainer
from transformers import PreTrainedTokenizerFast
from sloth_hatch.sloth import read_lines, write_lines, write_json, log_function_call, log_parsed_args, log_script, log_function_call

import utilities
from utilities.read_data import read_tokenizer_train_paths

@log_function_call
def train_unigram(
    files:list,
    save:str,
    vocab_size:int=32000,
    bos:str="<bos>",
    eos:str="<eos>",
    pad:str="<pad>",
    unk:str="<unk>",
    lang_toks:list=[],
    seed=42
):
    # validate
    if not os.path.exists(save):
        raise FileNotFoundError(f"Directory {save} does not exist!")
    
    # seed (#TODO - this may not be doing anything and may need to delete)
    from transformers import set_seed
    import numpy as np
    import random
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Metaspace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad, unk, bos, eos] + lang_toks,
        unk_token=unk
    )
    tokenizer.train(files, trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=pad,
        unk_token=unk,
        bos_token=bos,
        eos_token=eos
    )
    save = os.path.join(save, "UnigramTokenizer")
    fast_tokenizer.save_pretrained(save)
    return save

def load_tokenizer(tokenizer_path):
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def assemble_multilingual_tokenizer_data(
    tokenizer_train_data:dict, # the dictionary returned by make_tokenizer_data()
    config:dict
):
    pass
    #TODO
    # When doing a multilingual NLP system and need to make multilingual training data from the
    # files created by make_tokenizer_data
    # Make data files by sampling from the provided files
    # return those new data file paths


valid_data_ratio_dict_error = """
data_ratios must be a dictionary like so:
{<lang>: {ratio: <ratio>, file: <original data file>}}
where <lang> is the language code, <ratio> is an int indicating 
the number of shares of data given, and <original data file> is 
the path to the source data.
""".strip()

@log_function_call
def make_tokenizer_data(config, tokenizer_tag="", get_oc_data=False):
    #TODO implement logic to get OC data for an OC tokenizer
    tokenizer_dir = os.path.join(
        config["save"], 
        config["experiment_name"],
        f"NMT/tokenizers/{tokenizer_tag}tokenizer"
    )
    assert not os.path.exists(tokenizer_dir), f"Tokenizer directory {tokenizer_dir} already exists."
    
    sc_model_id_prefix = config["sc_model_id_prefix"] if get_oc_data else None
    scenario_to_data_ratios = _make_data_ratios(config["data"], 
                                                config["nmt_tokenizer_ratios"], 
                                                sc_model_id_prefix=sc_model_id_prefix)
    assert not os.path.exists(tokenizer_dir)
    os.makedirs(tokenizer_dir)
    tok_data_dir = os.path.join(tokenizer_dir, "data")
    os.mkdir(tok_data_dir)

    tokenizer_train_data = {}
    for scenario, data_ratios in scenario_to_data_ratios.items():
        pl, cl, tl = scenario
        scen_tok_dir = os.path.join(tok_data_dir, f"{pl}_{cl}-->{tl}")
        assert not os.path.exists(scen_tok_dir)
        os.mkdir(scen_tok_dir)
        print("\n\n\n#################################################################")
        print(f"MAKING TOKENIZER TRAINING DATA FOR SCENARIO: {pl}/{cl}-->{tl}")
        out_files = _make_scenario_data(
            data_ratios=data_ratios,
            save=scen_tok_dir,
            training_data_size=config["nmt_tokenizer_training_size"],
            seed=config["seed"]
        )
        assert scenario not in tokenizer_train_data
        tokenizer_train_data[scenario] = out_files
    return tokenizer_train_data, tokenizer_dir

@log_function_call
def _make_scenario_data(
    data_ratios:dict, # {<lang>: {ratio: <ratio>, files: [<original data file>, ...]}}
    save:str,
    training_data_size:int=500000,
    seed:int=42
):
    print(f"Tokenizer langs: {sorted(data_ratios.keys())}")
    if not _validate_data_ratios(data_ratios):
        raise ValueError(valid_data_ratio_dict_error)
    
    if not os.path.exists(save):
        raise FileNotFoundError(f"Directory does not exist: {save}")
    
    random.seed(seed)

    share_size = _get_share_size(data_ratios, training_data_size)

    total_lines_written = 0
    out_files = []
    notes = []
    for lang, item in data_ratios.items():
        item = copy(item)
        print(f"\n---------------- {lang} ----------------")
        og_data = []
        for f in item["files"]:
            print("reading data from", f)
            og_data += read_lines(f)
        print("Deduping")
        og_data = _dedupe(og_data)
        random.shuffle(og_data)

        quota = item["ratio"] * share_size

        if len(og_data) >= quota:
            print(f"Down sampling {len(og_data)} lines\n\tfrom {item['files']}\n\tto {quota}")
            sampled_data = random.sample(og_data, quota)
        else:
            print(f"Upsampling {len(og_data)} lines\n\tfrom {item['files']}\n\tto {quota}")
            sampled_data = _upsample(og_data, quota)
        assert len(sampled_data) == quota
        
        random.shuffle(sampled_data)

        out_path = os.path.join(save, f"{lang}.txt")
        print(f"{lang}: Writing {len(sampled_data)} lines ({item['ratio']} shares) to\n\t{out_path}")
        write_lines(sampled_data, out_path)
        total_lines_written += len(sampled_data)

        out_files.append(out_path)
        item["share_size"] = share_size
        item["quota"] = quota
        notes.append(item)

    print(f"\nFINISHED. Wrote {total_lines_written} lines.\n\n\n")

    notes_path = os.path.join(save, "notes.json")
    write_json(notes, notes_path)

    return out_files

def _dedupe(data):
    print("BEFORE:", len(data))
    unique_lines = set()
    new_data = []
    for line in data:
        if line not in unique_lines:
            new_data.append(line)
            unique_lines.add(line)
    print("AFTER:", len(new_data))
    return new_data

def _make_data_ratios(data_folders, nmt_tokenizer_ratios, sc_model_id_prefix=None):
    scenario_to_data_ratios = {}
    train_paths = read_tokenizer_train_paths(data_folders, sc_model_id_prefix=sc_model_id_prefix)
    for scenario, paths in train_paths.items():
        pl, cl, tl = scenario
        data_ratios = {}
        for i, lang in enumerate((pl, cl, tl)):
            if i == 0:
                lang_type = "pl"
            elif i == 1:
                lang_type = "cl"
            elif i == 2:
                lang_type = "tl"
            else:
                assert False
            assert lang not in data_ratios
            data_ratios[lang] = {
                "ratio": nmt_tokenizer_ratios[lang_type],
                "files": paths[lang]
            }
        assert len(data_ratios) == 3
        assert scenario not in scenario_to_data_ratios
        scenario_to_data_ratios[scenario] = data_ratios
    return scenario_to_data_ratios

def _upsample(data, quota):
    assert quota >= 0, "quota must be >= 0!"
    assert len(data) > 0, "length of data must be > 0!"
    while len(data) < quota:
        data = data + data
    data = data[:quota]
    assert len(data) == quota
    return data

def _get_share_size(data_ratios, training_data_size):
    total_shares = 0
    for lang, item in data_ratios.items():
        total_shares += item["ratio"]
    return training_data_size // total_shares

def _validate_data_ratios(data_ratios):
    if not isinstance(data_ratios, dict):
        return False
    for lang, item in data_ratios.items():
        if not isinstance(lang, str):
            return False
        if not isinstance(item, dict):
            return False
        if not sorted(item.keys()) == ["files", "ratio"]:
            return False
        if not isinstance(item["files"], list):
            return False
        for thing in item["files"]:
            if not isinstance(thing, str):
                return False
        if not isinstance(item["ratio"], int):
            return False
    return True

@log_parsed_args
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c")
    return parser.parse_args()

if __name__ == "__main__":
    log_script("NMT.train", __file__)
    args = _get_args()
    config = utilities.read_data.read_config(args.config)
    # For now, just monolingual
    tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config)
    assert len(config["data"]) == 1
    assert len(tokenizer_train_data) == 1
    scenario = tuple(config["data"][0][1:])
    print(f"Tokenizer data for scenario {scenario}.")
    data_files = tokenizer_train_data[scenario]
    for f in data_files:
        print(f"\t-{f}")
    tokenizer_path = train_unigram(files=data_files,
                                   save=tokenizer_dir,
                                   vocab_size=config["nmt_vocab_size"],
                                   seed=config["seed"])
    print(f"Tokenizer saved to {tokenizer_path}.\nYou should update config 'tokenizer'.")
