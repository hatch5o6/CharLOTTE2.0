import argparse
import os
from sloth_hatch.sloth import log_parsed_args, log_script

from NMT.train.NMTTokenizer import make_tokenizer_data, train_unigram
import utilities
from utilities.experiment_file_system import get_exp_dir, get_task_dir

def train_tokenizer(config, train_with_oc=False):
    # file structure
    exp_dir = get_exp_dir(config)
    NMT_dir = get_task_dir(exp_dir, task="NMT")
    tokenizers_dir = os.path.join(NMT_dir, "tokenizers")

    tag = "oc" if train_with_oc else "std"
    langs = ""
    for _, pl, cl, tl in config["data"]:
        langs += f"|{pl}-{cl}_{tl}"
    langs += "|"
    tag += langs

    gt_tokenizer_dir = os.path.join(tokenizers_dir, f"{tag}tokenizer")
    if not os.path.exists(gt_tokenizer_dir):
        # data
        print("Assembling tokenizer training data")
        tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config, 
                                                                  tokenizer_tag=tag,
                                                                  get_oc_data=train_with_oc)
        print(tokenizer_dir)
        print(gt_tokenizer_dir)
        assert tokenizer_dir == gt_tokenizer_dir
        assert len(config["data"]) == 1
        assert len(tokenizer_train_data) == 1
        scenario = tuple(config["data"][0][1:])
        data_files = tokenizer_train_data[scenario]

        # train
        print("Training Tokenizer")
        unigram_tokenizer_dir = train_unigram(files=data_files,
                                              save=gt_tokenizer_dir,
                                              vocab_size=config["nmt_vocab_size"],
                                              seed=config["seed"])
    else:
        print("Tokenizer already exists.")
        unigram_tokenizer_dir = os.path.join(gt_tokenizer_dir, "UnigramTokenizer")
        print(f"\tUsing `{unigram_tokenizer_dir}`")
        assert os.path.exists(unigram_tokenizer_dir)

    return unigram_tokenizer_dir

@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c")
    return parser.parse_args()

if __name__ == "__main__":
    log_script("NMT.train", __file__)
    args = get_args()
    unigram_tokenizer_dir = train_tokenizer(args.config)
    print(f"Tokenizer saved to: `{unigram_tokenizer_dir}`")
