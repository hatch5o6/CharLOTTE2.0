import argparse
import yaml
import os
import json
import functools
import lightning as L
import shutil
import torch
import subprocess
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from CharTokenizer import CharTokenizer
from OCLightning import OCLightning, OCDataModule

from metrics import calc_charBLEU, calc_chrF

torch.set_float32_matmul_precision('medium')

def log_mode_call(f):
    @functools.wraps(f)
    def wrapper(*args):
        config = args[0]
        rank_zero_info(f"\n---------------- Calling {f.__name__} ----------------")
        rank_zero_info("CONFIG:")
        for k, v in config.items():
            rank_zero_info(f"\t{k}=`{v}`")
        rank_zero_info("\n\n")
        result = f(*args)
        rank_zero_info(f"\n---------------- Ending {f.__name__} ----------------")
        return result
    return wrapper

def call_seed_everything(f):
    @functools.wraps(f)
    def wrapper(config):
        seed = config["seed"]
        rank_zero_info(f"Seeding everything with seed={seed}")
        L.seed_everything(seed, workers=True)
        result = f(config)
        return result
    return wrapper

def call_nvidia_smi(f):
    @functools.wraps(f)
    def wrapper(*args):
        subprocess_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(subprocess_result.stdout)
        result = f(*args)
        return result
    return wrapper

def get_save_dir(config):
    save_dir = os.path.join(config["save"], config["experiment_name"], "OC")
    validate_dir(save_dir)
    return save_dir

def validate_dir(d):
    if not os.path.exists(d):
        raise FileExistsError(f"Directory `{d}` does not exist!")

def get_save_subdirs(d):
    checkpoints_d = os.path.join(d, "checkpoints")
    data_d = os.path.join(d, "data")
    preds_d = os.path.join(d, "predictions")
    logs_d = os.path.join(d, "logs")
    tb_d = os.path.join(d, "tb")
    dirs = [checkpoints_d, data_d, preds_d, logs_d, tb_d]
    for d in dirs:
        validate_dir(d)
    return dirs

def read_file(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def get_tokenizers(f):
    src_tokenizer = CharTokenizer()
    tgt_tokenizer = CharTokenizer()

    data = read_file(f)
    src_data = ""
    tgt_data = ""
    for line in data:
        freq, src_word, tgt_word, nld = line.split(" ||| ")
        src_data += src_word.strip()
        tgt_data += tgt_word.strip()
    
    src_tokenizer.build_vocab(src_data)
    tgt_tokenizer.build_vocab(tgt_data)

    return src_tokenizer, tgt_tokenizer

def get_datamodule(config, src_tokenizer, tgt_tokenizer):
    dm = OCDataModule(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        train_f=config["oc_train"],
        val_f=config["oc_val"],
        batch_size=config["oc_batch_size"],
        max_length=config["oc_max_length"]
    )
    dm.setup()
    return dm

@log_mode_call
@call_seed_everything
@call_nvidia_smi
def train_model(config):
    save = get_save_dir(config)
    checkpoints_d, data_d, preds_d, logs_d, tb_d = get_save_subdirs(save)

    # tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config["oc_train"])
    
    # data
    dm = get_datamodule(config, src_tokenizer, tgt_tokenizer)

    # model
    model = OCLightning(
        config=config,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )

    # callbacks and loggers
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=config["oc_patience"], 
        verbose=True
    )
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_d,
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_top_k=config["oc_save_top_k"],
        monitor="val_loss",
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print_callback = PrintCallback()
    train_callbacks = [
        early_stopping,
        top_k_model_checkpoint,
        lr_monitor,
        print_callback
    ]
    logger = CSVLogger(save_dir=logs_d)
    tb_logger = TensorBoardLogger(save_dir=tb_d)

    # Trainer
    if config["oc_n_gpus"] >= 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = L.Trainer(
        max_steps=config["oc_max_steps"],
        val_check_interval=config["oc_val_interval"],
        accelerator=config["oc_device"],
        default_root_dir=save,
        callbacks=train_callbacks,
        logger=[logger, tb_logger],
        deterministic=True,
        strategy=strategy,
        gradient_clip_val=config["oc_gradient_clip_val"]
    )

    # train
    trainer.fit(model, dm)

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("######          TRAINING STARTED         ######")
        rank_zero_info("###############################################")
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("#######          TRAINING ENDED         #######")
        rank_zero_info("###############################################")


@log_mode_call
@call_seed_everything
@call_nvidia_smi
def eval_models(config):
    # tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config["oc_train"])
    
    # data
    dm = get_datamodule(config, src_tokenizer, tgt_tokenizer)

    # dirs
    save = get_save_dir(config)
    checkpoints_d, data_d, preds_d, logs_d, tb_d = get_save_subdirs(save)
    
    # inference
    scores = {}
    chkpt_preds = {}
    source_segs = None
    target_segs = None
    for chkpt_file in os.listdir(checkpoints_d):
        chkpt_file = os.path.join(checkpoints_d, chkpt_file)
        chkpt_source_segs = []
        chkpt_target_segs = []
        pred_segs = []
        inference_outputs = run_inference(chkpt_file, config, src_tokenizer, tgt_tokenizer, dm.val_dataloader())
        for source, target, pred in inference_outputs:
            chkpt_source_segs.append(source)
            chkpt_target_segs.append(target)
            pred_segs.append(pred)

        if source_segs == None:
            assert target_segs == None
            source_segs = chkpt_source_segs
            target_segs = chkpt_target_segs
        else:
            assert chkpt_source_segs == source_segs
            assert chkpt_target_segs == target_segs
        assert target_segs != None
        
        assert chkpt_file not in scores
        scores[chkpt_file] = {
            "chrF": calc_chrF(pred_segs, target_segs),
            "charBLEU": calc_charBLEU(pred_segs, target_segs)
        }
        assert chkpt_file not in chkpt_preds
        chkpt_preds[chkpt_file] = pred_segs
    get_best_scores(scores)

    # write
    write_preds(chkpt_preds, preds_d)
    write_scores(scores, preds_d)

def write_scores(scores, d):
    scores_f = os.path.join(d, "scores.json")
    with open(scores_f, "w") as outf:
        outf.write(json.dumps(scores, indent=2))

def write_preds(chkpt_preds, preds_d):
    for chkpt, preds in chkpt_preds.items():
        chkpt_name = chkpt.split("/")[-1]
        pred_dir = os.path.join(preds_d, chkpt_name)
        assert not os.path.exists(pred_dir)
        os.mkdir(pred_dir)
        pred_file = os.path.join(pred_dir, "validation.preds.txt")
        with open(pred_file, "w") as outf:
            outf.write("\n".join(preds) + "\n")

def get_best_scores(scores, use_metric="chrF"):
    assert use_metric in ["chrF", "charBLEU"]
    scores_list = [
        (scores[chkpt][use_metric], chkpt)
        for chkpt in scores
    ]
    best_score, best_checkpoint = max(scores_list)
    best_key = f"BEST_VAL_{use_metric}".upper()
    assert best_key not in scores
    scores[best_key] = {
        "checkpoint": best_checkpoint,
        "chrF": scores[best_checkpoint]["chrF"],
        "charBLEU": scores[best_checkpoint]["charBLEU"]
    }
    assert scores[best_key][use_metric] == scores[best_checkpoint][use_metric] == best_score
    

@log_mode_call
@call_nvidia_smi
def inference(config, chkpt_file, source_words_f, hyp_words_out):
    # tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config["oc_train"])
    
    # data
    dm = OCDataModule(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        train_f=source_words_f,
        val_f=source_words_f,
        batch_size=config["oc_batch_size"],
        max_length=config["oc_max_length"]
    )
    dm.setup()

    # inference
    inference_outputs = run_inference(chkpt_file, config, src_tokenizer, tgt_tokenizer, dm.val_dataloader())
    hyps = [pred for _, _, pred in inference_outputs]
    
    # write
    write_lines(hyps, hyp_words_out)

def write_lines(lines, out_f):
    with open(out_f, "w") as outf:
        outf.write("\n".join(lines) + "\n")

def run_inference(chkpt_file, config, src_tokenizer, tgt_tokenizer, dataloader):
    model = OCLightning.load_from_checkpoint(
        checkpoint_path=chkpt_file,
        config=config,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )
    model.eval()

    trainer = L.Trainer(accelerator=config["oc_device"])
    prediction_batches = trainer.predict(model, dataloader)

    predictions = []
    for batch in prediction_batches:
        predictions += batch
    
    return predictions


def read_yaml(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    config["oc_warmup_steps"] = config["oc_max_steps"] // 20
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices=["TRAIN", "EVAL", "INFERENCE"], default="TRAIN")
    parser.add_argument("-k", "--chkpt_file", help="INFERENCE with this checkpoint")
    parser.add_argument("-w", "--source_words", help="source words for INFERENCE")
    parser.add_argument("-o", "--hyp_words_out", help="INFERENCE hyp words out file")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    return args

if __name__ == "__main__":
    print("#########################")
    print("###### OC/train.py ######")
    print("#########################")
    args = get_args()
    config = read_yaml(args.config)
    f = {
        "TRAIN": train_model,
        "EVAL": eval_models,
        "INFERENCE": inference
    }[args.mode]
    f_args = [config]
    if args.mode  == "INFERENCE":
        f_args += [args.chkpt_file, args.source_words, args.hyp_words_out]
    f(*f_args)

