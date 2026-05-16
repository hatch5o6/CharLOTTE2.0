import argparse
import os
import json
import functools
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from sloth_hatch.sloth import read_yaml, read_lines, log_parsed_args, log_script

import utilities
from utilities.experiment_file_system import get_exp_dir, get_task_dir, get_train_dir
from OC.train.CharTokenizer import CharTokenizer
from OC.train.OCLightning import OCLightning, OCDataModule
from OC.utilities.utilities import read_oc_data
from utilities.metrics import calc_charBLEU, calc_chrF
from utilities.train_utilities import log_mode_call, call_nvidia_smi, call_seed_everything, PrintCallback

torch.set_float32_matmul_precision('medium')

def get_tokenizers(f):
    src_tokenizer = CharTokenizer()
    tgt_tokenizer = CharTokenizer()

    data = read_oc_data(f)
    src_data = ""
    tgt_data = ""
    for row in data:
        src_word, tgt_word = row[-3:-1]
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

def _get_save_dir(config, create=True):
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    model_dir_name = "-".join(config["oc_lang_pair"])
    save, save_subdirs = get_train_dir(OC_dir, model_dir_name, create=create)
    return save, save_subdirs

@log_mode_call
@call_seed_everything
@call_nvidia_smi
def train_model(config):
    save, save_subdirs = _get_save_dir(config)

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
        dirpath=save_subdirs["checkpoints"],
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
    logger = CSVLogger(save_dir=save_subdirs["logs"])
    tb_logger = TensorBoardLogger(save_dir=save_subdirs["tb"])

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


@log_mode_call
@call_seed_everything
@call_nvidia_smi
def eval_models(config):
    # tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config["oc_train"])
    
    # data
    dm = get_datamodule(config, src_tokenizer, tgt_tokenizer)

    # dirs
    save, save_subdirs = _get_save_dir(config, create=False)

    # save = get_save_dir(config)
    # checkpoints_d, data_d, preds_d, logs_d, tb_d = get_save_subdirs(save)
    
    # inference
    scores = {}
    chkpt_preds = {}
    source_segs = None
    target_segs = None
    for chkpt_file in os.listdir(save_subdirs["checkpoints"]):
        chkpt_file = os.path.join(save_subdirs["checkpoints"], chkpt_file)
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
    write_preds(chkpt_preds, save_subdirs["predictions"])
    write_scores(scores, save_subdirs["predictions"])

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
    best_key = f"BEST_VAL_{use_metric}"
    assert best_key not in scores
    scores[best_key] = {
        "checkpoint": best_checkpoint,
        "chrF": scores[best_checkpoint]["chrF"],
        "charBLEU": scores[best_checkpoint]["charBLEU"]
    }
    assert scores[best_key][use_metric] == scores[best_checkpoint][use_metric] == best_score
    

@log_mode_call
@call_seed_everything
@call_nvidia_smi
def inference(config, source_words_f, chkpt_file=None, best_metric="chrF"):
    assert best_metric in ["chrF", "charBLEU"]

    # dirs
    save, save_subdirs = _get_save_dir(config, create=False)

    if not chkpt_file:
        print("No checkpoint file provided. Getting best checkpoint based on validation scores.")
        chkpt_file = get_best_checkpoint(save_subdirs["predictions"], best_metric)
        print("BEST CHECKPOINT FOUND:", chkpt_file)

    # tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers(config["oc_train"])
    
    # data
    dm = OCDataModule(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        train=source_words_f,
        val=source_words_f,
        batch_size=config["oc_batch_size"],
        max_length=config["oc_max_length"]
    )
    dm.setup()

    # inference
    inference_outputs = run_inference(chkpt_file, config, src_tokenizer, tgt_tokenizer, dm.val_dataloader())
    hyps = [pred for _, _, pred in inference_outputs]
    
    # write
    scenario = config["oc_scenario"]
    output_tag = "." + config["sc_model_ids"][scenario]
    hyp_words_out = source_words_f + output_tag
    write_lines(hyps, hyp_words_out)

    return hyp_words_out, output_tag

def get_best_checkpoint(preds_d, best_metric):
    assert best_metric in ["chrF", "charBLEU"]

    scores_f = os.path.join(preds_d, "scores.json")
    if not os.path.exists(scores_f):
        raise FileExistsError(f"No scores file found: `{scores_f}`")
    scores = read_json(scores_f)
    best_key = f"BEST_VAL_{best_metric}".upper()
    if best_key not in scores:
        raise ValueError(f"No key `{best_key}` in scores file `{scores_f}`")
    checkpoint = scores[best_key]["checkpoint"]
    return checkpoint

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

def read_json(f):
    with open(f) as inf:
        data = json.load(inf)
    return data

@log_parsed_args
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
    log_script("OC.train", __file__)
    args = get_args()
    config = utilities.read_data.read_config(args.config)
    f = {
        "TRAIN": train_model,
        "EVAL": eval_models,
        "INFERENCE": inference
    }[args.mode]
    f_args = [config]
    if args.mode  == "INFERENCE":
        f_args += [args.chkpt_file, args.source_words, args.hyp_words_out]
    f(*f_args)

