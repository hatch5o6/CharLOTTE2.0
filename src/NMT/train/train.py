import argparse
import os
import functools
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from sloth_hatch.sloth import read_lines, log_parsed_args, log_script, write_json, write_lines

import utilities
from NMT.train.BARTLightning import BARTLightning, BARTDataModule
from NMT.train.NMTTokenizer import load_tokenizer
from utilities.metrics import calc_chrF_plus_plus, calc_spBLEU
from utilities.read_data import get_set
from utilities.train_utilities import log_mode_call, call_nvidia_smi, validate_dir, get_save_subdirs, PrintCallback

torch.set_float32_matmul_precision('medium')

def call_seed_everything(f):
    @functools.wraps(f)
    def wrapper(config):
        seed = config["seed"]
        rank_zero_info(f"Seeding everything with seed={seed}")
        L.seed_everything(seed, workers=True)
        result = f(config)
        return result
    return wrapper

def get_save_dir(config):
    save_dir = os.path.join(config["save"], config["experiment_name"], "NMT")
    validate_dir(save_dir)
    return save_dir

def get_datamodule(config, tokenizer):
    dm = BARTDataModule(
        tokenizer=tokenizer,
        data=config["data"],
        sc_model_ids=config["sc_model_ids"],
        reverse=config["nmt_reverse"],
        mode=config["nmt_corpus"],
        batch_size=config["nmt_batch_size"],
        max_length=config["nmt_max_length"],
        append_lang_tags=len(config["data"]) > 1 # True if multilingual
    )
    dm.setup()
    return dm

@log_mode_call
@call_seed_everything
@call_nvidia_smi
def train_model(config):
    save = get_save_dir(config)
    checkpoints_d, data_d, preds_d, logs_d, tb_d = get_save_subdirs(save)

    # tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # data
    dm = get_datamodule(config, tokenizer)

    # model
    model = BARTLightning(config, tokenizer)

    # callbacks and loggers
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config["nmt_patience"],
        verbose=True
    )
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_d,
        filename="{epoch}-{step}-{val_loss:.4f}",
        save_top_k=config["nmt_save_top_k"],
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
    if config["nmt_n_gpus"] >= 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = L.Trainer(
        max_steps=config["nmt_max_steps"],
        val_check_interval=config["nmt_val_interval"],
        accelerator=config["nmt_device"],
        default_root_dir=save,
        callbacks=train_callbacks,
        logger=[logger, tb_logger],
        deterministic=True,
        strategy=strategy,
        gradient_clip_val=config["nmt_gradient_clip_val"]
    )

    # train
    trainer.fit(model, dm)

@log_mode_call
@call_seed_everything
@call_nvidia_smi
def eval_models(config):
    save = get_save_dir(config)
    checkpoints_d, data_d, preds_d, logs_d, tb_d = get_save_subdirs(save)

    # tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # data
    dm = get_datamodule(config, tokenizer)

    # gt_source and gt_target sets
    # for now, just implement for bilingual directions
    # TODO Multilingual
    val_src, val_tgt = _get_div_set(config, "val")
    test_src, test_tgt = _get_div_set(config, "test")

    # scores
    scores = {}
    chkpt_preds = {}
    for chkpt_file in os.listdir(checkpoints_d):
        chkpt_file = os.path.join(checkpoints_d, chkpt_file)
        
        val_outputs = _run_inference(chkpt_file, config, tokenizer, dm.val_dataloader())
        test_outputs = _run_inference(chkpt_file, config, tokenizer, dm.test_dataloader())

        assert chkpt_file not in scores
        scores[chkpt_file] = _get_scores(val_outputs, val_src, val_tgt, div="VAL")
        scores[chkpt_file].update(_get_scores(test_outputs, test_src, test_tgt, div="TEST"))

        assert chkpt_file not in chkpt_preds
        chkpt_preds[chkpt_file] = {
            "val": _get_preds_from_outputs(val_outputs),
            "test": _get_preds_from_outputs(test_outputs)
        }
    assert sorted(scores.keys()) == sorted(chkpt_preds.keys())
    _get_best_val_scores(scores, use_metric="chrF++")
    
    # write
    _write_scores(scores, preds_d)
    _write_preds(chkpt_preds, preds_d)

def _write_scores(scores, d):
    assert os.path.exists(d)
    scores_f = os.path.join(d, "scores.json")
    write_json(scores, scores_f)

def _write_preds(chkpt_preds, d):
    assert os.path.exists(d)
    for chkpt, preds in chkpt_preds.items():
        chkpt_name = chkpt.split("/")[-1]
        chkpt_pred_d = os.path.join(d, chkpt_name)
        assert not os.path.exists(chkpt_pred_d)
        os.mkdir(chkpt_pred_d)
        write_lines(preds["val"], os.path.join(chkpt_pred_d, "validation.preds.txt"))
        write_lines(preds["test"], os.path.join(chkpt_pred_d, "test.preds.txt"))

def _get_best_val_scores(scores, use_metric="chrF++"):
    assert use_metric in ["chrF++", "spBLEU"]
    scores_list = [
        (scores[chkpt][f"VAL_{use_metric}"], chkpt)
        for chkpt in scores
    ]
    best_score, best_checkpoint = max(scores_list)
    best_key = f"BEST_VAL_{use_metric}"
    assert best_key not in scores
    scores[best_key] = {
        "checkpoint": best_checkpoint,
        "VAL_chrF++": scores[best_checkpoint]["VAL_chrF++"],
        "VAL_spBLEU": scores[best_checkpoint]["VAL_spBLEU"],
        "TEST_chrF++": scores[best_checkpoint]["TEST_chrF++"],
        "TEST_spBLEU": scores[best_checkpoint]["TEST_spBLEU"]
    }
    assert scores[best_key][f"VAL_{use_metric}"] == scores[best_checkpoint][f"VAL_{use_metric}"] == best_score

def _get_preds_from_outputs(outputs):
    return [pred for source, target, pred in outputs]

def _get_div_set(config, div):
    if div not in ["train", "val", "test"]:
        raise ValueError("div must be in 'train', 'val', 'test'.")
    # for getting source and target sets for a provided div
    # for now, just implement for bilingual directions
    # TODO Multilingual
    assert len(config["data"]) == 1
    assert config["nmt_corpus"] in ["parent", "child"]
    _, pl, cl, tl = config["data"][0]
    src_file, tgt_file = get_set(datasets=config["data"],
                                 scenario=(pl, cl, tl),
                                 pair=(pl, tl) if config["nmt_corpus"] == "parent" else (cl, tl),
                                 div=div)
    src = read_lines(src_file)
    tgt = read_lines(tgt_file)
    assert len(src) == len(tgt)
    return src, tgt


def _get_scores(outputs, gt_source, gt_target, div="VAL"):
    if div not in ["TEST", "VAL"]:
        raise ValueError(f"div must be 'TEST' or 'VAL'.")
    
    source_segs = []
    target_segs = []
    pred_segs = []
    for source, target, pred in outputs:
        source_segs.append(source)
        target_segs.append(target)
        pred_segs.append(pred)

    assert len(source_segs) == len(target_segs) == len(pred_segs) == len(gt_source) == len(gt_target)
    
    assert source_segs == gt_source
    assert target_segs == gt_target

    return {
        f"{div}_chrF++": calc_chrF_plus_plus(pred_segs, gt_target),
        f"{div}_spBLEU": calc_spBLEU(pred_segs, gt_target)
    }


def _run_inference(chkpt_file, config, tokenizer, dataloader):
    model = BARTLightning.load_from_checkpoint(
        checkpoint_path=chkpt_file,
        config=config,
        tokenizer=tokenizer
    )
    model.eval()

    trainer = L.Trainer(accelerator=config["nmt_device"])
    prediction_batches = trainer.predict(model, dataloader)

    predictions = []
    for batch in prediction_batches:
        predictions += batch
    
    return predictions

def inference():
    pass

@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices=["TRAIN", "EVAL", "INFERENCE"], default="TRAIN")
    parser.add_argument("-C", "--nmt_corpus", choices=["parent", "child"])
    parser.add_argument("--REVERSE", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    from NMT.train.NMTTokenizer import train_unigram, make_tokenizer_data, assemble_multilingual_tokenizer_data

    log_script("NMT.train", __file__)
    args = get_args()
    config = utilities.read_data.read_config(args.config, 
                                             nmt_corpus=args.nmt_corpus,
                                             reverse=args.REVERSE)
    
    if "tokenizer" not in config:
        raise ValueError("Must have config 'tokenizer' set in config.")
    
    # if "tokenizer" not in config:
    #     print("NMT.train.train: 'tokenizer' not set. Will train one.")
    #     tokenizer_train_data, tokenizer_dir = make_tokenizer_data(config)
    #     if len(config["data"]) == 1: 
    #         # is monolingual
    #         assert len(tokenizer_train_data) == 1
    #         scenario = tuple(config["data"][0][1:])
    #         print(f"Getting tokenizer data for scenario {scenario}.")
    #         data_files = tokenizer_train_data[scenario]
    #         for f in data_files:
    #             print(f"\t-{f}")
    #         config["tokenizer"] = train_unigram(files=data_files,
    #                                             save=tokenizer_dir,
    #                                             vocab_size=config["nmt_vocab_size"],
    #                                             seed=config["seed"])
    #         print(f"Set config 'tokenizer' to newly trained {config['tokenizer']}.")
    #     else:
    #         # is multilingual
    #         #TODO
    #         pass
    # else:
    #     print(f"NMT.train.train: Using pre-existing tokenizer `{config['tokenizer']}`")

    f = {
        "TRAIN": train_model,
        "EVAL": eval_models,
        "INFERENCE": inference
    }[args.mode]
    f(config)
