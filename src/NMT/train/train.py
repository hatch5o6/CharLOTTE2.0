import argparse
import os
import functools
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from sloth_hatch.sloth import read_lines, read_json, log_parsed_args, log_script, write_json, write_lines
from functools import partial
from copy import deepcopy

import utilities
from utilities.experiment_file_system import get_exp_dir, get_task_dir, get_train_dir
from NMT.train.BARTLightning import BARTLightning, BARTDataModule
from NMT.train.NMTTokenizer import load_tokenizer
from utilities import model_names
from utilities.metrics import calc_chrF_plus_plus, calc_spBLEU, calc_BLEU
from utilities.read_data import get_set
from utilities.train_utilities import log_mode_call, call_nvidia_smi, PrintCallback

torch.set_float32_matmul_precision('medium')

def _call_seed_everything(f):
    @functools.wraps(f)
    def wrapper(*args):
        config = args[0]
        seed = config["seed"]
        rank_zero_info(f"Seeding everything with seed={seed}")
        L.seed_everything(seed, workers=True)
        result = f(*args)
        return result
    return wrapper

def _nmt_config_key(config:dict, fine_tune:bool):
    if not isinstance(fine_tune, bool):
        raise ValueError("fine_tune must be a bool!")

    if fine_tune == False:
        if config["nmt_corpus"] == "parent":
            key = "parent"
        else:
            key = "simple"
    else:
        if config["nmt_corpus"] != "child":
            raise ValueError(f"When fine-tuning, nmt_corpus must be 'child'!")
        assert config["nmt_corpus"] == "child"
        key = "child"
    return key

def _set_nmt_config(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        config=args[0]
        fine_tune=kwargs["fine_tune"]
        if config["nmt_corpus"] not in ['parent', 'child']:
            raise ValueError(f"nmt_corpus must be 'parent' or 'child'!")
        
        key = _nmt_config_key(config, fine_tune=fine_tune)
        
        rank_zero_info(f"Setting {key} NMT parameters in the config.")

        prefix = f"{key}_"
        for k, v in list(config.items()):
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                config[new_k] = v
        
        result = f(config)
        return result
    return wrapper


def _get_datamodule(config, tokenizer):
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

def _get_save_dir(config, fine_tune, create=True):
    exp_dir = get_exp_dir(config)
    NMT_dir = get_task_dir(exp_dir, task="NMT")
    if config["nmt_corpus"] == "parent":
        model_dir_name = "NMT_parent"
    else:
        if fine_tune == True:
            model_dir_name = "NMT_child"
        else:
            model_dir_name = "NMT_simple"
    if config["sc_model_ids"] != None:
        model_dir_name = "OC_" + model_dir_name
    if config["nmt_reverse"] == True:
        model_dir_name += "_reverse"
    model_dir_name += "_" + config["nmt_model_id"]
    save, save_subdirs = get_train_dir(NMT_dir, model_dir_name, create=create)
    return save, NMT_dir, save_subdirs

@call_nvidia_smi
@_set_nmt_config
@log_mode_call
@_call_seed_everything
def train_model(config, fine_tune=False):
    if config["nmt_corpus"] not in ['parent', 'child']:
        raise ValueError(f"nmt_corpus must be 'parent' or 'child'!")
    
    # file structure
    save, NMT_dir, save_subdirs = _get_save_dir(config, fine_tune=fine_tune)

    # tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # data
    dm = _get_datamodule(config, tokenizer)

    # model
    if fine_tune == False:
        rank_zero_info("WILL TRAIN FROM SCRATCH")
        model = BARTLightning(config, tokenizer)
    else:
        if config["nmt_corpus"] != "child":
            raise ValueError(f"When fine-tuning, nmt_corpus must be 'child'!")
        parent_dir = "NMT_parent"
        if config["sc_model_ids"] != None:
            parent_dir = "OC_" + parent_dir
        parent_dir = os.path.join(NMT_dir, parent_dir)
        best_parent_chkpt = _best_checkpoint(parent_dir)
        rank_zero_info(f"WILL RESUME TRAINING OF `{best_parent_chkpt}`.")
        model = BARTLightning.load_from_checkpoint(
            checkpoint_path=best_parent_chkpt,
            config=config,
            tokenizer=tokenizer
        )

    # callbacks and loggers
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config["nmt_patience"],
        verbose=True
    )
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=save_subdirs["checkpoints"],
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
    logger = CSVLogger(save_dir=save_subdirs["logs"])
    tb_logger = TensorBoardLogger(save_dir=save_subdirs["tb"])

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

def _best_checkpoint(model_dir_stensil, use_metric="chrF++"):
    parent_of_stensil = "/".join(model_dir_stensil.split("/")[:-1])
    model_dirs = []
    for d in os.listdir(parent_of_stensil):
        d_path = os.path.join(parent_of_stensil, d)
        if d.startswith(model_dir_stensil + "_") and os.path.isdir(d_path):
            model_dirs.append(d_path)
    
    all_best_models = []
    for model_d in model_dirs:
        best_model_chkpt, best_model_chkpt_val_score = _best_checkpoint_in_model_dir(model_d, use_metric=use_metric)
        all_best_models.append((best_model_chkpt_val_score, best_model_chkpt))
    definitive_best_score, definitive_best_model = max(all_best_models)
    return definitive_best_model

def _best_checkpoint_in_model_dir(model_dir, use_metric="chrF++"):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Could not find NMT directory: `{model_dir}`")
    predictions_dir = os.path.join(model_dir, "predictions")
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Could not find NMT predictions: `{predictions_dir}`")
    scores_f = os.path.join(predictions_dir, "scores.json")
    if not os.path.exists(scores_f):
        raise FileNotFoundError(f"Could not find NMT scores: `{scores_f}`")
    
    scores = read_json(scores_f)
    best_key = f"BEST_VAL_{use_metric}"
    best_chkpt = scores[best_key]["checkpoint"]
    best_val_score = scores[best_key][f"VAL_{use_metric}"]
    rank_zero_info(f"Retrieved best checkpoint: {best_chkpt}")
    return best_chkpt, best_val_score

@call_nvidia_smi
@_set_nmt_config
@log_mode_call
@_call_seed_everything
def eval_models(config, fine_tune=False):
    # file structure
    save, NMT_dir, save_subdirs = _get_save_dir(config, fine_tune=fine_tune, create=False)

    # tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # data
    dm = _get_datamodule(config, tokenizer)

    # gt_source and gt_target sets
    # for now, just implement for bilingual directions
    # TODO Multilingual
    val_src, val_tgt = _get_div_set(config, "val")
    test_src, test_tgt = _get_div_set(config, "test")

    # scores
    scores = {}
    chkpt_preds = {}
    for chkpt_file in os.listdir(save_subdirs["checkpoints"]):
        chkpt_file = os.path.join(save_subdirs["checkpoints"], chkpt_file)
        
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
    _write_scores(scores, save_subdirs["predictions"])
    _write_preds(chkpt_preds, save_subdirs["predictions"])

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
        "VAL_BLEU": scores[best_checkpoint]["VAL_BLEU"],
        "TEST_chrF++": scores[best_checkpoint]["TEST_chrF++"],
        "TEST_spBLEU": scores[best_checkpoint]["TEST_spBLEU"],
        "TEST_BLEU": scores[best_checkpoint]["TEST_BLEU"]
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
        f"{div}_spBLEU": calc_spBLEU(pred_segs, gt_target),
        f"{div}_BLEU": calc_BLEU(pred_segs, gt_target)
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

@call_nvidia_smi
@_set_nmt_config
@log_mode_call
@_call_seed_everything
def inference(config, inference_file, src_lang, tgt_lang, fine_tune=True):
    config = deepcopy(config)
    reverse = config["nmt_reverse"]
    if reverse != True:
        rank_zero_info(f"inference: config[\"nmt_reverse\"] == `{reverse}`.")
        rank_zero_info("\treverse must be True for inference.\n\tSetting to `True`.")
        config["nmt_reverse"] = True
        reverse = True
    # include fine_tune argument to indicate if we're running inference on a child model or simple model.
    #   it's used by the _set_nmt_config decorator
    
    # file structure
    save, NMT_dir, save_subdirs = _get_save_dir(config, fine_tune=fine_tune, create=False)

    # tokenizer
    tokenizer = load_tokenizer(config["tokenizer"])

    # data
    dm = BARTDataModule(
        tokenizer=tokenizer,
        batch_size=config["nmt_batch_size"],
        max_length=config["nmt_max_length"],
        append_lang_tags=len(config["data"]) > 1, # True if multilingual
        inference_file=inference_file,
        inference_src=src_lang,
        inference_tgt=tgt_lang
    )
    dm.setup()

    nmt_config_key = _nmt_config_key(config, fine_tune=fine_tune)
    oc_tag = "OC_" if config["sc_model_ids"] != None else ""

    model_dir = os.path.join(NMT_dir, f"{oc_tag}NMT_{nmt_config_key}")
    best_chkpt = _best_checkpoint(model_dir)

    inf_outputs = _run_inference(best_chkpt, config, tokenizer, dm.inf_dataloader())

    inf_inputs = read_lines(inference_file)

    preds = []
    assert len(inf_inputs) == len(inf_outputs)
    for i, (source, target, pred) in inf_outputs:
        assert source == inf_inputs[i]
        assert target == "<to be generated>"
        preds.append(pred)
    
    output_file = inference_file + f".{src_lang}->{tgt_lang}." + config["nmt_model_id"]
    write_lines(preds, output_file)
    rank_zero_info(f"Wrote translations to `{output_file}`.")

    return output_file


# def run_job(
#     config,
#     script_mode,
#     fine_tune,
#     reverse,
#     HPC,
# ):
#     pass DON'T NEED THIS.


@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices=["TRAIN", "EVAL", "INFERENCE"], default="TRAIN")
    parser.add_argument("-C", "--nmt_corpus", choices=["parent", "child"])
    parser.add_argument("-f", "--fine_tune", action="store_true")
    parser.add_argument("-HPC", "--HPC", action="store_true")
    parser.add_argument("--REVERSE", action="store_true", default=False)
    parser.add_argument("--WITH_OC", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    from NMT.train.train_tokenizer import train_tokenizer
    
    log_script("NMT.train", __file__)
    args = get_args()
    config = utilities.read_data.read_config(args.config, 
                                             nmt_corpus=args.nmt_corpus,
                                             reverse=args.REVERSE,
                                             add_sc_model_ids=args.WITH_OC,
                                             nmt_model_id=model_names.get_new_name())

    assert "tokenizer" not in config
    config["tokenizer"] = train_tokenizer(config, train_with_oc=args.WITH_OC)
    
    f = {
        "TRAIN": train_model,
        "EVAL": eval_models,
        "INFERENCE": inference
    }[args.mode]

    if args.HPC:
        from utilities.hpc import submit_slurm

        nmt_config_key = _nmt_config_key(config, fine_tune=args.fine_tune)
        reverse_tag = "_reverse" if args.REVERSE else ""
        oc_tag = "OC_" if config["sc_model_ids"] != None else ""
        output_folder = os.path.join(config["save"], config["experiment_name"], f"NMT/{oc_tag}NMT_{nmt_config_key}{reverse_tag}_{config['nmt_model_id']}/SLURM")
        os.makedirs(output_folder, exist_ok=True)
        slurm_job_name = args.mode + "|" + config["experiment_name"] + f"|{oc_tag}NMT_{nmt_config_key}{reverse_tag}"

        job = submit_slurm(
            function=lambda: f(config, fine_tune=args.fine_tune),
            job_name=slurm_job_name,
            output_folder=output_folder,
            mail_user=config["email"],
            timeout=config[f"{nmt_config_key}_nmt_timeout"],
            ntasks_per_node=config[f"{nmt_config_key}_nmt_n_gpus"],
            mem_gb=config[f"{nmt_config_key}_nmt_mem"],
            n_gpus=config[f"{nmt_config_key}_nmt_n_gpus"],
            gpu_type=config["gpu_type"],
            qos=config[f"{nmt_config_key}_nmt_qos"]
        )

    else:
        f(config, fine_tune=args.fine_tune)


