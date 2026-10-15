import argparse
import os
from copy import deepcopy
import functools
import shutil
from sloth_hatch.sloth import read_content, write_content, write_json, read_lines, read_yaml, create_directory, log_parsed_args, log_script, log_function_call, read_json

from NMT.train import train_jobs as NMT_train_jobs
from NMT.train.train import _nmt_config_key
from OC.train.train import get_tokenizers, run_inference
from OC.train.OCLightning import OCDataModule
from OC.train import train_jobs as OC_train_jobs
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
    config["save"] = set_vars_in_path(config["save"])
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

        # prepare_OC_data(config)

        prepare_OC_data_results = hpc.abstract_function(
            function=lambda: prepare_OC_data(config), 
            job_name="prepare_OC_data_" + config["experiment_name"],
            output_folder=prepare_oc_data_folder,
            mail_user=config["email"],
            timeout=config["basic_timeout"],
            ntasks_per_node=1,
            nodes=1,
            mem_gb=config["basic_mem"],
            n_gpus=0,
            qos=config["basic_qos"],
            use_hpc=config["use_hpc"]
        )
        pipeline_results["prepare_OC"] = prepare_OC_data_results


    # ------------------------ OC ------------------------
    if "OC" in pipeline:
        # Train, Eval, and Infer OC
        OC_afterok = prepare_OC_data_results.job_id if config["use_hpc"] and "prepare_OC" in pipeline else None

        OC_results = OC(config, OC_afterok)
        pipeline_results["OC"] = OC_results

    # ------------------------ OC_reshape ------------------------
    if "OC_reshape" in pipeline:
        # OC reshape
        OC_reshape_folder = os.path.join(config["save"], config["experiment_name"], "OC/SLURM/OC_reshape")
        OC_reshape_afterok = OC_results.job_id if config["use_hpc"] and "OC" in pipeline else None
        OC_reshape_results = hpc.abstract_function(
            function=lambda: OC_reshape(config),
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
    print(_wrap_in_pounds("FAST PIVOT"))
    # get pl_tl dictionaries
    pl_tl_dict = {}
    pl_tl_info = {}
    for fr, pl, tl, theta in pl_tl_cognates:
        if not pl in pl_tl_dict:
            pl_tl_dict[pl] = tl
            pl_tl_info[pl] = (fr, theta)
        else:
            # take the duplicate with the smaller theta
            if theta < pl_tl_info[pl][1]:
                pl_tl_dict[pl] = tl
                pl_tl_info[pl] = (fr, theta)

    # get tl_cl dictionaries
    tl_cl_dict = {}
    tl_cl_info = {}
    for fr, cl, tl, theta in cl_tl_cognates:
        if not tl in tl_cl_dict:
            tl_cl_dict[tl] = cl
            tl_cl_info[tl] = (fr, theta)
        else:
            # take the duplicate with the smaller theta
            if theta < tl_cl_info[tl][1]:
                tl_cl_dict[tl] = cl
                tl_cl_info[tl] = (fr, theta)
        

    print(f"Removed {len(pl_tl_cognates) - len(pl_tl_dict)} source duplicates from pl_tl_cognates")
    print(f"Removed {len(cl_tl_cognates) - len(tl_cl_dict)} source duplicates from cl_tl_cognates")


    pl_cl_cognates = []
    for pl in pl_tl_dict:
        tl = pl_tl_dict[pl]
        if tl in tl_cl_dict:
            pl_fr, pl_theta = pl_tl_info[pl]
            cl_fr, cl_theta = tl_cl_info[tl]
            pl_cl_cognates.append((min(pl_fr, cl_fr), pl, tl_cl_dict[tl], max(pl_theta, cl_theta)))

    pl_cl_cognates = sorted(pl_cl_cognates, key=lambda t: t[0], reverse=True)

    print(f"pl_tl_cognates: {len(pl_tl_cognates)}\ncl_tl_cognates: {len(cl_tl_cognates)}\npivot_pl_cl_cognates: {len(pl_cl_cognates)}")
    return pl_cl_cognates

def write_pivot_file(pl_cl_cognates, OC_dir, pl, cl):
    cognate_dir = os.path.join(OC_dir, "cognates", f"{pl}_{cl}")
    os.makedirs(cognate_dir, exist_ok=True)
    cognate_f = os.path.join(cognate_dir, f"{pl}_{cl}.word_pairs")
    with open(cognate_f, "w") as f:
        lines = [f"({fr}, {p}, {c})" for fr, p, c, theta in pl_cl_cognates]
        f.write("\n".join(lines) + '\n')


def prepare_OC_data(config):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    data_folder, pl, cl, tl = config["data"]
    data_folder = set_vars_in_path(data_folder)

    if config["fast_pivot"]:
        # get pl_tl_pairs and cl_tl_pairs
        pl_tl_cognates = get_cognates(config, OC_dir, data_folder, pl, tl)
        cl_tl_cognates = get_cognates(config, OC_dir, data_folder, cl, tl)

        # run pivot
        pl_cl_cognates = fast_pivot(pl_tl_cognates, cl_tl_cognates)
        write_pivot_file(pl_cl_cognates, OC_dir, pl, cl)


    elif not config["fast_pivot"]:
        # get pl_cl_cognates
        pl_cl_cognates = get_cognates(config, OC_dir, data_folder, pl, cl)


    # write out train and val split for the cognates
    train_split, val_split = get_train_val_split(
          pairs=pl_cl_cognates,
          theta=config["theta"],
          size=config["oc_val_size"],
          n_buckets=config["oc_val_nld_buckets"],
          max_fraction=config["oc_val_max_bucket_fraction"],
          seed=config["seed"],
      )
    
    _assert_no_train_contamination(train_split, val_split)
    data_dir = os.path.join(OC_dir, "cognates", f"{pl}_{cl}")
    os.makedirs(data_dir, exist_ok=True)
    write_oc_data(train_split, os.path.join(data_dir, "train.txt"))
    write_oc_data(val_split, os.path.join(data_dir, "val.txt"))


def compile_hyperparam_results(config, metric="chrF"):
    assert metric in ["chrF", "charBLEU"], f"{metric} not a valid metric for selecting best OC model"
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")
    search_dir = os.path.join(OC_dir, "hyperparam_search")

    results = {}
    for hyperparams in sorted(os.listdir(search_dir)):
        run_dir = os.path.join(search_dir, hyperparams)
        if hyperparams == "SLURM" or not os.path.isdir(run_dir):
            continue

        scores_path = os.path.join(run_dir, "predictions", "scores.json")
        if not os.path.isfile(scores_path):
            raise FileNotFoundError(
                f"Missing hyperparameter scores file: {scores_path}"
            )

        scores = read_json(scores_path)
        best = scores["BEST_VAL_CHRF"]
        results[hyperparams] = {
            "chrF": best["chrF"],
            "charBLEU": best["charBLEU"],
            "checkpoint": best["checkpoint"],
        }

    if not results:
        raise ValueError(f"No hyperparameter results found in {search_dir}")

    best_config = max(results, key=lambda name: results[name][metric])
    report = {
        "configurations": results,
        "best_configuration": best_config,
        "best_scores": results[best_config],
    }
    write_json(
        report,
        os.path.join(search_dir, "hyperparam_results.json"),
        indent=2,
        ensure_ascii=False,
    )

    best_run_dir = os.path.join(search_dir, best_config)
    model_dir = os.path.join(OC_dir, "model")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(best_run_dir, model_dir)

    promoted_scores_path = os.path.join(model_dir, "predictions", "scores.json")
    promoted_scores = read_json(promoted_scores_path)
    old_checkpoint = promoted_scores["BEST_VAL_CHRF"]["checkpoint"]
    promoted_scores["BEST_VAL_CHRF"]["checkpoint"] = os.path.join(
        model_dir, "checkpoints", os.path.basename(old_checkpoint)
    )
    write_json(promoted_scores, promoted_scores_path, indent=2)

    return report

def _infer_selected_model(config, source_words_f, model_dir, scenario, output_tag):
    inference_config = deepcopy(config)

    if inference_config["hyperparam_search"]:
        report_path = os.path.join(
            get_task_dir(get_exp_dir(inference_config), task="OC"),
            "hyperparam_search",
            "hyperparam_results.json",
        )
        best_hyperparams = read_json(report_path)["best_configuration"]
        hidden, embed, batch, layers = map(int, best_hyperparams.split("-"))

        inference_config["oc_enc_hidden_dim"] = hidden
        inference_config["oc_dec_hidden_dim"] = hidden
        inference_config["oc_enc_embed_dim"] = embed
        inference_config["oc_dec_embed_dim"] = embed
        inference_config["oc_batch_size"] = batch
        inference_config["oc_enc_num_layers"] = layers
        inference_config["oc_dec_num_layers"] = layers

    scores_path = os.path.join(model_dir, "predictions", "scores.json")
    checkpoint = read_json(scores_path)["BEST_VAL_CHRF"]["checkpoint"]

    src_tokenizer, tgt_tokenizer = get_tokenizers(inference_config["oc_train"])
    dm = OCDataModule(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        train=source_words_f,
        val=source_words_f,
        batch_size=inference_config["oc_batch_size"],
        max_length=inference_config["oc_max_length"],
    )
    dm.setup()

    outputs = run_inference(
        checkpoint,
        inference_config,
        src_tokenizer,
        tgt_tokenizer,
        dm.val_dataloader(),
    )
    predictions = [pred for _, _, pred in outputs]

    hyp_words_f = source_words_f + output_tag
    with open(hyp_words_f, "w") as outf:
        outf.write("\n".join(predictions) + "\n")

    return hyp_words_f

def OC(config, OC_afterok):
    # Directory structure
    exp_dir = get_exp_dir(config)
    OC_dir = get_task_dir(exp_dir, task="OC")

    data_folder, pl, cl, tl = config["data"]
    data_folder = set_vars_in_path(data_folder)

    data_dir = os.path.join(OC_dir, "cognates", f"{pl}_{cl}")
    config["oc_train"] = os.path.join(data_dir, "train.txt")
    config["oc_val"] = os.path.join(data_dir, "val.txt")
    
    jobs = {}

    # hyperparam search
    if config["hyperparam_search"]:
        os.makedirs(os.path.join(OC_dir, "hyperparam_search"))
        eval_job_ids = []

        for hidden in config["oc_hidden_dim"]:
            for embed in config["oc_embed_dim"]:
                for batch in config["oc_batch"]:
                    for layers in config["oc_num_layers"]:
                        config["OC_hyperparams"] = f"{hidden}-{embed}-{batch}-{layers}"
                        config["oc_enc_hidden_dim"] = hidden
                        config["oc_dec_hidden_dim"] = hidden
                        config["oc_enc_embed_dim"] = embed
                        config["oc_dec_embed_dim"] = embed
                        config["oc_batch_size"] = batch
                        config["oc_enc_num_layers"] = layers
                        config["oc_dec_num_layers"] = layers


                        run_jobs = OC_train_jobs.train_and_eval(
                            config=config,
                            on_hpc=config["use_hpc"],
                            afterok=OC_afterok,
                        )
                        jobs[config["OC_hyperparams"]] = run_jobs

                        if config["use_hpc"]:
                            eval_job_ids.append(str(run_jobs["eval"][0].job_id))

        compile_results = hpc.abstract_function(function=lambda: compile_hyperparam_results(config, metric="chrF"),
                                                job_name="compile_OC_hyperparam_search_" + config["experiment_name"],
                                                output_folder=os.path.join(
                                                    OC_dir, "SLURM", "compile_hyperparam_search"),
                                                mail_user=config["email"],
                                                timeout=config["basic_timeout"],
                                                ntasks_per_node=1,
                                                nodes=1,
                                                mem_gb=config["basic_mem"],
                                                n_gpus=0,
                                                qos=config["basic_qos"],
                                                afterok=":".join(eval_job_ids) if eval_job_ids else None,
                                                use_hpc=config["use_hpc"],
                                            )


    else:
        hidden = config["oc_enc_hidden_dim"]
        embed = config["oc_enc_embed_dim"]
        batch = config["oc_batch_size"]
        layers = config["oc_enc_num_layers"]
        config["OC_hyperparams"] = f"{hidden}-{embed}-{batch}-{layers}"

        run_jobs = OC_train_jobs.train_and_eval(config=config,
                                                on_hpc=config["use_hpc"],
                                                afterok=OC_afterok)


    # Run inference on all PL words from train, validation, and test.
    scenario = (pl, cl, tl)
    pl_tl_dir = os.path.join(data_folder, f"{pl}-{tl}")

    pl_files = [
        os.path.join(pl_tl_dir, f"{split}.{pl}.txt")
        for split in ("train", "val", "test")
    ]
    source_words_f = os.path.join(pl_tl_dir, "words_for_inference.txt")
    reshape.prepare_source_words(
        pl_files,
        long_enough=config["oc_min_word_len_applied"],
        out_path=source_words_f,
    )

    model_dir = os.path.join(OC_dir, "model")
    config["oc_scenario"] = scenario
    config["oc_method"] = "OC"
    config["sc_model_ids"] = {
        scenario: f"{config['sc_model_id_prefix']}_{pl}-{cl}"
    }
    output_tag = f".{config['oc_method']}.{config['sc_model_ids'][scenario]}"

    if config["use_hpc"]:
        if config["hyperparam_search"]:
            inference_afterok = compile_results.job_id
        else:
            inference_afterok = run_jobs["eval"][0].job_id
    else:
        inference_afterok = None

    inference_results = hpc.abstract_function(function=lambda: _infer_selected_model(
                                                config, source_words_f, model_dir, scenario, output_tag),
                                                job_name=f"OC_INFER_{config['experiment_name']}",
                                                output_folder=os.path.join(OC_dir, "SLURM", "OC_INFER"),
                                                mail_user=config["email"],
                                                gpu_type=config["gpu_type"],
                                                qos=config["oc_qos"],
                                                timeout=config["basic_timeout"],
                                                ntasks_per_node=1,
                                                nodes=1,
                                                mem_gb=config["oc_mem"],
                                                n_gpus=1,
                                                afterok=inference_afterok,
                                                use_hpc=config["use_hpc"])

    return inference_results

        

def OC_reshape(config):
    data_folder, pl, cl, tl = config["data"]
    data_folder = set_vars_in_path(data_folder)

    pl_tl_dir = os.path.join(data_folder, f"{pl}-{tl}")
    source_words_f = os.path.join(pl_tl_dir, "words_for_inference.txt")

    # Match the tag used by inference in OC().
    output_tag = f".OC.{config['sc_model_id_prefix']}_{pl}-{cl}"
    hyp_words_f = source_words_f + output_tag

    for path in (source_words_f, hyp_words_f):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required inference file does not exist: {path}")

    source_rows = read_oc_data(source_words_f)
    source_words = [row[1] for row in source_rows]
    hyp_words = read_lines(hyp_words_f)

    if len(source_words) != len(hyp_words):
        raise ValueError(
            f"Source/prediction count mismatch: {len(source_words)} source words, "
            f"{len(hyp_words)} predictions in {hyp_words_f}"
        )

    word_mappings = dict(zip(source_words, hyp_words))

    reshaped_files = []
    for split in ("train", "val", "test"):
        pl_file = os.path.join(pl_tl_dir, f"{split}.{pl}.txt")
        if not os.path.isfile(pl_file):
            raise FileNotFoundError(f"PL corpus file does not exist: {pl_file}")

        reshape.reshape_data(
            pl_file,
            word_mappings=word_mappings,
            output_tag=output_tag,
            long_enough=config["oc_min_word_len_applied"],
        )
        reshaped_files.append(pl_file + output_tag)

    return reshaped_files


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