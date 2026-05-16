import os
from utilities.utilities import set_vars_in_path
from sloth_hatch.sloth import read_yaml

def read_config(
    config_f, 
    nmt_corpus=None, 
    # cognate_method=None, 
    reverse=None,
    warmup_divisor=20,
    add_sc_model_ids=False,
    nmt_model_id=None,
    oc_model_name=""
):
    if nmt_corpus not in ["parent", "child", None]:
        raise ValueError("nmt_corpus must be 'parent' or 'child'")
    # if cognate_method not in ["charlotte", "web", "fuzz", None]:
    #     raise ValueError("cognate_method must be 'charlotte', 'web', or 'fuzz'.")
    if reverse not in [True, False, None]:
        raise ValueError("reverse must be True or False")
    config = read_yaml(config_f)
    _validate_sets(config["data"])
    _validate_cognate_methods(config["methods"])

    config["save"] = set_vars_in_path(config["save"])
    config["oc_warmup_steps"] = config["oc_max_steps"] // warmup_divisor
    config["parent_nmt_warmup_steps"] = config["parent_nmt_max_steps"] // warmup_divisor
    config["child_nmt_warmup_steps"] = config["child_nmt_max_steps"] // warmup_divisor
    config["simple_nmt_warmup_steps"] = config["simple_nmt_max_steps"] // warmup_divisor
    config["sc_model_id_prefix"] = config["sc_model_id_prefix"].replace("{model_name}", oc_model_name)
    print(f"OC MODELS WILL HAVE TAG {config['sc_model_id_prefix']}")
    if add_sc_model_ids:
        config["sc_model_ids"] = _get_sc_model_ids(config["data"], config["sc_model_id_prefix"])
    else:
        config["sc_model_ids"] = None
    config["nmt_corpus"] = nmt_corpus
    config["nmt_reverse"] = reverse
    config["nmt_model_id"] = nmt_model_id

    return config

def _validate_cognate_methods(methods):
    visited = set()
    for method in methods:
        if method in visited:
            raise ValueError(f"Duplicate cognate method: {method}")
        visited.add(method)
        if method not in ["charlotte", "web", "fuzz"]:
            raise ValueError(f"Cognate methods must only include 'charlotte', 'web', and 'fuzz'.")

def _get_sc_model_ids(
    datasets:list, # list of (data folder, pl, cl, tl) tuples,
    sc_model_id_prefix:str
):
    _validate_sets(datasets)
    sc_model_ids = {}
    for data_folder, pl, cl, tl in datasets:
        scenario = pl, cl, tl
        assert scenario not in sc_model_ids
        sc_model_ids[scenario] = _get_one_sc_model_id(sc_model_id_prefix, (pl, cl, tl))
    return sc_model_ids

def get_pl_cl_pairs(
    datasets:list # list of (data folder, pl, cl, tl) tuples
):
    _validate_sets(datasets)
    return [(pl, cl) for _, pl, cl, _ in datasets]

def get_tl_pairs(
    datasets:list # list of (data folder, pl, cl, tl) tuples
):
    _validate_sets(datasets)
    return [
        ((pl, tl), (cl, tl))
        for _, pl, cl, tl in datasets
    ]

def read_pl_cl_paths( 
    datasets:list, # list of (data folder, pl, cl, tl) tuples
):
    _validate_sets(datasets)
    data = {}
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)
        lang_pair = pl, cl
        if lang_pair in data:
            raise ValueError(f"Duplicate OC (pl-cl) pair {lang_pair} in data!")
        
        directory = os.path.join(data_folder, f"{pl}-{cl}")
        if not os.path.exists(directory):
            print(f"No {pl}-{cl} data folder found in {data_folder}.")
            continue

        pl_path = os.path.join(directory, f"train.{pl}.txt")
        cl_path = os.path.join(directory, f"train.{cl}.txt")

        if os.path.exists(pl_path) and os.path.exists(cl_path):
            data[lang_pair] = pl_path, cl_path
        else:
            print(f"{pl} and/or {cl} train files could not be found in {directory}")
    
    print("charlotte OC PARALLEL SENTENCES FROM:")
    for lang_pair, paths in data.items():
        print(f"\t-{lang_pair}:{paths}")
    print("\n")
    return data

def read_pl_cl_web_paths(
    datasets:list, # list of (data folder, pl, cl, tl) tuples,
    tl_to_pl_tags:dict
):
    data = _read_pl_cl_paths_from_parallel_with_tl(datasets, tl_to_pl_tags)
    print("web OC PARALLEL SENTENCES FROM:")
    for lang_pair, paths in data.items():
        print(f"\t-{lang_pair}:{paths}")
    print("\n")
    return data

def read_pl_cl_fuzz_paths(
    datasets:list # list of (data folder, pl, cl, tl) tuples 
):
    data = _read_pl_cl_paths_from_parallel_with_tl(datasets)
    print("fuzz OC PARALLEL SENTENCES FROM:")
    for lang_pair, paths in data.items():
        print(f"\t-{lang_pair}:{paths}")
    print("\n")
    return data

def _read_pl_cl_paths_from_parallel_with_tl(
    datasets:list, # list of (data folder, pl, cl, tl) tuples
    tl_to_pl_tags:dict=None
):
    _validate_sets(datasets)
    data = {}
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)

        oc_pair = pl, cl
        if oc_pair in data:
            raise ValueError(f"Duplicate OC (pl-cl) pair {oc_pair} in data!")
        
        if tl_to_pl_tags:
            if (pl, cl, tl) not in tl_to_pl_tags:
                raise ValueError(f"Scenario {(pl, cl, tl)} missing from tl_to_pl_tags!")
            pl_path = os.path.join(data_folder, f"{cl}-{tl}", f"train.{tl}.txt" + tl_to_pl_tags[(pl, cl, tl)])
            cl_path = os.path.join(data_folder, f"{cl}-{tl}", f"train.{cl}.txt")
        else:
            pl_path = os.path.join(data_folder, f"{pl}-{tl}", f"train.{pl}.txt")
            cl_path = os.path.join(data_folder, f"{cl}-{tl}", f"train.{cl}.txt")

        assert os.path.exists(pl_path)
        assert os.path.exists(cl_path)

        data[oc_pair] = pl_path, cl_path
    return data

def read_pl_tl_data(
    datasets:list # list of (data folder, pl, cl, tl) tuples
):
    _validate_sets(datasets)
    data = {}
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)

        directory = os.path.join(data_folder, f"{pl}-{tl}")
        scenario = (pl, cl, tl)
        assert scenario not in data
        data[scenario] = directory
    return data

# Was read_pl_cl_web_paths
def read_pl_cl_parent_child_paths(
    datasets:list # list of (data folder, pl, cl, tl) tuples
):
    _validate_sets(datasets)
    data = {}
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)

        oc_pair = pl, cl
        if oc_pair in data:
            raise ValueError(f"Duplicate OC (pl-cl) pair {oc_pair} in data!")
        
        parent_data = os.path.join(data_folder, f"{pl}-{tl}")
        child_data = os.path.join(data_folder, f"{cl}-{tl}")

        assert os.path.exists(parent_data)
        assert os.path.exists(child_data)

        data[oc_pair] = parent_data, child_data, tl
    print("web OC PARALLEL SENTENCES FROM:")
    for lang_pair, paths in data.items():
        print(f"\t-{lang_pair}:{paths}")
    print("\n")
    return data

def read_tokenizer_train_paths(
    datasets:list, # list of (data folder, pl, cl, tl) tuples
    sc_model_id_prefix=None
):
    _validate_sets(datasets)
    train_paths_by_scenario = {}
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)
        paths = {pl: [], cl:[], tl:[]}
        for pair in [
            # (pl, cl), #Let's ignore this dataset and just use the NMT data for tokenization.
            (pl, tl), (cl, tl)
        ]:
            src_lang, tgt_lang = pair

            pair_str = f"{src_lang}-{tgt_lang}"
            directory = os.path.join(data_folder, pair_str)
            print("DIRECTORY", directory)
            assert os.path.exists(directory)

            src_file = os.path.join(directory, f"train.{src_lang}.txt")
            tgt_file = os.path.join(directory, f"train.{tgt_lang}.txt")
            
            if sc_model_id_prefix and (src_lang, tgt_lang) == (pl, tl):
                sc_model_id = _get_one_sc_model_id(sc_model_id_prefix, (pl, cl, tl))
                src_file += "." + sc_model_id
            
            assert os.path.exists(src_file)
            assert os.path.exists(tgt_file)

            paths[src_lang].append(src_file)
            paths[tgt_lang].append(tgt_file)
        
        scenario = (pl, cl, tl)
        assert scenario not in train_paths_by_scenario
        train_paths_by_scenario[scenario] = paths
    return train_paths_by_scenario

def _get_one_sc_model_id(sc_model_id_prefix, scenario):
    if not isinstance(sc_model_id_prefix, str):
        raise ValueError("sc_model_id_prefix must be a string!")
    if not isinstance(scenario, tuple):
        raise ValueError("scenario must be a tuple!")
    if not len(scenario) == 3:
        raise ValueError("scenario must be a tuple of length 3!")
    for item in scenario:
        if not isinstance(item, str):
            raise ValueError("scenario must be a tuple of pl, tl, cl languages (strings)!")
    
    pl, cl, tl = scenario
    return f"{sc_model_id_prefix}_{pl}-{cl}"

def get_set(
    datasets:list, # list of (data folder, pl, cl, tl) tuples
    scenario:tuple,
    pair:tuple,
    div:str,
):
    if div not in ["train", "val", "test"]:
        raise ValueError("div must be 'train', 'val', or 'test'.")
    _validate_sets(datasets)
    
    print(f"utilities.read_data.get_set: Getting {div} set --")
    for data_folder, pl, cl, tl in datasets:
        data_folder = set_vars_in_path(data_folder)
        if (pl, cl, tl) == scenario:
            if pair not in [(pl, cl), (pl, tl), (cl, tl)]:
                raise ValueError(f"Invalid pair {pair} for scenario {scenario}.")
            src, tgt = pair
            pair_dir = os.path.join(data_folder, f"{src}-{tgt}")
            src_file = os.path.join(pair_dir, f"{div}.{src}.txt")
            tgt_file = os.path.join(pair_dir, f"{div}.{tgt}.txt")

            assert os.path.exists(src_file)
            assert os.path.exists(tgt_file)
            
            print(f"\tsrc: {src_file}")
            print(f"\ttgt: {tgt_file}")

            return src_file, tgt_file

    raise ValueError(f"Scenario {scenario} could not be found.")

def _validate_sets(datasets):
    scenarios = set()
    pl_cls = set()
    for item in datasets:
        _validate_item(item)
        folder, pl, cl, tl = item
        if (pl, cl, tl) in scenarios:
            raise ValueError(f"Provided duplicate scenario {(pl, cl, tl)}.")
        if (pl, cl) in pl_cls:
            raise ValueError(f"Provided duplicate PL/CL in data: {(pl, cl)}")
        scenarios.add((pl, cl, tl))
        pl_cls.add((pl, cl))

def _validate_item(item):
    if not (isinstance(item, list) or isinstance(item, tuple)):
        raise ValueError("datasets must be a list of tuples/lists!")
    if len(item) != 4:
        raise ValueError("Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl")
