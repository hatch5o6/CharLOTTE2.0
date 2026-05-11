import os
from coolname import generate_slug

from utilities.utilities import set_vars_in_path

def get_new_name(model_name_cache="${DATA_HOME}/USED_MODEL_NAMES/cache.txt"):
    model_name_cache = set_vars_in_path(model_name_cache)
    if not os.path.exists(model_name_cache):
        _write_cache(model_name_cache, [])
    
    cache = _read_cache(model_name_cache)

    model_name = None
    while model_name == None or model_name in cache:
        model_name = generate_slug(4)
    cache.append(model_name)
    _write_cache(model_name_cache, cache)
    
    return model_name

def _read_cache(f):
    with open(f) as inf:
        cache = [l.strip() for l in inf.readlines()]
    if cache == [""]:
        cache = []
    return cache

def _write_cache(f, cache):
    parent_dir = "/".join(f.split("/")[:-1])
    os.makedirs(parent_dir, exist_ok=True)
    with open(f, "w") as outf:
        outf.write("\n".join(cache) + "\n")
