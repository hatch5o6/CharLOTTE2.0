import os

from sloth_hatch.sloth import create_directory

SUB_DIRS = ["checkpoints", "data", "predictions", "logs", "tb"]

def get_exp_dir(config):
    exp_dir = os.path.join(config["save"], config["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def get_task_dir(exp_dir, task="OC"):
    if task not in ["NMT", "OC"]:
        raise ValueError("task must be 'OC' or 'NMT'!")
    task_dir = os.path.join(exp_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir

def get_train_dir(task_dir, name, create=True):
    train_dir = os.path.join(task_dir, name)
    os.makedirs(train_dir, exist_ok=True)
    subdirs = _get_train_subdirs(train_dir)
    if create:
        for sub_d in subdirs.values():
            create_directory(sub_d)
    else:
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"TRAIN dir does not exist: `{train_dir}`")
        for sub_d in subdirs.values():
            if not os.path.exists(sub_d):
                raise FileNotFoundError(f"SUB TRAIN dir does not exist: `{sub_d}`")
    return train_dir, subdirs

def _get_train_subdirs(d):
    subdirs = {}
    for sub_d in SUB_DIRS:
        assert sub_d not in subdirs
        subdirs[sub_d] = os.path.join(d, sub_d)
    return subdirs
