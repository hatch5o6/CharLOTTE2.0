import functools
import lightning as L
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks import Callback
import subprocess
import os
import json

def log_mode_call(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        rank_zero_info(f"\n---------------- Calling {f.__name__} ----------------")
        rank_zero_info("CONFIG:")
        config = args[0]
        for k, v in config.items():
            rank_zero_info(f"\t{k}=`{v}`")
        rank_zero_info("\nOTHER ARGS:")
        for arg in args[1:]:
            rank_zero_info(f"\t{arg}")
        rank_zero_info("\nKWARGS:")
        for k, v in kwargs.items():
            rank_zero_info(f"\t{k}=`{v}`")
        rank_zero_info("\n\n")
        result = f(*args, **kwargs)
        rank_zero_info(f"\n---------------- Ending {f.__name__} ----------------")
        return result
    return wrapper

def call_seed_everything(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        config = args[0]
        seed = config["seed"]
        rank_zero_info(f"Seeding everything with seed={seed}")
        L.seed_everything(seed, workers=True)
        result = f(*args, **kwargs)
        return result
    return wrapper

def call_nvidia_smi(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        subprocess_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(subprocess_result.stdout)
        result = f(*args, **kwargs)
        return result
    return wrapper

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("######          TRAINING STARTED         ######")
        rank_zero_info("###############################################")
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("#######          TRAINING ENDED         #######")
        rank_zero_info("###############################################")

