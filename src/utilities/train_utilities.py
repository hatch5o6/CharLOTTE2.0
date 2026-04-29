import functools
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks import Callback
import subprocess
import os

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

def call_nvidia_smi(f):
    @functools.wraps(f)
    def wrapper(*args):
        subprocess_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(subprocess_result.stdout)
        result = f(*args)
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
