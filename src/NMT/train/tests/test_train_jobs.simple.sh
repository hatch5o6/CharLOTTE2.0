#!/bin/bash

#SBATCH --time=12:00:00   # walltime.  hours:minutes:seconds
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/CharLOTTE2.0/src/NMT/train/tests/%j_%x.out
#SBATCH --job-name=test_train_jobs.simple.sh
#SBATCH --qos=matrix

python src/NMT/train/tests/test_train_jobs.py --tests reproduce_simple
