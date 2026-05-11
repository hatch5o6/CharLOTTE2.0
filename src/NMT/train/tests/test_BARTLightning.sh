#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%u@byu.edu
#SBATCH --output=src/NMT/train/tests/%j_%x.out
#SBATCH --job-name=test_BARTLighting
#SBATCH --qos=cs

pytest src/NMT/train/tests/test_BARTLightning.py
