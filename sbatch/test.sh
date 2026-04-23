#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/CharLOTTE2.0/code/OC/out/slurm/%j_%x.out
#SBATCH --job-name=test_OC
#SBATCH --qos=matrix

python Pipeline/src/clean_slurm_outputs.py

nvidia-smi

srun python OC/src/train.py \
    --config configs/test.yaml

python Pipeline/src/clean_slurm_outputs.py
