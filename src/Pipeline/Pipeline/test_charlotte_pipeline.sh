#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000M 
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/hatch5o6/CharLOTTE2.0/src/Pipeline/Pipeline/%j_%x.out
#SBATCH --job-name=test_charlotte_pipeline
#SBATCH --qos matrix

python src/Pipeline/Pipeline/pipeline.py -m charlotte
