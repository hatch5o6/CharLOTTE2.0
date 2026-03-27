#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000M 
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output src/OC/extract_cognates/%j_%x.out
#SBATCH --job-name=test_cognates_from_nld
#SBATCH --qos matrix

python src/OC/extract_cognates/CognatesFromNLD.py
