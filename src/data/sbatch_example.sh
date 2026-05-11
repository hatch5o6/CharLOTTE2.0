#!/bin/bash
#SBATCH --time=4:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=32000M 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %pbickel@byu.edu
#SBATCH --output %j_%x.out
#SBATCH --job-name=clean_charlotte_2.0_data


# MUST RUN download_data.sh BEFORE THIS!
bash clean_data.sh