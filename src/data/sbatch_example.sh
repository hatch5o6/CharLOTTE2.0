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
#SBATCH --job-name=ngram_correspondences


# MUST RUN download_data.sh BEFORE THIS!
python src/OC/ngram_correspondences/ngram_correspondences.py -l uz-kaa -t -f 10
