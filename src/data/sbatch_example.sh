#!/bin/bash
#SBATCH --time=6:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=32000M 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %pbickel@byu.edu
#SBATCH --output %j_%x.out
#SBATCH --job-name=uz-kaa-char1-comp

python src/OC/ngram_correspondences/ngram_correspondences.py -l uz-kaa
# python src/OC/ngram_correspondences/ngram_correspondences.py -l bn-rhg -t -m nld -n 80 -cm fuzz
# python src/OC/ngram_correspondences/ngram_correspondences.py -l uz-kaa -t -m nld -n 42
# python src/OC/ngram_correspondences/ngram_correspondences.py -l ca-oc -t -n 80 -m nld -cm charlotte
# python src/OC/ngram_correspondences/ngram_correspondences.py -l ca-oc -t -m chrf -cm fuzz -n 40