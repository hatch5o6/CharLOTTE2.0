#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output src/OC/extract_cognates/%j_%x.out
#SBATCH --job-name=test_BLI.mfe.sh
#SBATCH --qos matrix

# SPANISH: dccuchile/bert-base-spanish-wwm-cased (BETO) or bertin-project/bertin-roberta-base-spanish (BERTIN)
# FRENCH: almanach/camembert-base (#TODO would need to change mask token from [MASK] to <mask>)

# TARGET_FILE_PATH: Path to target language monolingual data, one sentence per line
# hf_model_name: Can be a pretrained HuggingFace model identifier or a path to a locally trained mask filling model.
# OUTDIR: Directory where lexicons will be saved
# threshold (default:0.5): minimum normalized edit distance (NED) between a word and translation candidate.
# iterations (default:3): maximum number of times a single (sentence, word) instance will be processed
# batch_size: number of sentences processed at once by the mask filling pipeline.
# lang: Target language code, only required for naming purposes

source /home/hatch5o6/CharLOTTE2.0/src/BLI/.venv/bin/activate

TARGET_FILE_PATH=/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/MauritianCreole/Comprised/mfe.7k.txt
hf_model_name=almanach/camembert-base
OUTDIR=/home/hatch5o6/nobackup/archive/CharLOTTE2.0/test_bli_mfe
threshold=0.5
iterations=3
batch_size=100
lang=mfe
mask_token="<mask>"

export TRANSFORMERS_OFFLINE=1  

cd src/BLI
PYTHONPATH=. python bli/scripts/basic.py \
    --TARGET_FILE_PATH $TARGET_FILE_PATH \
    --hf_model_name $hf_model_name \
    --OUTDIR $OUTDIR \
    --threshold $threshold \
    --iterations $iterations \
    --batch_size $batch_size \
    --lang $lang \
    --mask_token $mask_token
