#!/bin/bash

#SBATCH --time=2:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output src/OC/train/tests/%j_%x.out
#SBATCH --job-name=test_eval.fuzz.out
#SBATCH --qos=matrix

python -m OC.train.train \
    --config "src/configs/test/test.xx_yy-->zz.oc.yaml" \
    --mode EVAL \
    --oc_model_id this-fuzz-model-is-made-with-train-py \
    --oc_method fuzz \
    --oc_train src/OC/train/tests/fixtures/train/train.monolingual.txt \
    --oc_val src/OC/train/tests/fixtures/train/val.monolingual.txt \
    --oc_scenario "('xx', 'yy', 'zz')"