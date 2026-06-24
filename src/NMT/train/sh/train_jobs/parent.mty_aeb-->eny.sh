python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mty_aeb-->eny.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.mty_aeb-->eny.out"