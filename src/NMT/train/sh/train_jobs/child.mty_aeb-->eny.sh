python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mty_aeb-->eny.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.mty_aeb-->eny.out"