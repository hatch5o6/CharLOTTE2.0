python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mty_aeb-->eny.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.mty_aeb-->eny.out"