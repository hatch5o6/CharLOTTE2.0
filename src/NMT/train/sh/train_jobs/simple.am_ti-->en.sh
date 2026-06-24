python -m NMT.train.train_jobs \
    --config "src/configs/experiments/am_ti-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.am_ti-->en.out"