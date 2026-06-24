python -m NMT.train.train_jobs \
    --config "src/configs/experiments/am_ti-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.am_ti-->en.out"