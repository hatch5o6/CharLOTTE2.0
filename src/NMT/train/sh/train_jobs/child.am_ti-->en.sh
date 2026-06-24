python -m NMT.train.train_jobs \
    --config "src/configs/experiments/am_ti-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.am_ti-->en.out"