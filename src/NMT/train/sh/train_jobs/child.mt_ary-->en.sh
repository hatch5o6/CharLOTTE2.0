python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mt_ary-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.mt_ary-->en.out"