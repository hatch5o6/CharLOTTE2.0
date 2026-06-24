python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mt_ary-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.mt_ary-->en.out"