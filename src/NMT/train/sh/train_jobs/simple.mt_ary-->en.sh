python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mt_ary-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.mt_ary-->en.out"