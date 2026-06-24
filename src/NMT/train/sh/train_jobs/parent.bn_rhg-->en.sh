python -m NMT.train.train_jobs \
    --config "src/configs/experiments/bn_rhg-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.bn_rhg-->en.out"