python -m NMT.train.train_jobs \
    --config "src/configs/experiments/bn_rhg-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.bn_rhg-->en.out"