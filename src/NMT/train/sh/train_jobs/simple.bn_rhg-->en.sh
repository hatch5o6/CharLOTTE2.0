python -m NMT.train.train_jobs \
    --config "src/configs/experiments/bn_rhg-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.bn_rhg-->en.out"