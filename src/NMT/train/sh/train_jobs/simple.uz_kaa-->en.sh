python -m NMT.train.train_jobs \
    --config "src/configs/experiments/uz_kaa-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.uz_kaa-->en.out"