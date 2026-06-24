python -m NMT.train.train_jobs \
    --config "src/configs/experiments/uz_kaa-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.uz_kaa-->en.out"