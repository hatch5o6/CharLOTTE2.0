python -m NMT.train.train_jobs \
    --config "src/configs/experiments/uz_kaa-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.uz_kaa-->en.out"