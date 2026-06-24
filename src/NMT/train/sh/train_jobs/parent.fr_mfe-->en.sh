python -m NMT.train.train_jobs \
    --config "src/configs/experiments/fr_mfe-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.fr_mfe-->en.out"