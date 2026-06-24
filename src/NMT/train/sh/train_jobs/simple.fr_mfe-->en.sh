python -m NMT.train.train_jobs \
    --config "src/configs/experiments/fr_mfe-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.fr_mfe-->en.out"