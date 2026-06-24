python -m NMT.train.train_jobs \
    --config "src/configs/experiments/fr_mfe-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.fr_mfe-->en.out"