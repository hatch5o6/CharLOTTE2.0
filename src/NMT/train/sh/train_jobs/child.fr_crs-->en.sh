python -m NMT.train.train_jobs \
    --config "src/configs/experiments/fr_crs-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.fr_crs-->en.out"