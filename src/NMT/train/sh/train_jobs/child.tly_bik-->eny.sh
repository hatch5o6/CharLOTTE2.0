python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tly_bik-->eny.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.tly_bik-->eny.out"