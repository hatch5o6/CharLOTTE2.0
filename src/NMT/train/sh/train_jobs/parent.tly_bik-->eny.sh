python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tly_bik-->eny.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.tly_bik-->eny.out"