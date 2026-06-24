python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tly_bik-->eny.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.tly_bik-->eny.out"