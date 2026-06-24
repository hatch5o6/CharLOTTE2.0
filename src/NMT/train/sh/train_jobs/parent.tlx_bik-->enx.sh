python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tlx_bik-->enx.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.tlx_bik-->enx.out"