python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tlx_bik-->enx.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.tlx_bik-->enx.out"