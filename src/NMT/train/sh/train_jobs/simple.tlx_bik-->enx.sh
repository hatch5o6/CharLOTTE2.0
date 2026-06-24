python -m NMT.train.train_jobs \
    --config "src/configs/experiments/tlx_bik-->enx.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.tlx_bik-->enx.out"