python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mtx_aeb-->enx.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.mtx_aeb-->enx.out"