python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mtx_aeb-->enx.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.mtx_aeb-->enx.out"