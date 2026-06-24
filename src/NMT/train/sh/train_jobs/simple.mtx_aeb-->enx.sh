python -m NMT.train.train_jobs \
    --config "src/configs/experiments/mtx_aeb-->enx.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.mtx_aeb-->enx.out"