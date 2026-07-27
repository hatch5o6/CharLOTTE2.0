python -m NMT.train.train_jobs \
    --config "src/configs/experiments/amx_tix-->enx.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.amx_tix-->enx.out"