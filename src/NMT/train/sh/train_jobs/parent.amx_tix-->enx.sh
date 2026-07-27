python -m NMT.train.train_jobs \
    --config "src/configs/experiments/amx_tix-->enx.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.amx_tix-->enx.out"