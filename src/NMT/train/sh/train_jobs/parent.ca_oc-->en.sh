python -m NMT.train.train_jobs \
    --config "src/configs/experiments/ca_oc-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.ca_oc-->en.out"