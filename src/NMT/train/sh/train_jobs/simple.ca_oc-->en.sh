python -m NMT.train.train_jobs \
    --config "src/configs/experiments/ca_oc-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.ca_oc-->en.out"