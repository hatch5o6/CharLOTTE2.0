python -m NMT.train.train_jobs \
    --config "src/configs/experiments/ca_oc-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.ca_oc-->en.out"