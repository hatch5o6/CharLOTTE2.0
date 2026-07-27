python -m NMT.train.train_jobs \
    --config "src/configs/experiments/ca_oc-->en.yaml" \
    --nmt_corpus parent \
    --REVERSE \
    --HPC > "src/NMT/train/sh/train_jobs/parent.en-->ca_oc.out"