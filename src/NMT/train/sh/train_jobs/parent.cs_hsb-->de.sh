python -m NMT.train.train_jobs \
    --config "src/configs/experiments/cs_hsb-->de.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.cs_hsb-->de.out"