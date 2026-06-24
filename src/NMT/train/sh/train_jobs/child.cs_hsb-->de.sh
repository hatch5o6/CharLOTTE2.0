python -m NMT.train.train_jobs \
    --config "src/configs/experiments/cs_hsb-->de.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.cs_hsb-->de.out"