python -m NMT.train.train_jobs \
    --config "src/configs/experiments/cs_hsb-->de.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.cs_hsb-->de.out"