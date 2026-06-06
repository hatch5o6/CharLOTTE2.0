python -m NMT.train.train_jobs \
    --config "src/configs/test.es_an-->en.yaml" \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/train_jobs/simple.es_an-->en.out"