python -m NMT.train.train_jobs \
    --config "src/configs/test.es_an-->en.yaml" \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/train_jobs/parent.es_an-->en.out"