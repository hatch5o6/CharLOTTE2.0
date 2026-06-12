python -m NMT.train.train_jobs \
    --config "src/configs/test.es_an-->en.yaml" \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/train_jobs/child.es_an-->en.out"