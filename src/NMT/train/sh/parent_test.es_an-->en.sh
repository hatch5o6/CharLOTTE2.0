python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m TRAIN \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/parent_test.es_an-->en.out"