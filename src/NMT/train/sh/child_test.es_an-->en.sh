python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m TRAIN \
    --nmt_corpus child \
    --fine_tune \
    --HPC