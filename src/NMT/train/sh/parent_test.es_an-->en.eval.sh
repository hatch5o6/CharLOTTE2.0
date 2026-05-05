python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus parent \
    --HPC